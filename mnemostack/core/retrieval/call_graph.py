"""Lightweight call graph for dependency-aware retrieval.

3 node types: File, Function, Class
3 edge types: CALLS, IMPORTS_FROM, CONTAINS

Stored in SQLite. Supports 2-hop BFS expansion for cross-file dependency chains.
Used to enrich retrieval results with related code that pure similarity would miss.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import deque
from enum import Enum
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from mnemostack.config.settings import settings

# Thread lock for CallGraph SQLite operations (graph.db is accessed from
# the file watcher's background thread via reindex_file -> remove_file).
_graph_lock = threading.Lock()

# Cached Python parser (Language + Parser are expensive to construct).
# Parser.parse() is NOT thread-safe — _py_parse_lock must be held during parse.
_PY_LANGUAGE: Language | None = None
_PY_PARSER: Parser | None = None
_parser_init_lock = threading.Lock()
_py_parse_lock = threading.Lock()


def _get_python_parser() -> tuple[Parser, threading.Lock]:
    """Return (Parser, lock). Lock must be held while calling parser.parse()."""
    global _PY_LANGUAGE, _PY_PARSER
    if _PY_PARSER is None:
        with _parser_init_lock:
            if _PY_PARSER is None:
                _PY_LANGUAGE = Language(tspython.language())
                _PY_PARSER = Parser(_PY_LANGUAGE)
    return _PY_PARSER, _py_parse_lock


class NodeType(str, Enum):
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"


class EdgeType(str, Enum):
    CALLS = "calls"
    IMPORTS_FROM = "imports_from"
    CONTAINS = "contains"


class CallGraph:
    """SQLite-backed lightweight dependency graph."""

    def __init__(self, store_dir: Path | None = None):
        self._store_dir = store_dir or settings.store.base_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._store_dir / "graph.db"
        self._db: sqlite3.Connection | None = None

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
        return self._db

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                qualified_name TEXT UNIQUE NOT NULL,
                node_type TEXT NOT NULL,
                file_path TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
            CREATE INDEX IF NOT EXISTS idx_nodes_qname ON nodes(qualified_name);

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES nodes(id),
                target_id INTEGER NOT NULL REFERENCES nodes(id),
                edge_type TEXT NOT NULL,
                UNIQUE(source_id, target_id, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        """)

    def add_node(self, qualified_name: str, node_type: NodeType, file_path: str) -> int:
        """Add a node (or get existing). Returns node ID.

        Does NOT auto-commit. Caller must call commit() to persist changes.
        """
        with _graph_lock:
            cursor = self.db.execute(
                "SELECT id FROM nodes WHERE qualified_name = ?", (qualified_name,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]

            cursor = self.db.execute(
                "INSERT INTO nodes (qualified_name, node_type, file_path) VALUES (?, ?, ?)",
                (qualified_name, node_type.value, file_path),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def add_edge(self, source_qname: str, target_qname: str, edge_type: EdgeType) -> None:
        """Add an edge between two nodes (by qualified name). No-op if edge exists."""
        with _graph_lock:
            source = self.db.execute(
                "SELECT id FROM nodes WHERE qualified_name = ?", (source_qname,)
            ).fetchone()
            target = self.db.execute(
                "SELECT id FROM nodes WHERE qualified_name = ?", (target_qname,)
            ).fetchone()
            if not source or not target:
                return

            self.db.execute(
                """INSERT OR IGNORE INTO edges (source_id, target_id, edge_type)
                   VALUES (?, ?, ?)""",
                (source[0], target[0], edge_type.value),
            )

    def commit(self) -> None:
        """Commit pending changes. Call after batch add_node/add_edge operations."""
        with _graph_lock:
            self.db.commit()

    def remove_file(self, file_path: str) -> None:
        """Remove all nodes and edges associated with a file."""
        with _graph_lock:
            node_ids = [
                row[0]
                for row in self.db.execute(
                    "SELECT id FROM nodes WHERE file_path = ?", (file_path,)
                ).fetchall()
            ]
            if not node_ids:
                return

            placeholders = ",".join("?" * len(node_ids))
            self.db.execute(
                f"DELETE FROM edges WHERE source_id IN ({placeholders})"
                f" OR target_id IN ({placeholders})",
                node_ids + node_ids,
            )
            self.db.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", node_ids)
            self.db.commit()

    def get_neighbors(
        self,
        qualified_name: str,
        hops: int = 2,
        direction: str = "outgoing",
    ) -> list[str]:
        """BFS expansion from a node. Returns qualified names of reachable nodes.

        Args:
            qualified_name: Starting node.
            hops: Maximum BFS depth (default 2).
            direction: 'outgoing', 'incoming', or 'both'.

        Returns:
            List of qualified names reachable within `hops` (excludes start node).
        """
        with _graph_lock:
            start = self.db.execute(
                "SELECT id FROM nodes WHERE qualified_name = ?", (qualified_name,)
            ).fetchone()
            if not start:
                return []

            visited: set[int] = {start[0]}
            queue: deque[tuple[int, int]] = deque([(start[0], 0)])
            result_ids: list[int] = []

            while queue:
                node_id, depth = queue.popleft()
                if depth >= hops:
                    continue

                neighbors = self._get_adjacent_unlocked(node_id, direction)
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        result_ids.append(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

            if not result_ids:
                return []

            placeholders = ",".join("?" * len(result_ids))
            rows = self.db.execute(
                f"SELECT qualified_name FROM nodes WHERE id IN ({placeholders})",
                result_ids,
            ).fetchall()
            return [row[0] for row in rows]

    def _get_adjacent_unlocked(self, node_id: int, direction: str) -> list[int]:
        """Get adjacent node IDs. Must be called while holding _graph_lock."""
        ids: list[int] = []
        if direction in ("outgoing", "both"):
            rows = self.db.execute(
                "SELECT target_id FROM edges WHERE source_id = ?", (node_id,)
            ).fetchall()
            ids.extend(row[0] for row in rows)
        if direction in ("incoming", "both"):
            rows = self.db.execute(
                "SELECT source_id FROM edges WHERE target_id = ?", (node_id,)
            ).fetchall()
            ids.extend(row[0] for row in rows)
        return ids

    @property
    def node_count(self) -> int:
        with _graph_lock:
            row = self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return row[0] if row else 0

    @property
    def edge_count(self) -> int:
        with _graph_lock:
            row = self.db.execute("SELECT COUNT(*) FROM edges").fetchone()
        return row[0] if row else 0

    def has_node(self, qualified_name: str) -> bool:
        """Check if a node exists by qualified name."""
        with _graph_lock:
            row = self.db.execute(
                "SELECT 1 FROM nodes WHERE qualified_name = ?", (qualified_name,)
            ).fetchone()
        return row is not None

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None


# --- Python-specific call/import extraction ---


def build_graph_for_python_file(
    file_path: Path,
    source: bytes | None = None,
    graph: CallGraph | None = None,
) -> CallGraph:
    """Parse a Python file and populate the call graph with nodes and edges.

    Extracts:
    - File node
    - Function/class CONTAINS edges from file
    - IMPORTS_FROM edges between files
    - CALLS edges between functions (within-file call sites)
    """
    if graph is None:
        graph = CallGraph()
    if source is None:
        source = file_path.read_bytes()

    parser, parse_lock = _get_python_parser()
    with parse_lock:
        tree = parser.parse(source)
    root = tree.root_node

    fpath_str = str(file_path)

    # Add file node
    graph.add_node(fpath_str, NodeType.FILE, fpath_str)

    # Track defined symbols for CONTAINS edges
    defined_functions: list[str] = []
    defined_classes: list[str] = []

    for child in root.children:
        if child.type == "function_definition":
            name = _py_node_name(child, source)
            qname = f"{fpath_str}::{name}"
            graph.add_node(qname, NodeType.FUNCTION, fpath_str)
            graph.add_edge(fpath_str, qname, EdgeType.CONTAINS)
            defined_functions.append(name)

        elif child.type == "class_definition":
            class_name = _py_node_name(child, source)
            class_qname = f"{fpath_str}::{class_name}"
            graph.add_node(class_qname, NodeType.CLASS, fpath_str)
            graph.add_edge(fpath_str, class_qname, EdgeType.CONTAINS)
            defined_classes.append(class_name)

            # Methods
            body = child.child_by_field_name("body")
            if body:
                for method in body.children:
                    if method.type == "function_definition":
                        mname = _py_node_name(method, source)
                        mqname = f"{fpath_str}::{class_name}.{mname}"
                        graph.add_node(mqname, NodeType.FUNCTION, fpath_str)
                        graph.add_edge(class_qname, mqname, EdgeType.CONTAINS)

        elif child.type in ("import_statement", "import_from_statement"):
            _extract_python_import(child, source, fpath_str, graph)

    # Extract call sites (function calls within the file)
    _extract_python_calls(root, source, fpath_str, defined_functions, defined_classes, graph)

    graph.commit()
    return graph


def _py_node_name(node: Node, source: bytes) -> str:
    name_node = node.child_by_field_name("name")
    if name_node:
        return source[name_node.start_byte:name_node.end_byte].decode()
    return "<anonymous>"


def _extract_python_import(
    node: Node, source: bytes, file_path: str, graph: CallGraph
) -> None:
    """Extract IMPORTS_FROM edges from import statements.

    Note: creates phantom FILE nodes for imported modules using a guessed path
    (dotted.module -> dotted/module.py). These may not correspond to real files
    on disk — they exist solely to enable dependency traversal in the graph.
    """
    if node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        if module_node:
            module_name = source[module_node.start_byte:module_node.end_byte].decode()
            module_path = module_name.replace(".", "/") + ".py"
            graph.add_node(module_path, NodeType.FILE, module_path)
            graph.add_edge(file_path, module_path, EdgeType.IMPORTS_FROM)


def _extract_python_calls(
    root: Node,
    source: bytes,
    file_path: str,
    defined_functions: list[str],
    defined_classes: list[str],
    graph: CallGraph,
) -> None:
    """Walk the AST to find function call sites and add CALLS edges."""
    # Find all call expressions
    calls = _find_nodes_by_type(root, "call")

    for call in calls:
        func_node = call.child_by_field_name("function")
        if not func_node:
            continue

        call_name = source[func_node.start_byte:func_node.end_byte].decode()

        # Determine the calling context (which function contains this call)
        caller = _find_enclosing_function(call, source, file_path)
        if not caller:
            continue

        # If the called function is defined in this file, add a CALLS edge
        # Simple name match (doesn't resolve imports — that's a static analysis problem)
        if call_name in defined_functions:
            target_qname = f"{file_path}::{call_name}"
            graph.add_edge(caller, target_qname, EdgeType.CALLS)
        elif "." in call_name:
            # Method call like self.method() or obj.method()
            parts = call_name.split(".")
            method = parts[-1]
            for cls in defined_classes:
                candidate = f"{file_path}::{cls}.{method}"
                if graph.has_node(candidate):
                    graph.add_edge(caller, candidate, EdgeType.CALLS)
                    break


def _find_enclosing_function(node: Node, source: bytes, file_path: str) -> str | None:
    """Walk up the tree to find the enclosing function/method qualified name."""
    current = node.parent
    func_name = None
    class_name = None
    while current:
        if current.type == "function_definition" and func_name is None:
            func_name = _py_node_name(current, source)
        elif current.type == "class_definition" and class_name is None:
            class_name = _py_node_name(current, source)
        current = current.parent

    if func_name is None:
        return None
    if class_name:
        return f"{file_path}::{class_name}.{func_name}"
    return f"{file_path}::{func_name}"


def _find_nodes_by_type(root: Node, node_type: str) -> list[Node]:
    """Recursively find all nodes of a given type."""
    results: list[Node] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == node_type:
            results.append(node)
        stack.extend(node.children)
    return results
