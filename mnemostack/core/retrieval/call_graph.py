"""Lightweight call graph for dependency-aware retrieval.

4 node types: File, Function, Class, External (an installed dependency)
4 edge types: CALLS, IMPORTS_FROM, CONTAINS, INHERITS

Stored in SQLite. Supports 2-hop BFS expansion for cross-file dependency chains.
Used to enrich retrieval results with related code that pure similarity would miss.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
from collections import deque
from collections.abc import Iterable
from enum import Enum
from functools import lru_cache
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from mnemostack.config.settings import settings
from mnemostack.core.retrieval.import_resolver import (
    ImportRecord,
    extract_imports,
    find_import_root,
    import_table_from_records,
    resolve_module_file,
)

# Thread lock for CallGraph SQLite operations (graph.db is accessed from
# the file watcher's background thread via reindex_file -> remove_file).
_graph_lock = threading.Lock()

# Cached Python parser (Language + Parser are expensive to construct).
# Parser.parse() is NOT thread-safe — _py_parse_lock must be held during parse.
_PY_LANGUAGE: Language | None = None
_PY_PARSER: Parser | None = None
_parser_init_lock = threading.Lock()
_py_parse_lock = threading.Lock()


def _repo_of(file_path: str) -> str:
    """Identity of the repo a file belongs to.

    The basename of the file's nearest ``.git`` ancestor, or of its import root
    when the file isn't inside a git repo. Two files with different repo tags on
    an edge mean the edge crosses a repo boundary.
    """
    p = Path(file_path)
    parts = p.parts
    if "site-packages" in parts:
        # An installed dependency: tag it by the package it belongs to, not by
        # the venv directory it happens to live in.
        i = parts.index("site-packages")
        if i + 1 < len(parts):
            return f"pkg:{Path(parts[i + 1]).stem}"
    for parent in p.parents:
        if (parent / ".git").exists():
            return parent.name
    root = find_import_root(p)
    return root.name or str(root)


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
    EXTERNAL = "external"


class EdgeType(str, Enum):
    CALLS = "calls"
    IMPORTS_FROM = "imports_from"
    CONTAINS = "contains"
    INHERITS = "inherits"


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
                file_path TEXT NOT NULL,
                repo TEXT
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
        # Migrate graph.db created before the repo column existed. CREATE TABLE
        # IF NOT EXISTS above leaves an old table untouched, so add the column.
        cols = {row[1] for row in self.db.execute("PRAGMA table_info(nodes)")}
        if "repo" not in cols:
            self.db.execute("ALTER TABLE nodes ADD COLUMN repo TEXT")

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
                "INSERT INTO nodes (qualified_name, node_type, file_path, repo) "
                "VALUES (?, ?, ?, ?)",
                (qualified_name, node_type.value, file_path, _repo_of(file_path)),
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

    def importer_files(self, file_path: str) -> list[str]:
        """Distinct file paths that have a graph edge pointing into ``file_path``.

        These are the files whose cross-file edges (IMPORTS_FROM / CALLS) target a
        node in ``file_path``. After ``file_path`` is reindexed its nodes get fresh
        ids, so these importers must be re-linked to re-establish the incoming
        edges that ``remove_file`` dropped.
        """
        with _graph_lock:
            rows = self.db.execute(
                """SELECT DISTINCT src.file_path
                   FROM edges e
                   JOIN nodes tgt ON e.target_id = tgt.id
                   JOIN nodes src ON e.source_id = src.id
                   WHERE tgt.file_path = ? AND src.file_path != ?""",
                (file_path, file_path),
            ).fetchall()
        return [row[0] for row in rows]

    def get_neighbors(
        self,
        qualified_name: str,
        hops: int = 2,
        direction: str = "outgoing",
        edge_types: Iterable[EdgeType] | None = None,
    ) -> list[str]:
        """BFS expansion from a node. Returns qualified names of reachable nodes.

        Args:
            qualified_name: Starting node.
            hops: Maximum BFS depth (default 2).
            direction: 'outgoing', 'incoming', or 'both'.
            edge_types: If given, only traverse edges of these types. Defaults to
                all edge types. Restrict to CALLS/IMPORTS_FROM for true dependency
                chains — traversing CONTAINS would reach every sibling symbol via
                the shared file node.

        Returns:
            List of qualified names reachable within `hops` (excludes start node).
        """
        edge_type_values = [e.value for e in edge_types] if edge_types is not None else None
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

                neighbors = self._get_adjacent_unlocked(node_id, direction, edge_type_values)
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

    def _get_adjacent_unlocked(
        self,
        node_id: int,
        direction: str,
        edge_type_values: list[str] | None = None,
    ) -> list[int]:
        """Get adjacent node IDs. Must be called while holding _graph_lock."""
        ids: list[int] = []
        type_clause = ""
        type_params: list[str] = []
        if edge_type_values:
            placeholders = ",".join("?" * len(edge_type_values))
            type_clause = f" AND edge_type IN ({placeholders})"
            type_params = edge_type_values
        if direction in ("outgoing", "both"):
            rows = self.db.execute(
                f"SELECT target_id FROM edges WHERE source_id = ?{type_clause}",
                (node_id, *type_params),
            ).fetchall()
            ids.extend(row[0] for row in rows)
        if direction in ("incoming", "both"):
            rows = self.db.execute(
                f"SELECT source_id FROM edges WHERE target_id = ?{type_clause}",
                (node_id, *type_params),
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

    def node_repo(self, qualified_name: str) -> str | None:
        """Repo a node belongs to, or None if the node doesn't exist."""
        with _graph_lock:
            row = self.db.execute(
                "SELECT repo FROM nodes WHERE qualified_name = ?", (qualified_name,)
            ).fetchone()
        return row[0] if row else None

    def source_files(self) -> list[str]:
        """Every indexed source file that has a graph builder, by path."""
        with _graph_lock:
            rows = self.db.execute(
                "SELECT file_path FROM nodes WHERE node_type = ?", (NodeType.FILE.value,)
            ).fetchall()
        return [row[0] for row in rows]

    def clear_external_imports(self, file_path: str) -> None:
        """Drop a file's boundary edges, before its imports are resolved again.

        A dependency that was outside the index can be indexed later; without
        this the stale boundary would keep claiming code we can now see.
        """
        with _graph_lock:
            self.db.execute(
                "DELETE FROM edges WHERE edge_type = ? AND source_id = "
                "(SELECT id FROM nodes WHERE qualified_name = ?) AND target_id IN "
                "(SELECT id FROM nodes WHERE node_type = ?)",
                (EdgeType.IMPORTS_FROM.value, file_path, NodeType.EXTERNAL.value),
            )

    def external_imports(self, file_path: str) -> list[str]:
        """Files a file imports that exist on disk but outside the index.

        External nodes hang off the importing *file*, so this is the boundary
        report for any chunk that file contains.
        """
        with _graph_lock:
            rows = self.db.execute(
                "SELECT t.qualified_name FROM edges e "
                "JOIN nodes s ON s.id = e.source_id "
                "JOIN nodes t ON t.id = e.target_id "
                "WHERE s.qualified_name = ? AND e.edge_type = ? AND t.node_type = ?",
                (file_path, EdgeType.IMPORTS_FROM.value, NodeType.EXTERNAL.value),
            ).fetchall()
        return sorted(row[0] for row in rows)

    def find_indexed_module(self, module: str) -> str | None:
        """Path of an indexed file for a dotted module, from any repo in the graph.

        The disk resolver only searches the importing file's own repo, so an
        import that crosses into another indexed repo resolves here instead:
        match a file whose path ends in ``a/b/c.py`` or ``a/b/c/__init__.py``.
        Returns None unless exactly one file matches — an ambiguous name (the
        same module path in two repos) stays quiet rather than guessing.
        """
        if not module:
            return None
        rel = "/".join(module.split("."))
        with _graph_lock:
            rows = self.db.execute(
                "SELECT qualified_name FROM nodes WHERE node_type = ? "
                "AND (file_path LIKE ? OR file_path LIKE ?) LIMIT 2",
                (NodeType.FILE.value, f"%/{rel}.py", f"%/{rel}/__init__.py"),
            ).fetchall()
        return rows[0][0] if len(rows) == 1 else None

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None


# --- Python-specific call/import extraction ---


@lru_cache(maxsize=256)
def _site_packages_dirs(repo_root: Path) -> tuple[Path, ...]:
    """site-packages directories of the virtualenvs belonging to a repo.

    An installed dependency's real source already sits on disk, so an import
    that leaves the repo can still be pointed at actual code. Looks at the repo's
    own ``.venv``/``venv`` and at an active ``VIRTUAL_ENV``.
    """
    candidates = [repo_root / ".venv", repo_root / "venv"]
    active = os.environ.get("VIRTUAL_ENV")
    if active:
        candidates.append(Path(active))
    dirs: list[Path] = []
    for venv in candidates:
        # posix: lib/pythonX.Y/site-packages, windows: Lib/site-packages
        dirs.extend(d for d in venv.glob("lib/*/site-packages") if d.is_dir())
        win = venv / "Lib" / "site-packages"
        if win.is_dir():
            dirs.append(win)
    return tuple(dict.fromkeys(dirs))


def _repo_root(file_path: Path) -> Path:
    """Nearest .git ancestor of a file, or its import root if it isn't in a repo."""
    for parent in file_path.parents:
        if (parent / ".git").exists():
            return parent
    return find_import_root(file_path)


def _installed_module_file(importing_path: Path, module: str) -> Path | None:
    """Resolve a module to an installed dependency's source file, if present."""
    if not module or module.split(".")[0] in sys.stdlib_module_names:
        return None
    for site_dir in _site_packages_dirs(_repo_root(importing_path)):
        resolved = resolve_module_file(site_dir / "__main__.py", module, 0, site_dir)
        if resolved is not None:
            return resolved
    return None


def _resolve_module(
    importing_path: Path,
    module: str,
    level: int,
    import_root: Path,
    graph: CallGraph,
) -> str | None:
    """File a module resolves to: on disk in this repo first, then across repos.

    A relative import can never leave its own package, so it is never looked up
    across repos.
    """
    resolved = resolve_module_file(importing_path, module, level, import_root)
    if resolved is not None:
        return str(resolved)
    if level > 0:
        return None
    return graph.find_indexed_module(module)


def build_nodes_for_python_file(
    file_path: Path,
    source: bytes | None = None,
    graph: CallGraph | None = None,
) -> CallGraph:
    """Add a file's nodes and intra-file edges.

    Populates the File node, Function/Class/method nodes, CONTAINS edges, and
    same-file CALLS edges. Cross-file edges (imports, calls into other files) are
    added separately by link_python_file_imports, which must run only after every
    file's nodes exist — otherwise add_edge no-ops on a missing target.
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

    # Same-file call sites (targets are defined in this file, so they exist now).
    _extract_python_calls(root, source, fpath_str, defined_functions, defined_classes, graph)

    graph.commit()
    return graph


def link_python_file_imports(
    file_path: Path,
    source: bytes | None = None,
    graph: CallGraph | None = None,
    import_root: Path | None = None,
) -> CallGraph:
    """Add cross-file edges for a Python file.

    Resolves each import to the real file on disk and adds an IMPORTS_FROM edge to
    it, and links calls to imported functions (``from m import f; f()`` and
    ``import m; m.f()``) with CALLS edges to the real definition. Only edges whose
    target node already exists are created, so run this after every file's nodes
    have been built. Imports that don't resolve to an indexed file are skipped —
    the graph stays quiet about code it can't see rather than guessing.
    """
    if graph is None:
        graph = CallGraph()
    if source is None:
        source = file_path.read_bytes()
    if import_root is None:
        import_root = find_import_root(file_path)

    parser, parse_lock = _get_python_parser()
    with parse_lock:
        tree = parser.parse(source)
    root = tree.root_node
    fpath_str = str(file_path)

    records = extract_imports(source)
    import_table = import_table_from_records(records)

    # IMPORTS_FROM edges to resolved, indexed files. For `from pkg import sub`
    # where sub is a submodule, the module ("pkg") resolves to pkg/__init__.py;
    # we also resolve "pkg.sub" so the edge points at the real submodule file.
    linked_targets: set[str] = set()
    for rec in records:
        modules = [rec.module]
        if rec.symbol is not None:
            modules.append(f"{rec.module}.{rec.symbol}" if rec.module else rec.symbol)
        resolved_any = False
        for module in modules:
            target = _resolve_module(file_path, module, rec.level, import_root, graph)
            if target is None:
                continue
            resolved_any = True
            if target in linked_targets:
                continue
            linked_targets.add(target)
            if graph.has_node(target):
                graph.add_edge(fpath_str, target, EdgeType.IMPORTS_FROM)

        if resolved_any or rec.level > 0:
            continue
        # The import left the indexed code. If it lands in an installed
        # dependency whose source is on disk, record the boundary and where that
        # code is, so the agent can see the edge of what we indexed.
        external = _installed_module_file(file_path, rec.module)
        if external is None:
            continue
        ext_path = str(external)
        if ext_path in linked_targets:
            continue
        linked_targets.add(ext_path)
        graph.add_node(ext_path, NodeType.EXTERNAL, ext_path)
        graph.add_edge(fpath_str, ext_path, EdgeType.IMPORTS_FROM)

    # Cross-file CALLS edges to imported functions.
    _extract_cross_file_calls(root, source, fpath_str, import_table, graph, file_path, import_root)
    _extract_inherits_edges(root, source, fpath_str, import_table, graph, file_path, import_root)

    graph.commit()
    return graph


def build_graph_for_python_file(
    file_path: Path,
    source: bytes | None = None,
    graph: CallGraph | None = None,
) -> CallGraph:
    """Full single-file build: nodes + same-file calls + cross-file links.

    Convenience wrapper used for incremental re-indexing of one file. Cross-file
    edges resolve against whatever nodes already exist in the graph; targets in
    not-yet-indexed files are simply skipped.
    """
    if graph is None:
        graph = CallGraph()
    if source is None:
        source = file_path.read_bytes()
    build_nodes_for_python_file(file_path, source, graph)
    link_python_file_imports(file_path, source, graph)
    return graph


def _py_node_name(node: Node, source: bytes) -> str:
    name_node = node.child_by_field_name("name")
    if name_node:
        return source[name_node.start_byte : name_node.end_byte].decode()
    return "<anonymous>"


_REEXPORT_MAX_HOPS = 4


def _follow_reexport(module_file: str, symbol: str, graph: CallGraph) -> str | None:
    """Resolve a symbol through a package __init__ that only re-exports it.

    ``from pkg import Thing`` resolves to ``pkg/__init__.py``, which for most
    distributed packages defines nothing itself, it just imports Thing from the
    module that does. Without this the target node never exists and the call
    edge is dropped, which is the common case, not an edge case.

    Follows the explicit top-level re-export idiom only, including renames
    (``from .impl import Thing as Other``) and relative imports. Star re-exports
    (``from .core import *``) and imports nested in ``try``/``if TYPE_CHECKING``
    blocks are not followed, because extract_imports does not record them. Those
    miss silently rather than resolving to something wrong.

    Returns None if the chain does not land on a real node, so callers keep
    whatever target they already had.

    ponytail: re-reads and re-parses the __init__ on every hop of every
    unresolved call site, no caching. Free on ordinary files (the __init__.py
    name check below rejects them before any IO) and unmeasurable on a normal
    repo, but a hub-style __init__ costs ~6ms per hop, and a consumer with
    thousands of call sites into one measured 3.7x slower to link. Cache the
    import table on (path, mtime) if indexing a hub library starts to hurt.
    """
    path = Path(module_file)
    for _ in range(_REEXPORT_MAX_HOPS):
        # Only a package __init__ re-exports. Anything else defines its symbols,
        # and if the node is missing there, chasing further would be guessing.
        # This also bounds cycles: a chain that loops burns hops and gives up.
        if path.name != "__init__.py":
            return None
        try:
            src = path.read_bytes()
        except OSError:
            return None
        rec = import_table_from_records(extract_imports(src)).get(symbol)
        if rec is None or rec.symbol is None:
            return None
        nxt = _resolve_module(path, rec.module, rec.level, find_import_root(path), graph)
        if nxt is None:
            return None
        path, symbol = Path(nxt), rec.symbol
        target = f"{path}::{symbol}"
        if graph.has_node(target):
            return target
    return None


def _node_for(resolved: str, symbol: str, graph: CallGraph) -> str | None:
    """Existing node for a symbol in a resolved file, chasing re-exports."""
    target = f"{resolved}::{symbol}"
    if graph.has_node(target):
        return target
    return _follow_reexport(resolved, symbol, graph)


def _resolve_imported_symbol(
    name: str,
    import_table: dict[str, ImportRecord],
    importing_path: Path,
    import_root: Path,
    graph: CallGraph,
) -> str | None:
    """Node a dotted reference to an imported symbol points at, or None.

      from mod import X;   X      -> resolved(mod)::X
      import mod [as m];   m.X    -> resolved(mod)::X
      from pkg import sub; sub.X  -> resolved(pkg.sub)::X

    Longest imported prefix wins for the attribute form. Returns None when the
    name is not an import, does not resolve to an indexed file, or lands on no
    real node, so callers never invent an edge.
    """
    if "." not in name:
        rec = import_table.get(name)
        if rec is None or rec.symbol is None:
            return None
        resolved = _resolve_module(importing_path, rec.module, rec.level, import_root, graph)
        return _node_for(resolved, rec.symbol, graph) if resolved else None

    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        rec = import_table.get(".".join(parts[:i]))
        if rec is None:
            continue
        remaining = parts[i:]
        if len(remaining) != 1:
            return None  # only <module>.symbol resolves to a definition
        if rec.symbol is None:
            module = rec.module  # import mod [as m]
        elif rec.module:
            module = f"{rec.module}.{rec.symbol}"  # from pkg import sub
        else:
            module = rec.symbol  # from . import sub
        resolved = _resolve_module(importing_path, module, rec.level, import_root, graph)
        return _node_for(resolved, remaining[0], graph) if resolved else None
    return None


def _symbol_node(
    name: str,
    file_path: str,
    import_table: dict[str, ImportRecord],
    importing_path: Path,
    import_root: Path,
    graph: CallGraph,
) -> str | None:
    """Node a bare or dotted symbol name refers to, same file first then imports."""
    local = f"{file_path}::{name}"
    if "." not in name and graph.has_node(local):
        return local
    return _resolve_imported_symbol(name, import_table, importing_path, import_root, graph)


def _annotation_name(node: Node | None, source: bytes) -> str | None:
    """Class name an annotation names directly, or None.

    Only a bare name counts. ``list[Client]`` and ``Optional[Client]`` annotate a
    container, not a Client, so unwrapping the subscript here would bind the
    variable to the wrong class.
    """
    if node is not None and node.type == "type":
        node = node.children[0] if node.children else None  # `type` wraps the name
    if node is None or node.type not in ("identifier", "attribute"):
        return None
    return source[node.start_byte : node.end_byte].decode()


def _rebindings_in(body: Node, source: bytes) -> list[tuple[Node | None, str]]:
    """Every plain-name rebinding in a function body, as (node, variable).

    Node is the assignment, or None for a binding whose shape carries no class
    at all (a for target, a with alias). Callers must treat None as poisoning
    the variable: it is a rebinding they cannot prove, not an absence of one.

    Does not descend into nested function, lambda, or class bodies. Those are
    separate scopes, and an inner assignment leaking out would bind the outer
    variable to a class it never holds.
    """
    found: list[tuple[Node | None, str]] = []
    stack = list(body.children)
    while stack:
        node = stack.pop()
        if node.type in ("function_definition", "lambda", "class_definition"):
            continue
        if node.type == "assignment":
            left = node.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                found.append((node, source[left.start_byte : left.end_byte].decode()))
        elif node.type in ("for_statement", "with_item"):
            target = node.child_by_field_name("left") or node.child_by_field_name("alias")
            if target is not None and target.type == "identifier":
                found.append((None, source[target.start_byte : target.end_byte].decode()))
        stack.extend(node.children)
    return found


def _receiver_classes(
    func: Node,
    source: bytes,
    file_path: str,
    import_table: dict[str, ImportRecord],
    importing_path: Path,
    import_root: Path,
    graph: CallGraph,
) -> dict[str, str]:
    """Variable -> class node, for receivers whose class is statically obvious.

    Three shapes carry a type with no inference needed:
      x = Client()        constructor call
      def f(x: Client)    annotated parameter
      x: Client = ...     annotated assignment
    plus ``self``, which is the enclosing class.

    Everything else is left alone. A factory return, a reassignment, an element
    pulled out of a container: guessing any of those would hang the call off the
    wrong class, which is worse than the edge being missing.
    """
    bindings: dict[str, str] = {}
    conflicted: set[str] = set()

    def bind(var: str, class_name: str) -> None:
        node = _symbol_node(class_name, file_path, import_table, importing_path, import_root, graph)
        if node is None:
            # Names a class we cannot resolve. The variable still holds
            # something, so treat it as unknown rather than as not-assigned.
            conflicted.add(var)
            return
        if var in bindings and bindings[var] != node:
            # Rebound to a different class inside one function. Which one a call
            # sees depends on where it sits, and that is flow analysis. Drop it.
            conflicted.add(var)
            return
        bindings[var] = node

    parent = func.parent
    while parent is not None and parent.type != "class_definition":
        parent = parent.parent
    if parent is not None:
        cls = f"{file_path}::{_py_node_name(parent, source)}"
        if graph.has_node(cls):
            bindings["self"] = cls

    params = func.child_by_field_name("parameters")
    if params is not None:
        for param in params.children:
            if param.type != "typed_parameter":
                continue
            name_node = next((c for c in param.children if c.type == "identifier"), None)
            ann = _annotation_name(param.child_by_field_name("type"), source)
            if name_node is not None and ann:
                bind(source[name_node.start_byte : name_node.end_byte].decode(), ann)

    body = func.child_by_field_name("body")
    for node, var in _rebindings_in(body, source) if body else []:
        if node is None:
            conflicted.add(var)  # for-target, with-alias: shape we cannot prove
            continue
        ann = _annotation_name(node.child_by_field_name("type"), source)
        if ann:
            bind(var, ann)
            continue
        right = node.child_by_field_name("right")
        ctor = (
            right.child_by_field_name("function")
            if right is not None and right.type == "call"
            else None
        )
        if ctor is not None and ctor.type in ("identifier", "attribute"):
            bind(var, source[ctor.start_byte : ctor.end_byte].decode())
        else:
            # A subscript, a bare name, an await, a comprehension, a literal.
            # The variable holds something this cannot prove, so every binding
            # for it is now suspect, including one made by an earlier branch.
            conflicted.add(var)

    for var in conflicted:
        bindings.pop(var, None)
    return bindings


def _receiver_bindings_by_caller(
    root: Node,
    source: bytes,
    file_path: str,
    import_table: dict[str, ImportRecord],
    importing_path: Path,
    import_root: Path,
    graph: CallGraph,
) -> dict[str, dict[str, str]]:
    """Receiver bindings per enclosing function qualified name.

    Bindings are per function, never per file: the same variable name routinely
    holds different classes in different functions, and a file-wide map would
    cross those over.

    ponytail: walks every function subtree once per file and does a has_node
    lookup per candidate class, costing about 14% of link time (measured on
    pydantic, mcp and litellm). Memoise _symbol_node per (file, name) if that
    ever matters; the answer cannot change within a single link pass.
    """
    out: dict[str, dict[str, str]] = {}
    ambiguous: set[str] = set()
    for func in _find_nodes_by_type(root, "function_definition"):
        body = func.child_by_field_name("body")
        qname = _find_enclosing_function(body, source, file_path) if body else None
        if qname is None:
            continue
        if qname in out:
            # Two functions resolving to one qualified name, e.g. a nested
            # function sharing a name. Drop both rather than merge them.
            ambiguous.add(qname)
            continue
        out[qname] = _receiver_classes(
            func, source, file_path, import_table, importing_path, import_root, graph
        )
    for qname in ambiguous:
        out.pop(qname, None)
    return out


def _extract_inherits_edges(
    root: Node,
    source: bytes,
    file_path: str,
    import_table: dict[str, ImportRecord],
    graph: CallGraph,
    importing_path: Path,
    import_root: Path,
) -> None:
    """Add INHERITS edges from a class to each base class that resolves.

    A base may be defined in the same file, imported from another file, or
    re-exported through a package __init__. Bases that are expressions rather
    than plain names (Generic[T], a metaclass keyword, a call) are skipped:
    nothing static resolves those to a definition.

    Without this a change to a base class propagates to nothing, since a
    subclass is connected to its parent by no edge at all.
    """
    for cls in root.children:
        if cls.type != "class_definition":
            continue
        # Top level only, matching build_nodes_for_python_file. A nested class
        # has no node of its own, and looking one up by bare name would attach
        # its bases to an unrelated top-level class that happens to share it.
        child = f"{file_path}::{_py_node_name(cls, source)}"
        if not graph.has_node(child):
            continue
        supers = cls.child_by_field_name("superclasses")
        if supers is None:
            continue
        for arg in supers.children:
            head = arg
            if head.type == "subscript":
                # Base[int], Request[P, T]: the parent is the subscripted value.
                # Dropping these would lose a fifth of all bases in typed code.
                head = head.child_by_field_name("value")
                if head is None:
                    continue
            if head.type not in ("identifier", "attribute"):
                continue  # metaclass=M, make_base(), *bases: nothing to resolve
            base = source[head.start_byte : head.end_byte].decode()
            local = f"{file_path}::{base}"
            if "." not in base and graph.has_node(local):
                graph.add_edge(child, local, EdgeType.INHERITS)  # same-file base
                continue
            target = _resolve_imported_symbol(
                base, import_table, importing_path, import_root, graph
            )
            if target:
                graph.add_edge(child, target, EdgeType.INHERITS)


def _extract_cross_file_calls(
    root: Node,
    source: bytes,
    file_path: str,
    import_table: dict[str, ImportRecord],
    graph: CallGraph,
    importing_path: Path,
    import_root: Path,
) -> None:
    """Add CALLS edges for calls to imported functions, resolved to real files.

    Handles the statically-resolvable shapes:
      from mod import func; func()       -> caller CALLS resolved(mod)::func
      import mod [as m];  m.func()        -> caller CALLS resolved(mod)::func
      from pkg import sub; sub.func()     -> caller CALLS resolved(pkg.sub)::func
    Instance/method calls and other dynamic dispatch are left alone. A name that
    is also defined locally is left to same-file resolution (the local definition
    shadows the import), so no spurious cross-file edge is added.
    """
    bindings = _receiver_bindings_by_caller(
        root, source, file_path, import_table, importing_path, import_root, graph
    )
    for call in _find_nodes_by_type(root, "call"):
        func_node = call.child_by_field_name("function")
        if not func_node:
            continue
        call_name = source[func_node.start_byte : func_node.end_byte].decode()
        caller = _find_enclosing_function(call, source, file_path)
        if not caller:
            continue
        if "." not in call_name and graph.has_node(f"{file_path}::{call_name}"):
            continue  # local definition shadows the import

        # receiver.method(): a variable whose class is known beats treating the
        # first segment as a module name, the same way a local def shadows an
        # import. Only single-segment receivers, so x.y.z() stays unresolved.
        receiver, _, rest = call_name.partition(".")
        cls = bindings.get(caller, {}).get(receiver) if rest and "." not in rest else None
        if cls:
            method = f"{cls}.{rest}"
            if graph.has_node(method):
                graph.add_edge(caller, method, EdgeType.CALLS)
            continue

        target = _resolve_imported_symbol(
            call_name, import_table, importing_path, import_root, graph
        )
        if target:
            graph.add_edge(caller, target, EdgeType.CALLS)


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

        call_name = source[func_node.start_byte : func_node.end_byte].decode()

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
