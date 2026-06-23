"""AST-aware code chunking using tree-sitter.

Chunks source files by meaningful boundaries: functions, classes, import blocks,
and module-level assignments. Each chunk carries metadata for indexing and ranking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser


class ChunkType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    IMPORT = "import"
    CONSTANT = "constant"


@dataclass(frozen=True, slots=True)
class Chunk:
    file_path: str
    symbol_name: str
    code: str
    line_start: int  # 1-indexed
    line_end: int  # 1-indexed, inclusive
    chunk_type: ChunkType
    last_modified: float  # unix timestamp
    qualified_name: str = ""  # file_path::ClassName.method_name
    dependencies: list[str] = field(default_factory=list)


# --- Language registry ---

_LANGUAGES: dict[str, Language] = {}
_PARSERS: dict[str, Parser] = {}
_parser_lock = threading.Lock()


def _get_language(ext: str) -> Language | None:
    """Return the tree-sitter Language for a file extension, or None if unsupported."""
    if ext not in _LANGUAGES:
        with _parser_lock:
            if ext not in _LANGUAGES:
                match ext:
                    case ".py":
                        _LANGUAGES[ext] = Language(tspython.language())
                    case ".js" | ".jsx" | ".mjs":
                        _LANGUAGES[ext] = Language(tsjavascript.language())
                    case ".ts":
                        _LANGUAGES[ext] = Language(tstypescript.language_typescript())
                    case ".tsx":
                        _LANGUAGES[ext] = Language(tstypescript.language_tsx())
                    case _:
                        return None
    return _LANGUAGES.get(ext)


def _get_parser(ext: str) -> Parser | None:
    """Return a cached Parser for a file extension, or None if unsupported."""
    lang = _get_language(ext)
    if lang is None:
        return None
    if ext not in _PARSERS:
        with _parser_lock:
            if ext not in _PARSERS:
                _PARSERS[ext] = Parser(lang)
    return _PARSERS[ext]


SUPPORTED_EXTENSIONS = frozenset({".py", ".js", ".jsx", ".mjs", ".ts", ".tsx"})


# --- Node type mappings per language ---

_FUNCTION_TYPES: dict[str, set[str]] = {
    ".py": {"function_definition"},
    ".js": {"function_declaration", "arrow_function", "function"},
    ".jsx": {"function_declaration", "arrow_function", "function"},
    ".mjs": {"function_declaration", "arrow_function", "function"},
    ".ts": {"function_declaration", "arrow_function", "function"},
    ".tsx": {"function_declaration", "arrow_function", "function"},
}

_CLASS_TYPES: dict[str, set[str]] = {
    ".py": {"class_definition"},
    ".js": {"class_declaration"},
    ".jsx": {"class_declaration"},
    ".mjs": {"class_declaration"},
    ".ts": {"class_declaration"},
    ".tsx": {"class_declaration"},
}

_IMPORT_TYPES: dict[str, set[str]] = {
    ".py": {"import_statement", "import_from_statement"},
    ".js": {"import_statement"},
    ".jsx": {"import_statement"},
    ".mjs": {"import_statement"},
    ".ts": {"import_statement"},
    ".tsx": {"import_statement"},
}


# --- Chunking logic ---


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_name(node: Node, source: bytes, ext: str) -> str:
    """Extract the symbol name from a function/class node."""
    # Python: function_definition has a 'name' child
    # JS/TS: function_declaration has a 'name' child
    name_node = node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node, source)

    # Arrow functions assigned to variables: const foo = () => ...
    # The parent is a variable_declarator with a 'name' field
    parent = node.parent
    if parent and parent.type == "variable_declarator":
        vname = parent.child_by_field_name("name")
        if vname:
            return _node_text(vname, source)

    return "<anonymous>"


def _qualified_name(file_path: str, symbol: str, parent_class: str | None = None) -> str:
    if parent_class:
        return f"{file_path}::{parent_class}.{symbol}"
    return f"{file_path}::{symbol}"


def chunk_file(file_path: Path, source: bytes | None = None) -> list[Chunk]:
    """Parse a file and return AST-aware chunks.

    Args:
        file_path: Path to the source file.
        source: Optional pre-read file contents. If None, reads from disk.

    Returns:
        List of Chunk objects representing meaningful code units.
    """
    ext = file_path.suffix.lower()
    parser = _get_parser(ext)
    if parser is None:
        return []

    if source is None:
        source = file_path.read_bytes()

    tree = parser.parse(source)
    root = tree.root_node

    # Only stat the file if we need to (source was read from disk)
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        mtime = time.time()
    fpath_str = str(file_path)

    func_types = _FUNCTION_TYPES.get(ext, set())
    class_types = _CLASS_TYPES.get(ext, set())
    import_types = _IMPORT_TYPES.get(ext, set())

    chunks: list[Chunk] = []
    import_nodes: list[Node] = []
    consumed_ranges: set[tuple[int, int]] = set()

    # First pass: collect top-level classes and their methods
    for child in root.children:
        if child.type in class_types:
            class_name = _extract_name(child, source, ext)
            chunks.append(Chunk(
                file_path=fpath_str,
                symbol_name=class_name,
                code=_node_text(child, source),
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                chunk_type=ChunkType.CLASS,
                last_modified=mtime,
                qualified_name=_qualified_name(fpath_str, class_name),
            ))
            consumed_ranges.add((child.start_byte, child.end_byte))

            # Also chunk individual methods within the class
            _chunk_class_methods(
                child, source, fpath_str, class_name, ext, func_types, mtime, chunks
            )

        elif child.type in func_types:
            name = _extract_name(child, source, ext)
            chunks.append(Chunk(
                file_path=fpath_str,
                symbol_name=name,
                code=_node_text(child, source),
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                chunk_type=ChunkType.FUNCTION,
                last_modified=mtime,
                qualified_name=_qualified_name(fpath_str, name),
            ))
            consumed_ranges.add((child.start_byte, child.end_byte))

        elif child.type in import_types:
            import_nodes.append(child)
            consumed_ranges.add((child.start_byte, child.end_byte))

        # JS/TS: exported functions/classes or variable declarations with arrow funcs
        elif child.type in ("export_statement", "lexical_declaration", "variable_declaration"):
            _chunk_declaration(
                child, source, fpath_str, ext, func_types, class_types, mtime, chunks,
                consumed_ranges
            )

    # Group imports into a single chunk
    if import_nodes:
        first = import_nodes[0]
        last = import_nodes[-1]
        import_code = "\n".join(_node_text(n, source) for n in import_nodes)
        chunks.append(Chunk(
            file_path=fpath_str,
            symbol_name="<imports>",
            code=import_code,
            line_start=first.start_point[0] + 1,
            line_end=last.end_point[0] + 1,
            chunk_type=ChunkType.IMPORT,
            last_modified=mtime,
            qualified_name=_qualified_name(fpath_str, "<imports>"),
        ))

    # Module-level assignments/constants not already consumed
    _chunk_constants(root, source, fpath_str, ext, mtime, consumed_ranges, chunks)

    # Sort by line number for stable output
    chunks.sort(key=lambda c: c.line_start)
    return chunks


def _chunk_class_methods(
    class_node: Node,
    source: bytes,
    file_path: str,
    class_name: str,
    ext: str,
    func_types: set[str],
    mtime: float,
    chunks: list[Chunk],
) -> None:
    """Extract individual method chunks from a class body."""
    body = class_node.child_by_field_name("body")
    if body is None:
        # Some languages nest methods directly under the class node
        body = class_node

    for child in body.children:
        if child.type in func_types:
            name = _extract_name(child, source, ext)
            chunks.append(Chunk(
                file_path=file_path,
                symbol_name=f"{class_name}.{name}",
                code=_node_text(child, source),
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                chunk_type=ChunkType.FUNCTION,
                last_modified=mtime,
                qualified_name=_qualified_name(file_path, name, class_name),
            ))
        # JS/TS class methods use "method_definition"
        elif child.type == "method_definition":
            name = _extract_name(child, source, ext)
            chunks.append(Chunk(
                file_path=file_path,
                symbol_name=f"{class_name}.{name}",
                code=_node_text(child, source),
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                chunk_type=ChunkType.FUNCTION,
                last_modified=mtime,
                qualified_name=_qualified_name(file_path, name, class_name),
            ))


def _chunk_declaration(
    node: Node,
    source: bytes,
    file_path: str,
    ext: str,
    func_types: set[str],
    class_types: set[str],
    mtime: float,
    chunks: list[Chunk],
    consumed_ranges: set[tuple[int, int]],
) -> None:
    """Handle export statements and variable declarations that may contain functions/classes."""
    # export default function foo() {} or export class Foo {}
    for child in node.children:
        if child.type in class_types:
            class_name = _extract_name(child, source, ext)
            chunks.append(Chunk(
                file_path=file_path,
                symbol_name=class_name,
                code=_node_text(node, source),
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                chunk_type=ChunkType.CLASS,
                last_modified=mtime,
                qualified_name=_qualified_name(file_path, class_name),
            ))
            consumed_ranges.add((node.start_byte, node.end_byte))
            _chunk_class_methods(
                child, source, file_path, class_name, ext, func_types, mtime, chunks
            )
            return

        if child.type in func_types:
            name = _extract_name(child, source, ext)
            chunks.append(Chunk(
                file_path=file_path,
                symbol_name=name,
                code=_node_text(node, source),
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                chunk_type=ChunkType.FUNCTION,
                last_modified=mtime,
                qualified_name=_qualified_name(file_path, name),
            ))
            consumed_ranges.add((node.start_byte, node.end_byte))
            return

        # variable_declarator with arrow function value
        if child.type == "variable_declarator":
            value = child.child_by_field_name("value")
            if value and value.type in func_types:
                name = _extract_name(value, source, ext)
                chunks.append(Chunk(
                    file_path=file_path,
                    symbol_name=name,
                    code=_node_text(node, source),
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    chunk_type=ChunkType.FUNCTION,
                    last_modified=mtime,
                    qualified_name=_qualified_name(file_path, name),
                ))
                consumed_ranges.add((node.start_byte, node.end_byte))
                return


def _chunk_constants(
    root: Node,
    source: bytes,
    file_path: str,
    ext: str,
    mtime: float,
    consumed_ranges: set[tuple[int, int]],
    chunks: list[Chunk],
) -> None:
    """Chunk module-level assignments/constants that weren't already consumed."""
    constant_types = {
        ".py": {"expression_statement"},
        ".js": {"lexical_declaration", "variable_declaration"},
        ".jsx": {"lexical_declaration", "variable_declaration"},
        ".mjs": {"lexical_declaration", "variable_declaration"},
        ".ts": {"lexical_declaration", "variable_declaration"},
        ".tsx": {"lexical_declaration", "variable_declaration"},
    }
    target_types = constant_types.get(ext, set())

    for child in root.children:
        if (child.start_byte, child.end_byte) in consumed_ranges:
            continue
        if child.type not in target_types:
            continue

        # Python: expression_statement containing assignment
        if ext == ".py":
            # Only chunk assignments (x = ...) not bare expressions
            if not any(
                gc.type == "assignment" for gc in child.children
            ):
                continue
            name = _extract_assignment_name(child, source)
        else:
            # JS/TS: already checked it's not a function (would be consumed)
            name = _extract_js_const_name(child, source)

        if name:
            chunks.append(Chunk(
                file_path=file_path,
                symbol_name=name,
                code=_node_text(child, source),
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                chunk_type=ChunkType.CONSTANT,
                last_modified=mtime,
                qualified_name=_qualified_name(file_path, name),
            ))


def _extract_assignment_name(node: Node, source: bytes) -> str:
    """Extract the variable name from a Python assignment expression_statement."""
    for child in node.children:
        if child.type == "assignment":
            left = child.child_by_field_name("left")
            if left:
                return _node_text(left, source)
    return ""


def _extract_js_const_name(node: Node, source: bytes) -> str:
    """Extract variable name from a JS/TS lexical/variable declaration."""
    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            if name_node:
                return _node_text(name_node, source)
    return ""


def chunk_file_fallback(file_path: Path, source: bytes | None = None) -> list[Chunk]:
    """Fallback chunker for unsupported languages: splits by blank-line-separated blocks.

    This ensures every text file gets indexed (with lower precision) even without
    a tree-sitter grammar. Each block of consecutive non-blank lines becomes one chunk.
    """
    if source is None:
        source = file_path.read_bytes()

    text = source.decode("utf-8", errors="replace")
    lines = text.split("\n")
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        mtime = time.time()
    fpath_str = str(file_path)

    chunks: list[Chunk] = []
    block_lines: list[str] = []
    block_start = 1

    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            if block_lines:
                code = "\n".join(block_lines)
                chunks.append(Chunk(
                    file_path=fpath_str,
                    symbol_name=f"<block L{block_start}>",
                    code=code,
                    line_start=block_start,
                    line_end=i - 1,
                    chunk_type=ChunkType.CONSTANT,
                    last_modified=mtime,
                    qualified_name=_qualified_name(fpath_str, f"<block L{block_start}>"),
                ))
                block_lines = []
        else:
            if not block_lines:
                block_start = i
            block_lines.append(line)

    # Final block
    if block_lines:
        code = "\n".join(block_lines)
        chunks.append(Chunk(
            file_path=fpath_str,
            symbol_name=f"<block L{block_start}>",
            code=code,
            line_start=block_start,
            line_end=block_start + len(block_lines) - 1,
            chunk_type=ChunkType.CONSTANT,
            last_modified=mtime,
            qualified_name=_qualified_name(fpath_str, f"<block L{block_start}>"),
        ))

    return chunks


def chunk_file_auto(file_path: Path, source: bytes | None = None) -> list[Chunk]:
    """Chunk a file using AST parsing if supported, fallback to block splitting otherwise.

    This is the primary entry point — guarantees results for any text file.
    """
    ext = file_path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return chunk_file(file_path, source=source)
    return chunk_file_fallback(file_path, source=source)


def chunk_files(paths: Sequence[Path]) -> list[Chunk]:
    """Chunk multiple files. Uses AST for supported languages, fallback for others."""
    all_chunks: list[Chunk] = []
    for p in paths:
        all_chunks.extend(chunk_file_auto(p))
    return all_chunks
