"""Python import resolution for cross-file call-graph edges.

The call graph used to link only calls to functions defined in the *same* file.
That made the graph blind to the most important relationships — a function in one
file calling a function imported from another. This module closes that gap for
Python by:

  1. Parsing a file's import statements into structured records.
  2. Resolving a (possibly relative) dotted module to the real file on disk,
     using the importing file's package layout to find the import root.

It deliberately resolves only what it can see statically. `from mod import func;
func()` and `import mod; mod.func()` resolve to real definitions; instance method
calls and other dynamic dispatch do not (the graph stays quiet rather than
guessing). Resolution is confined to files on disk — the same primitive that will
later let us follow a dependency out of one repo and into the next.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

_PY_LANGUAGE: Language | None = None
_PY_PARSER: Parser | None = None
_init_lock = threading.Lock()
_parse_lock = threading.Lock()


def _parser() -> tuple[Parser, threading.Lock]:
    global _PY_LANGUAGE, _PY_PARSER
    if _PY_PARSER is None:
        with _init_lock:
            if _PY_PARSER is None:
                _PY_LANGUAGE = Language(tspython.language())
                _PY_PARSER = Parser(_PY_LANGUAGE)
    return _PY_PARSER, _parse_lock


@dataclass(frozen=True)
class ImportRecord:
    """One name brought into a module's namespace by an import statement.

    local_name: the name bound in the importing file (alias if aliased).
    module:     dotted module path the name comes from. For `from a.b import c`
                this is "a.b"; for `import a.b.c` it is "a.b.c".
    symbol:     the imported symbol for `from ... import symbol`; None for a plain
                `import module` (the bound name refers to the module itself).
    level:      number of leading dots (0 = absolute, 1 = current package, ...).
    is_wildcard: True for `from a.b import *` (symbol/local_name are empty).
    """

    local_name: str
    module: str
    symbol: str | None
    level: int
    is_wildcard: bool = False


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode()


def _relative_module(rel_node: Node, source: bytes) -> tuple[int, str]:
    """Parse a `relative_import` node into (level, module).

    `.`        -> (1, "")
    `.mod`     -> (1, "mod")
    `..pkg.mod`-> (2, "pkg.mod")
    """
    level = 0
    module = ""
    for child in rel_node.children:
        if child.type == "import_prefix":
            level = _text(child, source).count(".")
        elif child.type == "dotted_name":
            module = _text(child, source)
    return level, module


def _imported_names(stmt: Node, source: bytes) -> list[tuple[str, str | None, bool]]:
    """Return (local_name, symbol, is_wildcard) for each name after `import`.

    Only the name children (field "name") are considered, so the module_name of a
    `from` import is never mistaken for an imported symbol.
    """
    names: list[tuple[str, str | None, bool]] = []
    for child in stmt.children:
        if child.type == "wildcard_import":
            names.append(("", None, True))
        elif child.type == "aliased_import":
            original = child.child_by_field_name("name")
            alias = child.child_by_field_name("alias")
            if original is not None and alias is not None:
                names.append((_text(alias, source), _text(original, source), False))
    # Plain `from a.b import c, d`: the imported names are the "name" fields.
    for child in stmt.children_by_field_name("name"):
        if child.type == "dotted_name":
            sym = _text(child, source)
            names.append((sym, sym, False))
    return names


def extract_imports(source: bytes) -> list[ImportRecord]:
    """Parse all import statements in a Python source buffer into ImportRecords."""
    parser, lock = _parser()
    with lock:
        tree = parser.parse(source)
    root = tree.root_node

    records: list[ImportRecord] = []
    for stmt in root.children:
        if stmt.type == "import_statement":
            # `import a.b`, `import a.b as x`, `import a, b`
            for child in stmt.children:
                if child.type == "dotted_name":
                    mod = _text(child, source)
                    records.append(ImportRecord(mod, mod, None, 0))
                elif child.type == "aliased_import":
                    name = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    if name is not None and alias is not None:
                        mod = _text(name, source)
                        records.append(ImportRecord(_text(alias, source), mod, None, 0))

        elif stmt.type == "import_from_statement":
            module_node = stmt.child_by_field_name("module_name")
            if module_node is None:
                continue
            if module_node.type == "relative_import":
                level, module = _relative_module(module_node, source)
            else:
                level, module = 0, _text(module_node, source)

            for local_name, symbol, is_wildcard in _imported_names(stmt, source):
                records.append(ImportRecord(local_name, module, symbol, level, is_wildcard))
    return records


def find_import_root(file_path: Path) -> Path:
    """Find the directory a file's top-level package is imported relative to.

    Climbs out of the package by following the chain of `__init__.py` directories;
    the first ancestor without one is the import root (what would be on sys.path).
    For a loose script with no package, that's just the file's own directory.
    """
    d = file_path.parent
    while (d / "__init__.py").exists() and d.parent != d:
        d = d.parent
    return d


def resolve_module_file(
    importing_file: Path,
    module: str,
    level: int,
    import_root: Path | None = None,
) -> Path | None:
    """Resolve a dotted module to a real .py file on disk, or None if not found.

    Absolute imports (level 0) resolve under ``import_root`` (derived from the
    importing file's package if not supplied), then — for PEP 420 namespace
    packages with no ``__init__.py`` to anchor the root — fall back to climbing
    the importing file's ancestors until the module resolves. Relative imports
    resolve against the importing file's package, climbing ``level - 1``
    directories first. Tries ``pkg/mod.py`` then ``pkg/mod/__init__.py``.
    """
    if level > 0:
        base = importing_file.parent
        for _ in range(level - 1):
            base = base.parent
        bases: list[Path] = [base]
    else:
        primary = import_root if import_root is not None else find_import_root(importing_file)
        # Try the package-derived root first; then walk up the importing file's
        # ancestors so namespace packages (no __init__.py marking the root) still
        # resolve. ponytail: the ancestor walk could match a same-named module at
        # the wrong depth; root-first ordering makes that rare. Add a real
        # project-root boundary if it ever misfires.
        bases = [primary, *importing_file.parents]

    parts = module.split(".") if module else []

    if parts:
        seen: set[Path] = set()
        for base in bases:
            if base in seen:
                continue
            seen.add(base)
            candidate = base.joinpath(*parts).with_suffix(".py")
            if candidate.is_file():
                return candidate
            pkg_init = base.joinpath(*parts) / "__init__.py"
            if pkg_init.is_file():
                return pkg_init
        return None

    # No module after the dots (e.g. `from . import x`): the base package itself.
    pkg_init = bases[0] / "__init__.py"
    return pkg_init if pkg_init.is_file() else None


def import_table_from_records(records: list[ImportRecord]) -> dict[str, ImportRecord]:
    """Map each bound local name to the import that introduced it.

    Later bindings win, mirroring Python's namespace. Wildcard imports are
    excluded (their bound names aren't known statically).
    """
    table: dict[str, ImportRecord] = {}
    for rec in records:
        if rec.is_wildcard or not rec.local_name:
            continue
        table[rec.local_name] = rec
    return table


def build_import_table(source: bytes, importing_file: Path) -> dict[str, ImportRecord]:
    """Parse a source buffer and return its bound-name -> import mapping."""
    return import_table_from_records(extract_imports(source))
