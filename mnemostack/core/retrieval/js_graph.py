"""Call-graph construction for JavaScript and TypeScript.

The chunker has always understood .js/.ts files, but the graph didn't, so for a
JS repo the dependency-aware half of retrieval was simply missing. This adds the
same two-pass build Python gets: file/function/class nodes first, then the edges
between files once every node exists.

It resolves what is statically visible — relative imports, `require()` of a
relative path, and calls to names those imports bind — and stays quiet about the
rest. Bare specifiers ("react") are not app code: they resolve into node_modules
as a boundary, the same way an installed Python package does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from tree_sitter import Node

from mnemostack.core.retrieval.ast_chunker import _get_parser
from mnemostack.core.retrieval.call_graph import (
    CallGraph,
    EdgeType,
    NodeType,
    _find_nodes_by_type,
)


class Binding(NamedTuple):
    """A name an import binds: its local name, the symbol it refers to (empty for
    a namespace), and whether it stands for the whole module."""

    local: str
    symbol: str
    is_namespace: bool


JS_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".ts", ".tsx"})

# Resolution order matters: a TS project importing "./util" must find util.ts
# before a stale util.js sitting next to it in a build output.
_TRY_EXTENSIONS = (".ts", ".tsx", ".mts", ".js", ".jsx", ".mjs", ".cjs")

_FUNCTION_NODES = frozenset({"function_declaration", "generator_function_declaration"})
_ARROW_NODES = frozenset({"arrow_function", "function_expression", "function"})


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _string_value(node: Node, source: bytes) -> str:
    """Contents of a tree-sitter string node, without its quotes."""
    return _text(node, source).strip("\"'`")


# --- Module resolution ---


def _resolve_file(base: Path) -> Path | None:
    """Node-style resolution of a path with no (or a rewritten) extension."""
    if base.is_file():
        return base
    for ext in _TRY_EXTENSIONS:
        candidate = base.with_name(base.name + ext)
        if candidate.is_file():
            return candidate
    # TS ESM writes "./util.js" for a file that is really util.ts on disk.
    if base.suffix in {".js", ".mjs", ".cjs"}:
        for ext in (".ts", ".tsx", ".mts"):
            candidate = base.with_suffix(ext)
            if candidate.is_file():
                return candidate
    for ext in _TRY_EXTENSIONS:
        candidate = base / f"index{ext}"
        if candidate.is_file():
            return candidate
    return None


def resolve_js_module(importing_file: Path, specifier: str) -> Path | None:
    """Resolve a relative import specifier to a real file on disk.

    Only relative specifiers resolve here. ponytail: tsconfig `paths` aliases
    ("@/lib/x") are not read, so those imports stay unlinked — wire up tsconfig
    parsing if alias-heavy repos turn out to be the common case.
    """
    if not specifier.startswith("."):
        return None
    return _resolve_file(importing_file.parent / specifier)


def _package_name(specifier: str) -> str:
    parts = specifier.split("/")
    return "/".join(parts[:2]) if specifier.startswith("@") else parts[0]


def find_js_package_entry(importing_file: Path, specifier: str) -> Path | None:
    """Entry file of an installed package in node_modules, if it's on disk.

    node_modules is never indexed, so this is a boundary: the point where the
    graph stops seeing code, reported with the real path rather than dropped.
    """
    if specifier.startswith("."):
        return None
    name = _package_name(specifier)
    for parent in importing_file.parents:
        pkg_dir = parent / "node_modules" / name
        if not pkg_dir.is_dir():
            continue
        manifest = pkg_dir / "package.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            for field in ("module", "main"):
                entry = data.get(field)
                if isinstance(entry, str) and (resolved := _resolve_file(pkg_dir / entry)):
                    return resolved
        return _resolve_file(pkg_dir)
    return None


# --- Import extraction ---


def _import_bindings(clause: Node, source: bytes) -> list[Binding]:
    """The names an import clause binds, and what each one refers to.

    A default import (`import Layout from "./Layout"`) names the target's default
    export, whose real symbol name the importing file can't see. The importer's
    own name for it is the only candidate available, so it is used — and since an
    edge is added only when the target file actually defines a symbol by that
    name, a mismatch produces no edge instead of a wrong one.
    """
    bindings: list[Binding] = []
    for child in clause.children:
        if child.type == "identifier":  # import Default from "m"
            local = _text(child, source)
            bindings.append(Binding(local, local, False))
        elif child.type == "namespace_import":  # import * as ns from "m"
            for sub in child.children:
                if sub.type == "identifier":
                    bindings.append(Binding(_text(sub, source), "", True))
        elif child.type == "named_imports":
            for spec in child.children:
                if spec.type != "import_specifier":
                    continue
                name = spec.child_by_field_name("name")
                alias = spec.child_by_field_name("alias")
                if name is None:
                    continue
                symbol = _text(name, source)
                bindings.append(
                    Binding(_text(alias, source) if alias else symbol, symbol, False)
                )
    return bindings


def extract_js_imports(root: Node, source: bytes) -> list[tuple[str, list[Binding]]]:
    """(specifier, bindings) for every import, re-export and require in a file."""
    out: list[tuple[str, list[Binding]]] = []

    for stmt in _find_nodes_by_type(root, "import_statement") + _find_nodes_by_type(
        root, "export_statement"
    ):
        src = stmt.child_by_field_name("source")
        if src is None:
            continue  # a local export, not a re-export: nothing to resolve
        bindings: list[Binding] = []
        for child in stmt.children:
            if child.type == "import_clause":
                bindings.extend(_import_bindings(child, source))
        out.append((_string_value(src, source), bindings))

    # const x = require("./m") — the CommonJS half of the same relationship.
    for call in _find_nodes_by_type(root, "call_expression"):
        func = call.child_by_field_name("function")
        args = call.child_by_field_name("arguments")
        if func is None or args is None or _text(func, source) != "require":
            continue
        strings = [c for c in args.children if c.type == "string"]
        if not strings:
            continue
        specifier = _string_value(strings[0], source)
        bindings = []
        declarator = call.parent
        if declarator is not None and declarator.type == "variable_declarator":
            name = declarator.child_by_field_name("name")
            if name is not None and name.type == "identifier":
                # `const lib = require("./m")` binds the module, like `* as lib`.
                bindings.append(Binding(_text(name, source), "", True))
        out.append((specifier, bindings))
    return out


# --- Node and edge construction ---


def _declared_name(node: Node, source: bytes) -> str | None:
    """Name of a function/class declaration, or of the variable an arrow is bound to."""
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(name, source)
    parent = node.parent
    if parent is not None and parent.type == "variable_declarator":
        vname = parent.child_by_field_name("name")
        if vname is not None and vname.type == "identifier":
            return _text(vname, source)
    return None


def _top_level_declarations(root: Node) -> list[Node]:
    """Declarations at module level, looking through `export` wrappers."""
    out: list[Node] = []
    for child in root.children:
        nodes = [child]
        if child.type == "export_statement":
            nodes = list(child.children)
        for node in nodes:
            if node.type in _FUNCTION_NODES or node.type == "class_declaration":
                out.append(node)
            elif node.type in {"lexical_declaration", "variable_declaration"}:
                for declarator in node.children:
                    if declarator.type != "variable_declarator":
                        continue
                    value = declarator.child_by_field_name("value")
                    if value is not None and value.type in _ARROW_NODES:
                        out.append(value)
    return out


def build_nodes_for_js_file(
    file_path: Path,
    source: bytes | None = None,
    graph: CallGraph | None = None,
) -> CallGraph:
    """Add a JS/TS file's nodes, CONTAINS edges, and same-file CALLS edges."""
    if graph is None:
        graph = CallGraph()
    if source is None:
        source = file_path.read_bytes()
    parsed = _get_parser(file_path.suffix.lower())
    if parsed is None:
        return graph
    parser, parse_lock = parsed
    with parse_lock:
        tree = parser.parse(source)
    root = tree.root_node
    fpath = str(file_path)

    graph.add_node(fpath, NodeType.FILE, fpath)

    defined_functions: list[str] = []
    defined_classes: list[str] = []
    for node in _top_level_declarations(root):
        name = _declared_name(node, source)
        if name is None:
            continue
        if node.type == "class_declaration":
            class_qname = f"{fpath}::{name}"
            graph.add_node(class_qname, NodeType.CLASS, fpath)
            graph.add_edge(fpath, class_qname, EdgeType.CONTAINS)
            defined_classes.append(name)
            body = node.child_by_field_name("body")
            for method in body.children if body else []:
                if method.type != "method_definition":
                    continue
                mname = method.child_by_field_name("name")
                if mname is None:
                    continue
                mqname = f"{fpath}::{name}.{_text(mname, source)}"
                graph.add_node(mqname, NodeType.FUNCTION, fpath)
                graph.add_edge(class_qname, mqname, EdgeType.CONTAINS)
        else:
            qname = f"{fpath}::{name}"
            graph.add_node(qname, NodeType.FUNCTION, fpath)
            graph.add_edge(fpath, qname, EdgeType.CONTAINS)
            defined_functions.append(name)

    _extract_js_calls(root, source, fpath, defined_functions, defined_classes, graph)
    graph.commit()
    return graph


def _enclosing_function(node: Node, source: bytes, file_path: str) -> str | None:
    """Qualified name of the function or method a node sits inside."""
    current = node.parent
    func_name: str | None = None
    class_name: str | None = None
    while current is not None:
        if func_name is None and (
            current.type in _FUNCTION_NODES or current.type in _ARROW_NODES
        ):
            func_name = _declared_name(current, source)
        elif current.type == "method_definition" and func_name is None:
            name = current.child_by_field_name("name")
            func_name = _text(name, source) if name is not None else None
        elif current.type == "class_declaration" and class_name is None:
            class_name = _declared_name(current, source)
        current = current.parent

    if func_name is None:
        return None
    return f"{file_path}::{class_name}.{func_name}" if class_name else f"{file_path}::{func_name}"


def _extract_js_calls(
    root: Node,
    source: bytes,
    file_path: str,
    defined_functions: list[str],
    defined_classes: list[str],
    graph: CallGraph,
) -> None:
    """Same-file CALLS edges: a call whose name is defined in this file."""
    for call in _find_nodes_by_type(root, "call_expression"):
        func_node = call.child_by_field_name("function")
        if func_node is None:
            continue
        call_name = _text(func_node, source)
        caller = _enclosing_function(call, source, file_path)
        if not caller:
            continue
        if call_name in defined_functions:
            graph.add_edge(caller, f"{file_path}::{call_name}", EdgeType.CALLS)
        elif "." in call_name:
            # this.method() / obj.method(): match a method defined in this file.
            method = call_name.split(".")[-1]
            for cls in defined_classes:
                candidate = f"{file_path}::{cls}.{method}"
                if graph.has_node(candidate):
                    graph.add_edge(caller, candidate, EdgeType.CALLS)
                    break


def link_js_file_imports(
    file_path: Path,
    source: bytes | None = None,
    graph: CallGraph | None = None,
) -> CallGraph:
    """Add a JS/TS file's cross-file edges. Run after every file's nodes exist.

    Relative imports become IMPORTS_FROM edges to the real file, and calls to the
    names they bind become CALLS edges to the real definition. An import of an
    installed package becomes an External boundary node at its entry file.
    """
    if graph is None:
        graph = CallGraph()
    if source is None:
        source = file_path.read_bytes()
    parsed = _get_parser(file_path.suffix.lower())
    if parsed is None:
        return graph
    parser, parse_lock = parsed
    with parse_lock:
        tree = parser.parse(source)
    root = tree.root_node
    fpath = str(file_path)

    imported_symbols: dict[str, tuple[str, str]] = {}  # local name -> (target file, symbol)
    namespaces: dict[str, str] = {}  # local name of a whole module -> target file
    for specifier, bindings in extract_js_imports(root, source):
        resolved = resolve_js_module(file_path, specifier)
        if resolved is not None:
            target = str(resolved)
            if graph.has_node(target):
                graph.add_edge(fpath, target, EdgeType.IMPORTS_FROM)
            for binding in bindings:
                if binding.is_namespace:
                    namespaces[binding.local] = target
                elif binding.symbol:
                    imported_symbols[binding.local] = (target, binding.symbol)
            continue
        entry = find_js_package_entry(file_path, specifier)
        if entry is None:
            continue
        ext_path = str(entry)
        graph.add_node(ext_path, NodeType.EXTERNAL, ext_path)
        graph.add_edge(fpath, ext_path, EdgeType.IMPORTS_FROM)

    # Uses of an imported name: a call, or a JSX element. Rendering <Widget /> is
    # how a React component depends on another one — without it a component tree
    # has no symbol-level edges at all, only file imports.
    uses: list[tuple[Node, str]] = []
    for call in _find_nodes_by_type(root, "call_expression"):
        func_node = call.child_by_field_name("function")
        if func_node is not None:
            uses.append((call, _text(func_node, source)))
    for element_type in ("jsx_opening_element", "jsx_self_closing_element"):
        for element in _find_nodes_by_type(root, element_type):
            name_node = element.child_by_field_name("name")
            if name_node is not None:
                uses.append((element, _text(name_node, source)))

    for node, used_name in uses:
        binding = imported_symbols.get(used_name)
        if binding is None and "." in used_name:
            # ns.helper() / lib.helper() through a namespace or require binding.
            prefix, _, member = used_name.rpartition(".")
            target_file = namespaces.get(prefix)
            if target_file is not None:
                binding = (target_file, member)
        if binding is None or graph.has_node(f"{fpath}::{used_name}"):
            continue  # a local definition shadows the import
        caller = _enclosing_function(node, source, fpath)
        if not caller:
            continue
        target_file, symbol = binding
        target = f"{target_file}::{symbol}"
        if graph.has_node(target):
            graph.add_edge(caller, target, EdgeType.CALLS)

    graph.commit()
    return graph
