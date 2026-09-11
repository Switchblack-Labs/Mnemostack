"""What breaks in your code if you upgrade a dependency, before you upgrade.

The information needed to answer this has always existed: the library's API
changed in a knowable way, and your call sites are right there on disk. Nobody
intersects the two, so the choice is bump-and-pray or pin forever.

griffe supplies both halves of the library side. load_pypi downloads any
published version without installing it, so the question is answerable before
committing to the upgrade, which is the part a type checker cannot do: mypy
tells you what broke after you upgrade, not what will.

The call-site half is the existing graph. A dependency's API becomes nodes like
any other module, so import resolution and receiver binding apply unchanged and
a method call into a compiled C extension resolves the same as one into a
sibling file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import griffe

from mnemostack.core.impact.api_diff import ApiChange, fqn_index
from mnemostack.core.impact.propagate import Impact, impact_report
from mnemostack.core.retrieval.call_graph import (
    CallGraph,
    NodeType,
    _find_enclosing_function,
    _find_nodes_by_type,
    _get_python_parser,
    build_nodes_for_python_file,
    link_python_file_imports,
)
from mnemostack.core.retrieval.constants import SKIP_DIRS


@dataclass(frozen=True)
class UpgradeReport:
    package: str
    from_version: str
    to_version: str
    total_changes: int
    verified_changes: int
    impacts: list[Impact]


def _walk_api(obj, prefix: str = ""):
    """Yield (symbol path below the package, griffe kind) for a loaded module."""
    for name, member in obj.members.items():
        if name.startswith("__") and name != "__init__":
            continue
        path = f"{prefix}{name}"
        kind = str(getattr(member, "kind", "")).lower()
        yield path, kind
        if "class" in kind or "module" in kind:
            try:
                yield from _walk_api(member, f"{path}.")
            except Exception:
                # A member that cannot be walked is not worth failing the run
                # over: it costs one symbol, and the alternative is no report.
                continue


def add_dependency_surface(graph: CallGraph, module) -> int:
    """Register a loaded package's API as graph nodes. Returns the count.

    The package's own file is the anchor. find_indexed_module resolves imports
    by matching a file path suffix, so registering that one file is what makes
    ``from tree_sitter import Language`` land on these nodes, and every rule
    downstream applies without knowing they came from griffe rather than from
    parsing source. That is what makes compiled packages work at all.
    """
    anchor = str(module.filepath)
    graph.add_node(anchor, NodeType.FILE, anchor)
    count = 0
    for symbol, kind in _walk_api(module):
        node_type = NodeType.CLASS if "class" in kind else NodeType.FUNCTION
        graph.add_node(f"{anchor}::{symbol}", node_type, anchor)
        count += 1
    graph.commit()
    return count


def _still_defined(new_root, fqn: str) -> bool:
    """Does `fqn` still exist in the new version, despite griffe saying removed?

    griffe drops overloaded members: a method declared as an ``@overload`` pair
    in a stub is absent from `.members` and absent from `.overloads`, so the
    diff reports it removed when it is plainly still there. Left unfiltered this
    fires on most typed and compiled packages, which is enough false positives
    to make the whole report untrustworthy.

    So a claimed removal is re-checked against the new version's own files,
    scoped to the owning class so a same-named method elsewhere cannot mask a
    real removal.
    """
    parent_path, _, name = fqn.rpartition(".")
    if not parent_path:
        return False

    # Read the files, not the object's `.source`. For a compiled package griffe
    # takes members from the .pyi stub while `.source` resolves against the .py
    # shim, which only re-exports from the binary: it returns unrelated text and
    # every check silently fails.
    text = ""
    for path in _api_sources(new_root):
        try:
            text += path.read_text(encoding="utf-8", errors="ignore") + "\n"
        except OSError:
            continue
    if not text:
        return False

    owner = parent_path.rsplit(".", 1)[-1]
    scope = text if owner == new_root.name else _class_block(text, owner)
    if scope is None:
        return False
    return (
        re.search(rf"(?m)^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", scope) is not None
    )


def _api_sources(module) -> list[Path]:
    """The module's own file plus its type stub, when it has one."""
    try:
        path = Path(str(module.filepath))
    except (AttributeError, TypeError):
        return []
    return [p for p in (path, path.with_suffix(".pyi")) if p.is_file()]


def _class_block(text: str, class_name: str) -> str | None:
    """Body of `class <class_name>`, by indentation."""
    match = re.search(rf"(?m)^(\s*)class\s+{re.escape(class_name)}\b", text)
    if match is None:
        return None
    indent = len(match.group(1))
    # Start at the line AFTER the class statement: the remainder of its own
    # line (the colon, a base list) sits at the class's indent and would end
    # the block before it began.
    newline = text.find("\n", match.end())
    if newline == -1:
        return None
    body = []
    for line in text[newline + 1 :].splitlines():
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return "\n".join(body)


def verify(changes: list[ApiChange], new_root) -> list[ApiChange]:
    """Drop changes that do not survive a check against the new version.

    Only removals are checked. A signature change names a symbol that exists on
    both sides by definition, so there is nothing to re-confirm.
    """
    return [c for c in changes if c.kind != "OBJECT_REMOVED" or not _still_defined(new_root, c.fqn)]


def _pair_with_nodes(index: dict[str, str], changes: list[ApiChange]):
    """Match changes to graph nodes, looking through constructors.

    Calling ``Language(...)`` produces an edge to the class, while griffe
    reports the change on ``Language.__init__``. Without this the most common
    breakage shape in practice, a constructor gaining a required argument,
    matches nothing.
    """
    paired = []
    for change in changes:
        node = None
        if change.fqn.endswith(".__init__"):
            # Prefer the class, always. `Language(...)` is an edge to the class,
            # so matching the `__init__` node first finds a node with no callers
            # and reports nothing, which is the silent-empty-report failure.
            node = index.get(change.fqn.removesuffix(".__init__"))
        if node is None:
            node = index.get(change.fqn)
        if node is not None:
            paired.append((node, change))
    return paired


_PARAM = re.compile(r"^\[(?P<kind>[^\]]+)\]\s*(?P<name>\w+)")


def _removed_keyword_param(change: ApiChange) -> str | None:
    """Name of the keyword-only parameter a PARAMETER_REMOVED names, if any.

    griffe renders the old value as ``[keyword-only] timeout_micros: int = None``.
    Positional removals are left alone: they break every caller that reaches
    that position, and proving otherwise needs arity arithmetic this does not do.
    """
    if change.kind != "PARAMETER_REMOVED" or not change.old:
        return None
    match = _PARAM.match(change.old.strip())
    if match is None or "keyword" not in match.group("kind"):
        return None
    return match.group("name")


def _passes_keyword(consumer_node: str, callee: str, keyword: str) -> bool:
    """Does the consumer actually pass `keyword` when it calls `callee`?

    Removing an optional keyword-only parameter breaks exactly the callers that
    pass it and nobody else. Without this check the report flags every call site
    of a function whose signature was merely tidied, which is most of them.
    """
    file_path, _, _ = consumer_node.partition("::")
    try:
        source = Path(file_path).read_bytes()
    except OSError:
        return True  # cannot read it, so cannot rule the call site out
    parser, lock = _get_python_parser()
    with lock:
        root = parser.parse(source).root_node

    for call in _find_nodes_by_type(root, "call"):
        if _find_enclosing_function(call, source, file_path) != consumer_node:
            continue
        func = call.child_by_field_name("function")
        if func is None:
            continue
        name = source[func.start_byte : func.end_byte].decode().split(".")[-1]
        if name != callee:
            continue
        args = call.child_by_field_name("arguments")
        for arg in args.children if args else []:
            if arg.type != "keyword_argument":
                continue
            key = arg.child_by_field_name("name")
            if key is not None and source[key.start_byte : key.end_byte].decode() == keyword:
                return True
    return False


def narrow(impacts: list[Impact]) -> list[Impact]:
    """Drop impacts the call site itself disproves."""
    kept = []
    for impact in impacts:
        keyword = _removed_keyword_param(impact.change)
        if keyword is not None:
            callee = impact.changed.split("::")[-1].split(".")[0]
            if not _passes_keyword(impact.consumer, callee, keyword):
                continue
        kept.append(impact)
    return kept


def _source_files(root: Path) -> list[Path]:
    return [
        p
        for p in sorted(root.rglob("*.py"))
        if not any(part in SKIP_DIRS for part in p.relative_to(root).parts)
    ]


def check_upgrade(
    repo: Path,
    package: str,
    to_version: str,
    distribution: str | None = None,
    from_version: str | None = None,
) -> UpgradeReport:
    """Which symbols in `repo` are touched by upgrading `package`.

    from_version defaults to whatever is installed, which is the question
    actually being asked: what does moving off what I have now cost me.
    """
    dist = distribution or package.replace("_", "-")

    if from_version is None:
        current = griffe.load(package, allow_inspection=True)
        from_label = "installed"
    else:
        current = griffe.load_pypi(package, dist, f"=={from_version}")
        from_label = from_version
    target = griffe.load_pypi(package, dist, f"=={to_version}")

    raw = [
        ApiChange(
            fqn=str(d.get("object_path")),
            kind=getattr(d.get("kind"), "name", str(d.get("kind"))),
            old=None if d.get("old_value") is None else str(d.get("old_value")),
            new=None if d.get("new_value") is None else str(d.get("new_value")),
        )
        for d in (b.as_dict() for b in griffe.find_breaking_changes(current, target))
    ]
    real = verify(raw, target)

    graph = CallGraph(store_dir=repo / ".mnemostack")
    try:
        add_dependency_surface(graph, current)
        files = _source_files(repo)
        for f in files:
            build_nodes_for_python_file(f, graph=graph)
        for f in files:
            link_python_file_imports(f, graph=graph)

        impacts = narrow(impact_report(graph, _pair_with_nodes(fqn_index(graph), real)))
    finally:
        graph.close()

    return UpgradeReport(
        package=package,
        from_version=from_label,
        to_version=to_version,
        total_changes=len(raw),
        verified_changes=len(real),
        impacts=impacts,
    )
