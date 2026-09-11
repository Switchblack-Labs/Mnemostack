"""Classified API changes between two git refs, joined to graph nodes.

Griffe does the classification. It models the API surface rather than hashing
it, so it knows direction: adding a required parameter breaks callers, adding
an optional one does not. A signature fingerprint cannot tell those apart, it
only reports that something changed, so it fires on the modal library commit
and is useless as a gate. Nothing here reimplements that.

What this module adds is the join. Griffe identifies an object by dotted module
path (``libx.core.verify``); the call graph identifies it by absolute file path
(``/abs/libx/core.py::verify``). Neither is convertible without knowing the
import root, so the mapping is computed here and the two halves meet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import griffe

from mnemostack.core.retrieval.call_graph import CallGraph
from mnemostack.core.retrieval.import_resolver import find_import_root


@dataclass(frozen=True)
class ApiChange:
    """One breaking change, addressed by dotted path.

    kind is griffe's BreakageKind name, e.g. PARAMETER_ADDED_REQUIRED. It is
    kept as the raw name rather than collapsed to a severity: what counts as
    breaking depends on the consumer, and a caller passing positionally cares
    about PARAMETER_MOVED while one passing by keyword does not.
    """

    fqn: str
    kind: str
    old: str | None
    new: str | None


def breaking_changes(package: str, repo: Path, old_ref: str, new_ref: str) -> list[ApiChange]:
    """Breaking changes in `package` between two git refs of `repo`.

    Do not pass search_paths to load_git. It takes precedence over the worktree
    griffe checks out for the ref, so the "old" side silently loads the current
    working tree and every diff comes back empty.
    """
    old = griffe.load_git(package, ref=old_ref, repo=repo)
    new = griffe.load_git(package, ref=new_ref, repo=repo)
    changes = []
    for breakage in griffe.find_breaking_changes(old, new):
        data = breakage.as_dict()
        kind = data.get("kind")
        changes.append(
            ApiChange(
                fqn=str(data.get("object_path")),
                kind=getattr(kind, "name", str(kind)),
                old=_render(data.get("old_value")),
                new=_render(data.get("new_value")),
            )
        )
    return changes


def _render(value: object) -> str | None:
    return None if value is None else str(value)


@lru_cache(maxsize=4096)
def _module_fqn(file_path: str) -> str | None:
    """Dotted module path for a file, or None if it has no import root."""
    path = Path(file_path)
    root = find_import_root(path)
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = list(rel.parts)
    parts[-1] = parts[-1].removesuffix(".py")
    if parts[-1] == "__init__":
        parts.pop()  # a package is named by its directory, not its __init__
    return ".".join(parts) if parts else None


def fqn_index(graph: CallGraph) -> dict[str, str]:
    """Dotted path -> graph node, for every function and class in the graph.

    A dotted path can collide across repos: two indexed projects may both ship
    ``utils.helpers.parse``. A colliding path is dropped rather than resolved to
    an arbitrary one of them, for the same reason find_indexed_module stays
    quiet on an ambiguous module name.
    """
    index: dict[str, str] = {}
    collisions: set[str] = set()
    rows = graph.db.execute(
        "SELECT qualified_name, file_path FROM nodes WHERE node_type IN (?, ?)",
        ("function", "class"),
    ).fetchall()
    for qname, file_path in rows:
        module = _module_fqn(file_path)
        if module is None:
            continue
        symbol = qname.split("::", 1)[1] if "::" in qname else None
        if not symbol:
            continue
        fqn = f"{module}.{symbol}"
        if fqn in index and index[fqn] != qname:
            collisions.add(fqn)
            continue
        index[fqn] = qname
    for fqn in collisions:
        index.pop(fqn, None)
    return index


def changed_nodes(graph: CallGraph, changes: list[ApiChange]) -> list[tuple[str, ApiChange]]:
    """Pair each change with the graph node it names, dropping unjoined ones.

    Usually a change with no node is uninteresting: the graph only covers repos
    that were indexed, and a package can break an object nobody here imports.

    OBJECT_REMOVED is the exception, and it is the severe one. A deleted object
    has no node at HEAD, and the consumer's edge to it was already dropped at
    link time for want of a target, so nothing here can find its callers. The
    fix is not in this function: index the library at the ref the consumer
    currently resolves against, not at HEAD. Then the removed symbol still has
    a node, the consumer's edges land on it, and the deletion reports like any
    other change. Until that is wired up, deletions are found by griffe and
    then silently lost here, so callers of this must not read an empty result
    as "nothing breaks".
    """
    index = fqn_index(graph)
    return [(index[c.fqn], c) for c in changes if c.fqn in index]
