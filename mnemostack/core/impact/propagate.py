"""Which symbols break when an API changes.

Deliberately one hop. A breaking change breaks the things that reference the
changed object directly. Whether it reaches a caller's caller depends on how
the direct caller is fixed: absorb the change and nothing propagates, re-raise
or widen its own signature and it does. That is a choice nobody has made yet at
the time this runs, so walking further produces plausible-looking results that
are not predictions of anything.

An earlier design scored paths with a per-hop decay so distant symbols ranked
lower. That is a way of expressing decreasing confidence, but the confidence
does not decrease smoothly with distance, it collapses at the first hop. A
ranking built on it reads as precision the analysis does not have.

What DOES vary, and is modelled here, is the pair of (what changed, how the
consumer references it). A removed base class breaks a subclass and merely
concerns a caller. A parameter that moved breaks a positional caller and does
nothing to a subclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.retrieval.call_graph import CallGraph, EdgeType


class Severity(str, Enum):
    """How much a consumer has to care."""

    BREAK = "break"  # will not run, or silently does the wrong thing
    REVIEW = "review"  # still runs, behaviour may differ
    NONE = "none"  # this reference is unaffected


# (breakage kind, edge type) -> severity. A pair that is absent is NONE, which
# is the useful half of the table: an added optional parameter never appears
# here at all because griffe does not report it as a breakage in the first
# place, and a return type change does not break a subclass that never calls it.
_TRANSMISSION: dict[tuple[str, EdgeType], Severity] = {
    # A caller of the changed object.
    ("OBJECT_REMOVED", EdgeType.CALLS): Severity.BREAK,
    ("OBJECT_CHANGED_KIND", EdgeType.CALLS): Severity.BREAK,
    ("PARAMETER_ADDED_REQUIRED", EdgeType.CALLS): Severity.BREAK,
    ("PARAMETER_REMOVED", EdgeType.CALLS): Severity.BREAK,
    ("PARAMETER_CHANGED_REQUIRED", EdgeType.CALLS): Severity.BREAK,
    ("PARAMETER_CHANGED_KIND", EdgeType.CALLS): Severity.BREAK,
    # Only breaks a caller passing positionally, which this cannot see. Called
    # out rather than ranked down, because the caller can check it in seconds
    # and guessing wrong in either direction is worse than asking.
    ("PARAMETER_MOVED", EdgeType.CALLS): Severity.REVIEW,
    ("PARAMETER_CHANGED_DEFAULT", EdgeType.CALLS): Severity.REVIEW,
    ("RETURN_CHANGED_TYPE", EdgeType.CALLS): Severity.REVIEW,
    ("ATTRIBUTE_CHANGED_TYPE", EdgeType.CALLS): Severity.REVIEW,
    ("ATTRIBUTE_CHANGED_VALUE", EdgeType.CALLS): Severity.REVIEW,
    ("CLASS_REMOVED_BASE", EdgeType.CALLS): Severity.REVIEW,
    # A subclass of the changed object.
    ("OBJECT_REMOVED", EdgeType.INHERITS): Severity.BREAK,
    ("OBJECT_CHANGED_KIND", EdgeType.INHERITS): Severity.BREAK,
    # The subclass keeps working, but every member it inherited from the
    # dropped base is gone. Nothing at the subclass site says so.
    ("CLASS_REMOVED_BASE", EdgeType.INHERITS): Severity.BREAK,
    # A base method signature moved under an override that did not. The
    # subclass still imports and still runs; it is now called with arguments
    # its override does not accept.
    ("PARAMETER_ADDED_REQUIRED", EdgeType.INHERITS): Severity.REVIEW,
    ("PARAMETER_REMOVED", EdgeType.INHERITS): Severity.REVIEW,
    ("PARAMETER_CHANGED_KIND", EdgeType.INHERITS): Severity.REVIEW,
    ("PARAMETER_CHANGED_REQUIRED", EdgeType.INHERITS): Severity.REVIEW,
}

_ORDER = {Severity.BREAK: 0, Severity.REVIEW: 1, Severity.NONE: 2}


@dataclass(frozen=True)
class Impact:
    """One consumer symbol affected by one change."""

    consumer: str  # graph node that references the changed object
    changed: str  # graph node that changed
    change: ApiChange
    edge: EdgeType  # how the consumer reaches it
    severity: Severity
    crosses_repo: bool


def impact_of(graph: CallGraph, changed_node: str, change: ApiChange) -> list[Impact]:
    """Symbols that reference `changed_node`, paired with how much they care.

    Incoming edges only. Retrieval traverses in both directions because it wants
    a neighbourhood; this wants dependents, and following outgoing edges would
    return the things the changed object itself uses, which are not affected by
    it changing.
    """
    out: list[Impact] = []
    changed_repo = graph.node_repo(changed_node)
    for edge in (EdgeType.CALLS, EdgeType.INHERITS):
        severity = _TRANSMISSION.get((change.kind, edge), Severity.NONE)
        if severity is Severity.NONE:
            continue
        for consumer in graph.get_neighbors(
            changed_node, hops=1, direction="incoming", edge_types=[edge]
        ):
            out.append(
                Impact(
                    consumer=consumer,
                    changed=changed_node,
                    change=change,
                    edge=edge,
                    severity=severity,
                    crosses_repo=graph.node_repo(consumer) != changed_repo,
                )
            )
    return out


def impact_report(graph: CallGraph, changes: list[tuple[str, ApiChange]]) -> list[Impact]:
    """Every affected symbol for a set of changes, worst first.

    Ordering is severity, then cross-repo before same-repo. A break in another
    repo is the one nobody is watching: the author of the change does not have
    that code open, and its tests do not run on this PR.
    """
    found = [i for node, change in changes for i in impact_of(graph, node, change)]
    return sorted(
        found,
        key=lambda i: (_ORDER[i.severity], not i.crosses_repo, i.consumer),
    )
