"""Classified API changes between two versions of a package.

Griffe does the classification. It models the API surface rather than hashing
it, so it knows direction: adding a required parameter breaks callers, adding
an optional one does not. A signature fingerprint cannot tell those apart, it
only reports that something changed, so it fires on the modal library commit
and is useless as a gate. Nothing here reimplements that.

This wraps the two-git-ref form, which the upgrade path does not use: that one
compares published versions. Kept for measuring against benchmark datasets of
(ref, ref) pairs, and for pointing this at a library author's own repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import griffe


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
