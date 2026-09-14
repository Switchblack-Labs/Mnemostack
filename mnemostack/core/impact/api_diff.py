"""Classified API changes between two versions of a package.

Griffe does the classification. It models the API surface rather than hashing
it, so it knows direction: adding a required parameter breaks callers, adding
an optional one does not.
"""

from __future__ import annotations

from dataclasses import dataclass


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
