"""How much a consumer has to care about a change, given how it uses the symbol.

The mapping below is the part of this project with no published equivalent.
Tools that diff an API tell you what changed; tools that find references tell
you where you touch it. Neither says that removing a base class breaks a
subclass and merely concerns a caller, or that a moved parameter breaks a
positional call and does nothing to a type annotation.

Deliberately one hop. Whether a break reaches a caller's caller depends on how
the direct caller is fixed, which nobody has decided when this runs, so walking
further produces plausible output that predicts nothing. The impact-analysis
literature prunes transitively with equivalence relations instead of truncating;
that is the better answer and it is not this one. The reason to stop at one hop
here is actionability: the reader can check a named line in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.reach import RefKind, Site


class Severity(str, Enum):
    BREAK = "break"  # will not run, or silently does the wrong thing
    REVIEW = "review"  # still runs, behaviour may differ
    NONE = "none"  # this use is unaffected


# (breakage kind, how the consumer uses it) -> severity. An absent pair is NONE,
# and the absences carry as much meaning as the entries: an added optional
# parameter never appears because griffe does not call it breaking, and a return
# type change cannot hurt a subclass that never calls the method.
_TRANSMISSION: dict[tuple[str, RefKind], Severity] = {
    # Calling the changed symbol.
    ("OBJECT_REMOVED", RefKind.CALL): Severity.BREAK,
    ("OBJECT_CHANGED_KIND", RefKind.CALL): Severity.BREAK,
    ("PARAMETER_ADDED_REQUIRED", RefKind.CALL): Severity.BREAK,
    ("PARAMETER_REMOVED", RefKind.CALL): Severity.BREAK,
    ("PARAMETER_CHANGED_REQUIRED", RefKind.CALL): Severity.BREAK,
    ("PARAMETER_CHANGED_KIND", RefKind.CALL): Severity.BREAK,
    # Breaks only a caller passing positionally, which the line does not always
    # reveal. Named rather than ranked down: the reader settles it in seconds
    # and guessing wrong in either direction is worse than asking.
    ("PARAMETER_MOVED", RefKind.CALL): Severity.REVIEW,
    ("PARAMETER_CHANGED_DEFAULT", RefKind.CALL): Severity.REVIEW,
    ("RETURN_CHANGED_TYPE", RefKind.CALL): Severity.REVIEW,
    ("ATTRIBUTE_CHANGED_TYPE", RefKind.CALL): Severity.REVIEW,
    ("ATTRIBUTE_CHANGED_VALUE", RefKind.CALL): Severity.REVIEW,
    ("CLASS_REMOVED_BASE", RefKind.CALL): Severity.REVIEW,
    # Subclassing it.
    ("OBJECT_REMOVED", RefKind.SUBCLASS): Severity.BREAK,
    ("OBJECT_CHANGED_KIND", RefKind.SUBCLASS): Severity.BREAK,
    # The subclass loses whatever it inherited from the dropped base, but only
    # breaks if it used one of those members, which is a hop further than a
    # source line shows. Measured on real code this was the single loudest and
    # least useful finding: pydantic 1 to 2 drops an internal `Representation`
    # mixin from BaseModel, and grading that BREAK shouts at every
    # `class X(BaseModel)` in the repo about something almost none of them use.
    ("CLASS_REMOVED_BASE", RefKind.SUBCLASS): Severity.REVIEW,
    # A base method's signature moved under an override that did not. The
    # override is now called with arguments it does not accept.
    ("PARAMETER_ADDED_REQUIRED", RefKind.SUBCLASS): Severity.REVIEW,
    ("PARAMETER_REMOVED", RefKind.SUBCLASS): Severity.REVIEW,
    ("PARAMETER_CHANGED_KIND", RefKind.SUBCLASS): Severity.REVIEW,
    ("PARAMETER_CHANGED_REQUIRED", RefKind.SUBCLASS): Severity.REVIEW,
    # Naming it in an annotation, or mentioning it as a value. Removal breaks
    # the name outright; a signature change only matters if whatever receives
    # the value calls it, which is past what a line of source shows.
    ("OBJECT_REMOVED", RefKind.ANNOTATION): Severity.BREAK,
    ("OBJECT_REMOVED", RefKind.MENTION): Severity.BREAK,
    # Importing it. A removed name fails on the import itself; any other change
    # only matters where the name is used, and each use is a site of its own.
    ("OBJECT_REMOVED", RefKind.IMPORT): Severity.BREAK,
    ("OBJECT_CHANGED_KIND", RefKind.ANNOTATION): Severity.REVIEW,
    ("OBJECT_CHANGED_KIND", RefKind.MENTION): Severity.REVIEW,
    ("PARAMETER_ADDED_REQUIRED", RefKind.MENTION): Severity.REVIEW,
    ("PARAMETER_REMOVED", RefKind.MENTION): Severity.REVIEW,
    ("RETURN_CHANGED_TYPE", RefKind.ANNOTATION): Severity.REVIEW,
    ("ATTRIBUTE_CHANGED_TYPE", RefKind.MENTION): Severity.REVIEW,
    ("ATTRIBUTE_CHANGED_VALUE", RefKind.MENTION): Severity.REVIEW,
}

_ORDER = {Severity.BREAK: 0, Severity.REVIEW: 1, Severity.NONE: 2}


@dataclass(frozen=True)
class Impact:
    """One place in the user's code affected by one change."""

    site: Site
    change: ApiChange
    severity: Severity


def severity_of(change: ApiChange, kind: RefKind) -> Severity:
    return _TRANSMISSION.get((change.kind, kind), Severity.NONE)


def impact_report(sites: list[Site], changes: list[ApiChange]) -> list[Impact]:
    """Affected sites, worst first, one entry per place and change."""
    by_symbol: dict[str, list[ApiChange]] = {}
    for change in changes:
        by_symbol.setdefault(change.fqn.split(".")[-1], []).append(change)

    found: dict[tuple[str, int, str], Impact] = {}
    for site in sites:
        leaf = site.symbol.split(".")[-1]
        for change in by_symbol.get(leaf, ()):
            # The leaf only narrows candidates; the match is on the full path.
            # Matching on the leaf alone made every `__init__`, `execute` and
            # `copy` in a package answer for every call with that name, across
            # unrelated classes. Measured over 20 repos we did not write, that
            # was most of the click and sqlalchemy false positives.
            if change.fqn != site.symbol and not change.fqn.endswith(f".{site.symbol}"):
                continue
            severity = severity_of(change, site.kind)
            if severity is Severity.NONE:
                continue
            if site.guarded and severity is Severity.BREAK:
                # A compatibility shim already expects this change; the line is
                # cleanup, not a break. Across twenty repos, two of the three
                # removals the witness confirmed were exactly this.
                severity = Severity.REVIEW
            # One entry per place and per change. The filters downstream judge each
            # change on its own; collapse() reduces a place to its worst change only
            # after they have run.
            key = (site.file, site.line, change.fqn, change.kind, change.old)
            current = found.get(key)
            if current is None or _ORDER[severity] < _ORDER[current.severity]:
                found[key] = Impact(site=site, change=change, severity=severity)

    return sorted(found.values(), key=_report_order)


def _report_order(impact: Impact) -> tuple:
    return (
        _ORDER[impact.severity],
        impact.site.file,
        impact.site.line,
    )


def collapse(impacts: list[Impact]) -> list[Impact]:
    """The worst surviving finding per place and symbol, in report order.

    Run last. Reducing a place to one change before the filters had judged each
    change let pydantic 2's constr keep its `curtail_length` removal as the
    representative; a filter then dropped it for not passing curtail_length=,
    and `constr(regex=...)`, which does break, went with it. Measured on dstack's
    real pydantic 2 migration.
    """
    best: dict[tuple[str, int, str], Impact] = {}
    for impact in impacts:
        key = (impact.site.file, impact.site.line, impact.change.fqn)
        current = best.get(key)
        if current is None or _ORDER[impact.severity] < _ORDER[current.severity]:
            best[key] = impact
    return sorted(best.values(), key=_report_order)
