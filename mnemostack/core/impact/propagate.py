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
    """Affected sites, worst first, one entry per place-and-change.

    Uncovered sites sort ahead of covered ones at equal severity. A covered site
    will fail loudly the moment the upgrade lands, so the test suite already
    handles it; an uncovered one is the thing nothing else will tell you.
    """
    by_symbol: dict[str, list[ApiChange]] = {}
    for change in changes:
        by_symbol.setdefault(change.fqn.split(".")[-1], []).append(change)

    found: dict[tuple[str, int, str], Impact] = {}
    for site in sites:
        leaf = site.symbol.split(".")[-1]
        for change in by_symbol.get(leaf, ()):
            severity = severity_of(change, site.kind)
            if severity is Severity.NONE:
                continue
            key = (site.file, site.line, change.fqn)
            current = found.get(key)
            if current is None or _ORDER[severity] < _ORDER[current.severity]:
                found[key] = Impact(site=site, change=change, severity=severity)

    return sorted(
        found.values(),
        key=lambda i: (
            _ORDER[i.severity],
            i.site.covered is True,
            i.site.file,
            i.site.line,
        ),
    )
