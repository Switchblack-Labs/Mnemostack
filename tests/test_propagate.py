"""The severity table: what a change costs, given how the code uses the symbol.

This mapping is the part of the project with no published equivalent, so the
tests are about the distinctions it makes rather than about plumbing. The
interesting cases are the ones where the same change means different things:
a removed base class breaks a subclass and merely concerns a caller.
"""

from __future__ import annotations

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.impact.propagate import Severity, impact_report, severity_of
from mnemostack.core.reach import RefKind, Site


def change(kind: str, fqn: str = "lib.thing", old: str | None = None) -> ApiChange:
    return ApiChange(fqn=fqn, kind=kind, old=old, new=None)


def site(kind: RefKind, symbol: str = "thing", line: int = 1, covered=None) -> Site:
    return Site(file="app.py", line=line, symbol=symbol, kind=kind, text="x", covered=covered)


def test_removed_base_is_review_for_everyone():
    """Measured on real code, and it changed the answer.

    A dropped base only breaks a subclass that used one of its members, which
    is a hop past what a source line shows. Grading it BREAK made pydantic 1 to
    2 shout at every `class X(BaseModel)` in a repo because an internal
    `Representation` mixin disappeared, burying the real findings under a fact
    the reader already knew.
    """
    removed_base = change("CLASS_REMOVED_BASE")
    assert severity_of(removed_base, RefKind.SUBCLASS) is Severity.REVIEW
    assert severity_of(removed_base, RefKind.CALL) is Severity.REVIEW


def test_a_moved_parameter_does_nothing_to_a_subclass():
    moved = change("PARAMETER_MOVED")
    assert severity_of(moved, RefKind.CALL) is Severity.REVIEW
    assert severity_of(moved, RefKind.SUBCLASS) is Severity.NONE


def test_removal_breaks_every_kind_of_reference():
    """A name that is gone is gone: calling, subclassing or naming it all fail."""
    removed = change("OBJECT_REMOVED")
    for kind in (RefKind.CALL, RefKind.SUBCLASS, RefKind.ANNOTATION, RefKind.MENTION):
        assert severity_of(removed, kind) is Severity.BREAK, kind


def test_a_signature_change_does_not_break_an_annotation():
    """Naming a type is unaffected by how its parameters moved."""
    assert severity_of(change("PARAMETER_ADDED_REQUIRED"), RefKind.ANNOTATION) is Severity.NONE


def test_unlisted_pairs_are_none():
    """The absences carry meaning: an unlisted pair must never be reported."""
    assert severity_of(change("RETURN_CHANGED_TYPE"), RefKind.SUBCLASS) is Severity.NONE
    assert severity_of(change("SOMETHING_GRIFFE_ADDED_LATER"), RefKind.CALL) is Severity.NONE


def test_report_matches_sites_to_changes_by_leaf_name():
    changes = [change("OBJECT_REMOVED", "lib.deep.module.verify")]
    sites = [site(RefKind.CALL, symbol="verify")]
    report = impact_report(sites, changes)

    assert len(report) == 1
    assert report[0].severity is Severity.BREAK
    assert report[0].change.fqn == "lib.deep.module.verify"


def test_report_is_ordered_worst_first_then_uncovered_first():
    """Uncovered outranks covered at equal severity.

    A covered site fails loudly the moment the upgrade lands, so the test suite
    already has it. An uncovered one is the thing nothing else will tell you.
    """
    changes = [change("OBJECT_REMOVED", "lib.gone"), change("RETURN_CHANGED_TYPE", "lib.shifted")]
    sites = [
        site(RefKind.CALL, "shifted", line=1, covered=True),
        site(RefKind.CALL, "gone", line=2, covered=True),
        site(RefKind.CALL, "gone", line=3, covered=False),
    ]
    report = impact_report(sites, changes)

    assert [i.severity for i in report] == [Severity.BREAK, Severity.BREAK, Severity.REVIEW]
    assert report[0].site.covered is False, "uncovered breaks come first"


def test_one_entry_per_place_and_change():
    """A line matching twice is one thing for the reader to fix."""
    changes = [change("OBJECT_REMOVED", "lib.gone")]
    sites = [site(RefKind.CALL, "gone", line=7), site(RefKind.MENTION, "gone", line=7)]
    report = impact_report(sites, changes)
    assert len(report) == 1
    assert report[0].severity is Severity.BREAK, "the worst reading of the line wins"


def test_sites_touching_nothing_changed_are_absent():
    assert (
        impact_report([site(RefKind.CALL, "untouched")], [change("OBJECT_REMOVED", "lib.gone")])
        == []
    )


def test_a_change_matches_only_its_own_symbol_not_every_same_named_one():
    """`Option.__init__` changing says nothing about `CliRunner()`.

    Matching on the last component made every `__init__`, `execute` and `copy`
    in a package answer for every call with that name. Over twenty repos we did
    not write, that flagged hundreds of constructor calls in click's own tests,
    `CliRunner()` among them, none of them broken.
    """
    changes = [change("PARAMETER_ADDED_REQUIRED", "click.core.Option.__init__")]
    sites = [
        site(RefKind.CALL, symbol="testing.CliRunner.__init__", line=1),
        site(RefKind.CALL, symbol="core.Option.__init__", line=2),
    ]
    assert [i.site.line for i in impact_report(sites, changes)] == [2]


def test_a_break_inside_a_compatibility_shim_is_review():
    guarded = Site(file="app.py", line=1, symbol="thing", kind=RefKind.CALL, text="x", guarded=True)
    (impact,) = impact_report([guarded], [change("OBJECT_REMOVED")])
    assert impact.severity is Severity.REVIEW


def test_two_parameter_changes_on_one_line_are_kept_apart():
    """pydantic 2 removes both curtail_length and regex from constr.

    One slot per symbol let the first removal win, and narrow() then dropped it
    for not passing curtail_length=, taking `constr(regex=...)` with it.
    """
    from mnemostack.core.impact.upgrade import narrow

    changes = [
        change("PARAMETER_REMOVED", old="[keyword-only] curtail_length: int = None"),
        change("PARAMETER_REMOVED", old="[keyword-only] regex: str = None"),
    ]
    line = Site(file="app.py", line=1, symbol="thing", kind=RefKind.CALL, text="thing(regex='x')")
    (kept,) = narrow(impact_report([line], changes))
    assert "regex" in kept.change.old


def test_collapse_keeps_the_worst_surviving_change_per_place():
    from mnemostack.core.impact.propagate import collapse

    line = site(RefKind.CALL)
    report = impact_report(
        [line],
        [change("PARAMETER_CHANGED_DEFAULT"), change("PARAMETER_REMOVED", old="[positional] x")],
    )
    assert len(report) == 2
    (worst,) = collapse(report)
    assert worst.severity is Severity.BREAK
