"""Witnessing removals: a claimed removal survives only if its import really fails.

griffe's models are wrong about re-exports, and across twenty repositories we did
not write that made OBJECT_REMOVED the largest false-positive class. These tests
pin both halves of the fix: the path a witness tests must be the one the code
uses, and the decision must follow what the witness saw.
"""

from __future__ import annotations

import shutil

import pytest

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.impact.propagate import Impact, Severity
from mnemostack.core.impact.witness import (
    uv_probe,
    uv_resolver,
    witness_removals,
    witness_removed_bases,
    witness_signatures,
)
from mnemostack.core.reach import RefKind, Site
from mnemostack.core.reach.static import via_path

# --- via_path: test the path the code uses, not where griffe says it lives ---


def test_from_import_of_a_re_export_uses_the_public_path():
    """The pydantic v2 false positive, which this repo reported as real."""
    via = via_path(
        "pydantic",
        "error_wrappers.ValidationError",
        {"ValidationError": "ValidationError"},
        "from pydantic import ValidationError",
        "ValidationError",
    )
    assert via == "pydantic.ValidationError"


def test_method_on_an_imported_class_extends_the_import():
    via = via_path("paylib", "Session.drain", {"Session": "Session"}, "return s.drain()", "drain")
    assert via == "paylib.Session.drain"


def test_module_import_is_read_off_the_line():
    via = via_path(
        "jinja2",
        "utils.contextfunction",
        {"jinja2": ""},
        "pass_context = jinja2.contextfunction",
        "contextfunction",
    )
    assert via == "jinja2.contextfunction"


def test_an_untyped_receiver_has_no_path():
    """`f.read()` reaches nothing a single import can test."""
    assert via_path("werkzeug", "wsgi.read", {"werkzeug": ""}, "expect = f.read()", "read") is None


# --- witness_removals: the decision follows the witness ---------------------


def _impact(kind: str, via: str | None, line: int = 1) -> Impact:
    site = Site(file="a.py", line=line, symbol="x", kind=RefKind.CALL, text="x", via=via)
    change = ApiChange(fqn="pkg.x", kind=kind, old=None, new=None)
    return Impact(site=site, change=change, severity=Severity.BREAK)


def test_removal_whose_path_still_resolves_is_dropped():
    kept, unwitnessed, _ = witness_removals(
        [_impact("OBJECT_REMOVED", "pkg.Still")],
        "pkg",
        "2.0",
        resolve=lambda d, v, p: {"pkg.Still": True},
    )
    assert kept == [] and unwitnessed == 0


def test_removal_whose_path_fails_is_kept():
    impact = _impact("OBJECT_REMOVED", "pkg.Gone")
    kept, _, _ = witness_removals(
        [impact], "pkg", "2.0", resolve=lambda d, v, p: {"pkg.Gone": False}
    )
    assert kept == [impact]


def test_unwitnessable_removal_is_dropped_and_counted():
    """At three percent measured precision, an unverifiable removal is noise."""
    kept, unwitnessed, _ = witness_removals(
        [_impact("OBJECT_REMOVED", None)], "pkg", "2.0", resolve=lambda d, v, p: {}
    )
    assert kept == [] and unwitnessed == 1


def test_nothing_is_dropped_if_the_witness_cannot_run():
    """Missing uv must not silently hide every finding."""
    impacts = [_impact("OBJECT_REMOVED", "pkg.Gone"), _impact("OBJECT_REMOVED", None, line=2)]
    kept, unwitnessed, _ = witness_removals(impacts, "pkg", "2.0", resolve=lambda d, v, p: None)
    assert kept == impacts and unwitnessed == 0


def test_non_removals_pass_through_untouched():
    other = _impact("PARAMETER_ADDED_REQUIRED", None)
    kept, _, _ = witness_removals([other], "pkg", "2.0", resolve=lambda d, v, p: {})
    assert kept == [other]


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_uv_resolver_really_imports_in_isolation():
    """The real witness, against a real published version."""
    verdict = uv_resolver("griffe", "1.5.0", ["griffe.load", "griffe.definitely_not_a_real_name"])
    assert verdict == {"griffe.load": True, "griffe.definitely_not_a_real_name": False}


# --- witness_signatures: compare the running code in both versions ----------


def _describe(signature, kind="callable", resolves=True):
    return {"resolves": resolves, "kind": kind, "signature": signature}


def _sig_impact(kind: str, via: str | None = "pkg.f", severity=Severity.BREAK) -> Impact:
    site = Site(file="a.py", line=1, symbol="f", kind=RefKind.CALL, text="f()", via=via)
    change = ApiChange(fqn="pkg.f", kind=kind, old=None, new=None)
    return Impact(site=site, change=change, severity=severity)


def _probe(before: dict, after: dict):
    return lambda dist, version, paths: before if version == "1.0" else after


def test_signature_identical_in_both_versions_is_dropped():
    """griffe said click.Argument gained a required parameter. It did not."""
    same = {"pkg.f": _describe([["x", "POSITIONAL_OR_KEYWORD", None]])}
    kept = witness_signatures(
        [_sig_impact("PARAMETER_ADDED_REQUIRED")], "pkg", "1.0", "2.0", probe=_probe(same, same)
    )
    assert kept == []


def test_signature_that_really_changed_is_kept():
    before = {"pkg.f": _describe([["x", "POSITIONAL_OR_KEYWORD", None]])}
    after = {
        "pkg.f": _describe(
            [["x", "POSITIONAL_OR_KEYWORD", None], ["token", "POSITIONAL_OR_KEYWORD", None]]
        )
    }
    impact = _sig_impact("PARAMETER_ADDED_REQUIRED")
    kept = witness_signatures([impact], "pkg", "1.0", "2.0", probe=_probe(before, after))
    assert kept == [impact]


def test_kind_change_compares_kind_not_signature():
    before = {"pkg.f": _describe(None, kind="callable")}
    after = {"pkg.f": _describe(None, kind="class")}
    impact = _sig_impact("OBJECT_CHANGED_KIND")
    assert witness_signatures([impact], "pkg", "1.0", "2.0", probe=_probe(before, after)) == [
        impact
    ]


def test_unprobeable_break_is_demoted_not_dropped():
    """A C extension with no signature is unverified, not wrong."""
    blank = {"pkg.f": _describe(None)}
    kept = witness_signatures(
        [_sig_impact("PARAMETER_REMOVED")], "pkg", "1.0", "2.0", probe=_probe(blank, blank)
    )
    assert len(kept) == 1 and kept[0].severity is Severity.REVIEW


def test_nothing_changes_when_the_probe_cannot_run_or_old_version_is_unknown():
    impact = _sig_impact("PARAMETER_REMOVED")
    assert witness_signatures([impact], "pkg", "1.0", "2.0", probe=lambda d, v, p: None) == [impact]
    assert witness_signatures([impact], "pkg", None, "2.0", probe=lambda d, v, p: {}) == [impact]


def test_removals_are_not_touched_by_signature_witnessing():
    removal = _sig_impact("OBJECT_REMOVED")
    assert witness_signatures([removal], "pkg", "1.0", "2.0", probe=lambda d, v, p: {}) == [removal]


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_uv_probe_describes_a_real_signature():
    described = uv_probe("griffe", "1.5.0", ["griffe.load"])
    assert described is not None
    info = described["griffe.load"]
    assert info["resolves"] is True and info["kind"] == "callable"
    assert any(name == "objspec" for name, _, _ in info["signature"])


# --- a changed signature is cleared only if this call still fits it ----------


def _call_impact(text: str, kind: str = "PARAMETER_ADDED_REQUIRED") -> Impact:
    site = Site(file="a.py", line=1, symbol="f", kind=RefKind.CALL, text=text, via="pkg.f")
    change = ApiChange(fqn="pkg.f", kind=kind, old=None, new=None)
    return Impact(site=site, change=change, severity=Severity.BREAK)


def test_changed_signature_that_this_call_still_fits_is_dropped():
    before = {"pkg.f": _describe([["x", "POSITIONAL_OR_KEYWORD", None]])}
    after = {"pkg.f": _describe([["x", "POSITIONAL_OR_KEYWORD", None], ["y", "KEYWORD_ONLY", "0"]])}
    kept = witness_signatures(
        [_call_impact("f(1)")], "pkg", "1.0", "2.0", probe=_probe(before, after)
    )
    assert kept == []


def test_changed_signature_that_this_call_no_longer_fits_is_kept():
    before = {"pkg.f": _describe([["x", "POSITIONAL_OR_KEYWORD", None]])}
    after = {
        "pkg.f": _describe([["x", "POSITIONAL_OR_KEYWORD", None], ["y", "KEYWORD_ONLY", None]])
    }
    impact = _call_impact("f(1)")
    assert witness_signatures([impact], "pkg", "1.0", "2.0", probe=_probe(before, after)) == [
        impact
    ]


def test_a_moved_parameter_is_not_cleared_by_binding():
    """It can bind while handing a positional argument to a different parameter."""
    before = {
        "pkg.f": _describe(
            [["a", "POSITIONAL_OR_KEYWORD", None], ["b", "POSITIONAL_OR_KEYWORD", None]]
        )
    }
    after = {
        "pkg.f": _describe(
            [["b", "POSITIONAL_OR_KEYWORD", None], ["a", "POSITIONAL_OR_KEYWORD", None]]
        )
    }
    impact = _call_impact("f(1, 2)", kind="PARAMETER_MOVED")
    assert witness_signatures([impact], "pkg", "1.0", "2.0", probe=_probe(before, after)) == [
        impact
    ]


def test_a_multi_line_call_is_bound_from_its_file(tmp_path):
    (tmp_path / "a.py").write_text("x = f(\n    1,\n)\n")
    before = {"pkg.f": _describe([["x", "POSITIONAL_OR_KEYWORD", None]])}
    after = {"pkg.f": _describe([["x", "POSITIONAL_OR_KEYWORD", None], ["y", "KEYWORD_ONLY", "0"]])}
    impact = _call_impact("x = f(")
    probe = _probe(before, after)
    # From the line alone the call is undecided: the change stands, graded REVIEW.
    (undecided,) = witness_signatures([impact], "pkg", "1.0", "2.0", probe=probe)
    assert undecided.severity is Severity.REVIEW
    assert witness_signatures([impact], "pkg", "1.0", "2.0", probe=probe, repo=tmp_path) == []


# --- a removal that still imports behind a deprecation warning ---------------


def test_removal_that_resolves_with_a_warning_is_returned_as_deprecated():
    impact = _impact("OBJECT_REMOVED", "pkg.generics.Old")
    kept, unwitnessed, warned = witness_removals(
        [impact], "pkg", "2.0", resolve=lambda d, v, p: {"pkg.generics.Old": "Old moved to pkg.New"}
    )
    assert kept == [] and unwitnessed == 0
    assert warned == [("pkg.generics.Old", "Old moved to pkg.New", impact.site)]


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_uv_resolver_reports_pydantic_2_migration_warnings():
    """Measured misses on dstack's and distiller's real pydantic 2 migrations."""
    verdict = uv_resolver(
        "pydantic", "2.9.2", ["pydantic.generics.GenericModel", "pydantic.parse_file_as"]
    )
    assert verdict is not None
    assert "GenericModel" in verdict["pydantic.generics.GenericModel"]
    assert verdict["pydantic.parse_file_as"] is False


# --- removed parameters, decided by binding rather than by reading one line --


def test_a_multi_line_call_passing_a_removed_parameter_is_kept(tmp_path):
    """`post(\\n "u",\\n verify=False,\\n)` with verify removed was silently dropped."""
    (tmp_path / "a.py").write_text('f(\n    "u",\n    verify=False,\n)\n')
    before = {
        "pkg.f": _describe(
            [["url", "POSITIONAL_OR_KEYWORD", None], ["verify", "POSITIONAL_OR_KEYWORD", "True"]]
        )
    }
    after = {"pkg.f": _describe([["url", "POSITIONAL_OR_KEYWORD", None]])}
    impact = _call_impact("f(", kind="PARAMETER_REMOVED")
    kept = witness_signatures(
        [impact], "pkg", "1.0", "2.0", probe=_probe(before, after), repo=tmp_path
    )
    assert kept == [impact]


def test_a_positional_call_reaching_a_removed_positional_parameter_is_kept():
    before = {
        "pkg.f": _describe(
            [["a", "POSITIONAL_OR_KEYWORD", None], ["b", "POSITIONAL_OR_KEYWORD", "0"]]
        )
    }
    after = {"pkg.f": _describe([["a", "POSITIONAL_OR_KEYWORD", None]])}
    reaching = _call_impact("f(1, 2)", kind="PARAMETER_REMOVED")
    short = _call_impact("f(1)", kind="PARAMETER_REMOVED")
    kept = witness_signatures([reaching, short], "pkg", "1.0", "2.0", probe=_probe(before, after))
    assert kept == [reaching]


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_uv_probe_drops_a_methods_instance_parameter():
    """Binding used to skip a first parameter only when it was spelled `self`."""
    described = uv_probe("griffe", "1.5.0", ["griffe.GriffeLoader.load"])
    assert described is not None
    names = [name for name, _, _ in described["griffe.GriffeLoader.load"]["signature"]]
    assert names and names[0] != "self"


def test_a_call_whose_arguments_are_not_in_the_source_is_review_not_break():
    """`Column(sa_type, *args, **kwargs)`: the change is real, the break is not shown.

    narrow() used to drop these silently; binding cannot decide them either way.
    """
    before = {
        "pkg.f": _describe(
            [["a", "POSITIONAL_OR_KEYWORD", None], ["b", "POSITIONAL_OR_KEYWORD", "0"]]
        )
    }
    after = {"pkg.f": _describe([["a", "POSITIONAL_OR_KEYWORD", None]])}
    impact = _call_impact("f(x, *args)", kind="PARAMETER_REMOVED")
    (kept,) = witness_signatures([impact], "pkg", "1.0", "2.0", probe=_probe(before, after))
    assert kept.severity is Severity.REVIEW


# --- a removed base class is only the public members it took ----------------


def _base_impact(via: str | None = "pkg.Model") -> Impact:
    site = Site(
        file="a.py",
        line=1,
        symbol="Model",
        kind=RefKind.SUBCLASS,
        text="class User(Model):",
        via=via,
    )
    change = ApiChange(
        fqn="pkg.main.Model", kind="CLASS_REMOVED_BASE", old="Representation", new=None
    )
    return Impact(site=site, change=change, severity=Severity.REVIEW)


def _class(members):
    return {
        "resolves": True,
        "kind": "class",
        "signature": None,
        "warning": None,
        "members": members,
    }


def test_a_removed_base_that_took_nothing_public_is_dropped():
    """click 8's CliRunner lost a base and no public member: 118 findings of nothing."""
    same = {"pkg.Model": _class(["invoke", "isolation"])}
    kept, lost = witness_removed_bases(
        [_base_impact()], "pkg", "1.0", "2.0", probe=_probe(same, same)
    )
    assert kept == [] and lost == {}


def test_a_removed_base_becomes_the_members_it_took():
    """pydantic 2's BaseModel lost its bases and exactly one public member, Config."""
    before = {"pkg.Model": _class(["Config", "dict"])}
    after = {"pkg.Model": _class(["dict", "model_dump"])}
    kept, lost = witness_removed_bases(
        [_base_impact()], "pkg", "1.0", "2.0", probe=_probe(before, after)
    )
    assert kept == [] and lost == {"pkg.main.Model": {"Config"}}


def test_a_class_that_cannot_be_described_keeps_its_finding():
    impact = _base_impact()
    kept, lost = witness_removed_bases([impact], "pkg", "1.0", "2.0", probe=_probe({}, {}))
    assert kept == [impact] and lost == {}
    assert witness_removed_bases([impact], "pkg", None, "2.0", probe=_probe({}, {})) == (
        [impact],
        {},
    )


# --- a changed default or a moved parameter, judged per call ----------------


def _described_change(text: str, kind: str, old: str) -> Impact:
    from dataclasses import replace

    impact = _call_impact(text, kind=kind)
    return replace(impact, change=replace(impact.change, old=old), severity=Severity.REVIEW)


def test_a_changed_default_does_not_reach_a_call_that_passes_the_parameter():
    before = {
        "pkg.f": _describe(
            [["x", "POSITIONAL_OR_KEYWORD", None], ["strict", "KEYWORD_ONLY", "False"]]
        )
    }
    after = {
        "pkg.f": _describe(
            [["x", "POSITIONAL_OR_KEYWORD", None], ["strict", "KEYWORD_ONLY", "True"]]
        )
    }
    old = "[keyword-only] strict: bool = False"
    passing = _described_change("f(1, strict=True)", "PARAMETER_CHANGED_DEFAULT", old)
    omitting = _described_change("f(1)", "PARAMETER_CHANGED_DEFAULT", old)
    kept = witness_signatures([passing, omitting], "pkg", "1.0", "2.0", probe=_probe(before, after))
    assert kept == [omitting]


def test_a_default_that_is_plainly_the_same_is_dropped_and_an_opaque_one_is_kept():
    """Other parameters changed, so the signatures differ; this default did not."""
    same = "[keyword-only] strict: bool = False"
    before = {
        "pkg.f": _describe(
            [["x", "POSITIONAL_OR_KEYWORD", None], ["strict", "KEYWORD_ONLY", "False"]]
        )
    }
    after = {
        "pkg.f": _describe(
            [["y", "POSITIONAL_OR_KEYWORD", None], ["strict", "KEYWORD_ONLY", "False"]]
        )
    }
    plain = _described_change("f(1)", "PARAMETER_CHANGED_DEFAULT", same)
    assert witness_signatures([plain], "pkg", "1.0", "2.0", probe=_probe(before, after)) == []

    opaque_before = {
        "pkg.f": _describe(
            [["x", "POSITIONAL_OR_KEYWORD", None], ["d", "KEYWORD_ONLY", "<object>"]]
        )
    }
    opaque_after = {
        "pkg.f": _describe(
            [["y", "POSITIONAL_OR_KEYWORD", None], ["d", "KEYWORD_ONLY", "<object>"]]
        )
    }
    opaque = _described_change(
        "f(1)", "PARAMETER_CHANGED_DEFAULT", "[keyword-only] d: Any = Undefined"
    )
    probe = _probe(opaque_before, opaque_after)
    assert witness_signatures([opaque], "pkg", "1.0", "2.0", probe=probe) == [opaque]


def test_a_moved_parameter_does_not_reach_a_call_passing_it_by_keyword():
    """click.style(text, fg=color): nothing positional moved under it."""
    before = {
        "pkg.f": _describe(
            [
                ["a", "POSITIONAL_OR_KEYWORD", None],
                ["b", "POSITIONAL_OR_KEYWORD", "0"],
                ["c", "POSITIONAL_OR_KEYWORD", "0"],
            ]
        )
    }
    after = {
        "pkg.f": _describe(
            [
                ["a", "POSITIONAL_OR_KEYWORD", None],
                ["c", "POSITIONAL_OR_KEYWORD", "0"],
                ["b", "POSITIONAL_OR_KEYWORD", "0"],
            ]
        )
    }
    old = "[positional or keyword] b: int = 0"
    by_keyword = _described_change("f(1, c=3)", "PARAMETER_MOVED", old)
    positional = _described_change("f(1, 2)", "PARAMETER_MOVED", old)
    kept = witness_signatures(
        [by_keyword, positional], "pkg", "1.0", "2.0", probe=_probe(before, after)
    )
    assert kept == [positional]
