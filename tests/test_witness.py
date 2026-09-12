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
from mnemostack.core.impact.witness import uv_resolver, witness_removals
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
    kept, unwitnessed = witness_removals(
        [_impact("OBJECT_REMOVED", "pkg.Still")],
        "pkg",
        "2.0",
        resolve=lambda d, v, p: {"pkg.Still": True},
    )
    assert kept == [] and unwitnessed == 0


def test_removal_whose_path_fails_is_kept():
    impact = _impact("OBJECT_REMOVED", "pkg.Gone")
    kept, _ = witness_removals([impact], "pkg", "2.0", resolve=lambda d, v, p: {"pkg.Gone": False})
    assert kept == [impact]


def test_unwitnessable_removal_is_dropped_and_counted():
    """At three percent measured precision, an unverifiable removal is noise."""
    kept, unwitnessed = witness_removals(
        [_impact("OBJECT_REMOVED", None)], "pkg", "2.0", resolve=lambda d, v, p: {}
    )
    assert kept == [] and unwitnessed == 1


def test_nothing_is_dropped_if_the_witness_cannot_run():
    """Missing uv must not silently hide every finding."""
    impacts = [_impact("OBJECT_REMOVED", "pkg.Gone"), _impact("OBJECT_REMOVED", None, line=2)]
    kept, unwitnessed = witness_removals(impacts, "pkg", "2.0", resolve=lambda d, v, p: None)
    assert kept == impacts and unwitnessed == 0


def test_non_removals_pass_through_untouched():
    other = _impact("PARAMETER_ADDED_REQUIRED", None)
    kept, _ = witness_removals([other], "pkg", "2.0", resolve=lambda d, v, p: {})
    assert kept == [other]


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_uv_resolver_really_imports_in_isolation():
    """The real witness, against a real published version."""
    verdict = uv_resolver("griffe", "1.5.0", ["griffe.load", "griffe.definitely_not_a_real_name"])
    assert verdict == {"griffe.load": True, "griffe.definitely_not_a_real_name": False}
