"""Sweeping every imported dependency.

The value of this command is the question it answers, which no test suite can:
of forty pins, which are free to take today. Tests can only answer that by
doing forty upgrades. So the tests here are about picking the right packages to
ask about and ranking the answers, not about the upgrade check itself.
"""

from __future__ import annotations

from pathlib import Path

from mnemostack.core.impact.sweep import SweepRow, candidates, imported_roots
from mnemostack.core.impact.upgrade import UpgradeReport


def _row(status_source: dict) -> SweepRow:
    return SweepRow(package="p", distribution="p", current="1.0", latest="2.0", **status_source)


def _report(impacts=(), deprecations=()) -> UpgradeReport:
    return UpgradeReport(
        package="p",
        from_version="1.0",
        to_version="2.0",
        total_changes=1,
        verified_changes=1,
        impacts=list(impacts),
        deprecations=list(deprecations),
    )


def test_imported_roots_reads_the_code_not_the_lockfile(tree):
    """A project's declared dependencies and its used ones drift apart.

    An upgrade only matters for code that names the package, so the roots come
    from the AST. Relative imports are not packages and must not appear.
    """
    repo = tree(
        {
            "a.py": "import requests\nimport os.path\n",
            "b/__init__.py": "",
            "b/c.py": "from flask import Flask\nfrom . import sibling\nfrom .deep import thing\n",
        }
    )
    roots = imported_roots(repo)

    assert {"requests", "os", "flask"} <= roots
    assert "sibling" not in roots and "deep" not in roots


def test_imported_roots_skips_virtualenvs(tree):
    repo = tree({".venv/lib/x.py": "import tensorflow\n", "app.py": "import os\n"})
    assert "tensorflow" not in imported_roots(repo)


def test_candidates_are_installed_and_imported(tmp_path: Path):
    """Both halves are required: a package must be used and have a version."""
    (tmp_path / "app.py").write_text("import griffe\nimport definitely_not_a_real_package_xyz\n")
    found = {dist for _, dist, _ in candidates(tmp_path)}

    assert any("griffe" in d for d in found), "an imported, installed package is a candidate"
    assert "definitely_not_a_real_package_xyz" not in found


def test_status_ranks_breakage_above_everything():
    from mnemostack.core.impact.propagate import Impact, Severity
    from mnemostack.core.reach import RefKind, Site

    site = Site(file="a.py", line=1, symbol="x", kind=RefKind.CALL, text="x()")
    change = type("C", (), {"kind": "OBJECT_REMOVED", "fqn": "p.x"})()

    breaking = _row({"report": _report(impacts=[Impact(site, change, Severity.BREAK)])})
    reviewing = _row({"report": _report(impacts=[Impact(site, change, Severity.REVIEW)])})
    clean = _row({"report": _report()})

    assert breaking.status == "breaks"
    assert reviewing.status == "review"
    assert clean.status == "safe"


def test_a_package_already_on_latest_is_current():
    row = SweepRow(package="p", distribution="p", current="2.0", latest="2.0")
    assert row.status == "current"


def test_an_unreachable_pypi_is_unknown_not_safe():
    """Silence from the network must never read as an all-clear."""
    row = SweepRow(package="p", distribution="p", current="1.0", latest=None)
    assert row.status == "unknown"


def test_a_failed_check_is_an_error_not_a_pass():
    row = SweepRow(package="p", distribution="p", current="1.0", latest="2.0", error="boom")
    assert row.status == "error"
