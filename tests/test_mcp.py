"""The MCP tool: the same report, as data an agent can read."""

from __future__ import annotations

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.impact.propagate import Impact, Severity
from mnemostack.core.impact.upgrade import Deprecation, UpgradeReport
from mnemostack.core.reach import RefKind, Site
from mnemostack.mcp import tools


def _report() -> UpgradeReport:
    site = Site(file="app.py", line=3, symbol="connect", kind=RefKind.CALL, text="connect(url)")
    change = ApiChange(fqn="paylib.connect", kind="PARAMETER_ADDED_REQUIRED", old=None, new=None)
    return UpgradeReport(
        package="paylib",
        from_version="1.0",
        to_version="2.0",
        total_changes=3,
        verified_changes=2,
        impacts=[Impact(site, change, Severity.BREAK)],
        deprecations=[Deprecation(symbol="Model.dict", message="use model_dump", sites=[site])],
        unwitnessed=1,
    )


async def test_the_tool_returns_every_finding(monkeypatch):
    """It read a field Impact no longer has, so any finding at all crashed it."""
    monkeypatch.setattr("mnemostack.core.impact.upgrade.check_upgrade", lambda **kwargs: _report())
    result = await tools.upgrade_check(package="paylib", to_version="2.0")

    (place,) = result.affected
    assert (place.file, place.line, place.severity, place.breakage) == (
        "app.py",
        3,
        "break",
        "PARAMETER_ADDED_REQUIRED",
    )
    assert result.deprecated[0].places == ["app.py:3"]
    assert result.unverified_removals == 1
