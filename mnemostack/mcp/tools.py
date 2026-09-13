"""MCP surface: one tool, the same question the CLI answers, as data."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("mnemostack")


class AffectedPlace(BaseModel):
    file: str
    line: int
    text: str
    severity: str
    breakage: str
    changed_symbol: str


class DeprecatedUse(BaseModel):
    symbol: str
    message: str
    places: list[str]


class UpgradeImpact(BaseModel):
    package: str
    from_version: str
    to_version: str
    breaking_api_changes: int
    affected: list[AffectedPlace]
    deprecated: list[DeprecatedUse]
    unverified_removals: int


@mcp.tool()
async def upgrade_check(
    package: str,
    to_version: str,
    repo: str = ".",
    from_version: str | None = None,
    distribution: str | None = None,
) -> UpgradeImpact:
    """Which places in `repo` an upgrade of `package` breaks or changes.

    from_version defaults to the version installed in the repo's own virtual
    environment (.venv or venv), not the environment this server runs in.
    """
    from mnemostack.core.impact.upgrade import check_upgrade

    report = check_upgrade(
        repo=Path(repo).resolve(),
        package=package,
        to_version=to_version,
        distribution=distribution,
        from_version=from_version,
    )
    return UpgradeImpact(
        package=report.package,
        from_version=report.from_version,
        to_version=report.to_version,
        breaking_api_changes=report.verified_changes,
        affected=[
            AffectedPlace(
                file=impact.site.file,
                line=impact.site.line,
                text=impact.site.text,
                severity=impact.severity.value,
                breakage=impact.change.kind,
                changed_symbol=impact.change.fqn,
            )
            for impact in report.impacts
        ],
        deprecated=[
            DeprecatedUse(
                symbol=dep.symbol,
                message=dep.message,
                places=[f"{site.file}:{site.line}" for site in dep.sites],
            )
            for dep in report.deprecations
        ],
        unverified_removals=report.unwitnessed,
    )
