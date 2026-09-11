"""MCP surface: one tool, the same question the CLI answers.

The server used to expose eight tools across semantic retrieval and session
memory. Both halves are gone; what survives is the upgrade check, so an agent
can ask it the same thing a human asks at the command line.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("mnemostack")


class AffectedSymbol(BaseModel):
    symbol: str
    file: str
    severity: str
    breakage: str
    changed_symbol: str


class UpgradeImpact(BaseModel):
    package: str
    from_version: str
    to_version: str
    breaking_changes: int
    affected: list[AffectedSymbol]


@mcp.tool()
async def upgrade_check(
    package: str,
    to_version: str,
    repo: str = ".",
    from_version: str | None = None,
    distribution: str | None = None,
) -> UpgradeImpact:
    """Which symbols in `repo` are affected by upgrading `package`.

    from_version defaults to the installed version, which is the question
    normally being asked: what does moving off what I have now cost me.
    """
    from mnemostack.core.impact.upgrade import check_upgrade

    root = Path(repo).resolve()
    report = check_upgrade(
        repo=root,
        package=package,
        to_version=to_version,
        distribution=distribution,
        from_version=from_version,
    )
    affected = []
    for impact in report.impacts:
        file_part, _, symbol = impact.consumer.partition("::")
        where = Path(file_part)
        try:
            where = where.relative_to(root)
        except ValueError:
            pass
        affected.append(
            AffectedSymbol(
                symbol=symbol,
                file=str(where),
                severity=impact.severity.value,
                breakage=impact.change.kind,
                changed_symbol=impact.change.fqn,
            )
        )
    return UpgradeImpact(
        package=report.package,
        from_version=report.from_version,
        to_version=report.to_version,
        breaking_changes=report.verified_changes,
        affected=affected,
    )
