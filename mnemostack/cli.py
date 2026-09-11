"""Command line entry point.

Bare `mnemostack` still starts the MCP server, because that is how every
configured client already invokes it. Subcommands are additive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# "serve" is not listed: bare `mnemostack` already starts the server, and
# having it match here made it fall through every branch to run() while
# silently discarding whatever arguments followed it.
SUBCOMMANDS = {"upgrade-check"}


def _fmt(impacts) -> list[str]:
    """Group by the change, list where it lands.

    One API change touching twenty classes is one thing to think about and
    twenty places to look, not twenty findings. Printing it per site was
    measured on real repos and buried three real breakages under twenty-two
    repetitions of a single fact.
    """
    grouped: dict[tuple[str, str, str], list] = {}
    for i in impacts:
        grouped.setdefault((i.severity.value, i.change.kind, i.change.fqn), []).append(i.site)

    lines = []
    for (severity, kind, fqn), sites in grouped.items():
        lines.append(f"  [{severity.upper():6}] {kind}  {fqn}")
        shown = sites[:5]
        for site in shown:
            cover = ""
            if site.covered is False:
                cover = "  (not covered by tests)"
            elif site.covered is True:
                cover = "  (covered)"
            lines.append(f"           {site.file}:{site.line}{cover}")
            lines.append(f"             {site.text}")
        if len(sites) > len(shown):
            lines.append(f"           ... and {len(sites) - len(shown)} more")
        lines.append("")
    return lines


def upgrade_check(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mnemostack upgrade-check",
        description="Show which symbols in your code an upgrade would touch.",
    )
    parser.add_argument("package", help="import name, e.g. tree_sitter")
    parser.add_argument("to_version", help="version to upgrade to, e.g. 0.26.0")
    parser.add_argument("--from-version", default=None, help="default: installed")
    parser.add_argument("--distribution", default=None, help="pypi name if it differs")
    parser.add_argument("--repo", default=".", type=Path)
    args = parser.parse_args(argv)

    from mnemostack.core.impact.upgrade import UpgradeError, check_upgrade

    repo = args.repo.resolve()
    try:
        report = check_upgrade(
            repo=repo,
            package=args.package,
            to_version=args.to_version,
            distribution=args.distribution,
            from_version=args.from_version,
        )
    except UpgradeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    dropped = report.total_changes - report.verified_changes
    note = f" ({dropped} unverified dropped)" if dropped else ""
    print(
        f"{report.package} {report.from_version} -> {report.to_version}: "
        f"{report.verified_changes} breaking change(s){note}"
    )

    if not report.impacts:
        print("\n  Nothing in your code touches what changed.")
        return 0

    places = len(report.impacts)
    kinds = len({(i.change.kind, i.change.fqn) for i in report.impacts})
    print(f"\n  {kinds} change(s) reach your code, across {places} place(s):\n")
    print("\n".join(_fmt(report.impacts)))
    return 1 if any(i.severity.value == "break" for i in report.impacts) else 0


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        command, rest = argv[0], argv[1:]
        if command == "upgrade-check":
            raise SystemExit(upgrade_check(rest))

    from mnemostack.mcp.server import run

    run()


if __name__ == "__main__":
    main()
