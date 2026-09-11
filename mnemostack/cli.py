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


def _fmt(impacts, repo: Path) -> list[str]:
    lines = []
    seen = set()
    for i in impacts:
        key = (i.consumer, i.change.fqn)
        if key in seen:
            continue
        seen.add(key)
        file_part, _, symbol = i.consumer.partition("::")
        where = Path(file_part)
        try:
            where = where.relative_to(repo)
        except ValueError:
            pass
        lines.append(f"  [{i.severity.value.upper():6}] {where}::{symbol}")
        detail = f"{i.change.kind}  {i.change.fqn}"
        lines.append(f"           {detail}")
        if i.change.old and i.change.new and i.change.old != i.change.new:
            lines.append(f"           {i.change.old} -> {i.change.new}")
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

    from mnemostack.core.impact.upgrade import check_upgrade

    repo = args.repo.resolve()
    report = check_upgrade(
        repo=repo,
        package=args.package,
        to_version=args.to_version,
        distribution=args.distribution,
        from_version=args.from_version,
    )

    dropped = report.total_changes - report.verified_changes
    note = f" ({dropped} unverified dropped)" if dropped else ""
    print(
        f"{report.package} {report.from_version} -> {report.to_version}: "
        f"{report.verified_changes} breaking change(s){note}"
    )

    if not report.impacts:
        print("\n  Nothing in your code touches what changed.")
        return 0

    print(f"\n  {len(report.impacts)} place(s) in your code touch what changed:\n")
    print("\n".join(_fmt(report.impacts, repo)))
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
