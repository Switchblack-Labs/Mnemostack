"""Command line entry point.

Bare `mnemostack` still starts the MCP server, because that is how every
configured client already invokes it. Subcommands are additive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUBCOMMANDS = {"upgrade-check", "sweep"}

_ORDER = {"break": 0, "review": 1, "none": 2}


def _fmt(impacts) -> list[str]:
    """Group by the change, rarest first.

    One API change touching twenty classes is one thing to think about and
    twenty places to look, not twenty findings.

    Within a severity, a change reaching few places sorts above one reaching
    many. A change that touches every file is structural and the reader already
    knows it; the one touching two places is the one they do not. Measured on a
    real repo, the loudest finding covered 56 of 56 sites and carried nothing.
    """
    lines: list[str] = []
    groups = finding_groups(impacts)
    for (severity, kind, _, _), (label, places) in sorted(
        groups.items(), key=lambda kv: (_ORDER.get(kv[0][0], 9), len(kv[1][1]), kv[0][2])
    ):
        lines.append(f"  [{severity.upper():6}] {kind}  {label}")
        for site in places[:5]:
            lines.append(f"           {site.file}:{site.line}")
            lines.append(f"             {site.text}")
        if len(places) > 5:
            lines.append(f"           ... and {len(places) - 5} more")
        lines.append("")
    return lines


def finding_groups(impacts) -> dict[tuple, tuple[str, list]]:
    """(severity, kind, name, places) -> (label, one site per place).

    griffe reaches one symbol through every module that re-exports it, so
    `FastMCP` arrives as both `mcp.server.FastMCP` and `mcp.server.fastmcp.FastMCP`
    at the same lines; those fold into one finding under the shortest path.
    Different symbols that share a name, `Session.close` and `Connection.close`,
    reach different lines and stay apart. Keying on the name alone merged them.
    """
    per_change: dict[tuple[str, str, str], dict] = {}
    for impact in impacts:
        key = (impact.severity.value, impact.change.kind, impact.change.fqn)
        per_change.setdefault(key, {}).setdefault((impact.site.file, impact.site.line), impact.site)

    groups: dict[tuple, tuple[str, list]] = {}
    for (severity, kind, fqn), sites in per_change.items():
        key = (severity, kind, fqn.split(".")[-1], frozenset(sites))
        current = groups.get(key)
        if current is None or fqn.count(".") < current[0].count("."):
            groups[key] = (fqn, list(sites.values()))
    return groups


def _fmt_deprecations(deprecations) -> list[str]:
    """Deprecations, rarest first.

    These are not breakage, and that is the point: warnings are suppressed by
    default, so an upgrade passes CI while quietly loading the debt that
    detonates at the next major. A green test run cannot tell you this.
    """
    lines: list[str] = []
    for dep in sorted(deprecations, key=lambda d: (len(d.sites), d.symbol)):
        lines.append(f"  {dep.symbol}")
        if dep.message:
            lines.append(f"           {dep.message}")
        for site in dep.sites[:3]:
            lines.append(f"           {site.file}:{site.line}")
        if len(dep.sites) > 3:
            lines.append(f"           ... and {len(dep.sites) - 3} more")
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
    note = f", ignoring {dropped} removal(s) still defined in its new source" if dropped else ""
    print(
        f"{report.package} {report.from_version} -> {report.to_version}: the library has "
        f"{report.verified_changes} breaking API change(s){note}. Only what reaches your "
        "code is listed."
    )
    if report.unwitnessed:
        print(
            f"  {report.unwitnessed} possible removal(s) not shown: reached through an "
            "object whose type a line cannot reveal, so no import could confirm them."
        )

    if not report.impacts and not report.deprecations:
        print("\n  Nothing in your code touches what changed.")
        return 0

    if report.impacts:
        places = len({(i.site.file, i.site.line) for i in report.impacts})
        # Count the findings the reader will see, not the griffe paths behind them.
        kinds = len(finding_groups(report.impacts))
        print(f"\n  {kinds} change(s) reach your code, across {places} place(s):\n")
        print("\n".join(_fmt(report.impacts)))
    else:
        print("\n  Nothing that changed reaches your code.")

    if report.deprecations:
        total = sum(len(d.sites) for d in report.deprecations)
        print(
            f"\n  Not breaking, but {len(report.deprecations)} thing(s) you use are "
            f"deprecated in {report.to_version}, across {total} place(s):\n"
        )
        print("\n".join(_fmt_deprecations(report.deprecations)))

    return 1 if any(i.severity.value == "break" for i in report.impacts) else 0


_STATUS_NOTE = {
    "breaks": "will break your code",
    "review": "touches your code, worth a look",
    "unverified": "possible removals could not be checked",
    "deprecations": "safe, but you use deprecated API",
    "safe": "nothing you use changed",
    "current": "already on latest",
    "unknown": "could not reach pypi",
    "error": "could not be checked",
}


def sweep_cmd(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mnemostack sweep",
        description="Check every dependency you import against its latest release.",
    )
    parser.add_argument("--repo", default=".", type=Path)
    parser.add_argument("--quiet", action="store_true", help="only show what is not already safe")
    args = parser.parse_args(argv)

    from mnemostack.core.impact.sweep import sweep
    from mnemostack.core.impact.upgrade import UpgradeError

    repo = args.repo.resolve()

    def progress(done: int, total: int, name: str) -> None:
        print(f"\r  checking {done}/{total}: {name:<30}", end="", file=sys.stderr)

    try:
        rows = sweep(repo, progress=progress)
    except UpgradeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\r" + " " * 60, end="\r", file=sys.stderr)

    if not rows:
        print("No installed packages are imported by this repo.")
        return 0

    interesting = [
        r for r in rows if r.status in ("breaks", "review", "unverified", "deprecations")
    ]
    safe = [r for r in rows if r.status == "safe"]
    current = [r for r in rows if r.status == "current"]

    for row in rows:
        if args.quiet and row.status in ("safe", "current"):
            continue
        arrow = f"{row.current} -> {row.latest}" if row.latest else row.current
        detail = ""
        if row.report is not None:
            places = len({(i.site.file, i.site.line) for i in row.report.impacts})
            deps = sum(len(d.sites) for d in row.report.deprecations)
            bits = []
            if places:
                bits.append(f"{places} place(s)")
            if row.report.unwitnessed:
                bits.append(f"{row.report.unwitnessed} unverified removal(s)")
            if deps:
                bits.append(f"{deps} deprecated use(s)")
            detail = f"  [{', '.join(bits)}]" if bits else ""
        if row.error:
            detail = f"  ({row.error})"
        print(f"  {row.status:<13} {row.distribution:<22} {arrow:<24}{detail}")

    print(
        f"\n  {len(safe)} of {len(rows) - len(current)} available upgrade(s) can be taken "
        f"with nothing to change; {len(current)} already current. "
        f"{len(interesting)} need attention."
    )
    return 1 if any(r.status == "breaks" for r in rows) else 0


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        if argv[0] == "upgrade-check":
            raise SystemExit(upgrade_check(argv[1:]))
        if argv[0] == "sweep":
            raise SystemExit(sweep_cmd(argv[1:]))

    from mnemostack.mcp.server import run

    run()


if __name__ == "__main__":
    main()
