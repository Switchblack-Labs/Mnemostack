"""Which of your pinned dependencies are safe to bump today.

This is the question no tool answers and no test suite can. Tests can tell you
whether one upgrade broke something, after you do it. They cannot rank forty
pins by what each would cost, because that needs forty upgrades and forty CI
runs. Dependabot's compatibility score tries and cannot be computed for 83% of
updates, because it is built from other people's CI rather than your code.

Only packages the repo actually imports are checked. A lockfile holds hundreds
of transitive dependencies, and a change in one you never name is not something
you can act on.
"""

from __future__ import annotations

import ast
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path

from mnemostack.core.impact.upgrade import UpgradeError, UpgradeReport, check_upgrade
from mnemostack.core.reach.static import SKIP_DIRS, parse_source

PYPI = "https://pypi.org/pypi/{name}/json"


@dataclass(frozen=True)
class SweepRow:
    """One dependency's verdict."""

    package: str  # import name
    distribution: str  # pypi name
    current: str
    latest: str | None
    report: UpgradeReport | None = None
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.latest is None:
            return "unknown"
        if self.latest == self.current:
            return "current"
        if self.report is None:
            return "unchecked"
        if any(i.severity.value == "break" for i in self.report.impacts):
            return "breaks"
        if self.report.impacts:
            return "review"
        if self.report.deprecations:
            return "deprecations"
        return "safe"


def imported_roots(repo: Path) -> set[str]:
    """Top-level module names the repo imports anywhere.

    Read from the AST rather than the lockfile: what a project declares and what
    it actually uses drift apart, and an upgrade only matters for code you name.
    """
    roots: set[str] = set()
    for path in repo.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(repo).parts):
            continue
        try:
            tree = parse_source(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def latest_version(distribution: str, timeout: float = 10.0) -> str | None:
    """Newest release on PyPI, or None if it cannot be determined."""
    try:
        with urllib.request.urlopen(PYPI.format(name=distribution), timeout=timeout) as response:
            return str(json.load(response)["info"]["version"])
    except (urllib.error.URLError, OSError, KeyError, ValueError, TimeoutError):
        return None


def candidates(repo: Path) -> list[tuple[str, str, str]]:
    """(import name, distribution, installed version) for what the repo imports.

    A package the repo imports but has not installed is skipped: without a
    current version there is no upgrade to reason about.
    """
    mapping = packages_distributions()
    found: dict[str, tuple[str, str, str]] = {}
    for root in sorted(imported_roots(repo)):
        for dist in mapping.get(root, []):
            try:
                installed = version(dist)
            except PackageNotFoundError:
                continue
            found.setdefault(dist, (root, dist, installed))
    return sorted(found.values())


def sweep(repo: Path, progress=None) -> list[SweepRow]:
    """Check every imported dependency against its latest release.

    Ordered worst first, so the answer to "what can I take today" is the tail
    and the answer to "what will cost me" is the head.
    """
    rows: list[SweepRow] = []
    todo = candidates(repo)
    for index, (package, distribution, current) in enumerate(todo, start=1):
        if progress:
            progress(index, len(todo), distribution)
        latest = latest_version(distribution)
        if latest is None or latest == current:
            rows.append(SweepRow(package, distribution, current, latest))
            continue
        try:
            report = check_upgrade(
                repo=repo,
                package=package,
                to_version=latest,
                distribution=distribution,
                from_version=current,
            )
        except UpgradeError as exc:
            rows.append(SweepRow(package, distribution, current, latest, error=str(exc)[:120]))
            continue
        rows.append(SweepRow(package, distribution, current, latest, report=report))

    rank = {
        "breaks": 0,
        "review": 1,
        "deprecations": 2,
        "safe": 3,
        "current": 4,
        "unknown": 5,
        "error": 6,
        "unchecked": 7,
    }
    return sorted(rows, key=lambda r: (rank.get(r.status, 9), r.distribution))
