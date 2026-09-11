"""What breaks in your code if you upgrade a dependency, before you upgrade.

griffe answers the library half: load_pypi fetches any published version without
installing it, and the diff is directional, so adding a required parameter is
breaking and adding an optional one is not. A type checker cannot answer this at
all before the upgrade, because there is nothing to check until you commit to it.

The consumer half is a scan, not a graph. The graph that used to live here
resolved names itself, reached 29% of what Pyright finds, hung on a quarter of
real packages, and wrote gigabytes into the user's repository. Reading imports
and scanning lines finds more, costs precision, and cannot hang.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import griffe

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.impact.propagate import Impact, impact_report
from mnemostack.core.reach import RefKind, Site
from mnemostack.core.reach.static import find_sites


class UpgradeError(RuntimeError):
    """A failure the user can act on, rather than a traceback."""


@dataclass(frozen=True)
class UpgradeReport:
    package: str
    from_version: str
    to_version: str
    total_changes: int
    verified_changes: int
    impacts: list[Impact]


def _load(package: str, distribution: str, version: str | None):
    try:
        if version is None:
            return griffe.load(package, allow_inspection=True)
        return griffe.load_pypi(package, distribution, f"=={version}")
    except Exception as exc:  # griffe raises several unrelated types
        target = version or "installed"
        raise UpgradeError(f"could not load {package} {target}: {exc}") from exc


def _still_defined(new_root, fqn: str) -> bool:
    """Does `fqn` survive in the new version, despite griffe saying removed?

    griffe drops members declared as an ``@overload`` pair, so it reports methods
    that are plainly still there as removed. Left alone this fires on most typed
    and compiled packages, which is enough noise to make a report untrustworthy.

    The check reads the file griffe says the symbol's owner lives in. An earlier
    version read the package's top-level __init__ instead, which for any modern
    package is a re-export facade containing none of the definitions, so it
    always returned False and the filter silently never fired.
    """
    owner_path, _, name = fqn.rpartition(".")
    if not owner_path:
        return False
    relative = owner_path.removeprefix(f"{new_root.name}.")
    try:
        owner = new_root if not relative else new_root[relative]
        source = Path(str(owner.filepath))
    except (KeyError, AttributeError, TypeError, ValueError):
        return False

    for candidate in (source, source.with_suffix(".pyi")):
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(rf"(?m)^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", text):
            return True
    return False


def verify(changes: list[ApiChange], new_root) -> list[ApiChange]:
    """Drop claimed removals that the new version's own source contradicts."""
    return [c for c in changes if c.kind != "OBJECT_REMOVED" or not _still_defined(new_root, c.fqn)]


_PARAM = re.compile(r"^\[(?P<kind>[^\]]+)\]\s*(?P<name>\w+)")


def _removed_keyword_param(change: ApiChange) -> str | None:
    """Name of the keyword-only parameter a PARAMETER_REMOVED names, if any.

    Positional removals are left alone: they break every caller that reaches
    that position, and proving otherwise needs arity arithmetic this does not do.
    """
    if change.kind != "PARAMETER_REMOVED" or not change.old:
        return None
    match = _PARAM.match(change.old.strip())
    if match is None or "keyword" not in match.group("kind"):
        return None
    return match.group("name")


def narrow(impacts: list[Impact]) -> list[Impact]:
    """Drop findings the source line itself disproves.

    Removing an optional keyword-only parameter breaks exactly the callers that
    pass it. The site carries its own line, so this is a regex rather than a
    second parse of the file, and it only applies to calls: a subclass or an
    annotation never passed the argument in the first place.
    """
    kept = []
    for impact in impacts:
        keyword = _removed_keyword_param(impact.change)
        if keyword is not None and impact.site.kind is RefKind.CALL:
            if not re.search(rf"\b{re.escape(keyword)}\s*=", impact.site.text):
                continue
        kept.append(impact)
    return kept


def changed_symbols(changes: list[ApiChange], package: str) -> set[str]:
    """Dotted paths below the package, for the symbols that changed."""
    root = package.split(".")[0]
    out = set()
    for change in changes:
        relative = change.fqn.removeprefix(f"{root}.")
        if relative and relative != change.fqn:
            out.add(relative)
    return out


def check_upgrade(
    repo: Path,
    package: str,
    to_version: str,
    distribution: str | None = None,
    from_version: str | None = None,
    sites: list[Site] | None = None,
) -> UpgradeReport:
    """Which places in `repo` are affected by upgrading `package`.

    from_version defaults to what is installed, which is the question normally
    being asked: what does moving off what I have now cost me.

    `sites` lets a caller supply reach measured some other way, such as from a
    test run, instead of the static scan.
    """
    dist = distribution or package.replace("_", "-")
    current = _load(package, dist, from_version)
    target = _load(package, dist, to_version)

    raw = [
        ApiChange(
            fqn=str(d.get("object_path")),
            kind=getattr(d.get("kind"), "name", str(d.get("kind"))),
            old=None if d.get("old_value") is None else str(d.get("old_value")),
            new=None if d.get("new_value") is None else str(d.get("new_value")),
        )
        for d in (b.as_dict() for b in griffe.find_breaking_changes(current, target))
    ]
    real = verify(raw, target)

    if sites is None:
        sites = find_sites(repo, package, changed_symbols(real, package))

    return UpgradeReport(
        package=package,
        from_version=from_version or "installed",
        to_version=to_version,
        total_changes=len(raw),
        verified_changes=len(real),
        impacts=narrow(impact_report(sites, real)),
    )
