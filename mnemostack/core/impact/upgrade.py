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
from dataclasses import dataclass, field
from pathlib import Path

import griffe
import platformdirs

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.impact.propagate import Impact, impact_report
from mnemostack.core.reach import RefKind, Site
from mnemostack.core.reach.static import find_sites


class UpgradeError(RuntimeError):
    """A failure the user can act on, rather than a traceback."""


@dataclass(frozen=True)
class Deprecation:
    """A symbol you use that still works, but is marked for removal."""

    symbol: str
    message: str
    sites: list[Site]


@dataclass(frozen=True)
class UpgradeReport:
    package: str
    from_version: str
    to_version: str
    total_changes: int
    verified_changes: int
    impacts: list[Impact]
    deprecations: list[Deprecation] = field(default_factory=list)
    unwitnessed: int = 0  # removals dropped because no import could confirm them


def _ensure_pypi_cache() -> None:
    """Create griffe's download cache directory if it does not exist.

    load_pypi opens a TemporaryDirectory inside platformdirs' griffe cache but
    never creates it, so on any machine that has not used griffe before, the
    first run dies with a FileNotFoundError naming a temp path the user has
    never heard of. That is every new user, and it was CI too.
    """
    try:
        Path(platformdirs.user_cache_dir("griffe")).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # if the cache is unwritable, griffe's own error is the better one


def _load(package: str, distribution: str, version: str | None):
    try:
        if version is None:
            return griffe.load(package, allow_inspection=True)
        _ensure_pypi_cache()
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


_DEPRECATED_MSG = re.compile(r"deprecated\s*\(\s*['\"](?P<msg>[^'\"]{0,160})")
_WALK_LIMIT = 20_000


def deprecated_symbols(module) -> dict[str, str]:
    """Dotted path below the package -> the deprecation message, if any.

    Not griffe's `is_deprecated`: it returns False for `@typing_extensions
    .deprecated`, which is what real packages use. pydantic 2.5 marks `dict`,
    `json`, `copy` and `parse_obj` that way and griffe reports none of them.
    The decorator text is populated though, so it is read directly.

    Deprecations are worth surfacing because they are the one thing a green
    test run actively hides: warnings are suppressed by default, so an upgrade
    passes CI while quietly loading the debt that detonates at the next major.

    The walk is bounded and tracks visited paths. An earlier traversal of
    griffe's module tree had neither and wrote 3.1 GB into a user's repository
    on a package with an import cycle.
    """
    found: dict[str, str] = {}
    seen: set[str] = set()
    root = module.name
    stack = [(module, 0)]
    budget = _WALK_LIMIT

    while stack and budget > 0:
        obj, depth = stack.pop()
        budget -= 1
        try:
            members = list(obj.members.items())
        except Exception:
            continue
        for name, member in members:
            if name.startswith("__"):
                continue
            try:
                path = str(member.path)
            except Exception:
                continue
            if path in seen:
                continue
            seen.add(path)

            # An Alias raises when its target is not in the loaded tree, and it
            # raises on attribute access rather than at construction, so every
            # read below has to be guarded. pydantic re-exports pydantic_core,
            # which is a separate distribution and never resolves.
            try:
                text = " ".join(
                    str(getattr(d, "value", "")) for d in getattr(member, "decorators", [])
                )
                kind = str(getattr(member, "kind", "")).lower()
            except Exception:
                continue

            if "deprecated" in text:
                match = _DEPRECATED_MSG.search(text)
                found[path.removeprefix(f"{root}.")] = match.group("msg") if match else ""

            if depth < 4 and ("class" in kind or "module" in kind):
                stack.append((member, depth + 1))
    return found


def _public_paths(deprecated: dict[str, str]) -> dict[str, str]:
    """Keep one path per deprecated thing: the shortest, which is the public one.

    griffe reaches the same class through every module that imports it, so
    `BaseModel.dict` also appears as `_internal._fields.BaseModel.dict` and
    thirty other spellings. Reporting all of them turned three deprecated
    methods into 105 findings for the same three lines of code.
    """
    best: dict[tuple[str, str], str] = {}
    for path, message in deprecated.items():
        key = (path.split(".")[-1], message)
        current = best.get(key)
        if current is None or path.count(".") < current.count("."):
            best[key] = path
    return {path: message for (_, message), path in best.items()}


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

    # Deprecations are looked for separately: they are not breaking changes, so
    # griffe never reports them, and they reach code the diff does not touch.
    deprecated = _public_paths(deprecated_symbols(target))
    dep_sites = find_sites(repo, package, set(deprecated)) if deprecated else []
    by_symbol: dict[str, list[Site]] = {}
    for site in dep_sites:
        by_symbol.setdefault(site.symbol, []).append(site)
    deprecations = [
        Deprecation(symbol=symbol, message=deprecated[symbol], sites=found)
        for symbol, found in sorted(by_symbol.items())
    ]

    # Imported here rather than at the top to keep witness, which depends on
    # propagate, out of the import path of anything that only needs the diff.
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as installed_version

    from mnemostack.core.impact.witness import witness_removals, witness_signatures

    # A removal is only reported if the import the code actually uses fails in
    # the target version. Measured across twenty repos we did not write, griffe's
    # removals were mostly re-exports that still import fine.
    impacts, unwitnessed = witness_removals(narrow(impact_report(sites, real)), dist, to_version)

    # Signature and kind changes are compared on the running code in both
    # versions. The ones that survived removal witnessing were griffe misreports
    # like click.Argument gaining a required parameter it does not have.
    try:
        old_version = from_version or installed_version(dist)
    except PackageNotFoundError:
        old_version = None
    impacts = witness_signatures(impacts, dist, old_version, to_version)

    return UpgradeReport(
        package=package,
        from_version=from_version or "installed",
        to_version=to_version,
        total_changes=len(raw),
        verified_changes=len(real),
        impacts=impacts,
        deprecations=deprecations,
        unwitnessed=unwitnessed,
    )
