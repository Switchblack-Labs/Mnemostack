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

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

import griffe
import platformdirs

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.impact.environment import project_environment
from mnemostack.core.impact.propagate import Impact, collapse, impact_report
from mnemostack.core.reach import Site
from mnemostack.core.reach.static import find_sites, parse_source


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


def _load(package: str, distribution: str, version: str):
    try:
        _ensure_pypi_cache()
        return griffe.load_pypi(package, distribution, f"=={version}")
    except Exception as exc:  # griffe raises several unrelated types
        raise UpgradeError(f"could not load {package} {version}: {exc}") from exc


def _still_defined(new_root, fqn: str) -> bool:
    """Does `fqn` survive in the new version, despite griffe saying removed?

    griffe drops members declared as an ``@overload`` pair, so it reports methods
    that are plainly still there as removed. Left alone this fires on most typed
    and compiled packages, which is enough noise to make a report untrustworthy.

    The check reads the owner's own scope, the module body or the body of the
    class the member belongs to, in the file griffe says it lives in. One earlier
    version read the package's top-level __init__, a re-export facade, so it
    never fired. The next matched `def <name>` anywhere in the file, so removing
    `A.close` looked fake whenever a `B.close` sat in the same module.
    """
    owner_path, _, name = fqn.rpartition(".")
    if not owner_path:
        return False
    relative = "" if owner_path == new_root.name else owner_path.removeprefix(f"{new_root.name}.")
    try:
        owner = new_root if not relative else new_root[relative]
        if owner.is_alias:
            owner = owner.final_target
        module = owner if owner.is_module else owner.module
        source = Path(str(module.filepath))
        scope = [] if owner.is_module else owner.path.removeprefix(f"{module.path}.").split(".")
    except Exception:  # an unresolvable alias raises on access  # noqa: BLE001
        return False

    for candidate in (source, source.with_suffix(".pyi")):
        if not candidate.is_file():
            continue
        try:
            tree = parse_source(candidate.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if tree is not None and _defines(tree.body, [*scope, name]):
            return True
    return False


def _defines(body: list, chain: list[str]) -> bool:
    """Whether `chain`, enclosing class names then a name, is defined in this scope.

    Statements under if, try and with belong to the scope: a definition behind
    `if TYPE_CHECKING` or `except ImportError` is still a definition.
    """
    head, rest = chain[0], chain[1:]
    for node in _scope_statements(body):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != head:
            continue
        if not rest:
            return True
        if isinstance(node, ast.ClassDef) and _defines(node.body, rest):
            return True
    return False


def _scope_statements(body: list):
    for node in body:
        yield node
        if isinstance(node, (ast.If, ast.With, ast.AsyncWith)):
            yield from _scope_statements(node.body)
            yield from _scope_statements(getattr(node, "orelse", []))
        elif isinstance(node, (ast.Try, ast.TryStar)):
            yield from _scope_statements(node.body)
            for handler in node.handlers:
                yield from _scope_statements(handler.body)
            yield from _scope_statements(node.orelse)
            yield from _scope_statements(node.finalbody)


def verify(changes: list[ApiChange], new_root) -> list[ApiChange]:
    """Drop claimed removals that the new version's own source contradicts."""
    return [c for c in changes if c.kind != "OBJECT_REMOVED" or not _still_defined(new_root, c.fqn)]


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
    public: dict[str, tuple[str, str]] = {}  # identity -> (shortest path, message)
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
                # griffe reaches one class through every module that imports it:
                # `BaseModel.dict` is also `_internal._fields.BaseModel.dict` and
                # thirty other spellings, which once made three deprecated methods
                # 105 findings. They are one object, so keep its shortest path.
                # Folding by name instead merged different methods that share one.
                try:
                    identity = str(member.canonical_path)
                except Exception:  # noqa: BLE001
                    identity = path
                relative = path.removeprefix(f"{root}.")
                current = public.get(identity)
                if current is None or relative.count(".") < current[0].count("."):
                    public[identity] = (relative, match.group("msg") if match else "")

            if depth < 4 and ("class" in kind or "module" in kind):
                stack.append((member, depth + 1))
    return dict(public.values())


def changed_symbols(changes: list[ApiChange], package: str) -> set[str]:
    """Dotted paths below the package, for the symbols that changed."""
    root = package.split(".")[0]
    out = set()
    for change in changes:
        relative = change.fqn.removeprefix(f"{root}.")
        if relative and relative != change.fqn:
            out.add(relative)
    return out


def class_members(root, symbols: set[str]) -> set[str]:
    """The symbols whose owner is a class in `root`, read from the API, not the spelling."""
    found = set()
    for symbol in symbols:
        owner = symbol.rpartition(".")[0]
        if not owner:
            continue
        try:
            if root[owner].is_class:
                found.add(symbol)
        except Exception:  # an unresolvable alias raises on access  # noqa: BLE001
            continue
    return found


def check_upgrade(
    repo: Path,
    package: str,
    to_version: str,
    distribution: str | None = None,
    from_version: str | None = None,
    sites: list[Site] | None = None,
) -> UpgradeReport:
    """Which places in `repo` are affected by upgrading `package`.

    from_version defaults to the version installed in the repo's own environment
    (.venv or venv), which is the question normally being asked: what does moving
    off what I have now cost me. Never mnemostack's own environment.

    `sites` lets a caller supply uses found some other way instead of the static
    scan.
    """
    env = project_environment(repo) if from_version is None or distribution is None else None
    dist = (
        distribution
        or (env.distribution_for(package) if env else None)
        or package.replace("_", "-")
    )
    if from_version is None:
        from_version = env.version_of(dist) if env else None
        if from_version is None:
            where = f"the environment at {env.python}" if env else f"a .venv or venv in {repo}"
            raise UpgradeError(f"{dist} is not installed in {where}; pass --from-version")
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
        symbols = changed_symbols(real, package)
        members = class_members(current, symbols) | class_members(target, symbols)
        sites = find_sites(repo, package, symbols, members=members)

    # Deprecations are looked for separately: they are not breaking changes, so
    # griffe never reports them, and they reach code the diff does not touch.
    deprecated = deprecated_symbols(target)
    dep_sites = (
        find_sites(repo, package, set(deprecated), members=class_members(target, set(deprecated)))
        if deprecated
        else []
    )
    by_symbol: dict[str, list[Site]] = {}
    for site in dep_sites:
        by_symbol.setdefault(site.symbol, []).append(site)
    deprecations = [
        Deprecation(symbol=symbol, message=deprecated[symbol], sites=found)
        for symbol, found in sorted(by_symbol.items())
    ]

    # Imported here rather than at the top to keep witness, which depends on
    # propagate, out of the import path of anything that only needs the diff.
    from mnemostack.core.impact.witness import witness_removals, witness_signatures

    # A removal is only reported if the import the code actually uses fails in
    # the target version. Measured across twenty repos we did not write, griffe's
    # removals were mostly re-exports that still import fine.
    impacts, unwitnessed, warned = witness_removals(impact_report(sites, real), dist, to_version)

    # A removal that still imports behind a deprecation warning is a migration
    # the decorator scan cannot see: pydantic 2 warns from a module __getattr__.
    moved: dict[str, tuple[str, dict[tuple[str, int], Site]]] = {}
    for via, message, site in warned:
        moved.setdefault(via, (message, {}))[1].setdefault((site.file, site.line), site)
    deprecations += [
        Deprecation(
            symbol=via.removeprefix(f"{package.split('.')[0]}."),
            message=message.splitlines()[0][:160],
            sites=list(found.values()),
        )
        for via, (message, found) in sorted(moved.items())
    ]

    # Signature and kind changes are compared on the running code in both
    # versions. The ones that survived removal witnessing were griffe misreports
    # like click.Argument gaining a required parameter it does not have.
    impacts = collapse(witness_signatures(impacts, dist, from_version, to_version, repo=repo))

    return UpgradeReport(
        package=package,
        from_version=from_version,
        to_version=to_version,
        total_changes=len(raw),
        verified_changes=len(real),
        impacts=impacts,
        deprecations=deprecations,
        unwitnessed=unwitnessed,
    )
