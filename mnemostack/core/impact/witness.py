"""Check a claimed change by trying it, not by reading about it.

griffe diffs API models, and those models are wrong often enough to sink a
report. Across twenty repositories we did not write, most OBJECT_REMOVED
findings named symbols that import perfectly well in the new version, because
what moved was an internal definition the public name still points at. The
signature changes that survived were the same story: `click.Argument([...])` is
reported as gaining a required parameter and works fine.

Reading source cannot settle either. Running it can, in a throwaway environment
that never touches the user's own:

- a removal is checked by importing the path the code uses in the new version;
- a signature or kind change is checked by describing that path in both
  versions and comparing what the running code actually exposes.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from mnemostack.core.impact.propagate import Impact, Severity
from mnemostack.core.reach import Site

Resolver = Callable[[str, str, list[str]], "dict[str, bool | str] | None"]
Prober = Callable[[str, str, list[str]], "dict[str, dict] | None"]

SIGNATURE_KINDS = frozenset(
    {
        "PARAMETER_ADDED_REQUIRED",
        "PARAMETER_REMOVED",
        "PARAMETER_MOVED",
        "PARAMETER_CHANGED_KIND",
        "PARAMETER_CHANGED_REQUIRED",
        "PARAMETER_CHANGED_DEFAULT",
        "OBJECT_CHANGED_KIND",
    }
)

_PROBE = r"""
import importlib, inspect, json, sys, warnings
warnings.simplefilter("ignore")

def resolve(path):
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        parent = None
        try:
            for part in parts[cut:]:
                parent, obj = obj, getattr(obj, part)
        except Exception:
            return False, None, None
        return True, obj, parent
    return False, None, None

def default(value):
    if value is inspect.Parameter.empty:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    return "<object>"

def describe(path):
    leaf = path.rsplit(".", 1)[-1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok, obj, parent = resolve(path)
    # Any category: pydantic 2.9 warns that GenericModel moved as a UserWarning,
    # 2.12 as a DeprecationWarning. What makes it evidence is that resolving this
    # path raised a warning naming it; importing a package can warn about
    # unrelated things on its own, and those do not name the leaf.
    warning = next((str(w.message) for w in caught if leaf in str(w.message)), None)
    if not ok:
        return {"resolves": False, "kind": None, "signature": None, "warning": None}
    if inspect.isclass(obj):
        kind = "class"
    elif inspect.ismodule(obj):
        kind = "module"
    elif callable(obj):
        kind = "callable"
    else:
        kind = "attribute"
    try:
        params = list(inspect.signature(obj).parameters.values())
        # A plain function looked up on a class is a method, and its first
        # parameter is the instance a call passes implicitly, whatever it is named.
        if (
            params
            and inspect.isclass(parent)
            and inspect.isfunction(obj)
            and not isinstance(inspect.getattr_static(parent, leaf, None), staticmethod)
        ):
            params = params[1:]
        signature = [[p.name, p.kind.name, default(p.default)] for p in params]
    except (TypeError, ValueError):
        signature = None
    return {"resolves": True, "kind": kind, "signature": signature, "warning": warning}

print(json.dumps({p: describe(p) for p in json.loads(sys.argv[1])}))
"""


def uv_probe(distribution: str, version: str, paths: list[str]) -> dict[str, dict] | None:
    """Describe dotted paths with `distribution==version` installed, or None.

    Each path maps to whether it resolves, what kind of object it is, and its
    parameters as (name, kind, default). Defaults are compared only when they
    are plain values; anything else is recorded as an opaque object, so a
    change between two object defaults is not seen.

    Runs through `uv run --no-project` in a throwaway environment, so nothing in
    the user's environment is installed, upgraded or imported. None means the
    probe could not run at all (no uv, a failed install, a crash), which callers
    must read as "unverified", never as "unchanged".

    Resolution goes through getattr, so a module-level __getattr__ shim answers
    the way it would for real code.
    """
    uv = shutil.which("uv")
    if uv is None:
        return None
    if not paths:
        return {}
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    command = [
        uv,
        "run",
        "--quiet",
        "--no-project",
        "--python",
        python,
        "--with",
        f"{distribution}=={version}",
        "python",
        "-c",
        _PROBE,
        json.dumps(paths),
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    try:
        # Last line only: importing a library can print to stdout on its own.
        answer = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return {str(path): info for path, info in answer.items() if isinstance(info, dict)}


def uv_resolver(distribution: str, version: str, paths: list[str]) -> dict[str, bool | str] | None:
    """Which dotted paths resolve with `distribution==version`, or None if unknown.

    A path that resolves while warning that it is deprecated maps to the warning
    instead of True.
    """
    described = uv_probe(distribution, version, paths)
    if described is None:
        return None
    return {
        path: (info.get("warning") or True) if info.get("resolves") else False
        for path, info in described.items()
    }


def witness_removals(
    impacts: Iterable[Impact],
    distribution: str,
    version: str,
    resolve: Resolver = uv_resolver,
) -> tuple[list[Impact], int, list[tuple[str, str, Site]]]:
    """Keep a removal only if the path the code uses really fails to resolve.

    Returns the surviving impacts, and how many removals were dropped because
    nothing could witness them.

    A removal whose path still resolves is dropped without comment: it is not a
    removal as far as this code is concerned. A removal reached through an
    object whose type the line does not show has no path to test; at a measured
    three percent precision an unverifiable removal is far more likely noise
    than news, so it is dropped and counted rather than shown.

    If the witness cannot run at all, nothing is dropped. Hiding every finding
    because uv is missing would be worse than showing unverified ones.

    A removal whose path still resolves but warns that it is deprecated comes back
    third, as (path, warning, site). pydantic 2 keeps `pydantic.generics
    .GenericModel` importable behind a warning: not a break, but exactly the
    migration a maintainer has to do. Measured on dstack's and distiller's real
    pydantic 2 migrations, both paths were changed and neither was reported.
    """
    impacts = list(impacts)
    removals = [i for i in impacts if i.change.kind == "OBJECT_REMOVED"]
    if not removals:
        return impacts, 0, []

    paths = sorted({i.site.via for i in removals if i.site.via})
    verdict = resolve(distribution, version, paths)
    if verdict is None:
        return impacts, 0, []

    kept: list[Impact] = []
    unwitnessed = 0
    warned: list[tuple[str, str, Site]] = []
    for impact in impacts:
        if impact.change.kind != "OBJECT_REMOVED":
            kept.append(impact)
            continue
        result = verdict.get(impact.site.via) if impact.site.via else None
        if result is False:
            kept.append(impact)  # witnessed: the path this code uses is gone
        elif isinstance(result, str):
            warned.append((impact.site.via, result, impact.site))
        elif result is None:
            unwitnessed += 1
    return kept, unwitnessed, warned


def witness_signatures(
    impacts: Iterable[Impact],
    distribution: str,
    old_version: str | None,
    new_version: str,
    probe: Prober = uv_probe,
    repo: Path | None = None,
) -> list[Impact]:
    """Check signature and kind changes against the running code in both versions.

    If the path the code uses has the same signature (or, for a kind change, is
    the same kind of object) before and after, griffe reported a change the
    running code does not show, and the finding is dropped. If it differs, the
    finding stands, now witnessed.

    If either side cannot be described, a builtin or C extension with no
    introspectable signature for instance, a BREAK is demoted to REVIEW rather
    than dropped. Unverified is not the same as wrong, but it should not be the
    loudest line in the report either.

    If the old version is unknown or the probe cannot run, nothing changes.

    With `repo`, a call is bound from its file's parse, so calls split across
    lines are decided too; without it, only from the site's own line. A call
    whose binding cannot be decided keeps its finding at REVIEW, not BREAK.
    """
    from mnemostack.core.impact.binding import BIND_DECIDES, still_binds
    from mnemostack.core.reach.static import parse_source

    trees: dict[str, ast.Module | None] = {}

    def tree_of(file: str) -> ast.Module | None:
        if repo is None:
            return None
        if file not in trees:
            try:
                source = (repo / file).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                source = None
            trees[file] = parse_source(source) if source is not None else None
        return trees[file]

    impacts = list(impacts)
    if not old_version or not any(i.change.kind in SIGNATURE_KINDS for i in impacts):
        return impacts

    paths = sorted({i.site.via for i in impacts if i.change.kind in SIGNATURE_KINDS and i.site.via})
    before = probe(distribution, old_version, paths)
    after = probe(distribution, new_version, paths)
    if before is None or after is None:
        return impacts

    kept: list[Impact] = []
    for impact in impacts:
        if impact.change.kind not in SIGNATURE_KINDS:
            kept.append(impact)
            continue
        field = "kind" if impact.change.kind == "OBJECT_CHANGED_KIND" else "signature"
        via = impact.site.via
        old = before.get(via) if via else None
        new = after.get(via) if via else None
        comparable = (
            old is not None
            and new is not None
            and old.get("resolves")
            and new.get("resolves")
            and old.get(field) is not None
            and new.get(field) is not None
        )
        if comparable:
            if old[field] == new[field]:
                continue  # griffe reported a change the running code does not show
            if (
                field == "signature"
                and impact.site.kind.value == "call"
                and impact.change.kind in BIND_DECIDES
            ):
                fits = still_binds(
                    impact.site.text,
                    impact.change.fqn,
                    new["signature"],
                    tree=tree_of(impact.site.file),
                    line=impact.site.line,
                )
                if fits is True:
                    continue  # the signature changed, and this call still fits it
                if fits is None and impact.severity is Severity.BREAK:
                    # The change is real, but the call's arguments are not all in
                    # the source, `Column(sa_type, *args, **kwargs)`, so nothing
                    # shows that this call breaks. Asking is honest; BREAK is not.
                    kept.append(dataclasses.replace(impact, severity=Severity.REVIEW))
                    continue
            kept.append(impact)  # witnessed: the running code really changed
            continue
        if impact.severity is Severity.BREAK:
            kept.append(dataclasses.replace(impact, severity=Severity.REVIEW))
        else:
            kept.append(impact)
    return kept
