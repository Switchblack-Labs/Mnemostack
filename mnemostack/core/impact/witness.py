"""Check a claimed removal by trying it, not by reading about it.

griffe diffs API models, and those models are wrong about re-exports. Across
twenty repositories we did not write, most OBJECT_REMOVED findings named symbols
that import perfectly well in the new version, because what moved was an
internal definition the public name still points at. Reading source cannot
settle that. Importing it can, in an isolated environment that never touches
the user's own.

Only removals are witnessed here. They were the largest false-positive class and
the only one a bare import can decide. A changed signature needs the arguments
the code actually passes, which is a different mechanism.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable

from mnemostack.core.impact.propagate import Impact

Resolver = Callable[[str, str, list[str]], "dict[str, bool] | None"]

_PROBE = r"""
import importlib, json, sys, warnings
warnings.simplefilter("ignore")

def resolves(path):
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        try:
            for part in parts[cut:]:
                obj = getattr(obj, part)
        except Exception:
            return False
        return True
    return False

print(json.dumps({p: resolves(p) for p in json.loads(sys.argv[1])}))
"""


def uv_resolver(distribution: str, version: str, paths: list[str]) -> dict[str, bool] | None:
    """Which dotted paths resolve with `distribution==version`, or None if unknown.

    Runs through `uv run --no-project` in a throwaway environment, so nothing in
    the user's environment is installed, upgraded or imported. None means the
    witness could not run at all (no uv, a failed install, a crashed probe), and
    callers must read that as "unverified", never as "not removed".

    Resolution goes through getattr, not a static lookup, so a module-level
    __getattr__ shim answers the way it would for real code: pydantic's
    `parse_file_as` raises through its shim and correctly counts as removed.
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
    return {str(path): bool(ok) for path, ok in answer.items()}


def witness_removals(
    impacts: Iterable[Impact],
    distribution: str,
    version: str,
    resolve: Resolver = uv_resolver,
) -> tuple[list[Impact], int]:
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
    """
    impacts = list(impacts)
    removals = [i for i in impacts if i.change.kind == "OBJECT_REMOVED"]
    if not removals:
        return impacts, 0

    paths = sorted({i.site.via for i in removals if i.site.via})
    verdict = resolve(distribution, version, paths)
    if verdict is None:
        return impacts, 0

    kept: list[Impact] = []
    unwitnessed = 0
    for impact in impacts:
        if impact.change.kind != "OBJECT_REMOVED":
            kept.append(impact)
            continue
        result = verdict.get(impact.site.via) if impact.site.via else None
        if result is False:
            kept.append(impact)  # witnessed: the path this code uses is gone
        elif result is None:
            unwitnessed += 1
    return kept, unwitnessed
