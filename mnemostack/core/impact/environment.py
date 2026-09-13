"""The project's installed packages, not the ones mnemostack runs with.

Installed with `uv tool install`, mnemostack lives in an environment of its own,
and asking importlib there what is installed answers for mnemostack's
dependencies. A repo importing torch and numpy was reported as using pydantic
and PyYAML at mnemostack's versions, and torch and numpy were skipped without a
word. So the question is put to the repo's own interpreter instead.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_QUERY = r"""
import json
from importlib.metadata import distributions, packages_distributions
print(json.dumps({
    "imports": packages_distributions(),
    "versions": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
}))
"""


def canonical(name: str) -> str:
    """A distribution name as PyPI compares them: `Typing.Extensions` is `typing-extensions`."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Environment:
    python: Path
    imports: dict[str, list[str]]  # import name -> distributions providing it
    versions: dict[str, str]  # canonical distribution name -> version

    def distribution_for(self, package: str) -> str | None:
        """The PyPI name behind an import name: `yaml` is PyYAML, `PIL` is Pillow."""
        found = self.imports.get(package.split(".")[0])
        return found[0] if found else None

    def version_of(self, distribution: str) -> str | None:
        return self.versions.get(canonical(distribution))


def project_python(repo: Path) -> Path | None:
    """The interpreter of the repo's virtual environment, if it has one.

    Looked for as `.venv` or `venv` in the repo, then an activated VIRTUAL_ENV,
    unless that is the environment mnemostack itself is running in.
    """
    bindir, exe = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    places = [repo / ".venv", repo / "venv"]
    active = os.environ.get("VIRTUAL_ENV")
    if active and Path(active).resolve() != Path(sys.prefix).resolve():
        places.append(Path(active))
    for place in places:
        python = place / bindir / exe
        if python.is_file():
            return python
    return None


def project_environment(repo: Path) -> Environment | None:
    """What the repo's own interpreter has installed, or None if there is none to ask."""
    python = project_python(repo)
    if python is None:
        return None
    try:
        done = subprocess.run(
            [str(python), "-c", _QUERY], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    try:
        data = json.loads(done.stdout.strip().splitlines()[-1])
        return Environment(
            python=python,
            imports={str(k): [str(d) for d in v] for k, v in data["imports"].items()},
            versions={canonical(str(k)): str(v) for k, v in data["versions"].items()},
        )
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        return None
