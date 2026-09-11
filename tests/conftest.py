"""Shared helpers.

Most tests build what they need inline: the pieces are a repo of text files and
a griffe module, both cheap enough that a fixture would hide more than it saves.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def write_tree(root: Path, files: dict[str, str]) -> Path:
    """Create `files` (relative path -> source) under `root`."""
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


@pytest.fixture
def tree(tmp_path: Path):
    """Make a source tree in a fresh directory."""

    def _make(files: dict[str, str], name: str = "repo") -> Path:
        return write_tree(tmp_path / name, files)

    return _make


def git_init(repo: Path) -> None:
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
