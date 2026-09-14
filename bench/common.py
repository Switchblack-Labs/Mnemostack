"""Shared by the benchmark scripts: fetch a repository at an exact commit."""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "mnemostack-bench"


def git(*args: str, cwd: Path) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout


def checkout(url: str, sha: str, dest: Path, depth: int = 1) -> Path:
    """`dest` holding `url` at exactly `sha`, fetching only the history that needs.

    Reused as-is when it is already there, so a rerun costs no network.
    """
    if (dest / ".git").is_dir():
        try:
            if git("rev-parse", "HEAD", cwd=dest).strip() == sha:
                return dest
        except subprocess.CalledProcessError:
            pass
    else:
        dest.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=dest)
        git("remote", "add", "origin", url, cwd=dest)
    git("fetch", "-q", "--depth", str(depth), "origin", sha, cwd=dest)
    git("checkout", "-q", "--force", sha, cwd=dest)
    return dest
