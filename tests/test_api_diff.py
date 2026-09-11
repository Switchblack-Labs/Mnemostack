"""Griffe-backed API diff, and its join to call graph nodes.

The join is the part worth testing. Griffe addresses an object by dotted module
path and the graph addresses it by absolute file path, so a silent mismatch
would leave every change unpaired and the impact set empty while everything
still looked green.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mnemostack.core.impact.api_diff import (
    breaking_changes,
    changed_nodes,
    fqn_index,
)
from mnemostack.core.retrieval.call_graph import (
    CallGraph,
    build_nodes_for_python_file,
    link_python_file_imports,
)

V1 = {
    "libx/__init__.py": "from libx.core import verify\n",
    "libx/core.py": (
        "def verify(token):\n"
        "    return True\n"
        "\n"
        "\n"
        "def retired(x):\n"
        "    return x\n"
        "\n"
        "\n"
        "class Client:\n"
        "    def fetch(self, url):\n"
        "        return url\n"
    ),
}

V2 = {
    "libx/__init__.py": "from libx.core import verify\n",
    "libx/core.py": (
        # verify gains a REQUIRED parameter: breaking.
        "def verify(token, strict):\n"
        "    return strict\n"
        "\n"
        "\n"
        "class Client:\n"
        # fetch gains an OPTIONAL parameter: not breaking, and the point of
        # using griffe rather than comparing signature hashes.
        "    def fetch(self, url, timeout=30):\n"
        "        return url\n"
    ),
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


@pytest.fixture
def versioned_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "libx_repo"
    repo.mkdir()
    _write(repo, V1)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "v1")
    _git(repo, "tag", "v1")

    (repo / "libx" / "core.py").unlink()
    _write(repo, V2)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "v2")
    _git(repo, "tag", "v2")
    return repo


def test_required_parameter_is_breaking_and_optional_is_not(versioned_repo):
    """The whole reason for griffe instead of a signature fingerprint.

    A hash changes for both of these. Only one of them breaks a caller.
    """
    changes = breaking_changes("libx", versioned_repo, "v1", "v2")
    by_fqn = {c.fqn: c.kind for c in changes}

    assert by_fqn.get("libx.core.verify") == "PARAMETER_ADDED_REQUIRED"
    assert "libx.core.Client.fetch" not in by_fqn, (
        "an added optional parameter is backward compatible and must not be reported as breaking"
    )


def test_removed_object_is_breaking(versioned_repo):
    changes = breaking_changes("libx", versioned_repo, "v1", "v2")
    kinds = {c.fqn: c.kind for c in changes}
    assert kinds.get("libx.core.retired") == "OBJECT_REMOVED"


def test_fqn_index_maps_dotted_paths_to_graph_nodes(tmp_path, versioned_repo):
    """The join: griffe's `libx.core.verify` must find the graph's node."""
    graph = CallGraph(store_dir=tmp_path / "store")
    files = sorted(versioned_repo.rglob("*.py"))
    for f in files:
        build_nodes_for_python_file(f, graph=graph)
    for f in files:
        link_python_file_imports(f, graph=graph)

    index = fqn_index(graph)
    core = versioned_repo / "libx" / "core.py"

    assert index.get("libx.core.verify") == f"{core}::verify"
    # Methods keep the Class.method tail on both sides of the join.
    assert index.get("libx.core.Client.fetch") == f"{core}::Client.fetch"
    graph.close()


def test_changed_nodes_pairs_breakages_with_nodes(tmp_path, versioned_repo):
    graph = CallGraph(store_dir=tmp_path / "store")
    files = sorted(versioned_repo.rglob("*.py"))
    for f in files:
        build_nodes_for_python_file(f, graph=graph)
    for f in files:
        link_python_file_imports(f, graph=graph)

    changes = breaking_changes("libx", versioned_repo, "v1", "v2")
    paired = changed_nodes(graph, changes)

    core = versioned_repo / "libx" / "core.py"
    assert (f"{core}::verify", next(c for c in changes if c.fqn == "libx.core.verify")) in paired

    # `retired` is gone at HEAD, so it has no node to pair with. Dropping it is
    # correct: the graph describes the current tree, not the old one.
    assert all("retired" not in node for node, _ in paired)
    graph.close()
