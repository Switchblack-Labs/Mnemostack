"""End to end: a real API change in one repo, the symbols it breaks in another.

This is the whole pipeline in one place. Two git repos, a breaking commit in
the library, griffe classifying it, the graph resolving who references it, and
a report naming the consumer symbols. If any seam is wrong the report comes
back empty, so every assertion names the consumer it expects.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mnemostack.core.impact.api_diff import breaking_changes, changed_nodes
from mnemostack.core.impact.propagate import Severity, impact_report
from mnemostack.core.retrieval.call_graph import (
    CallGraph,
    build_nodes_for_python_file,
    link_python_file_imports,
)

LIB_V1 = {
    "payments/__init__.py": "from payments.core import Processor, charge\n",
    "payments/core.py": (
        "def charge(token):\n"
        "    return token\n"
        "\n"
        "\n"
        "def refund(charge_id):\n"
        "    return charge_id\n"
        "\n"
        "\n"
        "class Processor:\n"
        "    def run(self, job):\n"
        "        return job\n"
    ),
}

LIB_V2 = {
    "payments/__init__.py": "from payments.core import Processor, charge\n",
    "payments/core.py": (
        # charge gains a required parameter: breaks callers.
        "def charge(token, idempotency_key):\n"
        "    return idempotency_key\n"
        "\n"
        "\n"
        # refund is gone: breaks callers.
        "class Processor:\n"
        # run gains an optional parameter: breaks nobody.
        "    def run(self, job, retries=0):\n"
        "        return job\n"
    ),
}

SVC = {
    "billing/handlers.py": (
        "from payments import Processor, charge\n"
        "from payments.core import refund\n"
        "\n"
        "\n"
        "def do_charge(token):\n"
        "    return charge(token)\n"
        "\n"
        "\n"
        "def do_refund(charge_id):\n"
        "    return refund(charge_id)\n"
        "\n"
        "\n"
        "def do_run(job):\n"
        "    p = Processor()\n"
        "    return p.run(job)\n"
        "\n"
        "\n"
        "class CustomProcessor(Processor):\n"
        "    def run(self, job):\n"
        "        return job\n"
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
def two_repos(tmp_path: Path):
    lib = tmp_path / "lib_side" / "payments_repo"
    lib.mkdir(parents=True)
    _write(lib, LIB_V1)
    _git(lib, "init", "-q")
    _git(lib, "add", "-A")
    _git(lib, "commit", "-qm", "v1")
    _git(lib, "tag", "v1")
    (lib / "payments" / "core.py").unlink()
    _write(lib, LIB_V2)
    _git(lib, "add", "-A")
    _git(lib, "commit", "-qm", "v2")
    _git(lib, "tag", "v2")

    svc = tmp_path / "svc_side" / "billing_repo"
    svc.mkdir(parents=True)
    _write(svc, SVC)
    _git(svc, "init", "-q")
    _git(svc, "add", "-A")
    _git(svc, "commit", "-qm", "init")

    graph = CallGraph(store_dir=tmp_path / "store")
    files = sorted(lib.rglob("*.py")) + sorted(svc.rglob("*.py"))
    for f in files:
        build_nodes_for_python_file(f, graph=graph)
    for f in files:
        link_python_file_imports(f, graph=graph)

    yield lib, svc, graph
    graph.close()


def _report(lib: Path, graph: CallGraph):
    changes = breaking_changes("payments", lib, "v1", "v2")
    return impact_report(graph, changed_nodes(graph, changes))


def test_required_parameter_breaks_the_calling_symbol(two_repos):
    lib, svc, graph = two_repos
    handlers = svc / "billing" / "handlers.py"

    hits = [
        i
        for i in _report(lib, graph)
        if i.consumer == f"{handlers}::do_charge" and i.change.kind == "PARAMETER_ADDED_REQUIRED"
    ]
    assert hits, "a caller of charge() must be reported"
    assert hits[0].severity is Severity.BREAK
    assert hits[0].crosses_repo


def test_optional_parameter_breaks_nobody(two_repos):
    """`Processor.run` gained `retries=0`. Nothing should be reported for it.

    This is the end-to-end version of the claim that justifies griffe. A
    fingerprint would have flagged do_run and CustomProcessor here.
    """
    lib, svc, graph = two_repos
    handlers = svc / "billing" / "handlers.py"
    consumers = {i.consumer for i in _report(lib, graph)}

    assert f"{handlers}::do_run" not in consumers
    assert f"{handlers}::CustomProcessor" not in consumers


def test_report_is_ordered_worst_first(two_repos):
    lib, svc, graph = two_repos
    report = _report(lib, graph)
    assert report, "the report must not be empty"

    severities = [i.severity for i in report]
    assert severities == sorted(severities, key=lambda s: 0 if s is Severity.BREAK else 1)


def test_only_referencing_symbols_appear(two_repos):
    """Nothing in the library shows up as a consumer of its own change."""
    lib, svc, graph = two_repos
    assert all("payments_repo" not in i.consumer for i in _report(lib, graph))


def test_deleted_object_is_classified_but_not_yet_joined(two_repos):
    """A known hole, pinned so it cannot be mistaken for working.

    `refund` was deleted in v2. griffe reports OBJECT_REMOVED, which is the
    most severe breakage there is, and `do_refund` calls it. But a deleted
    object has no node at HEAD, and the consumer's call edge was dropped at
    link time for want of a target, so the join finds nothing and the caller
    is never reported.

    Indexing the library at the ref the consumer resolves against, instead of
    at HEAD, fixes this properly: the symbol still exists there, the edges
    land, and a deletion reports like any other change. This test flips to
    asserting `do_refund` IS reported when that lands.
    """
    lib, svc, graph = two_repos
    handlers = svc / "billing" / "handlers.py"

    kinds = {c.kind for c in breaking_changes("payments", lib, "v1", "v2")}
    assert "OBJECT_REMOVED" in kinds, "griffe still classifies the deletion"

    reported = {i.consumer for i in _report(lib, graph)}
    assert f"{handlers}::do_refund" not in reported, (
        "if this now passes, the base-ref indexing landed: flip this assertion"
    )
