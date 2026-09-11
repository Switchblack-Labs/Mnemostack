"""Symbol-level edge recall across two repos.

The cross-repo claim is only worth anything if the graph links *symbols*, not
files. File-level reachability lights up through the File node's CONTAINS edges
whether or not a single real symbol edge exists, so asserting reachability would
pass on a graph that knows nothing. Every assertion here names both endpoints.

The fixture is small but deliberately not flat: a package that exposes its API
through ``__init__.py`` re-exports, a class consumed by instance method call,
and a base class subclassed downstream. That is what a real dependency looks
like, and each pattern is scored separately so the table says which mechanism
is missing rather than just "recall is low".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemostack.core.retrieval.call_graph import (
    CallGraph,
    EdgeType,
    build_nodes_for_python_file,
    link_python_file_imports,
)

# --- fixture repos -----------------------------------------------------------

LIB_FILES = {
    "libcore/__init__.py": (
        "from libcore.auth import verify\n"
        "from libcore.client import BaseHandler, Client\n"
        "\n"
        '__all__ = ["verify", "Client", "BaseHandler"]\n'
    ),
    "libcore/auth.py": ("def verify(token):\n    return bool(token)\n"),
    "libcore/client.py": (
        "class Client:\n"
        "    def fetch(self, url):\n"
        "        return url\n"
        "\n"
        "\n"
        "class BaseHandler:\n"
        "    def handle(self, event):\n"
        "        return event\n"
    ),
}

SVC_FILES = {
    "app.py": (
        "from libcore.auth import verify\n"
        "from libcore import verify as reexported_verify\n"
        "from libcore.client import BaseHandler, Client\n"
        "\n"
        "\n"
        "def use_direct(token):\n"
        "    return verify(token)\n"
        "\n"
        "\n"
        "def use_reexport(token):\n"
        "    return reexported_verify(token)\n"
        "\n"
        "\n"
        "def use_client(url):\n"
        "    client = Client()\n"
        "    return client.fetch(url)\n"
        "\n"
        "\n"
        "class MyHandler(BaseHandler):\n"
        "    def handle(self, event):\n"
        "        return event\n"
    ),
}


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


@pytest.fixture
def two_repos(tmp_path: Path):
    """Index a library repo and a consumer repo into one graph.

    Nodes for every file first, then links: link_python_file_imports no-ops on a
    target whose node does not exist yet, so interleaving would make the result
    depend on file order.
    """
    # Separate parent directories, deliberately. resolve_module_file falls back to
    # searching every parent of the importing file, so two repos under one shared
    # parent resolve each other off disk like a single tree and the cross-repo
    # path is never taken. Split parents force the resolution to go through
    # find_indexed_module, which is the mechanism under test.
    lib = _write(tmp_path / "lib_side" / "libcore_repo", LIB_FILES)
    svc = _write(tmp_path / "svc_side" / "svc_repo", SVC_FILES)

    graph = CallGraph(store_dir=tmp_path / "store")
    for root, files in ((lib, LIB_FILES), (svc, SVC_FILES)):
        for rel in files:
            build_nodes_for_python_file(root / rel, graph=graph)
    for root, files in ((lib, LIB_FILES), (svc, SVC_FILES)):
        for rel in files:
            link_python_file_imports(root / rel, graph=graph, import_root=root)

    yield lib, svc, graph
    graph.close()


def _has_edge(graph: CallGraph, source: str, target: str, edge_type: EdgeType | None) -> bool:
    """One hop from source to target. edge_type None means any kind of edge."""
    types = [edge_type] if edge_type is not None else None
    return target in graph.get_neighbors(source, hops=1, edge_types=types)


def _ground_truth(lib: Path, svc: Path) -> list[tuple[str, str, str, EdgeType | None, bool]]:
    """(pattern, source qname, target qname, edge type, resolves today).

    The subclass row asks for any edge at all: there is no INHERITS type yet, so
    naming one would be asserting against a mechanism that does not exist.

    The last column is per-pattern rather than one recall number on purpose. A
    scalar lets two rows swap states and still read 25%, and it lets a fixed gap
    pass silently against a `>=` bound while the recorded figure goes stale.
    Flip a False to True in the same commit that closes the gap.
    """
    app, auth = svc / "app.py", lib / "libcore" / "auth.py"
    client = lib / "libcore" / "client.py"
    return [
        ("direct-import call", f"{app}::use_direct", f"{auth}::verify", EdgeType.CALLS, True),
        (
            "__init__ re-export call",
            f"{app}::use_reexport",
            f"{auth}::verify",
            EdgeType.CALLS,
            False,
        ),
        (
            "instance method call",
            f"{app}::use_client",
            f"{client}::Client.fetch",
            EdgeType.CALLS,
            False,
        ),
        ("subclass of imported base", f"{app}::MyHandler", f"{client}::BaseHandler", None, False),
    ]


def test_ground_truth_endpoints_exist(two_repos):
    """Every row must be *able* to match before its result means anything.

    _has_edge cannot tell "the edge is missing" from "that qname was never a
    node", so a rename in the fixture that is not mirrored in _ground_truth
    would pin a row at MISS forever and the baseline would keep looking honest
    while measuring nothing. This is the guard that makes the table
    self-verifying rather than correct-because-somebody-checked-once.
    """
    lib, svc, graph = two_repos
    for name, source, target, _, _ in _ground_truth(lib, svc):
        assert graph.has_node(source), f"{name}: source node {source} does not exist"
        assert graph.has_node(target), f"{name}: target node {target} does not exist"


def test_cross_repo_symbol_recall(two_repos, capsys):
    lib, svc, graph = two_repos
    rows = _ground_truth(lib, svc)

    actual = {name: _has_edge(graph, s, t, e) for name, s, t, e, _ in rows}
    expected = {name: exp for name, _, _, _, exp in rows}
    recall = sum(actual.values()) / len(actual)

    with capsys.disabled():
        print(f"\n  cross-repo symbol edge recall: {recall:.0%}")
        for name, hit in actual.items():
            print(f"    {'HIT ' if hit else 'MISS'}  {name}")

    assert actual == expected, (
        "cross-repo edge resolution changed. If a gap was closed, flip that "
        "row's last column in _ground_truth in the same commit."
    )


def test_direct_import_call_resolves(two_repos):
    """The one pattern that works today, pinned by name.

    The table above only checks the set of results, so on its own it would stay
    green if this pattern broke while another started working. This says which
    one is load-bearing.
    """
    lib, svc, graph = two_repos
    assert _has_edge(
        graph,
        f"{svc / 'app.py'}::use_direct",
        f"{lib / 'libcore' / 'auth.py'}::verify",
        EdgeType.CALLS,
    )


def test_reachability_reaches_symbols_with_no_symbol_edge(two_repos):
    """Why recall is measured on edges, not reachability.

    The consumer reaches the library's File node by IMPORTS_FROM, and every
    symbol hangs off that File node by CONTAINS. So a reachability query returns
    library *symbols* whether or not any symbol edge to them exists, which is
    why asserting reachability would pass on a graph that knows nothing.

    Deliberately no assertion that a given edge is absent: the expected-status
    table owns that, and pinning an absence here would turn implementing a
    missing mechanism into a test failure.
    """
    lib, svc, graph = two_repos
    client = lib / "libcore" / "client.py"

    reachable = graph.get_neighbors(str(svc / "app.py"), hops=2)
    assert f"{client}::Client" in reachable
    assert f"{client}::BaseHandler" in reachable
