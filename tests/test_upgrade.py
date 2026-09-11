"""Upgrade impact: the precision rules that decide whether this is trustworthy.

A tool that flags every call site of a changed function is worse than no tool,
because it trains you to ignore it. These tests pin the cases where the honest
answer is "this does not affect you", alongside the ones where it does.

Offline by design: two versions of a fake package on disk, loaded through the
same griffe entry point that load_pypi feeds, so no network and no flakiness.
"""

from __future__ import annotations

from pathlib import Path

import griffe
import pytest

from mnemostack.core.impact.api_diff import ApiChange, fqn_index
from mnemostack.core.impact.propagate import impact_report
from mnemostack.core.impact.upgrade import (
    _pair_with_nodes,
    add_dependency_surface,
    narrow,
    verify,
)
from mnemostack.core.retrieval.call_graph import (
    CallGraph,
    build_nodes_for_python_file,
    link_python_file_imports,
)

V1 = """
class Session:
    def __init__(self, url, *, timeout=None, retries=0):
        self.url = url

    def send(self, body):
        return body

    def drain(self):
        return None


def connect(url):
    return Session(url)
"""

V2 = """
class Session:
    # `timeout` is gone: breaks only callers that passed it.
    def __init__(self, url, *, retries=0):
        self.url = url

    def send(self, body):
        return body

    # `drain` is gone: breaks every caller.


def connect(url, token):
    # `token` is now required: breaks every caller.
    return Session(url)
"""

CONSUMER = """
from paylib import Session, connect


def passes_the_removed_kwarg():
    return Session("http://x", timeout=5)


def does_not_pass_it():
    return Session("http://x", retries=2)


def calls_removed_method():
    s = Session("http://x")
    return s.drain()


def calls_changed_function():
    return connect("http://x")
"""


def _write_pkg(root: Path, body: str) -> Path:
    pkg = root / "paylib"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(body)
    return root


@pytest.fixture
def scenario(tmp_path: Path):
    old_root = _write_pkg(tmp_path / "v1", V1)
    new_root = _write_pkg(tmp_path / "v2", V2)
    old = griffe.load("paylib", search_paths=[old_root])
    new = griffe.load("paylib", search_paths=[new_root])

    consumer = tmp_path / "app"
    consumer.mkdir()
    (consumer / "handlers.py").write_text(CONSUMER)

    graph = CallGraph(store_dir=tmp_path / "store")
    add_dependency_surface(graph, old)
    files = sorted(consumer.rglob("*.py"))
    for f in files:
        build_nodes_for_python_file(f, graph=graph)
    for f in files:
        link_python_file_imports(f, graph=graph, import_root=consumer)

    changes = [
        ApiChange(
            fqn=str(d.get("object_path")),
            kind=getattr(d.get("kind"), "name", str(d.get("kind"))),
            old=None if d.get("old_value") is None else str(d.get("old_value")),
            new=None if d.get("new_value") is None else str(d.get("new_value")),
        )
        for d in (b.as_dict() for b in griffe.find_breaking_changes(old, new))
    ]
    yield consumer, graph, new, changes
    graph.close()


def _report(graph, new, changes):
    real = verify(changes, new)
    return narrow(impact_report(graph, _pair_with_nodes(fqn_index(graph), real)))


def _consumers(report) -> set[str]:
    return {i.consumer.split("::")[1] for i in report}


def test_removed_keyword_reported_only_where_it_is_passed(scenario):
    """The rule that makes the report worth reading.

    `timeout` was removed. One function passes it and breaks; the other passes a
    different keyword and is fine. Flagging both would make every tidy-up of a
    signature look like a breakage at every call site.
    """
    consumer, graph, new, changes = scenario
    names = _consumers(_report(graph, new, changes))

    assert "passes_the_removed_kwarg" in names
    assert "does_not_pass_it" not in names


def test_removed_method_is_reported(scenario):
    """`Session.drain` is gone, so its caller breaks regardless of arguments."""
    consumer, graph, new, changes = scenario
    assert "calls_removed_method" in _consumers(_report(graph, new, changes))


def test_newly_required_parameter_is_reported(scenario):
    """`connect` gained a required argument: every existing caller breaks."""
    consumer, graph, new, changes = scenario
    assert "calls_changed_function" in _consumers(_report(graph, new, changes))


def test_verify_keeps_real_removals(scenario):
    """The overload filter must not swallow genuine deletions."""
    consumer, graph, new, changes = scenario
    kinds = {(c.fqn, c.kind) for c in verify(changes, new)}
    assert ("paylib.Session.drain", "OBJECT_REMOVED") in kinds


def test_dependency_surface_registers_symbols(tmp_path):
    """A dependency's API becomes nodes, which is what lets imports resolve."""
    root = _write_pkg(tmp_path / "v1", V1)
    module = griffe.load("paylib", search_paths=[root])
    graph = CallGraph(store_dir=tmp_path / "store")
    count = add_dependency_surface(graph, module)

    assert count > 0
    index = fqn_index(graph)
    assert "paylib.Session" in index
    assert "paylib.Session.send" in index
    graph.close()


# --- griffe already filters private symbols ------------------------------

PRIV_V1 = """
__all__ = ["public_api"]


def public_api(x):
    return _helper(x)


def _helper(x, mode="fast"):
    return x
"""

PRIV_V2 = """
__all__ = ["public_api"]


def public_api(x, required):
    return x


def _helper(x):
    return x
"""


def test_griffe_reports_no_breakage_for_private_symbols(tmp_path):
    """Why there is no severity downgrade for private symbols in this codebase.

    The literature recommends downgrading a break in an underscore-prefixed or
    non-__all__ symbol, because the library never promised it. AexPy needs that
    rule since it computes its own diff. griffe does not: it drops private
    symbols before reporting, so a downgrade rule here could never fire.

    Both functions change signature below and only the public one is reported.
    If this ever fails, griffe changed and the rule becomes worth adding.
    """
    old = griffe.load("paylib", search_paths=[_write_pkg(tmp_path / "v1", PRIV_V1)])
    new = griffe.load("paylib", search_paths=[_write_pkg(tmp_path / "v2", PRIV_V2)])

    reported = {str(b.as_dict().get("object_path")) for b in griffe.find_breaking_changes(old, new)}
    assert "paylib.public_api" in reported
    assert "paylib._helper" not in reported
