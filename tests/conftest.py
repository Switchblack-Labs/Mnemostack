"""Shared fixtures.

The embedding stub and index fixtures that used to live here went with the
retrieval stack. What the impact tests need is a graph and a temp store, both
of which are cheap enough that most tests build them inline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemostack.core.retrieval.call_graph import CallGraph


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
def graph(tmp_store: Path):
    g = CallGraph(store_dir=tmp_store)
    yield g
    g.close()
