"""Shared test fixtures.

The important one is `deterministic_embeddings`. Real embeddings need a running
Ollama, which CI does not have, and random vectors make retrieval assertions
meaningless — a test that passes on noise proves nothing. So the fixture hashes
tokens into a fixed-width bag-of-words vector: fully offline and reproducible,
while still placing text that shares vocabulary close together. That is enough
for the pipeline's ranking behaviour to be asserted end to end.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from mnemostack.core.retrieval.call_graph import CallGraph
from mnemostack.core.retrieval.faiss_index import FaissIndex, create_chunks_db
from mnemostack.core.retrieval.fts_index import FTSIndex

EMBED_DIM = 64
_TOKEN = re.compile(r"[a-z0-9]+")


def _hash_embed(text: str) -> np.ndarray:
    """Bag of hashed tokens, L2-normalised so distance tracks overlap.

    Identifiers are also split on case and underscores, so a query saying
    "resolve module" lands near a chunk defining `resolve_module_file`.
    """
    words = _TOKEN.findall(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ").lower())
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for word in words:
        # crc32, not hash(): str hashing is salted per process, so hash()
        # would make the 'deterministic' embedder differ between runs.
        vec[zlib.crc32(word.encode()) % EMBED_DIM] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


@pytest.fixture
def deterministic_embeddings(monkeypatch):
    """Point every embedding call at the offline hash embedder."""
    import mnemostack.core.retrieval.indexer as indexer_mod
    import mnemostack.core.retrieval.query as query_mod

    def embed_texts(texts):
        return np.stack([_hash_embed(t) for t in texts])

    monkeypatch.setattr(indexer_mod, "embed_texts", embed_texts)
    monkeypatch.setattr(query_mod, "embed_query", lambda q, model=None: _hash_embed(q))
    return _hash_embed


@pytest.fixture
def tmp_store(tmp_path):
    """Provides a temporary store directory."""
    return tmp_path / "store"


@pytest.fixture
def shared_db(tmp_store):
    """Shared SQLite connection for FAISS + FTS."""
    return create_chunks_db(tmp_store)


@pytest.fixture
def faiss_idx(shared_db, tmp_store):
    idx = FaissIndex(store_dir=tmp_store, dimension=4, db=shared_db)
    yield idx
    idx.close()


@pytest.fixture
def fts_idx(shared_db, tmp_store):
    idx = FTSIndex(store_dir=tmp_store, db=shared_db)
    yield idx
    idx.close()


@pytest.fixture
def graph(tmp_store):
    g = CallGraph(store_dir=tmp_store)
    yield g
    g.close()


@pytest.fixture
def indexes(shared_db, tmp_store):
    """A full set of indexes sized for the hash embedder, for pipeline tests."""
    faiss_idx = FaissIndex(store_dir=tmp_store, dimension=EMBED_DIM, db=shared_db)
    fts = FTSIndex(store_dir=tmp_store, db=shared_db)
    call_graph = CallGraph(store_dir=tmp_store)
    yield faiss_idx, fts, call_graph
    faiss_idx.close()
    fts.close()
    call_graph.close()


@pytest.fixture
def workspace(tmp_path):
    """Two Python repos and a JS app on disk, wired like real projects.

    service imports shared_lib (a separate repo) and requests (installed only in
    service's virtualenv, never indexed). The JS app imports its own module and
    a package in node_modules. This is the shape every cross-repo and boundary
    claim is checked against.
    """
    ws = tmp_path / "ws"

    service = ws / "work" / "service"
    (service / ".git").mkdir(parents=True)
    (service / "main.py").write_text(
        "import requests\n"
        "from shared_lib.util import normalise\n\n"
        "def handle_request(payload):\n"
        "    return normalise(payload)\n"
    )
    site = service / ".venv" / "lib" / "python3.12" / "site-packages" / "requests"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("def get(url):\n    return url\n")

    shared = ws / "elsewhere" / "shared-lib"
    (shared / ".git").mkdir(parents=True)
    pkg = shared / "shared_lib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "util.py").write_text("def normalise(text):\n    return text.strip()\n")

    app = ws / "web" / "app"
    (app / ".git").mkdir(parents=True)
    (app / "Widget.jsx").write_text("export function Widget() { return <div />; }\n")
    (app / "Page.jsx").write_text(
        "import { Widget } from './Widget';\n"
        "import pad from 'left-pad';\n"
        "export function Page() { return <div><Widget />{pad('x')}</div>; }\n"
    )
    dep = app / "node_modules" / "left-pad"
    dep.mkdir(parents=True)
    (dep / "package.json").write_text('{"main": "./index.js"}')
    (dep / "index.js").write_text("module.exports = function pad(s) { return s; };\n")

    return {
        "root": ws,
        "service": service,
        "shared": shared,
        "app": app,
        "service_main": service / "main.py",
        "shared_util": pkg / "util.py",
        "installed_requests": site / "__init__.py",
        "page": app / "Page.jsx",
        "widget": app / "Widget.jsx",
        "installed_pad": dep / "index.js",
    }


def index_all(roots: list[Path], indexes) -> int:
    """Index each root in order, as index_project would across several repos."""
    from mnemostack.core.retrieval.indexer import index_directory

    faiss_idx, fts, call_graph = indexes
    return sum(
        index_directory(root=root, faiss_idx=faiss_idx, fts_idx=fts, graph=call_graph)
        for root in roots
    )
