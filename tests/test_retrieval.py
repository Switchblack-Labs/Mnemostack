"""Tests for the core retrieval pipeline.

Covers: AST chunking, FAISS add/search/remove, FTS5 search, RRF fusion,
recency/dependency reranking, call graph construction + BFS, and the
full query pipeline (with mocked embeddings).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mnemostack.core.retrieval.ast_chunker import (
    Chunk,
    ChunkType,
    chunk_file,
    chunk_file_auto,
    chunk_file_fallback,
)
from mnemostack.core.retrieval.call_graph import (
    CallGraph,
    EdgeType,
    NodeType,
    build_graph_for_python_file,
)
from mnemostack.core.retrieval.faiss_index import FaissIndex, create_chunks_db
from mnemostack.core.retrieval.fts_index import FTSIndex
from mnemostack.core.retrieval.ranker import (
    RankedResult,
    apply_query_intent_boost,
    compute_recency_score,
    reciprocal_rank_fusion,
    rerank,
)

# --- Fixtures ---


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


def _make_chunks(file_path: str, names: list[str], mtime: float = 0.0) -> list[Chunk]:
    """Helper to create test chunks."""
    return [
        Chunk(
            file_path=file_path,
            symbol_name=name,
            code=f"def {name}(): pass",
            line_start=i + 1,
            line_end=i + 1,
            chunk_type=ChunkType.FUNCTION,
            last_modified=mtime or time.time(),
            qualified_name=f"{file_path}::{name}",
            dependencies=[],
        )
        for i, name in enumerate(names)
    ]


def _random_embeddings(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    return vecs


# --- AST Chunker Tests ---


class TestASTChunker:
    def test_chunk_python_function(self, tmp_path):
        f = tmp_path / "example.py"
        f.write_text("def hello():\n    return 'world'\n")
        chunks = chunk_file(f)
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "hello"
        assert chunks[0].chunk_type == ChunkType.FUNCTION
        assert chunks[0].line_start == 1
        assert chunks[0].line_end == 2

    def test_chunk_python_class_with_methods(self, tmp_path):
        f = tmp_path / "cls.py"
        f.write_text(
            "class Foo:\n"
            "    def bar(self):\n"
            "        pass\n"
            "    def baz(self):\n"
            "        pass\n"
        )
        chunks = chunk_file(f)
        names = [c.symbol_name for c in chunks]
        assert "Foo" in names
        assert "Foo.bar" in names
        assert "Foo.baz" in names

    def test_chunk_python_imports_grouped(self, tmp_path):
        f = tmp_path / "imp.py"
        f.write_text("import os\nimport sys\nfrom pathlib import Path\n")
        chunks = chunk_file(f)
        import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
        assert len(import_chunks) == 1
        assert "os" in import_chunks[0].code

    def test_chunk_unsupported_ext_returns_empty(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("hello world")
        chunks = chunk_file(f)
        assert chunks == []

    def test_chunk_file_auto_fallback(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome content here\n")
        chunks = chunk_file_auto(f)
        assert len(chunks) > 0
        assert chunks[0].chunk_type == ChunkType.CONSTANT

    def test_chunk_file_fallback_splits_on_blank_lines(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("block one\nline two\n\nblock two\n")
        chunks = chunk_file_fallback(f)
        assert len(chunks) == 2

    def test_chunk_javascript(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text("function greet(name) {\n  return `Hello ${name}`;\n}\n")
        chunks = chunk_file(f)
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "greet"

    def test_qualified_name_format(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def process(): pass\n")
        chunks = chunk_file(f)
        assert "::" in chunks[0].qualified_name
        assert "process" in chunks[0].qualified_name


# --- FAISS Index Tests ---


class TestFaissIndex:
    def test_add_and_search_roundtrip(self, faiss_idx):
        chunks = _make_chunks("a.py", ["foo", "bar", "baz"])
        embeddings = _random_embeddings(3)
        ids = faiss_idx.add(chunks, embeddings)
        assert len(ids) == 3
        assert faiss_idx.total_chunks == 3

        results = faiss_idx.search(embeddings[0], top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == ids[0]  # closest to itself

    def test_remove_by_file(self, faiss_idx):
        chunks = _make_chunks("a.py", ["foo", "bar"])
        embeddings = _random_embeddings(2)
        faiss_idx.add(chunks, embeddings)
        assert faiss_idx.total_chunks == 2

        removed = faiss_idx.remove_by_file("a.py")
        assert removed == 2
        assert faiss_idx.total_chunks == 0

    def test_remove_nonexistent_file(self, faiss_idx):
        removed = faiss_idx.remove_by_file("does_not_exist.py")
        assert removed == 0

    def test_get_chunks_by_file(self, faiss_idx):
        chunks = _make_chunks("a.py", ["foo"]) + _make_chunks("b.py", ["bar"])
        embeddings = _random_embeddings(2)
        faiss_idx.add(chunks, embeddings)

        results = faiss_idx.get_chunks_by_file("a.py")
        assert len(results) == 1
        assert results[0].symbol_name == "foo"

    def test_get_chunk_by_id(self, faiss_idx):
        chunks = _make_chunks("a.py", ["foo"])
        embeddings = _random_embeddings(1)
        ids = faiss_idx.add(chunks, embeddings)

        result = faiss_idx.get_chunk_by_id(ids[0])
        assert result is not None
        assert result.symbol_name == "foo"

    def test_get_chunk_by_id_nonexistent(self, faiss_idx):
        result = faiss_idx.get_chunk_by_id(9999)
        assert result is None

    def test_get_chunk_ids_by_qnames(self, faiss_idx):
        chunks = _make_chunks("a.py", ["foo", "bar"])
        embeddings = _random_embeddings(2)
        ids = faiss_idx.add(chunks, embeddings)

        mapping = faiss_idx.get_chunk_ids_by_qnames(["a.py::foo", "a.py::bar", "a.py::nope"])
        assert mapping["a.py::foo"] == ids[0]
        assert mapping["a.py::bar"] == ids[1]
        assert "a.py::nope" not in mapping

    def test_search_empty_index(self, faiss_idx):
        query = np.zeros(4, dtype=np.float32)
        results = faiss_idx.search(query, top_k=5)
        assert results == []

    def test_embeddings_shape_mismatch_raises(self, faiss_idx):
        chunks = _make_chunks("a.py", ["foo"])
        wrong_shape = np.zeros((1, 8), dtype=np.float32)  # dim=8 but index is dim=4
        with pytest.raises(ValueError, match="Expected embeddings shape"):
            faiss_idx.add(chunks, wrong_shape)


# --- FTS5 Index Tests ---


class TestFTSIndex:
    def test_sync_added_and_search(self, faiss_idx, fts_idx):
        chunks = _make_chunks("a.py", ["validate_token", "parse_request"])
        embeddings = _random_embeddings(2)
        ids = faiss_idx.add(chunks, embeddings)
        fts_idx.sync_added(ids)

        results = fts_idx.search("validate_token")
        assert len(results) >= 1
        assert results[0].symbol_name == "validate_token"

    def test_sync_removed(self, faiss_idx, fts_idx):
        chunks = _make_chunks("a.py", ["handler"])
        embeddings = _random_embeddings(1)
        ids = faiss_idx.add(chunks, embeddings)
        fts_idx.sync_added(ids)

        # Remove FTS entries
        fts_idx.sync_removed("a.py")
        results = fts_idx.search("handler")
        assert results == []

    def test_search_empty_query(self, fts_idx):
        results = fts_idx.search("")
        assert results == []

    def test_search_no_matches(self, fts_idx):
        results = fts_idx.search("nonexistent_symbol_xyz")
        assert results == []

    def test_fts_results_carry_dependencies(self, faiss_idx, fts_idx):
        chunks = [
            Chunk(
                file_path="a.py",
                symbol_name="caller",
                code="def caller(): callee()",
                line_start=1,
                line_end=1,
                chunk_type=ChunkType.FUNCTION,
                last_modified=time.time(),
                qualified_name="a.py::caller",
                dependencies=["a.py::callee"],
            )
        ]
        embeddings = _random_embeddings(1)
        ids = faiss_idx.add(chunks, embeddings)
        fts_idx.sync_added(ids)

        results = fts_idx.search("caller")
        assert len(results) == 1
        assert results[0].dependencies == ["a.py::callee"]


# --- Call Graph Tests ---


class TestCallGraph:
    def test_add_node_and_get_count(self, graph):
        graph.add_node("mod.py", NodeType.FILE, "mod.py")
        graph.add_node("mod.py::foo", NodeType.FUNCTION, "mod.py")
        graph.commit()
        assert graph.node_count == 2

    def test_add_edge_and_get_count(self, graph):
        graph.add_node("a.py", NodeType.FILE, "a.py")
        graph.add_node("a.py::foo", NodeType.FUNCTION, "a.py")
        graph.add_edge("a.py", "a.py::foo", EdgeType.CONTAINS)
        graph.commit()
        assert graph.edge_count == 1

    def test_get_neighbors_bfs(self, graph):
        graph.add_node("a.py::foo", NodeType.FUNCTION, "a.py")
        graph.add_node("a.py::bar", NodeType.FUNCTION, "a.py")
        graph.add_node("a.py::baz", NodeType.FUNCTION, "a.py")
        graph.add_edge("a.py::foo", "a.py::bar", EdgeType.CALLS)
        graph.add_edge("a.py::bar", "a.py::baz", EdgeType.CALLS)
        graph.commit()

        # 1-hop from foo should get bar
        neighbors_1 = graph.get_neighbors("a.py::foo", hops=1, direction="outgoing")
        assert "a.py::bar" in neighbors_1
        assert "a.py::baz" not in neighbors_1

        # 2-hop from foo should get bar and baz
        neighbors_2 = graph.get_neighbors("a.py::foo", hops=2, direction="outgoing")
        assert "a.py::bar" in neighbors_2
        assert "a.py::baz" in neighbors_2

    def test_get_neighbors_nonexistent_node(self, graph):
        result = graph.get_neighbors("does_not_exist")
        assert result == []

    def test_remove_file(self, graph):
        graph.add_node("a.py", NodeType.FILE, "a.py")
        graph.add_node("a.py::foo", NodeType.FUNCTION, "a.py")
        graph.add_edge("a.py", "a.py::foo", EdgeType.CONTAINS)
        graph.commit()

        graph.remove_file("a.py")
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_has_node(self, graph):
        graph.add_node("x.py::func", NodeType.FUNCTION, "x.py")
        graph.commit()
        assert graph.has_node("x.py::func")
        assert not graph.has_node("x.py::nope")

    def test_build_graph_for_python_file(self, tmp_path, graph):
        f = tmp_path / "sample.py"
        f.write_text(
            "import os\n\n"
            "def caller():\n"
            "    callee()\n\n"
            "def callee():\n"
            "    pass\n"
        )
        build_graph_for_python_file(f, graph=graph)

        assert graph.node_count >= 3  # file + 2 functions
        assert graph.has_node(f"{f}::caller")
        assert graph.has_node(f"{f}::callee")

        # caller -> callee CALLS edge
        neighbors = graph.get_neighbors(f"{f}::caller", hops=1, direction="outgoing")
        assert f"{f}::callee" in neighbors


class TestCrossFileLinking:
    """Cross-file CALLS / IMPORTS_FROM edges via real import resolution."""

    def _build(self, graph, files):
        from mnemostack.core.retrieval.call_graph import (
            build_nodes_for_python_file,
            link_python_file_imports,
        )

        for f in files:
            build_nodes_for_python_file(f, graph=graph)
        for f in files:
            link_python_file_imports(f, graph=graph)

    def test_from_import_call_links_across_files(self, tmp_path, graph):
        proj = tmp_path / "proj"
        proj.mkdir()
        util = proj / "util.py"
        main = proj / "main.py"
        util.write_text("def helper():\n    return 1\n")
        main.write_text("from util import helper\n\ndef run():\n    helper()\n")
        self._build(graph, [util, main])

        calls = graph.get_neighbors(
            f"{main}::run", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        )
        assert f"{util}::helper" in calls, "cross-file CALLS edge not created"

        imports = graph.get_neighbors(
            str(main), hops=1, direction="outgoing", edge_types=(EdgeType.IMPORTS_FROM,)
        )
        assert str(util) in imports, "cross-file IMPORTS_FROM edge not created"

    def test_aliased_module_call_links_across_files(self, tmp_path, graph):
        proj = tmp_path / "proj"
        proj.mkdir()
        util = proj / "util.py"
        main = proj / "main.py"
        util.write_text("def helper():\n    return 1\n")
        main.write_text("import util as u\n\ndef run():\n    u.helper()\n")
        self._build(graph, [util, main])

        calls = graph.get_neighbors(
            f"{main}::run", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        )
        assert f"{util}::helper" in calls

    def test_relative_import_links_within_package(self, tmp_path, graph):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        a = pkg / "a.py"
        b = pkg / "b.py"
        a.write_text("def fa():\n    return 1\n")
        b.write_text("from .a import fa\n\ndef fb():\n    fa()\n")
        self._build(graph, [a, b])

        calls = graph.get_neighbors(
            f"{b}::fb", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        )
        assert f"{a}::fa" in calls

    def test_external_import_creates_no_edge(self, tmp_path, graph):
        proj = tmp_path / "proj"
        proj.mkdir()
        main = proj / "main.py"
        main.write_text("import os\n\ndef run():\n    os.getpid()\n")
        self._build(graph, [main])

        # Nothing off-disk resolves, so no IMPORTS_FROM / CALLS edge is invented.
        imports = graph.get_neighbors(
            str(main), hops=2, direction="outgoing", edge_types=(EdgeType.IMPORTS_FROM,)
        )
        assert imports == []

    def test_query_surfaces_cross_file_dependency(
        self, tmp_store, shared_db, graph, monkeypatch
    ):
        """End to end: a function in another file, reachable only through an
        imported call, surfaces in query results."""
        import mnemostack.core.retrieval.indexer as indexer_mod
        import mnemostack.core.retrieval.query as query_mod
        from mnemostack.core.retrieval.indexer import index_directory
        from mnemostack.core.retrieval.query import query_pipeline

        faiss_idx = FaissIndex(store_dir=tmp_store, dimension=4, db=shared_db)
        fts_idx = FTSIndex(store_dir=tmp_store, db=shared_db)

        proj = tmp_store / "proj"
        proj.mkdir(parents=True)
        (proj / "crypto.py").write_text(
            "def zzhashpw_unique(pw):\n    return pw[::-1]\n"
        )
        (proj / "auth.py").write_text(
            "from crypto import zzhashpw_unique\n\n"
            "def authenticate_user(pw):\n"
            "    return zzhashpw_unique(pw)\n"
        )

        def fake_embed(texts):
            rng = np.random.default_rng(len(texts))
            return rng.standard_normal((len(texts), 4)).astype(np.float32)

        monkeypatch.setattr(indexer_mod, "embed_texts", fake_embed)
        index_directory(root=proj, faiss_idx=faiss_idx, fts_idx=fts_idx, graph=graph)
        monkeypatch.setattr(
            query_mod, "embed_query", lambda q, model=None: np.zeros(4, dtype=np.float32)
        )

        results = query_pipeline(
            query="authenticate_user",
            faiss_idx=faiss_idx,
            fts_idx=fts_idx,
            graph=graph,
            top_k=5,
        )
        qnames = {r.qualified_name for r in results}
        auth_q = f"{proj / 'auth.py'}::authenticate_user"
        crypto_q = f"{proj / 'crypto.py'}::zzhashpw_unique"
        assert auth_q in qnames
        assert crypto_q in qnames, "cross-file dependency did not surface in retrieval"

        faiss_idx.close()
        fts_idx.close()


# --- Ranker Tests ---


class TestRanker:
    def _make_faiss_result(self, chunk_id, score=0.5, symbol="foo", qname="a.py::foo"):
        from mnemostack.core.retrieval.faiss_index import SearchResult
        return SearchResult(
            chunk_id=chunk_id,
            score=score,
            file_path="a.py",
            symbol_name=symbol,
            code="def foo(): pass",
            line_start=1,
            line_end=1,
            chunk_type="function",
            qualified_name=qname,
            last_modified=time.time(),
            dependencies=[],
        )

    def _make_fts_result(self, chunk_id, bm25=1.0, symbol="foo", qname="a.py::foo"):
        from mnemostack.core.retrieval.fts_index import FTSResult
        return FTSResult(
            chunk_id=chunk_id,
            bm25_score=bm25,
            file_path="a.py",
            symbol_name=symbol,
            qualified_name=qname,
            code="def foo(): pass",
            line_start=1,
            line_end=1,
            chunk_type="function",
            last_modified=time.time(),
            dependencies=["a.py::bar"],
        )

    def test_rrf_merges_both_lists(self):
        faiss_results = [self._make_faiss_result(1, symbol="a", qname="a.py::a")]
        fts_results = [self._make_fts_result(2, symbol="b", qname="a.py::b")]
        fused = reciprocal_rank_fusion(faiss_results, fts_results)
        ids = {r.chunk_id for r in fused}
        assert 1 in ids
        assert 2 in ids

    def test_rrf_shared_chunk_scores_higher(self):
        # Chunk 1 appears in both lists, chunk 2 only in FAISS
        faiss_results = [
            self._make_faiss_result(1, symbol="shared", qname="a.py::shared"),
            self._make_faiss_result(2, symbol="only_faiss", qname="a.py::only_faiss"),
        ]
        fts_results = [
            self._make_fts_result(1, symbol="shared", qname="a.py::shared"),
        ]
        fused = reciprocal_rank_fusion(faiss_results, fts_results)
        assert fused[0].chunk_id == 1  # shared should rank first

    def test_rrf_fts_only_carries_dependencies(self):
        faiss_results = []
        fts_results = [self._make_fts_result(5, symbol="x", qname="a.py::x")]
        fused = reciprocal_rank_fusion(faiss_results, fts_results)
        assert fused[0].dependencies == ["a.py::bar"]

    def test_recency_score_recent_is_high(self):
        now = time.time()
        score = compute_recency_score(now - 10, now)  # 10 seconds ago
        assert score > 0.9

    def test_recency_score_old_is_low(self):
        now = time.time()
        score = compute_recency_score(now - 86400 * 7, now)  # 1 week ago
        assert score < 0.1

    def test_query_intent_boost_pascal_class(self):
        results = [
            RankedResult(
                chunk_id=1, file_path="a.py", symbol_name="AuthService",
                code="class AuthService: ...", line_start=1, line_end=1,
                chunk_type="class", qualified_name="a.py::AuthService",
                last_modified=time.time(), dependencies=[], final_score=1.0,
            ),
            RankedResult(
                chunk_id=2, file_path="a.py", symbol_name="auth_service",
                code="def auth_service(): ...", line_start=5, line_end=5,
                chunk_type="function", qualified_name="a.py::auth_service",
                last_modified=time.time(), dependencies=[], final_score=1.0,
            ),
        ]
        apply_query_intent_boost(results, "AuthService")
        assert results[0].final_score > results[1].final_score

    def test_query_intent_boost_snake_function(self):
        results = [
            RankedResult(
                chunk_id=1, file_path="a.py", symbol_name="validate_input",
                code="def validate_input(): ...", line_start=1, line_end=1,
                chunk_type="function", qualified_name="a.py::validate_input",
                last_modified=time.time(), dependencies=[], final_score=1.0,
            ),
        ]
        apply_query_intent_boost(results, "validate_input")
        assert results[0].final_score == 1.5

    def test_rerank_dependency_bonus(self):
        results = [
            RankedResult(
                chunk_id=1, file_path="a.py", symbol_name="foo",
                code="...", line_start=1, line_end=1,
                chunk_type="function", qualified_name="a.py::foo",
                last_modified=time.time(), dependencies=[], final_score=0.01,
            ),
            RankedResult(
                chunk_id=2, file_path="a.py", symbol_name="bar",
                code="...", line_start=2, line_end=2,
                chunk_type="function", qualified_name="a.py::bar",
                last_modified=time.time(), dependencies=[], final_score=0.01,
            ),
        ]
        # chunk 2 is a dependency
        reranked = rerank(results, dependency_ids={2})
        assert reranked[0].chunk_id == 2  # dependency bonus pushes it up


# --- Indexer Integration Test ---


class TestIndexer:
    def test_index_directory_roundtrip(self, tmp_store, shared_db, graph):
        from mnemostack.core.retrieval.indexer import index_directory

        faiss_idx = FaissIndex(store_dir=tmp_store, dimension=4, db=shared_db)
        fts_idx = FTSIndex(store_dir=tmp_store, db=shared_db)

        # Create a small project
        project = tmp_store / "project"
        project.mkdir()
        (project / "main.py").write_text("def main():\n    helper()\n\ndef helper():\n    pass\n")
        (project / "util.py").write_text("def util_func():\n    return 42\n")

        # Mock embed_texts to avoid real API calls
        import mnemostack.core.retrieval.indexer as indexer_mod
        original_embed = indexer_mod.embed_texts

        def fake_embed(texts):
            rng = np.random.default_rng(len(texts))
            return rng.standard_normal((len(texts), 4)).astype(np.float32)

        indexer_mod.embed_texts = fake_embed
        try:
            count = index_directory(
                root=project,
                faiss_idx=faiss_idx,
                fts_idx=fts_idx,
                graph=graph,
            )
        finally:
            indexer_mod.embed_texts = original_embed

        assert count >= 3  # main, helper, util_func at minimum
        assert faiss_idx.total_chunks == count

        # FTS should find our functions
        fts_results = fts_idx.search("helper")
        assert len(fts_results) >= 1

        # Graph should have nodes
        assert graph.node_count > 0

        faiss_idx.close()
        fts_idx.close()


# --- Query Pipeline: graph expansion ---


class TestQueryPipelineExpansion:
    def test_pure_call_graph_dependency_surfaces(
        self, tmp_store, shared_db, graph, monkeypatch
    ):
        """A callee with no semantic/keyword overlap with the query must still
        appear in results purely because a top hit depends on it, and the seed's
        ``dependencies`` field must name that callee."""
        import mnemostack.core.retrieval.indexer as indexer_mod
        import mnemostack.core.retrieval.query as query_mod
        from mnemostack.core.retrieval.indexer import index_directory
        from mnemostack.core.retrieval.query import query_pipeline

        faiss_idx = FaissIndex(store_dir=tmp_store, dimension=4, db=shared_db)
        fts_idx = FTSIndex(store_dir=tmp_store, db=shared_db)

        # alphaqxz_entrypoint matches the query by name and CALLS zetawvu_helper,
        # which shares no token with the query. Enough noise functions (>fetch_k)
        # ensure zeta is pushed out of the FAISS candidate window, so it can only
        # reach the result set through call-graph expansion.
        lines = [
            "def alphaqxz_entrypoint():",
            "    zetawvu_helper()",
            "",
            "def zetawvu_helper():",
            "    return 1",
            "",
        ]
        for i in range(16):
            lines += [f"def noise_{i}():", f"    return {i}", ""]
        project = tmp_store / "project"
        project.mkdir()
        (project / "mod.py").write_text("\n".join(lines))

        mod = str(project / "mod.py")
        alpha_q = f"{mod}::alphaqxz_entrypoint"
        zeta_q = f"{mod}::zetawvu_helper"

        # Deterministic embeddings: query == alpha (distance 0, FAISS rank 1),
        # noise just off-axis (tiny distance), zeta orthogonal (farthest).
        def fake_embed(texts):
            vecs = []
            for j, text in enumerate(texts):
                if "def zetawvu_helper" in text:
                    vecs.append([0.0, 1.0, 0.0, 0.0])
                elif "def alphaqxz_entrypoint" in text:
                    vecs.append([1.0, 0.0, 0.0, 0.0])
                else:
                    vecs.append([1.0, 0.001 * (j + 1), 0.0, 0.0])
            return np.array(vecs, dtype=np.float32)

        monkeypatch.setattr(indexer_mod, "embed_texts", fake_embed)
        index_directory(root=project, faiss_idx=faiss_idx, fts_idx=fts_idx, graph=graph)

        monkeypatch.setattr(
            query_mod,
            "embed_query",
            lambda q, model=None: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

        results = query_pipeline(
            query="alphaqxz_entrypoint",
            faiss_idx=faiss_idx,
            fts_idx=fts_idx,
            graph=graph,
            top_k=5,
        )

        qnames = {r.qualified_name for r in results}
        assert alpha_q in qnames, "direct hit missing"
        assert zeta_q in qnames, "call-graph dependency was not merged into results"

        alpha = next(r for r in results if r.qualified_name == alpha_q)
        assert zeta_q in alpha.dependencies, "dependency chain not surfaced on seed"

        faiss_idx.close()
        fts_idx.close()

    def test_dependency_chain_recorded_without_real_chunk(
        self, tmp_store, shared_db, graph, monkeypatch
    ):
        """Neighbors that don't resolve to indexed chunks (e.g. phantom import
        targets) must not appear in ``dependencies`` — only resolvable ones do."""
        import mnemostack.core.retrieval.indexer as indexer_mod
        import mnemostack.core.retrieval.query as query_mod
        from mnemostack.core.retrieval.indexer import index_directory
        from mnemostack.core.retrieval.query import query_pipeline

        faiss_idx = FaissIndex(store_dir=tmp_store, dimension=4, db=shared_db)
        fts_idx = FTSIndex(store_dir=tmp_store, db=shared_db)

        project = tmp_store / "project"
        project.mkdir()
        # solo_fn imports an external module (phantom node, no chunk) and has no
        # in-repo callees, so its dependency chain should be empty.
        (project / "solo.py").write_text(
            "import os\n\ndef solofn_unique():\n    return os.getpid()\n"
        )

        def fake_embed(texts):
            return np.ones((len(texts), 4), dtype=np.float32)

        monkeypatch.setattr(indexer_mod, "embed_texts", fake_embed)
        index_directory(root=project, faiss_idx=faiss_idx, fts_idx=fts_idx, graph=graph)
        monkeypatch.setattr(
            query_mod,
            "embed_query",
            lambda q, model=None: np.ones(4, dtype=np.float32),
        )

        results = query_pipeline(
            query="solofn_unique",
            faiss_idx=faiss_idx,
            fts_idx=fts_idx,
            graph=graph,
            top_k=5,
        )
        solo = next(
            r for r in results if r.qualified_name.endswith("::solofn_unique")
        )
        assert solo.dependencies == [], "unresolvable neighbors leaked into chain"

        faiss_idx.close()
        fts_idx.close()
