"""Properties the ranking must hold whatever the query or the corpus.

Ranking is tuned against numbers that only hold for one corpus. These are the
statements that must hold regardless: the order is stable, the scores are sane,
and relevance decides the top before any bonus does.
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import time

import pytest

from mnemostack.core.retrieval.query import query_pipeline
from mnemostack.core.retrieval.ranker import (
    RankedResult,
    reciprocal_rank_fusion,
    rerank,
)

from .conftest import index_all

QUERIES = [
    "how do we normalise text",
    "handle_request",
    "render the widget component",
    "left pad a string",
]


def _ranked(chunk_id, qname, rrf_ranks, age_seconds, chunk_type="function"):
    from mnemostack.core.retrieval.ranker import _RRF_K

    return RankedResult(
        chunk_id=chunk_id,
        file_path=qname.split("::")[0],
        symbol_name=qname.split("::")[1],
        code="pass",
        line_start=1,
        line_end=1,
        chunk_type=chunk_type,
        qualified_name=qname,
        last_modified=time.time() - age_seconds,
        dependencies=[],
        final_score=sum(1.0 / (_RRF_K + r) for r in rrf_ranks),
    )


class TestPipelineInvariants:
    @pytest.fixture
    def populated(self, workspace, indexes, deterministic_embeddings):
        index_all([workspace["shared"], workspace["service"], workspace["app"]], indexes)
        return indexes

    def test_results_are_deterministic(self, populated):
        faiss_idx, fts, graph = populated
        runs = [
            [r.qualified_name for r in query_pipeline(query=QUERIES[0], faiss_idx=faiss_idx,
                                                      fts_idx=fts, graph=graph, top_k=5)]
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2]

    def test_primary_results_are_sorted_and_scores_finite(self, populated):
        faiss_idx, fts, graph = populated
        for query in QUERIES:
            out = query_pipeline(query=query, faiss_idx=faiss_idx, fts_idx=fts,
                                 graph=graph, top_k=3)
            primaries = [r.final_score for r in out[:3]]
            assert primaries == sorted(primaries, reverse=True)
            assert all(math.isfinite(r.final_score) and r.final_score >= 0 for r in out)

    def test_no_duplicate_chunks(self, populated):
        faiss_idx, fts, graph = populated
        for query in QUERIES:
            out = query_pipeline(query=query, faiss_idx=faiss_idx, fts_idx=fts,
                                 graph=graph, top_k=5)
            ids = [r.chunk_id for r in out]
            assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("top_k", [1, 2, 5, 50, 500])
    def test_top_k_bounds(self, populated, top_k):
        faiss_idx, fts, graph = populated
        out = query_pipeline(query=QUERIES[0], faiss_idx=faiss_idx, fts_idx=fts,
                             graph=graph, top_k=top_k)
        # top_k primaries, plus at most top_k appended dependency chunks.
        assert len(out) <= 2 * top_k

    @pytest.mark.parametrize("query", ["", "   ", "the a of", "\n"])
    def test_empty_and_stopword_queries_are_safe(self, populated, query):
        faiss_idx, fts, graph = populated
        assert isinstance(
            query_pipeline(query=query, faiss_idx=faiss_idx, fts_idx=fts, graph=graph, top_k=5),
            list,
        )

    def test_concurrent_queries(self, populated):
        faiss_idx, fts, graph = populated

        def run(i):
            return query_pipeline(query=QUERIES[i % len(QUERIES)], faiss_idx=faiss_idx,
                                  fts_idx=fts, graph=graph, top_k=5)

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(run, range(40)))
        assert all(isinstance(o, list) for o in outputs)


class TestRankingProperties:
    def test_rerank_and_fusion_accept_empty_input(self):
        assert rerank([], query="anything") == []
        assert reciprocal_rank_fusion([], [], top_k=5) == []

    @pytest.mark.parametrize("age_days", [0, 1, 7, 30, 365])
    def test_a_better_match_wins_at_every_age(self, age_days):
        """Recency may break ties; it may not overturn a clearly better match.

        Both are ranked by one search only, which is the case that actually
        broke: two hits in one list sit close enough together that an RRF k too
        large for the list length lets the recency bonus decide between them.
        """
        best = _ranked(1, "right.py::answer", (1,), age_days * 86400)
        fresh = _ranked(2, "wrong.py::helper", (9,), 0)
        assert rerank([fresh, best], query="q")[0].qualified_name == "right.py::answer"

    def test_import_chunk_loses_a_tie_to_real_code(self):
        imports = _ranked(1, "a.py::<imports>", (1, 1), 0, chunk_type="import")
        code = _ranked(2, "a.py::render", (1, 1), 0)
        assert rerank([imports, code], query="q")[0].qualified_name == "a.py::render"

    def test_dependency_bonus_does_not_overturn_relevance(self):
        best = _ranked(1, "right.py::answer", (1,), 0)
        dep = _ranked(2, "dep.py::callee", (12,), 0)
        ranked = rerank([dep, best], query="q", dependency_ids={2})
        assert ranked[0].qualified_name == "right.py::answer"

    def test_scores_survive_a_zero_timestamp(self):
        """A file with no usable mtime must not produce inf or nan."""
        stale = _ranked(1, "a.py::f", (1,), time.time())  # last_modified = 0
        out = rerank([stale], query="q")
        assert math.isfinite(out[0].final_score)
