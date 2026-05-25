"""Reciprocal Rank Fusion + recency-weighted ranking.

Merges FAISS semantic results and FTS5 keyword results into a single ranked list
using RRF, then applies recency decay and dependency bonuses.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from mnemostack.config.settings import settings
from mnemostack.core.retrieval.faiss_index import SearchResult
from mnemostack.core.retrieval.fts_index import FTSResult


@dataclass
class RankedResult:
    """Final ranked search result after fusion and re-ranking."""

    chunk_id: int
    file_path: str
    symbol_name: str
    code: str
    line_start: int
    line_end: int
    chunk_type: str
    qualified_name: str
    last_modified: float
    dependencies: list[str]
    final_score: float
    # Component scores for debugging/tuning
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    recency_score: float = 0.0
    dependency_score: float = 0.0


# RRF constant (standard value from literature)
_RRF_K = 60


def reciprocal_rank_fusion(
    faiss_results: list[SearchResult],
    fts_results: list[FTSResult],
    top_k: int = 10,
) -> list[RankedResult]:
    """Merge FAISS and FTS5 ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum(1 / (k + rank_i(d))) across all ranked lists.
    No score normalization needed — only rank positions matter.
    """
    # Build a map of chunk_id -> combined RRF score + metadata
    scores: dict[int, float] = {}
    metadata: dict[int, dict] = {}

    # FAISS results (ranked by similarity — index = rank)
    for rank, r in enumerate(faiss_results, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
        if r.chunk_id not in metadata:
            metadata[r.chunk_id] = {
                "file_path": r.file_path,
                "symbol_name": r.symbol_name,
                "code": r.code,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "chunk_type": r.chunk_type,
                "qualified_name": r.qualified_name,
                "last_modified": r.last_modified,
                "dependencies": r.dependencies,
            }

    # FTS5 results (ranked by BM25 — index = rank)
    for rank, r in enumerate(fts_results, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
        if r.chunk_id not in metadata:
            metadata[r.chunk_id] = {
                "file_path": r.file_path,
                "symbol_name": r.symbol_name,
                "code": r.code,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "chunk_type": r.chunk_type,
                "qualified_name": r.qualified_name,
                "last_modified": r.last_modified,
                "dependencies": [],
            }

    # Sort by RRF score descending, take top_k
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    results: list[RankedResult] = []
    for cid in sorted_ids:
        meta = metadata[cid]
        results.append(RankedResult(
            chunk_id=cid,
            file_path=meta["file_path"],
            symbol_name=meta["symbol_name"],
            code=meta["code"],
            line_start=meta["line_start"],
            line_end=meta["line_end"],
            chunk_type=meta["chunk_type"],
            qualified_name=meta["qualified_name"],
            last_modified=meta["last_modified"],
            dependencies=meta["dependencies"],
            final_score=scores[cid],
            semantic_score=scores[cid],  # Will be overwritten in rerank
        ))

    return results


def compute_recency_score(last_modified: float, now: float | None = None) -> float:
    """Exponential decay based on time since last modification.

    Half-life is configurable (default 60 minutes). Returns value in [0, 1].
    """
    if now is None:
        now = time.time()
    half_life_seconds = settings.retrieval.recency_half_life_minutes * 60
    age_seconds = max(0.0, now - last_modified)
    return math.exp(-math.log(2) * age_seconds / half_life_seconds)


def apply_query_intent_boost(results: list[RankedResult], query: str) -> None:
    """Apply score multipliers based on query format heuristics.

    - PascalCase query -> boost class chunks by 1.5x
    - snake_case query -> boost function chunks by 1.5x
    - dotted.path query -> boost exact qualified name matches by 2.0x
    """
    if not query or not results:
        return

    is_pascal = bool(re.match(r"^[A-Z][a-zA-Z0-9]+$", query))
    is_snake = bool(re.match(r"^[a-z][a-z0-9_]+$", query)) and "_" in query
    is_dotted = "." in query and not query.startswith(".")

    for r in results:
        if is_pascal and r.chunk_type == "class":
            r.final_score *= 1.5
        elif is_snake and r.chunk_type == "function":
            r.final_score *= 1.5
        elif is_dotted and query.lower() in r.qualified_name.lower():
            r.final_score *= 2.0


def rerank(
    results: list[RankedResult],
    query: str = "",
    dependency_ids: set[int] | None = None,
) -> list[RankedResult]:
    """Apply recency weighting, dependency bonus, and query intent boost.

    Mutates and re-sorts `results` in place. Returns the same list for chaining.

    Final score = a * rrf_score + b * recency + c * dependency_bonus
    where a, b, c are from settings.retrieval.ranking_weights.
    """
    weights = settings.retrieval.ranking_weights
    now = time.time()

    for r in results:
        recency = compute_recency_score(r.last_modified, now)
        dep_bonus = 1.0 if (dependency_ids and r.chunk_id in dependency_ids) else 0.0

        r.recency_score = recency
        r.dependency_score = dep_bonus

        # Scale RRF score for meaningful weighting against recency/dependency [0,1]
        # Rank-1 in one list → ~1.0, rank-1 in both lists → ~2.0
        normalized_rrf = r.final_score * (_RRF_K + 1)

        r.semantic_score = normalized_rrf
        r.final_score = (
            weights.semantic * normalized_rrf
            + weights.recency * recency
            + weights.dependency * dep_bonus
        )

    # Apply query intent boost
    apply_query_intent_boost(results, query)

    # Sort by final score descending
    results.sort(key=lambda r: r.final_score, reverse=True)
    return results
