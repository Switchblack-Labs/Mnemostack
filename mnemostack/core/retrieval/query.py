"""Query pipeline.

Orchestrates: embed query -> FAISS search -> FTS5 search -> RRF fusion ->
call graph expansion -> rerank -> return results.
"""

from __future__ import annotations

from mnemostack.config.settings import settings
from mnemostack.core.retrieval.call_graph import CallGraph
from mnemostack.core.retrieval.embed import embed_query
from mnemostack.core.retrieval.faiss_index import FaissIndex
from mnemostack.core.retrieval.fts_index import FTSIndex
from mnemostack.core.retrieval.ranker import RankedResult, reciprocal_rank_fusion, rerank


def query_pipeline(
    query: str,
    faiss_idx: FaissIndex,
    fts_idx: FTSIndex,
    graph: CallGraph,
    top_k: int = 5,
) -> list[RankedResult]:
    """Run the full retrieval pipeline for a query.

    1. Short-circuit if nothing is indexed
    2. Embed query
    3. FAISS semantic search (fetch 3x top_k candidates)
    4. FTS5 keyword search (fetch 3x top_k candidates)
    5. RRF fusion
    6. Call graph expansion (2-hop BFS on top results)
    7. Rerank with recency + dependency bonus
    8. Return top_k
    """
    # Short-circuit if nothing is indexed
    if faiss_idx.total_chunks == 0:
        return []

    # Over-fetch candidates for better fusion
    fetch_k = top_k * 3

    # Parallel retrieval from both indexes
    query_vec = embed_query(query)
    faiss_results = faiss_idx.search(query_vec, top_k=fetch_k)
    fts_results = fts_idx.search(query, top_k=fetch_k)

    # Fuse ranked lists
    fused = reciprocal_rank_fusion(faiss_results, fts_results, top_k=fetch_k)

    if not fused:
        return []

    # Graph expansion: collect dependency IDs for top candidates
    hops = settings.retrieval.dependency_hops
    dependency_ids: set[int] = set()

    # Collect all neighbor qualified names first
    all_neighbor_qnames: list[str] = []
    for result in fused[:top_k]:
        neighbor_qnames = graph.get_neighbors(
            result.qualified_name, hops=hops, direction="both"
        )
        all_neighbor_qnames.extend(neighbor_qnames)

    # Batch-resolve qualified names to chunk IDs via public API
    if all_neighbor_qnames:
        qname_to_id = faiss_idx.get_chunk_ids_by_qnames(all_neighbor_qnames)
        dependency_ids = set(qname_to_id.values())

    # Rerank with recency + dependency bonus + query intent boost
    reranked = rerank(fused, query=query, dependency_ids=dependency_ids)

    return reranked[:top_k]
