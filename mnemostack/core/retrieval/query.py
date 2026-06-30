"""Query pipeline.

Orchestrates: embed query -> FAISS search -> FTS5 search -> RRF fusion ->
call graph expansion -> rerank -> return results.
"""

from __future__ import annotations

from mnemostack.config.settings import settings
from mnemostack.core.retrieval.call_graph import CallGraph, EdgeType
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
    6. Call graph expansion: 2-hop BFS on the top candidates, resolving neighbors
       to real chunks. Resolved dependencies are (a) merged into the candidate
       set so a callee can rank on its own merits, and (b) recorded on each seed's
       ``dependencies`` field so the caller sees the chain.
    7. Rerank with recency + dependency bonus
    8. Return the top_k primary results plus the dependency chain of those
       primaries (so a pure call-graph dependency surfaces even when it has no
       semantic/keyword overlap with the query).
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

    # --- Graph expansion ---
    # For each top candidate, BFS its call-graph neighbors and keep only those
    # that resolve to indexed chunks. Record the resolvable chain per seed.
    hops = settings.retrieval.dependency_hops
    neighbors_by_qname: dict[str, list[str]] = {}
    all_neighbor_qnames: list[str] = []
    for result in fused[:top_k]:
        # Follow only dependency edges (CALLS / IMPORTS_FROM). Traversing CONTAINS
        # would reach every sibling symbol through the shared file node, drowning
        # the real dependency chain in same-file noise.
        neighbor_qnames = graph.get_neighbors(
            result.qualified_name,
            hops=hops,
            direction="both",
            edge_types=(EdgeType.CALLS, EdgeType.IMPORTS_FROM),
        )
        neighbors_by_qname[result.qualified_name] = neighbor_qnames
        all_neighbor_qnames.extend(neighbor_qnames)

    qname_to_id: dict[str, int] = {}
    if all_neighbor_qnames:
        qname_to_id = faiss_idx.get_chunk_ids_by_qnames(all_neighbor_qnames)
    dependency_ids: set[int] = set(qname_to_id.values())

    # Surface the resolvable dependency chain on each seed result.
    for result in fused[:top_k]:
        result.dependencies = sorted(
            qn for qn in neighbors_by_qname.get(result.qualified_name, []) if qn in qname_to_id
        )

    # Merge dependency chunks that hybrid search didn't already surface into the
    # candidate set, so the graph expands retrieval (not just re-ranks it). A
    # newly-injected dependency carries no hybrid score (final_score=0.0); the
    # dependency bonus + recency in rerank position it below direct hits.
    candidates_by_id: dict[int, RankedResult] = {r.chunk_id: r for r in fused}
    missing_ids = [cid for cid in dependency_ids if cid not in candidates_by_id]
    fetched_chunks = faiss_idx.get_chunks_by_ids(missing_ids)
    for cid in missing_ids:
        chunk = fetched_chunks.get(cid)
        if chunk is None:
            continue
        injected = RankedResult(
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            symbol_name=chunk.symbol_name,
            code=chunk.code,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            chunk_type=chunk.chunk_type,
            qualified_name=chunk.qualified_name,
            last_modified=chunk.last_modified,
            dependencies=[],
            final_score=0.0,
        )
        candidates_by_id[cid] = injected
        fused.append(injected)

    # Rerank with recency + dependency bonus + query intent boost
    reranked = rerank(fused, query=query, dependency_ids=dependency_ids)

    # Primary results: the top_k by score.
    primary = reranked[:top_k]
    selected_ids = {r.chunk_id for r in primary}
    output = list(primary)

    # A dependency injected above can itself be promoted into the primaries; its
    # `dependencies` was never computed (only seeds were). Fill in the chain for
    # any primary we didn't already seed so its callees can still be surfaced.
    unseeded = [r for r in primary if r.qualified_name not in neighbors_by_qname]
    if unseeded:
        new_qnames: list[str] = []
        for result in unseeded:
            nb = graph.get_neighbors(
                result.qualified_name,
                hops=hops,
                direction="both",
                edge_types=(EdgeType.CALLS, EdgeType.IMPORTS_FROM),
            )
            neighbors_by_qname[result.qualified_name] = nb
            new_qnames.extend(qn for qn in nb if qn not in qname_to_id)
        if new_qnames:
            qname_to_id.update(faiss_idx.get_chunk_ids_by_qnames(new_qnames))
        for result in unseeded:
            result.dependencies = sorted(
                qn for qn in neighbors_by_qname[result.qualified_name] if qn in qname_to_id
            )
        # Make the newly-resolved dependency chunks available to the expansion
        # loop below so a promoted primary's chain can surface like a seed's.
        new_dep_ids = [
            qname_to_id[qn]
            for result in unseeded
            for qn in result.dependencies
            if qname_to_id[qn] not in candidates_by_id
        ]
        for cid, chunk in faiss_idx.get_chunks_by_ids(new_dep_ids).items():
            candidates_by_id[cid] = RankedResult(
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                symbol_name=chunk.symbol_name,
                code=chunk.code,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                chunk_type=chunk.chunk_type,
                qualified_name=chunk.qualified_name,
                last_modified=chunk.last_modified,
                dependencies=[],
                final_score=0.0,
            )

    # Guarantee the dependency chain of the primary results is present, even if a
    # pure dependency couldn't out-score direct hits for a top_k slot. Append in
    # primary-rank order, capped at top_k extra chunks to keep the response bounded.
    expansion_budget = top_k
    for result in primary:
        if expansion_budget <= 0:
            break
        for dep_qname in result.dependencies:
            dep_id = qname_to_id.get(dep_qname)
            if dep_id is None or dep_id in selected_ids:
                continue
            dep_chunk = candidates_by_id.get(dep_id)
            if dep_chunk is None:
                continue
            output.append(dep_chunk)
            selected_ids.add(dep_id)
            expansion_budget -= 1
            if expansion_budget <= 0:
                break

    return output
