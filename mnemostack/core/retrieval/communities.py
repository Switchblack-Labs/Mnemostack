"""Leiden community detection on the lightweight call graph.

Clusters related code into communities (e.g., 'auth-core', 'db-layer') for
community-tagged session snapshots. Runs locally on the graph, no LLM calls.
"""

from __future__ import annotations

import igraph as ig
import leidenalg

from mnemostack.core.retrieval.call_graph import CallGraph


def detect_communities(graph: CallGraph) -> dict[str, int]:
    """Run Leiden community detection on the call graph.

    Returns:
        Mapping of qualified_name -> community_id for every node.
    """
    # Fetch all nodes and edges
    nodes = graph.db.execute(
        "SELECT id, qualified_name FROM nodes ORDER BY id"
    ).fetchall()
    edges = graph.db.execute(
        "SELECT source_id, target_id FROM edges"
    ).fetchall()

    if not nodes:
        return {}

    # Build igraph Graph
    id_to_idx: dict[int, int] = {}
    idx_to_qname: dict[int, str] = {}
    for idx, (node_id, qname) in enumerate(nodes):
        id_to_idx[node_id] = idx
        idx_to_qname[idx] = qname

    g = ig.Graph(n=len(nodes), directed=True)

    # Add edges (skip any with missing node IDs — shouldn't happen but defensive)
    edge_list = []
    for source_id, target_id in edges:
        if source_id in id_to_idx and target_id in id_to_idx:
            edge_list.append((id_to_idx[source_id], id_to_idx[target_id]))

    if edge_list:
        g.add_edges(edge_list)

    # Leiden expects undirected for modularity-based detection
    g_undirected = g.as_undirected(mode="collapse")

    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        g_undirected,
        leidenalg.ModularityVertexPartition,
    )

    # Map back to qualified names
    result: dict[str, int] = {}
    for idx, community_id in enumerate(partition.membership):
        result[idx_to_qname[idx]] = community_id

    return result


def get_community_for_chunks(
    graph: CallGraph,
    qualified_names: list[str],
) -> dict[str, int]:
    """Get community assignments for specific chunks.

    Runs full detection and returns only the requested nodes.
    Results are cached per graph state in production (not implemented in MVP).
    """
    all_communities = detect_communities(graph)
    return {qn: all_communities[qn] for qn in qualified_names if qn in all_communities}


def get_community_members(graph: CallGraph, community_id: int) -> list[str]:
    """Get all qualified names belonging to a community."""
    all_communities = detect_communities(graph)
    return [qn for qn, cid in all_communities.items() if cid == community_id]
