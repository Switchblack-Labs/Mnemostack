"""End-to-end tests over a realistic multi-repo workspace.

Everything here goes through the real indexer, graph and query pipeline against
files on disk, with only the embedding call replaced by an offline deterministic
one. These are the claims the product makes: several repos in one graph, links
that cross between them, an honest boundary where the index stops, and both
languages behaving the same way.
"""

from __future__ import annotations

from mnemostack.core.retrieval.call_graph import EdgeType
from mnemostack.core.retrieval.indexer import reindex_file
from mnemostack.core.retrieval.query import query_pipeline

from .conftest import index_all


class TestCrossRepoIndexing:
    def test_import_into_another_repo_becomes_an_edge(
        self, workspace, indexes, deterministic_embeddings
    ):
        _, _, graph = indexes
        index_all([workspace["shared"], workspace["service"]], indexes)

        imports = graph.get_neighbors(
            str(workspace["service_main"]), hops=1, direction="outgoing",
            edge_types=(EdgeType.IMPORTS_FROM,),
        )
        assert str(workspace["shared_util"]) in imports
        calls = graph.get_neighbors(
            f"{workspace['service_main']}::handle_request", hops=1, direction="outgoing",
            edge_types=(EdgeType.CALLS,),
        )
        assert f"{workspace['shared_util']}::normalise" in calls

    def test_repo_tags_mark_the_boundary_between_repos(
        self, workspace, indexes, deterministic_embeddings
    ):
        _, _, graph = indexes
        index_all([workspace["shared"], workspace["service"]], indexes)

        assert graph.node_repo(str(workspace["service_main"])) == "service"
        assert graph.node_repo(str(workspace["shared_util"])) == "shared-lib"
        assert graph.node_repo(str(workspace["installed_requests"])) == "pkg:requests"

    def test_either_indexing_order_gives_the_same_graph(
        self, workspace, indexes, deterministic_embeddings
    ):
        """Dependency-first and dependent-first must converge."""
        _, _, graph = indexes
        index_all([workspace["service"], workspace["shared"]], indexes)

        imports = graph.get_neighbors(
            str(workspace["service_main"]), hops=1, direction="outgoing",
            edge_types=(EdgeType.IMPORTS_FROM,),
        )
        assert str(workspace["shared_util"]) in imports
        assert graph.external_imports(str(workspace["service_main"])) == [
            str(workspace["installed_requests"])
        ], "only the genuinely unindexed dependency stays a boundary"


class TestBoundaryReporting:
    def test_unindexed_dependency_is_named_not_dropped(
        self, workspace, indexes, deterministic_embeddings
    ):
        _, _, graph = indexes
        index_all([workspace["service"]], indexes)

        # shared_lib is not indexed and not installed: nothing to report.
        # requests is installed in the venv: report where its source lives.
        assert graph.external_imports(str(workspace["service_main"])) == [
            str(workspace["installed_requests"])
        ]

    def test_boundary_reaches_the_query_result(
        self, workspace, indexes, deterministic_embeddings
    ):
        faiss_idx, fts, graph = indexes
        index_all([workspace["service"]], indexes)

        results = query_pipeline(query="handle request payload", faiss_idx=faiss_idx,
                                 fts_idx=fts, graph=graph, top_k=5)
        assert any(
            str(workspace["installed_requests"]) in r.external_dependencies for r in results
        )

    def test_javascript_boundary_resolves_through_node_modules(
        self, workspace, indexes, deterministic_embeddings
    ):
        _, _, graph = indexes
        index_all([workspace["app"]], indexes)

        assert graph.external_imports(str(workspace["page"])) == [
            str(workspace["installed_pad"])
        ]


class TestBothLanguagesInOneGraph:
    def test_python_and_javascript_share_the_index(
        self, workspace, indexes, deterministic_embeddings
    ):
        faiss_idx, _, graph = indexes
        index_all([workspace["shared"], workspace["service"], workspace["app"]], indexes)

        assert graph.has_node(f"{workspace['service_main']}::handle_request")
        assert graph.has_node(f"{workspace['page']}::Page")
        # JSX use of an imported component is a dependency edge like a call.
        calls = graph.get_neighbors(
            f"{workspace['page']}::Page", hops=1, direction="outgoing",
            edge_types=(EdgeType.CALLS,),
        )
        assert f"{workspace['widget']}::Widget" in calls
        assert faiss_idx.total_chunks > 0


class TestIncrementalUpdate:
    def test_editing_a_file_updates_its_chunks_and_graph(
        self, workspace, indexes, deterministic_embeddings
    ):
        faiss_idx, fts, graph = indexes
        index_all([workspace["shared"], workspace["service"]], indexes)
        main = workspace["service_main"]

        main.write_text(
            "from shared_lib.util import normalise\n\n"
            "def handle_request(payload):\n"
            "    return normalise(payload)\n\n"
            "def health_check():\n"
            "    return 'ok'\n"
        )
        reindex_file(main, faiss_idx, fts, graph)

        assert graph.has_node(f"{main}::health_check"), "new symbol missing from the graph"
        assert graph.external_imports(str(main)) == [], "dropped import still reported"
        assert str(workspace["shared_util"]) in graph.get_neighbors(
            str(main), hops=1, direction="outgoing", edge_types=(EdgeType.IMPORTS_FROM,)
        ), "surviving import lost on reindex"

    def test_deleting_a_file_removes_it_everywhere(
        self, workspace, indexes, deterministic_embeddings
    ):
        faiss_idx, fts, graph = indexes
        index_all([workspace["shared"], workspace["service"]], indexes)
        util = workspace["shared_util"]

        util.unlink()
        reindex_file(util, faiss_idx, fts, graph)

        assert not graph.has_node(f"{util}::normalise")
        results = query_pipeline(query="normalise text", faiss_idx=faiss_idx, fts_idx=fts,
                                 graph=graph, top_k=5)
        assert all(str(util) != r.file_path for r in results), "deleted file still retrievable"


class TestRetrievalQuality:
    """The pipeline must actually find the right code, not merely return rows."""

    def test_question_finds_the_function_that_answers_it(
        self, workspace, indexes, deterministic_embeddings
    ):
        faiss_idx, fts, graph = indexes
        index_all([workspace["shared"], workspace["service"], workspace["app"]], indexes)

        results = query_pipeline(query="how do we normalise text", faiss_idx=faiss_idx,
                                 fts_idx=fts, graph=graph, top_k=3)
        assert results, "a question about indexed code returned nothing"
        assert results[0].qualified_name == f"{workspace['shared_util']}::normalise"

    def test_dependency_surfaces_without_matching_the_query(
        self, workspace, indexes, deterministic_embeddings
    ):
        """A callee reached only through the graph still comes back."""
        faiss_idx, fts, graph = indexes
        index_all([workspace["shared"], workspace["service"]], indexes)

        results = query_pipeline(query="handle_request", faiss_idx=faiss_idx, fts_idx=fts,
                                 graph=graph, top_k=3)
        names = {r.qualified_name for r in results}
        assert f"{workspace['service_main']}::handle_request" in names
        assert f"{workspace['shared_util']}::normalise" in names, (
            "call-graph dependency did not surface"
        )
