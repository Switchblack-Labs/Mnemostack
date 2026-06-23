Local MCP daemon that builds a live dependency graph of your codebase and gives AI coding assistants graph-aware code retrieval. Ask about any function and get back not just that function, but everything it calls and everything that calls it — across files, automatically.

The problem: AI coding assistants have no idea how your code connects. They retrieve isolated files, miss cross-file dependencies, and have zero understanding of your call graph. When you ask about authentication, they find `auth.py` but miss the `db.py` query it calls and the `crypto.py` hash function two hops away. You waste tokens manually pasting related files.

How it works:

Tree-sitter AST parsing builds a lightweight call graph (functions, classes, imports — who calls what)
2-hop BFS expansion on the graph retrieves entire dependency chains, not isolated snippets
Hybrid search: FAISS HNSW (semantic similarity) + FTS5/BM25 (exact identifier matching) fused with Reciprocal Rank Fusion
Recency-weighted ranking ensures recently edited code outranks stale matches
Leiden community detection clusters related code automatically
Persistent compressed session memory survives across sessions
Runs locally as an MCP server — plugs into Claude Code, Cursor, or any MCP-compatible tool

Status: Retrieval pipeline built and tested (AST chunker, FAISS+FTS5 hybrid, call graph, community detection, file watcher). Compression pipeline in progress.
