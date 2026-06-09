Local MCP daemon with persistent compressed session memory and fast semantic code retrieval. Shrinks context windows and eliminates redundant token usage by giving LLMs long-term memory over your codebase.
The problem: Every time you start a new session with an LLM, it forgets everything. You re-paste files, re-explain your project, burn through tokens feeding it context it's already seen. Developers and companies are wasting significant budget on redundant context.
How it works:

AST-aware chunking via tree-sitter — retrieves functions and classes, not arbitrary text blocks
FAISS HNSW index for fast semantic search over your codebase
Compressed session summaries that persist across sessions
Runs locally as an MCP server — plugs into Claude, Cursor, or any MCP-compatible tool

Status: Early development. Architecture designed, core scaffolding in place.
