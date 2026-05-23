from __future__ import annotations

from pydantic import BaseModel, Field

from mnemostack.mcp.server import mcp

# --- Response models ---


class CodeChunk(BaseModel):
    file_path: str
    symbol_name: str
    code: str
    line_start: int
    line_end: int
    score: float
    dependencies: list[str] = Field(default_factory=list)


class SessionMemory(BaseModel):
    decisions: list[dict] = Field(default_factory=list)
    constraints: list[dict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    architecture_state: dict = Field(default_factory=dict)
    resolved: list[str] = Field(default_factory=list)
    local_extractions_pending: int = 0
    last_consolidation_turn: int | None = None


class CombinedContext(BaseModel):
    session: SessionMemory
    chunks: list[CodeChunk]


class ConsolidationResult(BaseModel):
    success: bool
    turns_consolidated: int
    snapshot_id: str
    token_count: int


class MemoryStats(BaseModel):
    snapshot_count: int
    total_memory_tokens: int
    index_chunk_count: int
    graph_node_count: int
    graph_edge_count: int
    last_consolidation_turn: int | None


class Confirmation(BaseModel):
    success: bool
    message: str


# --- MCP Tools ---


@mcp.tool()
async def query_codebase(query: str, top_k: int = 5) -> list[CodeChunk]:
    """Search the codebase semantically. Returns top-k relevant code chunks
    using hybrid FTS5+FAISS retrieval with dependency expansion and recency ranking."""
    # TODO: wire to core/retrieval pipeline
    return []


@mcp.tool()
async def get_session_context() -> SessionMemory:
    """Get the current compressed session memory. Returns the latest LLM consolidation
    combined with any local extractions since the last consolidation."""
    # TODO: wire to core/compression/memory_store
    return SessionMemory()


@mcp.tool()
async def get_full_context(query: str) -> CombinedContext:
    """Get session memory and relevant code chunks in a single call.
    Convenience tool that combines get_session_context + query_codebase."""
    session = await get_session_context()
    chunks = await query_codebase(query)
    return CombinedContext(session=session, chunks=chunks)


@mcp.tool()
async def force_consolidate() -> ConsolidationResult:
    """Manually trigger an LLM consolidation cycle. Useful when major
    architectural decisions have been made and you want them persisted immediately."""
    # TODO: wire to core/compression/llm_consolidator
    return ConsolidationResult(
        success=False,
        turns_consolidated=0,
        snapshot_id="",
        token_count=0,
    )


@mcp.tool()
async def get_memory_stats() -> MemoryStats:
    """Get current memory system statistics. Returns snapshot count, index size,
    graph size, and last consolidation info."""
    # TODO: wire to memory_store + faiss_index + call_graph
    return MemoryStats(
        snapshot_count=0,
        total_memory_tokens=0,
        index_chunk_count=0,
        graph_node_count=0,
        graph_edge_count=0,
        last_consolidation_turn=None,
    )


@mcp.tool()
async def update_constraint(constraint: str) -> Confirmation:
    """Manually add a constraint to session memory. Bypasses automatic extraction
    so you can explicitly tell the system to remember something."""
    # TODO: wire to core/compression/memory_store
    return Confirmation(
        success=False,
        message="Not implemented yet",
    )
