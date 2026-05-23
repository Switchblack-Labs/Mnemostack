from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mnemostack.mcp.server import mcp

# extra="forbid" on all wire models so a server-side typo (e.g. emitting a misspelled
# field) raises in tests instead of silently dropping data on the way to the client.
_WIRE_MODEL_CONFIG = ConfigDict(extra="forbid")


# --- Response models ---


class CodeChunk(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    file_path: str = Field(min_length=1)
    symbol_name: str
    code: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    score: float = Field(ge=0.0)
    dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_line_range(self) -> CodeChunk:
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self


class SessionMemory(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    decisions: list[dict] = Field(default_factory=list)
    constraints: list[dict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    architecture_state: dict = Field(default_factory=dict)
    resolved: list[str] = Field(default_factory=list)
    local_extractions_pending: int = Field(default=0, ge=0)
    last_consolidation_turn: int | None = None


class CombinedContext(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    session: SessionMemory
    chunks: list[CodeChunk]


class ConsolidationResult(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    success: bool
    turns_consolidated: int = Field(ge=0)
    snapshot_id: str
    token_count: int = Field(ge=0)


class MemoryStats(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    snapshot_count: int = Field(ge=0)
    total_memory_tokens: int = Field(ge=0)
    index_chunk_count: int = Field(ge=0)
    graph_node_count: int = Field(ge=0)
    graph_edge_count: int = Field(ge=0)
    last_consolidation_turn: int | None = None


class Confirmation(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    success: bool
    message: str


# --- MCP Tools ---


@mcp.tool()
async def query_codebase(query: str, top_k: int = 5) -> list[CodeChunk]:
    """Search the codebase semantically. Returns top-k relevant code chunks
    using hybrid FTS5+FAISS retrieval with dependency expansion and recency ranking."""
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
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
    if not query:
        raise ValueError("query must not be empty")
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
    if not constraint:
        raise ValueError("constraint must not be empty")
    # TODO: wire to core/compression/memory_store
    return Confirmation(
        success=False,
        message="Not implemented yet",
    )
