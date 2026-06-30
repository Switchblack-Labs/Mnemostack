from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mnemostack.mcp.server import mcp

log = logging.getLogger(__name__)

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
async def index_project(root_dir: str) -> Confirmation:
    """Index a project directory. Chunks all files, embeds them, and populates
    the search index and call graph. Must be called before query_codebase will
    return results."""
    if not root_dir:
        raise ValueError("root_dir must not be empty")

    from pathlib import Path

    from mnemostack.core.retrieval.indexer import index_directory
    from mnemostack.core.state import state

    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    count = index_directory(
        root=root,
        faiss_idx=state.faiss,
        fts_idx=state.fts,
        graph=state.graph,
    )
    state.start_watching(root)
    return Confirmation(
        success=True,
        message=f"Indexed {count} chunks from {root} (file watcher active)",
    )


@mcp.tool()
async def query_codebase(query: str, top_k: int = 5) -> list[CodeChunk]:
    """Search the codebase using graph-aware retrieval. Returns the top-k relevant code
    chunks (hybrid FTS5+FAISS search, RRF fusion, recency ranking) plus the call-graph
    dependency chain of those results, so callees/callers a top hit relies on are
    included even when they don't match the query directly. Each chunk's `dependencies`
    lists the qualified names it calls or imports."""
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    from mnemostack.core.retrieval.query import query_pipeline
    from mnemostack.core.state import state

    results = query_pipeline(
        query=query,
        faiss_idx=state.faiss,
        fts_idx=state.fts,
        graph=state.graph,
        top_k=top_k,
    )

    return [
        CodeChunk(
            file_path=r.file_path,
            symbol_name=r.symbol_name,
            code=r.code,
            line_start=r.line_start,
            line_end=r.line_end,
            score=r.final_score,
            dependencies=r.dependencies,
        )
        for r in results
    ]


@mcp.tool()
async def get_session_context() -> SessionMemory:
    """Get the current compressed session memory. Returns the latest LLM consolidation
    combined with any local extractions since the last consolidation."""
    from mnemostack.core.state import state

    return SessionMemory(**state.memory.session_view())


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
    from mnemostack.core.compression.llm_consolidator import consolidate
    from mnemostack.core.state import state

    out = consolidate(state.memory)
    return ConsolidationResult(
        success=out.success,
        turns_consolidated=out.turns_consolidated,
        snapshot_id=out.snapshot_id,
        token_count=out.token_count,
    )


@mcp.tool()
async def get_memory_stats() -> MemoryStats:
    """Get current memory system statistics. Returns snapshot count, index size,
    graph size, and last consolidation info."""
    from mnemostack.core.state import state

    mem = state.memory.stats()
    return MemoryStats(
        snapshot_count=mem["snapshot_count"],
        total_memory_tokens=mem["total_memory_tokens"],
        index_chunk_count=state.faiss.total_chunks,
        graph_node_count=state.graph.node_count,
        graph_edge_count=state.graph.edge_count,
        last_consolidation_turn=mem["last_consolidation_turn"],
    )


@mcp.tool()
async def update_constraint(constraint: str) -> Confirmation:
    """Manually add a constraint to session memory. Bypasses automatic extraction
    so you can explicitly tell the system to remember something."""
    if not constraint:
        raise ValueError("constraint must not be empty")
    from mnemostack.core.state import state

    state.memory.add_pinned("constraint", constraint)
    return Confirmation(
        success=True,
        message="Constraint pinned to session memory.",
    )


@mcp.tool()
async def record_turn(text: str) -> Confirmation:
    """Record a raw conversation turn into session memory. Once enough turns
    accumulate (compression.consolidation_interval), an LLM consolidation fires
    automatically to compress them into the session snapshot."""
    if not text:
        raise ValueError("text must not be empty")
    from mnemostack.config.settings import settings
    from mnemostack.core.compression.llm_consolidator import ConsolidationError, consolidate
    from mnemostack.core.state import state

    count = state.memory.add_turn(text)
    # Fire every `interval`-th pending turn (not just `>= interval`): if a
    # consolidation fails, pending stays high, and `>=` would re-fire the blocking
    # LLM call on every single turn. `% interval` backs off to one retry per interval.
    interval = settings.compression.consolidation_interval
    if state.memory.pending_count() % interval != 0:
        return Confirmation(success=True, message=f"Recorded turn {count}.")

    # ponytail: synchronous LLM call blocks the event loop; offload to a thread if
    # it stalls clients. The turn is already persisted, so a failed consolidation
    # just leaves pending intact to retry on the next trigger — never lose data.
    try:
        consolidate(state.memory)
        return Confirmation(success=True, message=f"Recorded turn {count}; consolidated.")
    except ConsolidationError as exc:
        log.warning("Auto-consolidation failed at turn %d: %s", count, exc)
        return Confirmation(
            success=True, message=f"Recorded turn {count}; consolidation deferred ({exc})."
        )
