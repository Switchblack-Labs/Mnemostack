from __future__ import annotations

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import ValidationError

from mnemostack.mcp import tools as mcp_tools
from mnemostack.mcp.server import mcp
from mnemostack.mcp.tools import (
    CodeChunk,
    CombinedContext,
    Confirmation,
    ConsolidationResult,
    MemoryStats,
    SessionMemory,
)

# --- CodeChunk invariants (M4) ---


def test_code_chunk_valid():
    c = CodeChunk(
        file_path="a.py",
        symbol_name="foo",
        code="def foo(): ...",
        line_start=1,
        line_end=5,
        score=0.42,
    )
    assert c.dependencies == []


def test_code_chunk_line_end_before_start_rejected():
    with pytest.raises(ValidationError, match="line_end"):
        CodeChunk(
            file_path="a.py",
            symbol_name="foo",
            code="x",
            line_start=10,
            line_end=1,
            score=0.5,
        )


def test_code_chunk_line_start_zero_rejected():
    with pytest.raises(ValidationError):
        CodeChunk(
            file_path="a.py",
            symbol_name="foo",
            code="x",
            line_start=0,
            line_end=1,
            score=0.5,
        )


def test_code_chunk_negative_score_rejected():
    with pytest.raises(ValidationError):
        CodeChunk(
            file_path="a.py",
            symbol_name="foo",
            code="x",
            line_start=1,
            line_end=1,
            score=-0.1,
        )


def test_code_chunk_empty_file_path_rejected():
    with pytest.raises(ValidationError):
        CodeChunk(
            file_path="",
            symbol_name="foo",
            code="x",
            line_start=1,
            line_end=1,
            score=0.5,
        )


def test_code_chunk_extra_field_rejected():
    with pytest.raises(ValidationError):
        CodeChunk(
            file_path="a.py",
            symbol_name="foo",
            code="x",
            line_start=1,
            line_end=1,
            score=0.5,
            unknown_extra_field="oops",
        )


def test_code_chunk_single_line_allowed():
    """A chunk that's exactly one line: line_start == line_end."""
    c = CodeChunk(
        file_path="a.py",
        symbol_name="X",
        code="X = 1",
        line_start=7,
        line_end=7,
        score=0.0,
    )
    assert c.line_start == c.line_end


# --- Response model invariants ---


def test_session_memory_default_isolation():
    """Two SessionMemory instances must have independent list defaults."""
    m1 = SessionMemory()
    m2 = SessionMemory()
    m1.decisions.append({"text": "x"})
    assert m2.decisions == []


def test_session_memory_extra_field_rejected():
    with pytest.raises(ValidationError):
        SessionMemory(unknown="x")


def test_session_memory_negative_pending_rejected():
    with pytest.raises(ValidationError):
        SessionMemory(local_extractions_pending=-1)


def test_consolidation_result_negative_token_count_rejected():
    with pytest.raises(ValidationError):
        ConsolidationResult(
            success=True,
            turns_consolidated=10,
            snapshot_id="snap_001",
            token_count=-1,
        )


def test_memory_stats_negative_count_rejected():
    with pytest.raises(ValidationError):
        MemoryStats(
            snapshot_count=-1,
            total_memory_tokens=0,
            index_chunk_count=0,
            graph_node_count=0,
            graph_edge_count=0,
            last_consolidation_turn=None,
        )


def test_combined_context_composes():
    c = CombinedContext(session=SessionMemory(), chunks=[])
    assert c.session.decisions == []
    assert c.chunks == []


def test_confirmation_basic():
    c = Confirmation(success=True, message="ok")
    assert c.success and c.message == "ok"


# --- Tool-function-level behavior ---


async def test_query_codebase_returns_empty_stub():
    result = await mcp_tools.query_codebase("anything")
    assert result == []


async def test_query_codebase_rejects_empty_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        await mcp_tools.query_codebase("")


async def test_query_codebase_rejects_nonpositive_top_k():
    with pytest.raises(ValueError, match="top_k must be positive"):
        await mcp_tools.query_codebase("x", top_k=0)
    with pytest.raises(ValueError, match="top_k must be positive"):
        await mcp_tools.query_codebase("x", top_k=-3)


@pytest.fixture(autouse=True)
def _isolate_memory(tmp_path, monkeypatch):
    """Point the shared state singleton at a fresh memory store per test so
    constraint writes don't pollute ./store or bleed across tests."""
    from mnemostack.core.compression.memory_store import MemoryStore
    from mnemostack.core.state import state

    monkeypatch.setattr(state, "_memory", MemoryStore(store_dir=tmp_path))


async def test_get_session_context_empty_on_fresh_store():
    s = await mcp_tools.get_session_context()
    assert isinstance(s, SessionMemory)
    assert s.decisions == [] and s.constraints == []
    assert s.local_extractions_pending == 0


async def test_update_constraint_shows_up_in_session_context():
    await mcp_tools.update_constraint("we use postgres")
    s = await mcp_tools.get_session_context()
    assert {"kind": "constraint", "text": "we use postgres"} in s.constraints


async def test_get_full_context_composes():
    c = await mcp_tools.get_full_context("hello")
    assert isinstance(c, CombinedContext)
    assert c.chunks == []


async def test_get_full_context_rejects_empty_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        await mcp_tools.get_full_context("")


async def test_force_consolidate_noop_when_nothing_pending():
    r = await mcp_tools.force_consolidate()
    assert r.success is True
    assert r.turns_consolidated == 0


async def test_force_consolidate_consolidates_pending(monkeypatch):
    import types

    from mnemostack.core.compression import llm_consolidator
    from mnemostack.core.state import state

    canned = '{"decisions": [{"d": "use postgres"}], "open_questions": []}'
    monkeypatch.setattr(
        llm_consolidator,
        "completion",
        lambda **kw: types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=canned))]
        ),
    )
    state.memory.add_turn("we picked postgres")
    state.memory.add_turn("dropped redis")

    r = await mcp_tools.force_consolidate()
    assert r.success is True
    assert r.turns_consolidated == 2
    assert state.memory.pending_count() == 0
    s = await mcp_tools.get_session_context()
    assert s.decisions == [{"d": "use postgres"}]


async def test_get_memory_stats_stub():
    r = await mcp_tools.get_memory_stats()
    assert r.snapshot_count == 0


async def test_update_constraint_confirms():
    r = await mcp_tools.update_constraint("we use postgres")
    assert isinstance(r, Confirmation)
    assert r.success is True


async def test_update_constraint_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        await mcp_tools.update_constraint("")


async def test_record_turn_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        await mcp_tools.record_turn("")


def _set_interval(monkeypatch, n):
    """settings is frozen; swap the module singleton for a copy with a new interval."""
    import mnemostack.config.settings as cfg

    new = cfg.settings.model_copy(
        update={
            "compression": cfg.settings.compression.model_copy(update={"consolidation_interval": n})
        }
    )
    monkeypatch.setattr(cfg, "settings", new)


async def test_record_turn_accumulates_without_consolidating(monkeypatch):
    from mnemostack.core.state import state

    _set_interval(monkeypatch, 3)
    r = await mcp_tools.record_turn("a")
    assert r.success is True and "consolidated" not in r.message
    await mcp_tools.record_turn("b")
    assert state.memory.pending_count() == 2  # below threshold, no LLM call


async def test_record_turn_auto_consolidates_at_threshold(monkeypatch):
    import types

    from mnemostack.core.compression import llm_consolidator
    from mnemostack.core.state import state

    _set_interval(monkeypatch, 2)
    canned = '{"decisions": [{"d": "x"}]}'
    monkeypatch.setattr(
        llm_consolidator,
        "completion",
        lambda **kw: types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=canned))]
        ),
    )
    await mcp_tools.record_turn("a")
    r = await mcp_tools.record_turn("b")  # hits threshold -> fires consolidation
    assert "consolidated" in r.message
    assert state.memory.pending_count() == 0


# --- MCP protocol-level dispatch ---


EXPECTED_TOOLS = {
    "index_project",
    "query_codebase",
    "get_session_context",
    "get_full_context",
    "force_consolidate",
    "get_memory_stats",
    "update_constraint",
    "record_turn",
}


async def test_mcp_registers_expected_tools():
    listed = await mcp.list_tools()
    assert {t.name for t in listed} == EXPECTED_TOOLS


async def test_mcp_dispatch_query_codebase():
    _, structured = await mcp.call_tool("query_codebase", {"query": "auth", "top_k": 3})
    assert structured == {"result": []}


async def test_mcp_dispatch_get_session_context():
    _, structured = await mcp.call_tool("get_session_context", {})
    assert structured["decisions"] == []
    assert structured["local_extractions_pending"] == 0


async def test_mcp_dispatch_get_full_context():
    _, structured = await mcp.call_tool("get_full_context", {"query": "auth"})
    assert structured["chunks"] == []
    assert "session" in structured


async def test_mcp_dispatch_force_consolidate():
    _, structured = await mcp.call_tool("force_consolidate", {})
    assert structured["success"] is True


async def test_mcp_dispatch_get_memory_stats():
    _, structured = await mcp.call_tool("get_memory_stats", {})
    assert structured["snapshot_count"] == 0


async def test_mcp_dispatch_update_constraint():
    _, structured = await mcp.call_tool("update_constraint", {"constraint": "use postgres"})
    assert structured["success"] is True


async def test_mcp_dispatch_query_codebase_empty_query_errors():
    with pytest.raises(ToolError, match="query must not be empty"):
        await mcp.call_tool("query_codebase", {"query": "", "top_k": 5})


async def test_mcp_dispatch_query_codebase_bad_top_k_errors():
    with pytest.raises(ToolError, match="top_k must be positive"):
        await mcp.call_tool("query_codebase", {"query": "x", "top_k": 0})


async def test_mcp_dispatch_update_constraint_empty_errors():
    with pytest.raises(ToolError, match="must not be empty"):
        await mcp.call_tool("update_constraint", {"constraint": ""})


async def test_mcp_dispatch_unknown_tool_errors():
    with pytest.raises(Exception):
        await mcp.call_tool("does_not_exist", {})


# --- daemon lifecycle ---


def test_run_closes_state_even_if_server_raises(monkeypatch):
    from mnemostack.core.state import state
    from mnemostack.mcp import server

    closed = []
    monkeypatch.setattr(state, "close", lambda: closed.append(True))

    # Patch the class method (not the instance) — an instance setattr would leak a
    # shadowing attribute past monkeypatch teardown and break other run() tests.
    def _boom(self, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(type(server.mcp), "run", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        server.run()
    assert closed == [True]  # finally ran teardown
