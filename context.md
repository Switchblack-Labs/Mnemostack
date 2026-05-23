# Mnemostack — Build Context

## What This Is

A local daemon (MCP server) that gives any MCP-compatible AI coding assistant persistent compressed session memory and fast semantic codebase retrieval. Model-agnostic — any client that speaks MCP gets the full memory system for free.

One-liner: "An MCP server that gives any AI coding assistant persistent session memory and fast semantic codebase retrieval."

---

## Core Problem

LLMs have fixed context windows. As sessions grow:
- Early decisions/constraints get pushed out
- Model re-debates settled questions
- Dumping everything back in costs more tokens (linear growth), slows inference (O(n^2) attention), and reduces accuracy (lost-in-the-middle effect)

## Core Thesis

Treat context like a scalpel, not a bucket. Inject only what's relevant. Three simultaneous wins: cheaper (fewer tokens), faster (O(k^2) where k << n), more accurate (clean targeted context).

---

## Architecture — Two-Tier Memory System

### Tier 1: Compression Pipeline (Session Memory)

**Layer 1A — Local Extraction (every turn, free, zero API)**
- Runs locally on every turn
- Extracts: keywords, named entities, AST diffs, dependency graph updates, explicit user constraints/decisions, action items, open questions
- Stores as structured JSON, append-only log between consolidation cycles
- Memory is never stale — always a fresh extraction from the most recent turn

**Layer 1B — LLM Consolidation (every N turns, default 25)**
- Takes accumulated local extraction log + previous consolidated memory
- Structured prompt to LLM: extract decisions, merge, prune contradictions
- Outputs fixed-size structured JSON memory blob (budget: ~3000 tokens)
- Versioned snapshot stack (snap_001.json, snap_002.json, ...)
- Lazy merge: only merges two oldest snapshots when approaching token budget ceiling

**Consolidation output schema:**
```json
{
  "decisions": [{"text": "...", "communities": ["..."], "impact_score": 0.72}],
  "constraints": [{"text": "...", "communities": ["..."], "impact_score": 0.45}],
  "open_questions": [...],
  "architecture_state": {...},
  "resolved": [...],
  "file_relationships": [{"from": "...", "to": "...", "community": "..."}]
}
```

**Config params:**
- `consolidation_interval`: turns between LLM consolidation (default 25)
- `memory_token_budget`: max consolidated memory size (default 3000 tokens)
- `consolidation_model`: configurable via litellm (Ollama local for dev, frontier for prod)

### Tier 2: Semantic Retrieval (Codebase Search)

**2A — AST-Aware Code Chunking**
- tree-sitter for multi-language AST parsing (100+ languages)
- Chunks by: functions/methods, classes, module-level constants, import blocks
- Each chunk stores: file path, symbol name, line range, last modified timestamp, dependencies

**2B — Lightweight Call Graph**
- 3 node types: File, Function, Class (qualified names: `file_path::ClassName.method_name`)
- 3 edge types: CALLS, IMPORTS_FROM, CONTAINS
- Stored in SQLite (graph.db)
- 2-hop BFS expansion on retrieval for cross-file dependency chains

**2C — Hybrid Retrieval: FTS5 + FAISS + RRF**
- FAISS HNSW for semantic similarity (O(log n), local CPU)
- FTS5 (SQLite) for exact identifier/keyword search (BM25 + Porter stemming)
- Reciprocal Rank Fusion to merge both ranked lists: `score(d) = sum(1 / (60 + rank_i(d)))`
- Query intent boost: PascalCase -> +1.5x Class, snake_case -> +1.5x Func, dotted.path -> +2.0x qualified name match

**2D — Incremental Index Updates**
- watchdog monitors filesystem for file saves (debounce 500ms)
- On save: re-parse changed file AST, diff chunks, update FAISS + FTS5 + call graph
- Batch rebuild available as fallback (branch switch, initial load)

**2E — Recency-Weighted Ranking**
- `score = a * semantic_similarity + b * recency_score + c * dependency_relevance`
- Weights: a=0.6, b=0.25, c=0.15 (tunable)
- Recency: exponential decay, half-life ~1 hour during active dev

### Tier 1 + Tier 2 Integration: Community-Tagged Snapshots

- Leiden community detection on the lightweight graph clusters related code (e.g., `auth-core`, `db-layer`)
- Compression output is tagged with community IDs
- `get_session_context(query)` filters snapshots to only return communities relevant to the current query
- Reduces ~3000 token flat dump to ~400-800 targeted tokens per query

### Graph-Aware Snapshot Merging

- Before LLM merge, annotate each decision with structural importance score from graph
- `importance = f(caller_count, cross_community_callers, test_coverage)`
- LLM instruction: drop decisions below 0.15, always keep above 0.60
- Prevents silent loss of load-bearing architectural decisions during repeated merges

---

## MCP Tools (6)

```
query_codebase(query: str, top_k: int = 5) -> list[CodeChunk]
    Hybrid FTS5+FAISS search, RRF fusion, 2-hop BFS expansion, recency ranking

get_session_context() -> SessionMemory
    Latest consolidated memory + local extractions since last consolidation
    Community-filtered when query context is available

get_full_context(query: str) -> CombinedContext
    Convenience: session memory + relevant code chunks in one call

force_consolidate() -> ConsolidationResult
    Manual LLM consolidation trigger

get_memory_stats() -> Stats
    Memory size, index size, last consolidation time

update_constraint(constraint: str) -> Confirmation
    Manually inject a constraint into session memory
```

---

## File Structure

All source lives under `mnemostack/` package (avoids namespace collision with `mcp` pip package).

```
Mnemostack/
  mnemostack/                  # Python package root
    core/
      compression/
        local_extractor.py     -- keyword, AST diff, constraint extraction (every turn)
        llm_consolidator.py    -- LLM compression via litellm, community-tagged output
        memory_store.py        -- versioned snapshot stack, graph-aware lazy merge
      retrieval/
        ast_chunker.py         -- tree-sitter code chunking by symbol boundaries
        call_graph.py          -- lightweight graph (3 node, 3 edge), 2-hop BFS
        faiss_index.py         -- FAISS HNSW index management
        fts_index.py           -- FTS5 keyword index + RRF fusion
        ranker.py              -- recency-weighted semantic + dependency ranking
        file_watcher.py        -- watchdog file save listener, incremental updates
        communities.py         -- Leiden community detection on graph
      router.py                -- model-agnostic compression model selection (litellm)
    mcp/
      server.py                -- FastMCP server, stdio transport, entrypoint
      tools.py                 -- 6 tool defs with Pydantic response models (stubs wired)
    adapters/
      rest_adapter.py          -- FastAPI REST server (post-MVP)
      library.py               -- Python SDK surface (post-MVP)
    daemon/
      lifecycle.py             -- daemon start/stop/restart, PID management
    config/
      settings.py              -- Pydantic config models, yaml loader, deep merge, path resolution
      defaults.yaml            -- default configuration values
  store/                       -- runtime data (outside package, gitignored)
    snapshots/                 -- snap_001.json, snap_002.json ...
    graph.db                   -- SQLite: nodes, edges, communities
    index.faiss                -- FAISS HNSW index
    chunks.db                  -- SQLite: chunk metadata
    relevance.db               -- SQLite: query/chunk feedback log
  pyproject.toml               -- Python 3.11+, hatchling build, all deps
  .gitignore
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| AST parsing | tree-sitter | Multi-language, production-grade, clean chunk boundaries |
| Semantic search | FAISS HNSW | O(log n), runs on CPU, zero cost |
| Keyword search | SQLite FTS5 | Built into SQLite, zero new deps, BM25 scoring |
| Rank fusion | Reciprocal Rank Fusion | Parameter-light, no score normalization needed |
| Graph storage | SQLite | Already a dependency, no new process |
| Community detection | Leiden algorithm | Model-agnostic, runs locally on graph |
| LLM routing | litellm | Unified interface across all providers |
| File watching | watchdog | Python standard for filesystem events |
| MCP transport | stdio (default) | Microsecond overhead, no network stack |
| Embeddings | nomic-embed-text (Ollama) or text-embedding-3-small | Local for dev, API for prod |

## Dev Strategy

- Ollama + local model (Llama 3 8B) for dev — free
- FAISS on CPU — zero cost
- litellm for model routing so any provider works
- Paid APIs only for final benchmarks
- Test across multiple MCP clients from day one

---

## Token Economics

```
WITHOUT Mnemo (turn 200):  ~50,000-200,000 tokens/turn (growing linearly)
WITH Mnemo (turn 200):     ~2,400-3,800 tokens/turn (flat, community-filtered)
Consolidation cost:        ~$0.001 per cycle (negligible)
```

---

## Build Order

### DONE (Day 1)
- Project scaffolding (mnemostack/ package, pyproject.toml, .gitignore, venv)
- MCP skeleton: `mcp/server.py` + `mcp/tools.py` — 6 tools with Pydantic models, stub returns
- Config system: `config/settings.py` + `config/defaults.yaml` — typed config, yaml loading, deep merge overrides, path resolution
- All verified: ruff clean, 11 tests passed (tool dispatch, protocol-level calls, error handling, config edge cases, serialization)

### Next: Retrieval Pipeline
1. `core/retrieval/ast_chunker.py` — tree-sitter chunking
2. `core/retrieval/faiss_index.py` — FAISS HNSW index
3. `core/retrieval/fts_index.py` — FTS5 index
4. `core/retrieval/ranker.py` — RRF fusion + recency ranking
5. `core/retrieval/call_graph.py` — lightweight graph + 2-hop BFS
6. `core/retrieval/communities.py` — Leiden community detection
7. `core/retrieval/file_watcher.py` — incremental index updates

### Then: Compression Pipeline
8. `core/router.py` — litellm model router
9. `core/compression/local_extractor.py` — per-turn extraction
10. `core/compression/llm_consolidator.py` — LLM consolidation
11. `core/compression/memory_store.py` — snapshot stack + graph-aware merge

### Then: Wire + Ship
12. Wire real implementations into MCP tool stubs
13. `daemon/lifecycle.py` — daemon management
14. End-to-end integration tests
15. Multi-client testing (Claude Code, Cursor, etc.)

## Known Traps

1. **`_PROJECT_ROOT` is fragile** — `Path(__file__).parent.parent.parent` counts on exact nesting depth of `mnemostack/config/settings.py`. If that file moves, store paths silently break. Consider `MNEMOSTACK_ROOT` env var before shipping.
2. **`get_full_context` calls Python functions directly**, not via MCP dispatch — works fine, but if we add MCP middleware (logging, metrics, auth), those calls will bypass it. Revisit when wiring real implementations.
3. **`settings` singleton loads at import time** — no runtime reload. Fine for daemon (restart to pick up config changes), but worth knowing.
4. **Namespace collision avoided** — everything under `mnemostack/` package specifically because a top-level `mcp/` dir shadows the `mcp` pip package. Do not flatten back to top-level.

---

## Unique Differentiators (What Nobody Else Has)

1. **Community-tagged snapshots** — session memory filtered by code community per query
2. **Graph-aware snapshot merging** — structural importance scoring prevents silent info loss
3. **Two-tier compression** — local extraction (never stale) + periodic LLM consolidation (deep understanding)
4. **Model-agnostic** — works with any MCP client and any LLM provider

## Explicitly Out of Scope

- Execution flow detection (code-review-graph's domain)
- Risk-scored change detection via git diff
- Full 7-edge structural graph (3 edges sufficient for retrieval)
- VS Code extension
- Architecture overview / community listing as MCP tools

---

## Research Backing

- arxiv.org/abs/2601.07190 — 22.7% token reduction via autonomous compression (2026)
- arxiv.org/abs/2510.00615 — 26-54% memory reduction, 95%+ accuracy preserved (2025)
- arxiv.org/abs/2601.02553v1 — three-stage pipeline: compression, consolidation, adaptive retrieval (2026)
- arxiv.org/html/2506.15655v1 — validates AST chunking approach
- Stanford "lost in the middle" — models perform worst on info in middle of long contexts
