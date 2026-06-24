# Mnemostack — Build Context

## What This Is

A local daemon (MCP server) that builds a live dependency graph of your codebase and gives any MCP-compatible AI coding assistant graph-aware code retrieval. Ask about any function and get back the entire dependency chain — across files, automatically. Model-agnostic — any client that speaks MCP gets it for free.

One-liner: "An MCP server that builds a live call graph of your codebase and gives any AI coding assistant graph-aware code retrieval."

---

## Core Problem

AI coding assistants have no understanding of how code connects. They retrieve isolated files, miss cross-file dependency chains, and chunk code into arbitrary token windows that split functions mid-body. When you ask about authentication, they find `auth.py` but miss the DB query it calls and the crypto function two hops away. On top of that, LLMs forget everything between sessions.

## Core Thesis

Code is a graph, not a bag of text. Any retrieval system that ignores call-graph structure is fundamentally broken. Mnemostack builds a live dependency graph from your AST, then uses that graph to retrieve complete dependency chains — not isolated snippets. Graph-targeted retrieval gives complete context with fewer tokens and higher accuracy.

---

## Architecture — Two-Tier System

### Tier 1: Graph-Aware Code Retrieval

**1A — AST-Aware Code Chunking**
- tree-sitter for multi-language AST parsing (100+ languages)
- Chunks by: functions/methods, classes, module-level constants, import blocks
- Each chunk stores: file path, symbol name, line range, last modified timestamp, dependencies

**1B — Lightweight Call Graph**
- 3 node types: File, Function, Class (qualified names: `file_path::ClassName.method_name`)
- 3 edge types: CALLS, IMPORTS_FROM, CONTAINS
- Stored in SQLite (graph.db)
- 2-hop BFS expansion on retrieval for cross-file dependency chains

**1C — Hybrid Retrieval: FTS5 + FAISS + RRF**
- FAISS HNSW for semantic similarity (O(log n), local CPU)
- FTS5 (SQLite) for exact identifier/keyword search (BM25 + Porter stemming)
- Reciprocal Rank Fusion to merge both ranked lists: `score(d) = sum(1 / (60 + rank_i(d)))`
- Query intent boost: PascalCase -> +1.5x Class, snake_case -> +1.5x Func, dotted.path -> +2.0x qualified name match

**1D — Incremental Index Updates**
- watchdog monitors filesystem for file saves (debounce 500ms)
- On save: re-parse changed file AST, diff chunks, update FAISS + FTS5 + call graph
- Batch rebuild available as fallback (branch switch, initial load)

**1E — Recency-Weighted Ranking**
- `score = a * semantic_similarity + b * recency_score + c * dependency_relevance`
- Weights: a=0.6, b=0.25, c=0.15 (tunable)
- Recency: exponential decay, half-life ~1 hour during active dev

### Tier 2: Compressed Session Memory

**Layer 2A — Local Extraction (every turn, free, zero API)**
- Runs locally on every turn
- Extracts: keywords, named entities, AST diffs, dependency graph updates, explicit user constraints/decisions, action items, open questions
- Stores as structured JSON, append-only log between consolidation cycles
- Memory is never stale — always a fresh extraction from the most recent turn

**Layer 2B — LLM Consolidation (every N turns, default 25)**
- Takes accumulated local extraction log + previous consolidated memory
- Structured prompt to LLM: extract decisions, merge, prune contradictions
- Outputs fixed-size structured JSON memory blob (budget: ~3000 tokens)
- Versioned snapshot stack (snap_001.json, snap_002.json, ...)
- Lazy merge: only merges two oldest snapshots when approaching token budget ceiling

**Consolidation output schema:**
```json
{
  "decisions": [{"text": "...", "impact_score": 0.72}],
  "constraints": [{"text": "...", "impact_score": 0.45}],
  "open_questions": [...],
  "architecture_state": {...},
  "resolved": [...],
  "file_relationships": [{"from": "...", "to": "..."}]
}
```

**Config params:**
- `consolidation_interval`: turns between LLM consolidation (default 25)
- `memory_token_budget`: max consolidated memory size (default 3000 tokens)
- `consolidation_model`: configurable via litellm (Ollama local for dev, frontier for prod)

### Graph-Aware Snapshot Merging

- Before LLM merge, annotate each decision with structural importance score from graph
- `importance = f(caller_count, test_coverage)`
- LLM instruction: drop decisions below 0.15, always keep above 0.60
- Prevents silent loss of load-bearing architectural decisions during repeated merges

---

## MCP Tools (6)

```
query_codebase(query: str, top_k: int = 5) -> list[CodeChunk]
    Hybrid FTS5+FAISS search, RRF fusion, 2-hop BFS expansion, recency ranking

get_session_context() -> SessionMemory
    Latest consolidated memory + local extractions since last consolidation

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
      retrieval/
        ast_chunker.py         -- tree-sitter code chunking by symbol boundaries
        call_graph.py          -- lightweight graph (3 node, 3 edge), 2-hop BFS
        faiss_index.py         -- FAISS HNSW index management
        fts_index.py           -- FTS5 keyword index + RRF fusion
        ranker.py              -- recency-weighted semantic + dependency ranking
        file_watcher.py        -- watchdog file save listener, incremental updates
      compression/
        local_extractor.py     -- keyword, AST diff, constraint extraction (every turn)
        llm_consolidator.py    -- LLM compression via litellm
        memory_store.py        -- versioned snapshot stack, graph-aware lazy merge
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
    graph.db                   -- SQLite: nodes, edges
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
WITH Mnemo (turn 200):     ~2,400-3,800 tokens/turn (flat)
Consolidation cost:        ~$0.001 per cycle (negligible)
```

---

## Build Order

### DONE (Day 1)
- Project scaffolding (mnemostack/ package, pyproject.toml, .gitignore, venv)
- MCP skeleton: `mcp/server.py` + `mcp/tools.py` — 6 tools with Pydantic models, stub returns
- Config system: `config/settings.py` + `config/defaults.yaml` — typed config, yaml loading, deep merge overrides, path resolution
- All verified: ruff clean, 11 tests passed (tool dispatch, protocol-level calls, error handling, config edge cases, serialization)

### DONE: Retrieval Pipeline
1. `core/retrieval/ast_chunker.py` — tree-sitter chunking (Python/JS/TS + fallback)
2. `core/retrieval/faiss_index.py` — FAISS HNSW index with SQLite persistence
3. `core/retrieval/fts_index.py` — FTS5 index (BM25 + Porter stemming)
4. `core/retrieval/ranker.py` — RRF fusion + recency ranking + query intent boost
5. `core/retrieval/call_graph.py` — lightweight graph + 2-hop BFS (Python extraction)
6. `core/retrieval/file_watcher.py` — debounced incremental index updates

### Next: Compression Pipeline
7. `core/router.py` — litellm model router
8. `core/compression/local_extractor.py` — per-turn extraction
9. `core/compression/llm_consolidator.py` — LLM consolidation
10. `core/compression/memory_store.py` — snapshot stack + graph-aware merge

### Then: Wire + Ship
11. Wire real implementations into MCP tool stubs
12. `daemon/lifecycle.py` — daemon management
13. End-to-end integration tests
14. Multi-client testing (Claude Code, Cursor, etc.)

## Known Traps

1. **`_PROJECT_ROOT` is fragile** — `Path(__file__).parent.parent.parent` counts on exact nesting depth of `mnemostack/config/settings.py`. If that file moves, store paths silently break. Consider `MNEMOSTACK_ROOT` env var before shipping.
2. **`get_full_context` calls Python functions directly**, not via MCP dispatch — works fine, but if we add MCP middleware (logging, metrics, auth), those calls will bypass it. Revisit when wiring real implementations.
3. **`settings` singleton loads at import time** — no runtime reload. Fine for daemon (restart to pick up config changes), but worth knowing.
4. **Namespace collision avoided** — everything under `mnemostack/` package specifically because a top-level `mcp/` dir shadows the `mcp` pip package. Do not flatten back to top-level.

---

## Unique Differentiators (What Nobody Else Has)

1. **Live call graph** — tree-sitter AST → dependency graph (functions, classes, imports) → 2-hop BFS expansion retrieves entire dependency chains, not isolated snippets
2. **Hybrid graph + search** — FAISS semantic + FTS5 exact identifiers fused via RRF, then expanded through the call graph
3. **Graph-aware snapshot merging** — structural importance scoring from the call graph prevents silent info loss during memory compression
4. **Model-agnostic** — works with any MCP client and any LLM provider

## Explicitly Out of Scope

- Execution flow detection (code-review-graph's domain)
- Risk-scored change detection via git diff
- Full 7-edge structural graph (3 edges sufficient for retrieval)
- VS Code extension
- Architecture overview as an MCP tool

---

## Research Backing

- arxiv.org/abs/2601.07190 — 22.7% token reduction via autonomous compression (2026)
- arxiv.org/abs/2510.00615 — 26-54% memory reduction, 95%+ accuracy preserved (2025)
- arxiv.org/abs/2601.02553v1 — three-stage pipeline: compression, consolidation, adaptive retrieval (2026)
- arxiv.org/html/2506.15655v1 — validates AST chunking approach
- Stanford "lost in the middle" — models perform worst on info in middle of long contexts
