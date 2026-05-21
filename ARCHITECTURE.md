# Context Memory Daemon — Architecture v2

## Project Summary

A background daemon exposed as an MCP server that gives any MCP-compatible AI coding assistant (Claude Code, Cursor, Copilot, Continue.dev, local LLMs) persistent session memory and fast semantic codebase retrieval. Model-agnostic by design — the MCP interface means any client that speaks MCP gets the full memory system for free with zero custom integration.

---

## Core Problem

AI coding assistants use transformer-based LLMs with fixed context windows. As sessions grow:
- Early context (architecture decisions, constraints, file relationships) gets pushed out
- The model re-debates settled questions and forgets stack choices
- Current fix is dumping everything back into context, which:
  - Costs more tokens (linear growth per turn)
  - Slows inference (transformers have O(n^2) attention complexity — more tokens = quadratically more compute)
  - Reduces accuracy (Stanford "lost in the middle" research — models perform worst on info buried in the middle of large contexts)
- Bigger context windows (Gemini 1M tokens) make this WORSE, not better — the lost-in-the-middle problem scales with window size

## Core Thesis

Current tools treat context like a bucket (dump everything). We treat it like a scalpel (inject only what's relevant). This gives three wins simultaneously — not as tradeoffs:
1. Less tokens → cheaper
2. Faster responses → O(n^2) attention over fewer tokens
3. Better accuracy → model works with clean, targeted context instead of searching through noise

---

## Architecture — Two-Tier Memory System

### Tier 1: Two-Tier Compression Pipeline

**Issue with naive approach:** Hitting an LLM every N turns for compression is the slowest, most expensive, and least reliable part of the system. Between LLM calls, memory is stale.

**Solution: Two-tier compression — local extraction + periodic LLM consolidation**

#### Layer 1A: Lightweight Local Extraction (every turn, free)
- Runs locally with zero API cost
- Extracts on every turn:
  - Keywords and named entities (libraries, frameworks, file names, variable names mentioned)
  - AST diffs — what functions/classes/files changed this turn
  - Dependency graph updates — new imports, new function calls, new file relationships
  - Explicit user constraints and decisions (pattern match on phrases like "don't use X", "we decided Y", "the constraint is Z")
  - Action items and open questions
- Stores as structured JSON, append-only log between LLM consolidation cycles
- This ensures memory is NEVER stale — there's always a fresh local extraction from the most recent turn

#### Layer 1B: LLM Consolidation (every N turns, configurable — start with 20-30)
- Takes the accumulated local extraction log + previous consolidated memory
- Sends to LLM API with structured prompt:
  - Extract: decisions made + rationale, active constraints, current architecture state, open questions, resolved questions
  - Merge: recursive consolidation — old compressed memories merge with new ones
  - Prune: remove contradicted or superseded information
  - Output: fixed-size structured memory blob that stays under a token budget regardless of session length
- Writes to CLAUDE.md (or equivalent memory file the coding assistant reads)
- The local extraction log is cleared after successful consolidation

#### Compression tuning parameters:
- `local_extraction_mode`: what to extract locally (keywords, AST diffs, constraints, all)
- `consolidation_interval`: turns between LLM consolidation (default 25)
- `memory_token_budget`: max size of consolidated memory blob (default 3000 tokens)
- `consolidation_model`: which LLM to use (can be cheap local model for dev, expensive model for production)

#### Flow

```
session transcript (every turn)
        |
        v
+-------------------+
|  local_extractor  |  runs every turn, zero API cost
|                   |  extracts: keywords, AST diffs,
|  (core/           |  constraints, entities, action items
|   compression/    |
|   local_          |  stores as append-only JSON log
|   extractor.py)   |
+--------+----------+
         |
         | accumulated log (every N turns)
         v
+-------------------+
|  llm_consolidator |  sends log + previous memory to LLM
|                   |  structured JSON output schema:
|  (core/           |   {
|   compression/    |     "decisions": [...],
|   llm_            |     "constraints": [...],
|   consolidator.py)|     "open_questions": [...],
|                   |     "architecture_state": {...},
|                   |     "resolved": [...]
|                   |   }
+--------+----------+
         |
         | consolidated memory blob
         v
+-------------------+
|  memory_store     |  versioned snapshot files
|  snapshots/       |  snap_001.json, snap_002.json ...
|                   |  lazy merge: only merges two oldest
|                   |  when approaching token budget ceiling
|                   |  latest pointer: latest.json (symlink)
+--------+----------+
         |
         v
  served via get_session_context() MCP tool
  returns: latest consolidated memory + any local
  extractions since last consolidation
```

#### Why versioned snapshots instead of a merged blob

Single merged blob: information loss compounds every cycle. By cycle 5 you've silently dropped nuance you didn't know you needed, with no way to recover it.

Versioned stack: each snapshot is immutable once written. Merge only happens when you're near the token budget, and only the two oldest snapshots merge — the recent ones stay intact. You can inspect history and roll back if something critical got dropped.

---

### Tier 2: Dependency-Aware Semantic Retrieval

**Issue with naive approach:** AST chunking treats every function/class as isolated. Code doesn't work in isolation — function A calls function B in a different file. Pure semantic search retrieves A but misses B, giving incomplete context.

**Issue with static indexing:** Building the FAISS index once and querying it means retrieval serves stale results during active development — exactly when accuracy matters most.

**Issue with pure semantic ranking:** A helper function written 6 months ago might be more semantically similar to a query than code edited 30 seconds ago. Recency matters.

**Solution: Dependency-aware retrieval with incremental updates and recency-weighted ranking**

#### 2A: AST-Aware Code Chunking
- Parse codebase using tree-sitter (supports 100+ languages)
- Chunk by meaningful code boundaries:
  - Functions/methods (with docstrings and decorators)
  - Classes (with method signatures)
  - Module-level constants and configurations
  - Import blocks
- Each chunk stores metadata:
  - File path
  - Function/class name
  - Line range
  - Last modified timestamp
  - Dependencies (imports, function calls — see 2B)
- AST chunking reduces irrelevant retrievals by ~40% vs naive token-window splitting

#### 2B: Lightweight Call Graph (Dependency-Aware Retrieval)
- Build a directed graph of function/class dependencies:
  - Function A calls Function B → edge A→B
  - Class X imports module Y → edge X→Y
  - File Z imports from File W → edge Z→W
- When a query retrieves chunk A, also retrieve:
  - Direct dependencies of A (one hop in the call graph)
  - Optionally: two-hop dependencies for complex queries (configurable)
- Graph is stored alongside the FAISS index and updated incrementally
- Implementation: use tree-sitter queries to extract call sites and imports, store as adjacency list
- This ensures cross-file relationships are never missed

#### 2C: FAISS HNSW Index
- Embed each AST chunk using a code embedding model (candidates: OpenAI text-embedding-3-small, CodeBERT, or local model like nomic-embed-text via Ollama)
- Store embeddings in FAISS HNSW index for O(log n) approximate nearest neighbor retrieval
- HNSW parameters to tune:
  - `M`: number of connections per layer (default 32)
  - `efConstruction`: index build quality (default 200)
  - `efSearch`: query-time accuracy/speed tradeoff (default 128)
- FAISS runs locally on CPU — zero cost regardless of query volume

#### 2D: Incremental Index Updates
- Watch for file system events using watchdog (Python)
- On file save:
  - Re-parse only the changed file's AST
  - Identify which chunks changed (diff against previous AST)
  - Remove old chunk embeddings from FAISS index
  - Embed new/modified chunks
  - Insert into FAISS index
  - Update call graph edges for affected nodes
- This ensures retrieval is never stale during active development
- Batch rebuild is still available as a fallback (e.g., on branch switch or initial project load)

#### 2E: Recency-Weighted Ranking
- Final retrieval score = `a * semantic_similarity + b * recency_score + c * dependency_relevance`
- `semantic_similarity`: cosine similarity from FAISS query (0 to 1)
- `recency_score`: decaying function based on last-modified timestamp (e.g., exponential decay with half-life of 1 hour during active dev)
- `dependency_relevance`: bonus if chunk is a direct dependency of another highly-ranked chunk (0 or fixed bonus)
- a, b, c are tunable weights — start with a=0.6, b=0.25, c=0.15
- Return top-k chunks after re-ranking (k configurable, default 5-10)

#### Indexing Flow

```
codebase files
      |
      v
+-----------------+
|  file_watcher   |  watches filesystem for file saves
|  (core/         |  debounces rapid saves (500ms window)
|   retrieval/    |  queues changed files for re-indexing
|   file_         |
|   watcher.py)   |
+--------+--------+
         | file changed
         v
+-----------------+
|  ast_chunker    |  tree-sitter based code chunking
|                 |  splits by function/class boundaries
|  per chunk:     |  not arbitrary token windows
|  - code text    |
|  - file path    |
|  - symbol name  |
|  - line range   |
|  - last modified|
+--------+--------+
         | chunks
         v
+-----------------+
|  call_graph     |  extracts function calls and imports
|                 |  builds directed dependency graph
|  A calls B →    |  stored as adjacency list
|  edge A→B       |  updated incrementally per file
+--------+--------+
         | chunks + graph
         v
+-----------------+
|  faiss_index    |  embeds each chunk (local model or API)
|                 |  upserts into FAISS HNSW index
|  O(log n)       |  stores chunk metadata in sqlite sidecar
|  approximate    |  (file path, symbol, line range, deps)
|  nearest        |
|  neighbor       |
+--------+--------+
         |
         v
   index.faiss + chunks.db
```

#### Query Flow

```
user query (from MCP client)
        |
        v
+-------------------+
|  query_codebase() |  MCP tool handler
|  (mcp/tools.py)   |
+--------+----------+
         |
         v
+-------------------+
|  faiss_index      |  embed query
|                   |  FAISS HNSW search → top-k candidates
+--------+----------+
         |
         v
+-------------------+
|  call_graph       |  for each retrieved chunk:
|  expansion        |  fetch direct dependencies (one hop)
|                   |  e.g., auth.py → config.py constants
+--------+----------+
         |
         v
+-------------------+
|  ranker           |  re-rank expanded set:
|                   |  score = a*semantic + b*recency + c*dep
|                   |  return final top-k
+--------+----------+
         |
         v
  return chunks to MCP client
  client injects into prompt
```

---

## MCP Server Interface

The entire system is exposed as an MCP server. This is the thinnest layer — just the API surface.

### Transport
- **stdio** for local use (default) — stdin/stdout pipes, microsecond overhead
- **SSE/Streamable HTTP** for remote use (future, post-launch)

### Exposed MCP Tools

```
query_codebase(query: string, top_k: int = 5) -> list[CodeChunk]
  # Queries FAISS index with recency-weighted ranking
  # Returns top-k relevant code chunks with metadata
  # Automatically includes dependency-linked chunks

get_session_context() -> SessionMemory
  # Returns the current compressed session memory
  # Combines latest LLM consolidation + local extractions since last consolidation

get_full_context(query: string) -> CombinedContext
  # Convenience tool: returns session memory + relevant code chunks
  # Single call that gives the model everything it needs

force_consolidate() -> ConsolidationResult
  # Manually trigger LLM consolidation cycle
  # Useful when user knows they've made major decisions

get_memory_stats() -> Stats
  # Returns current memory size, index size, last consolidation time
  # Useful for debugging and monitoring

update_constraint(constraint: string) -> Confirmation
  # Manually add a constraint to session memory
  # Bypasses extraction — user explicitly says "remember this"
```

### Why MCP
- One integration point instead of N custom plugins per client
- Any MCP-compatible client connects and gets full memory system for free
- Frictionless setup: point MCP config at the server, no plugins, no forks
- The coding assistant calls tools like `query_codebase("auth flow")` the same way it calls bash or file read

### Performance
- MCP over stdio adds microseconds of overhead
- FAISS HNSW query is O(log n) — milliseconds even on large codebases
- The bottleneck is the LLM consolidation call — but that runs async in the background, never blocking the user
- Total per-turn latency: retrieval (ms) + context injection (ms) = negligible

---

## How It Replaces (Not Adds To) Current Context

Critical distinction: we are NOT adding retrieval on top of existing context stuffing. We REPLACE the dumb context stuffing.

- **Before**: coding assistant grabs files → stuffs all into prompt → model processes O(n^2) attention over massive bloated prompt
- **After**: coding assistant calls our MCP tools → we return compressed memory + top-k relevant chunks → model processes O(k^2) attention where k << n

The model's transformer architecture doesn't change. We control what it sees before it thinks. Every API call is a fresh forward pass on whatever tokens are in the prompt — we just make those tokens cleaner and more relevant.

---

## Token Economics

```
WITHOUT Mnemo (turn 200):
  +------------------------------------------+
  | prompt = all files + full transcript      |
  | ~50,000-200,000 tokens per turn           |
  | grows linearly every turn                 |
  | O(n^2) attention cost                     |
  +------------------------------------------+

WITH Mnemo (turn 200):
  +------------------------------------------+
  | prompt = session summary (~3k tokens)     |
  |        + top-k chunks (~2-4k tokens)      |
  | ~5,000-7,000 tokens per turn              |
  | stays flat regardless of session length   |
  | O(k^2) attention where k << n             |
  +------------------------------------------+

  one-time consolidation cost: ~10k tokens per cycle
  saves: 40k-190k tokens on every subsequent turn
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Memory format | Versioned snapshot stack | Prevents silent info loss from recursive merge |
| Compression | Two-tier (local + LLM) | Memory is never stale; LLM cost amortized over N turns |
| Retrieval | FAISS HNSW | O(log n) vs O(n) for Chroma/naive cosine |
| Chunking | AST boundaries (tree-sitter) | Retrieves complete functions/classes, not arbitrary slices |
| Dependency expansion | Call graph (one-hop default) | Catches cross-file relationships at function granularity |
| Ranking | Recency-weighted semantic | Recently edited code outranks stale semantic matches |
| Index updates | Incremental on file save | Retrieval is never stale during active development |
| Compression output | Structured JSON schema | Prevents LLM from dropping fields, easier to inspect/debug |
| Dev LLM | Ollama (Llama 3 8B) | Free, good enough for compression, no API credits burned |
| Transport | stdio | Microsecond overhead, no network stack needed for local use |

---

## Project Structure

```
core/
  compression/
    local_extractor.py      — keyword, AST diff, constraint extraction (runs every turn)
    llm_consolidator.py     — LLM-based recursive memory consolidation
    memory_store.py         — read/write compressed memory blob, snapshot stack management
  retrieval/
    ast_chunker.py          — tree-sitter based code chunking
    call_graph.py           — lightweight dependency graph builder
    faiss_index.py          — HNSW index management (build, query, incremental update)
    ranker.py               — recency-weighted semantic ranking
    file_watcher.py         — file save event listener for incremental updates
mcp/
  server.py                 — MCP server setup and lifecycle
  tools.py                  — tool definitions (query_codebase, get_session_context, etc.)
daemon/
  lifecycle.py              — daemon start/stop/restart, PID management
config/
  settings.py               — all tunable parameters
  defaults.yaml             — default configuration
store/
  snapshots/                — snap_001.json, snap_002.json ...
  index.faiss               — FAISS HNSW index
  chunks.db                 — sqlite: chunk metadata + call graph
  relevance.db              — sqlite: query/chunk feedback log
```

---

## Development Strategy

- Use Ollama with local model (Llama 3 8B or similar) for all dev and integration testing — free
- FAISS runs on CPU locally — zero cost for retrieval testing
- Only use paid APIs for final benchmarks and demos
- No cloud infrastructure — everything runs locally as a daemon
- Test across multiple MCP clients from day one to catch client-specific quirks early

---

## Deferred (Post-MVP)

These are architecturally valid but premature before the core works:

- **Chat mode** — conversational chunking for non-code sessions. Same FAISS pipeline, different chunker. Add after code mode is validated.
- **Security layer** — trust scoring (git blame), integrity checks, anomaly detection, quarantine logic. Important for shared codebases, not for local single-user MVP.
- **SSE/HTTP transport** — remote MCP server. Ship stdio first, add network transport when there's demand.

---

## Research Backing

- https://arxiv.org/abs/2601.07190 — 22.7% token reduction via autonomous compression (2026)
- https://arxiv.org/abs/2510.00615 — 26-54% memory reduction, 95%+ accuracy preserved (2025)
- https://arxiv.org/abs/2601.02553v1 — three stage pipeline: compression, consolidation, adaptive retrieval (2026)
- https://arxiv.org/html/2506.15655v1 — directly validates AST chunking approach
- Stanford "lost in the middle" — models perform worst on info in middle of long contexts

## Existing Tools to Leverage

- https://github.com/supermemoryai/code-chunk — AST-aware chunking library (can use directly)
- tree-sitter — multi-language AST parsing
- FAISS — Facebook's similarity search library
- watchdog (Python) — file system event monitoring

---

## Team Ownership

- **Person 1 (Devang)**: Core architecture — compression pipeline (both tiers), FAISS indexing, AST chunking, call graph, ranking algorithm, MCP server interface
- **Person 2**: Security layer (post-MVP), adversarial evaluation harness, red-team attack suite
- **Person 3**: CLI, daemon lifecycle, file watcher, config system, documentation, benchmarking infrastructure

---

## Evaluation Plan

### Baselines:
- vs **no memory system** (any coding assistant raw — Claude Code, Cursor, Copilot)
- vs **claude-mem** (O(n) Chroma retrieval, Claude-locked)
- vs **raw RAG baseline** (naive token-window chunking + vector search, no compression)
- vs **our system with components ablated**

### Cross-Client Testing:
- Test across Claude Code, Cursor, Copilot, Continue.dev (local LLM)
- Proves the architecture generalizes across models and clients

### Ablation Study (4 experiments):
1. Remove recency weighting → measure retrieval relevance degradation
2. Remove dependency-aware retrieval → measure cross-file query failure rate
3. Use only LLM compression (no local extraction) → measure staleness and token cost increase
4. Use batch index rebuilds (no incremental updates) → measure retrieval staleness during active editing

### Metrics:
- Token usage per turn (flat vs growing)
- Task completion accuracy on long-session benchmarks
- Retrieval precision/recall (relevant chunks retrieved vs total)
- Latency per turn (end-to-end)
- Cost per session (API dollars)
- Memory staleness (how often does retrieved context reflect current code state)

---

## One-Liner Pitch

"An MCP server that gives any AI coding assistant persistent session memory and fast semantic codebase retrieval."
