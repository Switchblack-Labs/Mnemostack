# Mnemo — Architectural Suggestions

## Preface

This document proposes concrete architectural improvements to Mnemo based on a rigorous comparison with the state of the art in codebase knowledge graphs, retrieval systems, and LLM memory research. Every suggestion is evaluated against three criteria:

1. **Does it objectively improve performance** in Mnemo's core domain?
2. **Does it keep Mnemo model-agnostic and transport-agnostic?**
3. **Does it stay differentiated** — i.e., not turn Mnemo into a worse version of an existing tool?

---

## The Single Most Important Reframe

The current architecture positions Mnemo as an MCP server for Claude Code users. This is the wrong identity. The session memory problem is not a Claude Code problem — it is a universal LLM problem.

```
Turn 1:    user explains their architecture decisions
Turn 40:   model still remembers most of it
Turn 100:  model starts hallucinating earlier decisions
Turn 200:  model has no memory of turn 1-50 at all
           user is manually re-pasting context from scratch
```

This happens in ChatGPT, Claude.ai, Cursor, Windsurf, Copilot Chat — every long-running LLM session. Nobody has shipped a clean, universal fix for it.

**Mnemo should own this problem across all LLM interfaces, not just Claude Code.**

MCP is one transport. It should not be the identity.

---

## Suggestion 1: Pluggable Transport Architecture

### Current

Mnemo is designed exclusively as an MCP stdio server. The session watcher writes to a local transcript file. The MCP client (Claude Code) is the only way in.

### Problem

This locks Mnemo to:
- MCP-compatible clients only (Claude Code, Cursor, Windsurf)
- Local file-based session transcripts only
- No path to web users, API developers, or other chat interfaces

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MNEMO CORE ENGINE                            │
│                                                                     │
│   compression.py     consolidation.py     store/                   │
│   (model-agnostic)   (snapshot stack)     (snapshots, graph.db)    │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
  │  MCP Adapter    │ │  REST Adapter   │ │  Python Library     │
  │                 │ │                 │ │                     │
  │  stdio / SSE    │ │  POST /compress │ │  import mnemo       │
  │  transport      │ │  GET /context   │ │  s = mnemo.Session()│
  │                 │ │  POST /query    │ │  s.push(transcript) │
  │  Claude Code    │ │                 │ │  s.context()        │
  │  Cursor         │ │  Any API client │ │                     │
  │  Windsurf       │ │  Web apps       │ │  SDK for builders   │
  └─────────────────┘ └─────────────────┘ └─────────────────────┘
           │
           ▼ (future, not v1)
  ┌─────────────────┐
  │  Browser Ext    │
  │                 │
  │  Intercepts     │
  │  ChatGPT /      │
  │  Claude.ai DOM  │
  │  Injects memory │
  │  into prompt    │
  └─────────────────┘
```

### Why it's better

- MCP stays fully supported — no regression for existing use case
- REST adapter unlocks every API developer building on top of GPT/Claude/Gemini without needing MCP support in their stack
- Python library lets developers embed Mnemo directly into agent pipelines
- The browser extension path (v2+) is the unlock for non-technical LLM users — the largest segment of the market
- Core engine has zero knowledge of transport — compression and consolidation logic is written once

### Implementation note

Adapters are thin wrappers. The REST adapter is ~80 lines of FastAPI calling the same `compression.py` and `consolidation.py` functions the MCP adapter calls. No logic duplication.

---

## Suggestion 2: Model-Agnostic Compression Engine

### Current

The architecture specifies Ollama (Llama 3 8B) as the compression model. The compression prompt calls a single hardcoded provider.

### Problem

- Couples the compression quality to one model choice
- Users running Claude Code have Claude available — using an 8B local model for compression when a frontier model is already in the loop is a quality regression
- Users without GPU cannot run Ollama at all
- No way to benchmark compression quality across models

### Proposed Architecture

```
compression.py
      │
      ▼
┌─────────────────────────────────────────────────┐
│              Model Router                        │
│                                                  │
│  priority order (configurable in settings.py):  │
│                                                  │
│  1. MNEMO_COMPRESSION_MODEL env var             │
│     (user explicit override)                    │
│                                                  │
│  2. Cheapest available from detected providers: │
│     - Ollama local    → llama3:8b               │
│     - Anthropic key   → claude-haiku-3-5        │
│     - OpenAI key      → gpt-4o-mini             │
│     - Gemini key      → gemini-1.5-flash        │
│     - Groq key        → llama3-8b-8192 (free)  │
│                                                  │
│  3. Fallback: raise ConfigError with guidance   │
└─────────────────────────────────────────────────┘
      │
      ▼
  litellm.completion(model=resolved_model, ...)
  unified interface — one call regardless of provider
```

### Why it's better

- Compression quality matches the user's available resources — frontier model users get frontier compression, local users get local compression
- `litellm` handles auth, retries, and API format differences across all providers — Mnemo doesn't maintain provider-specific code
- Users can point Mnemo at any future model without a code change
- Benchmark compression quality by swapping models via env var — this is how you improve the compression prompt over time without guessing

### Cost reality

Compressing 25 turns of a coding session is ~2,000-4,000 input tokens. At claude-haiku-3-5 pricing ($0.25/MTok input), that is **$0.001 per compression cycle**. The argument for using a degraded local model to save money does not hold up at this scale. Default to quality, let users opt down.

---

## Suggestion 3: Replace FAISS-only with FTS5 Hybrid + Reciprocal Rank Fusion

### Current

Mnemo's Tier 2 uses FAISS HNSW as the sole retrieval backend. The query flow is:

```
query → embed → FAISS top-k → one-hop import expansion → return chunks
```

### Problem

Pure vector search has a well-documented failure mode: **exact identifier retrieval**. If a user asks about `validate_token` or `AuthService`, FAISS returns semantically similar code — but may rank the exact function lower than thematically related but less relevant code. Identifier names are the most important signal in code search and BM25 handles them better than cosine similarity.

### Proposed Query Pipeline

```
query
  │
  ├──────────────────────────────────────────────────────────┐
  │                                                          │
  ▼                                                          ▼
FTS5 search                                           FAISS HNSW search
(BM25 + Porter stemming)                              (semantic similarity)
(SQLite virtual table,                                (approximate nearest
 no extra process)                                     neighbor, O(log n))
  │                                                          │
  │  ranked list R1                                          │  ranked list R2
  └──────────────────────┬───────────────────────────────────┘
                         │
                         ▼
              Reciprocal Rank Fusion
              score(d) = Σ 1 / (60 + rank_i(d))
              (k=60 is the standard RRF constant)
                         │
                         ▼
              merged ranked list
                         │
                         ▼
              Query intent boost
              ┌──────────────────────────────────┐
              │ PascalCase query  → +1.5x Class  │
              │ snake_case query  → +1.5x Func   │
              │ dotted.path query → +2.0x QN match│
              └──────────────────────────────────┘
                         │
                         ▼
              2-hop BFS dependency expansion
              (replaces one-hop import expansion)
                         │
                         ▼
              return top-k expanded chunks
```

### Why it's better

RRF is a proven, parameter-light fusion method. It does not require score normalization between FTS5 (BM25) and FAISS (cosine) — rank positions are the only input, so different score scales don't interfere. Academic benchmarks consistently show hybrid BM25 + dense retrieval outperforms either alone by 5-15% on code retrieval tasks.

FTS5 is built into SQLite — zero new dependencies, zero new processes. The FAISS index you already have becomes one of two signals rather than the only signal.

---

## Suggestion 4: Replace One-Hop Import Expansion with 2-Hop BFS on a Lightweight Graph

### Current

After FAISS retrieval, Mnemo does one-hop import expansion: for each retrieved chunk, fetch files it imports from. This is file-level and import-only.

### Problem

```
auth.py  →  imports  →  config.py      ← one-hop catches this
auth.py  →  calls    →  db.py          ← one-hop MISSES this (call, not import)
auth.py  →  imports  →  utils.py
utils.py →  calls    →  crypto.py      ← two hops, completely missed
```

When a user asks "how does authentication work", the most relevant code is often two hops away via function calls, not imports. One-hop import expansion misses entire dependency chains.

### Proposed Lightweight Graph

This is **not** a full structural graph with 7 edge types and BFS-based impact analysis — that is code-review-graph's domain and reimplementing it here would be redundant. Mnemo needs only enough graph to serve better retrieval context.

```
Nodes:  File, Function, Class
        qualified_name: "file_path::ClassName.method_name"

Edges:  CALLS         (function → function)
        IMPORTS_FROM  (file → file)
        CONTAINS      (file → function/class)

Storage: SQLite (graph.db, reuses existing SQLite dependency)
Parsing: Tree-sitter (replaces code-chunk lib, same AST quality,
                      supports 19 languages)
```

```
2-hop BFS expansion (bounded):

retrieved chunk: auth.py::validate_token
       │
       ├── hop 1: follow CALLS + IMPORTS_FROM edges outward
       │          auth.py::validate_token → CALLS → db.py::query_user
       │          auth.py → IMPORTS_FROM → config.py
       │
       └── hop 2: follow same edges from hop-1 results
                  db.py::query_user → CALLS → crypto.py::hash_password
                  config.py → (no outgoing CALLS)

final context: [validate_token, query_user, hash_password, config.py]
               4 chunks, all genuinely relevant, 0 irrelevant noise
```

### Why it's better

Two hops catches cross-file call chains that one-hop import expansion misses entirely. The graph stays minimal — 3 node types, 3 edge types, no flow detection, no community detection, no risk scoring. That scope belongs to code-review-graph. Mnemo's graph exists only to serve better retrieval context for session memory compression.

Tree-sitter replaces code-chunk because it is the standard library for production-grade AST parsing. It handles edge cases (nested classes, decorators, lambda bodies) that simpler chunkers mishandle, producing cleaner chunk boundaries and more accurate qualified names.

---

## Suggestion 5: Community-Aware Session Snapshots

### Current

Session snapshots are flat JSON blobs compressing the full transcript uniformly. `get_session_context()` returns the latest snapshot regardless of what the user is currently asking about.

### Problem

By session turn 100, the snapshot contains compressed context about authentication, database schema, API design, frontend components, and CI configuration. When the user asks a question about the database layer, they receive a 3,000-token context dump covering everything. Most of it is irrelevant to the query at hand.

### Proposed Architecture

Leiden community detection runs on the lightweight graph (Suggestion 4). Each community is a natural cluster of related code — e.g., `auth-core`, `api-gateway`, `db-layer`.

```
Compression flow (upgraded):

transcript slice (25 turns)
         │
         ▼
identify which code communities were discussed
(match mentioned file paths / function names → community_id via graph lookup)
         │
         ▼
structured compression output, now community-tagged:

{
  "decisions": [
    {
      "text": "JWT tokens expire after 24h",
      "communities": ["auth-core"],
      "impact_score": 0.72
    }
  ],
  "constraints": [
    {
      "text": "database must be Postgres-compatible",
      "communities": ["db-layer"],
      "impact_score": 0.45
    }
  ],
  "open_questions": [...],
  "file_relationships": [
    {
      "from": "auth.py",
      "to": "config.py",
      "community": "auth-core"
    }
  ]
}

         │
         ▼
snapshot written with community_scope metadata:
snap_003.json { "community_scope": ["auth-core"], "tokens": 420 }


get_session_context(query="how does password hashing work"):
         │
         ▼
query_codebase("password hashing") → returns chunks in "auth-core" community
         │
         ▼
filter snapshots: only return snaps where community_scope ∩ {"auth-core"} ≠ ∅
         │
         ▼
return: ~600 targeted tokens instead of 3,000 token full dump
        all of it relevant to the current query
```

### Why it's better

This is the architectural feature that makes Tier 1 and Tier 2 work as a unified system rather than two independent modules. Without this, session memory and codebase retrieval are parallel tools that the LLM has to reconcile manually. With community-tagging, the memory layer knows which decisions belong to which part of the codebase, and surfaces only what matters for the current query.

Token reduction is concrete: a 3,000-token flat snapshot filtered to one community is typically 400-800 tokens. Over a 200-turn session with many queries, this compounds into significant cost savings with zero information loss for the query at hand.

Leiden community detection is model-agnostic, runs locally on the graph, and requires no LLM calls. It is a one-time post-processing step after graph construction.

---

## Suggestion 6: Graph-Aware Snapshot Merging

### Current

The lazy merge strategy merges the two oldest snapshots when total token count approaches the budget ceiling (8k tokens). The merge is an LLM call that combines two JSON blobs.

### Problem

The LLM doing the merge has no signal for which decisions are still architecturally load-bearing. A decision like "we're using JWT for auth" (referenced in 15 functions across 4 files) should survive every merge. A decision like "we decided to name this variable `ctx` instead of `context`" should be dropped immediately. The current merge cannot distinguish these.

### Proposed Merge Strategy

Before sending snapshots to the LLM for merging, annotate each decision with a structural importance score derived from the graph:

```
for each decision in (snap_oldest + snap_second_oldest):

  1. resolve decision.file_relationships → graph nodes
  2. if no matching nodes found:
       importance = 0.1  (likely stale/minor decision)
  3. if matching nodes found:
       caller_count   = count(CALLS edges pointing to node)
       cross_community = count(callers in different community)
       importance = min(caller_count / 20, 0.5)
                  + min(cross_community * 0.1, 0.3)
                  + (0.2 if node has no TESTED_BY edge)

merge prompt includes importance scores:
  "decision [importance: 0.87]: JWT tokens expire after 24h
   decision [importance: 0.03]: renamed ctx to context in handler.py
   ..."

LLM instruction: drop decisions below 0.15 importance,
                 always keep decisions above 0.60
```

### Why it's better

The current merge is a blind LLM summarization. The upgraded merge uses structural graph signals to guide which information survives — decisions that touch heavily-called, cross-community code are objectively more important than decisions about isolated low-traffic code. The LLM is still doing the natural language merge, but it is given objective importance weights derived from the codebase structure rather than having to infer importance from text alone.

This improves long-session quality where silent information loss in repeated merges is the most damaging failure mode of compressed memory systems.

---

## What Not to Build

The following features would objectively make Mnemo a worse version of an existing tool. They are explicitly out of scope.

| Feature | Why to skip |
|---|---|
| Execution flow detection (forward BFS from API entry points) | This is the core feature of code-review-graph's `flows.py`. Reimplementing it here adds no differentiation and splits maintenance. |
| Risk-scored change detection via git diff | This is code-review-graph's `changes.py` — multi-factor risk scoring, security keyword detection, test gap analysis. Same story. |
| Full 7-edge structural graph (INHERITS, IMPLEMENTS, TESTED_BY, DEPENDS_ON) | The 3-edge lightweight graph (Suggestions 4-5) is sufficient for Mnemo's retrieval use case. The full graph is code-review-graph's domain. |
| Architecture overview / community listing as MCP tools | code-review-graph exposes 22 MCP tools for code review workflows. Mnemo's MCP surface should stay at 4-6 tools focused on memory operations. |
| VS Code extension | code-review-graph ships one. Building a competing extension for the same graph data is redundant. |

---

## Revised Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER'S MACHINE                                │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │  MCP Client  │   │  REST Client │   │ Python SDK   │                │
│  │ (Claude Code │   │ (any app)    │   │ (agent devs) │                │
│  │  Cursor etc) │   │              │   │              │                │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                │
│         │                  │                  │                         │
│         └──────────────────┴──────────────────┘                        │
│                            │                                            │
│                            ▼                                            │
│              ┌─────────────────────────────┐                           │
│              │        Mnemo Core           │                           │
│              │                             │                           │
│              │  query_codebase()           │                           │
│              │  get_session_context()      │                           │
│              │  compress_session()         │                           │
│              │  log_relevance()            │                           │
│              └──────────────┬──────────────┘                           │
│                             │                                           │
│              ┌──────────────┴──────────────┐                           │
│              │                             │                           │
│              ▼                             ▼                           │
│   ┌──────────────────┐         ┌──────────────────────┐               │
│   │   TIER 1         │         │   TIER 2             │               │
│   │   Session Memory │         │   Codebase Retrieval │               │
│   │                  │         │   (optional plugin)  │               │
│   │  session_watcher │         │                      │               │
│   │  compression.py  │         │  FTS5 + FAISS + RRF  │               │
│   │  consolidation.py│◄────────│  2-hop BFS expansion │               │
│   │                  │ community│  Tree-sitter parser  │               │
│   │  snapshot stack  │ tagging  │  lightweight graph   │               │
│   │  snap_001.json   │         │  (3 node, 3 edge)    │               │
│   │  snap_002.json   │         │  Leiden communities  │               │
│   └──────────────────┘         └──────────────────────┘               │
│              │                             │                           │
│              └──────────────┬──────────────┘                           │
│                             │                                           │
│              ┌──────────────▼──────────────┐                           │
│              │         store/              │                           │
│              │                             │                           │
│              │  snapshots/   graph.db      │                           │
│              │  index.faiss  relevance.db  │                           │
│              └─────────────────────────────┘                           │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐                                   │
│  │ code_watcher │   │session_watch │                                   │
│  │ (file saves) │   │ (transcript) │                                   │
│  └──────────────┘   └──────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Revised File Structure

```
Mnemo/
├── core/
│   ├── compression.py        — LLM compression via litellm, community-tagged JSON output
│   ├── consolidation.py      — versioned snapshot stack, graph-aware lazy merge
│   ├── graph.py              — SQLite graph (3 node types, 3 edge types, 2-hop BFS)
│   ├── parser.py             — Tree-sitter chunker, qualified name generation
│   ├── indexer.py            — FTS5 + FAISS + RRF hybrid retrieval
│   ├── communities.py        — Leiden community detection on lightweight graph
│   └── router.py             — model-agnostic compression model selection
│
├── adapters/
│   ├── mcp_adapter.py        — MCP stdio/SSE server (wraps core tools)
│   ├── rest_adapter.py       — FastAPI REST server (wraps same core tools)
│   └── library.py            — Python SDK surface (Session class)
│
├── watcher/
│   ├── code_watcher.py       — file watch → re-index + rebuild lightweight graph
│   └── session_watcher.py    — transcript watch, triggers compression
│
├── store/
│   ├── snapshots/            — snap_001.json, snap_002.json (community-tagged)
│   ├── graph.db              — SQLite: nodes, edges, communities
│   ├── index.faiss           — FAISS HNSW index
│   └── relevance.db          — query/chunk feedback log
│
├── daemon/
│   └── lifecycle.py          — start/stop/restart, PID management
│
└── config/
    └── settings.py           — compression freq, token budgets, top-k,
                                 compression model, adapter selection
```

---

## Token Economics (Revised)

```
WITHOUT Mnemo (turn 200):
┌─────────────────────────────────────────┐
│ prompt = all files + full transcript    │
│ ~50,000–200,000 tokens per turn         │
│ grows linearly, O(n²) attention cost    │
└─────────────────────────────────────────┘

WITH Mnemo original design (turn 200):
┌─────────────────────────────────────────┐
│ session summary (~3,000 tokens flat)    │
│ + top-k chunks (~2,000–4,000 tokens)   │
│ ~5,000–7,000 tokens per turn            │
│ stays flat regardless of session length │
└─────────────────────────────────────────┘

WITH Mnemo upgraded (turn 200, community-filtered):
┌─────────────────────────────────────────┐
│ community-filtered session summary      │
│ (~400–800 tokens, only relevant         │
│  communities for current query)         │
│ + top-k chunks with 2-hop expansion     │
│ (~2,000–3,000 tokens, higher precision) │
│ ~2,400–3,800 tokens per turn            │
│ 35–55% further reduction vs original   │
│ zero information loss for focused query │
└─────────────────────────────────────────┘
```

---

## Summary Table

| Suggestion | Performance gain | Model agnostic | Unique to Mnemo | Complexity |
|---|---|---|---|---|
| Pluggable transport | Larger user base | Yes | Yes | Low (thin adapters) |
| litellm compression router | Quality + availability | Yes | No (standard practice) | Low |
| FTS5 + FAISS + RRF hybrid | +5–15% retrieval quality | Yes | No (proven approach) | Medium |
| 2-hop BFS on lightweight graph | Higher context completeness | Yes | Partial | Medium |
| Community-tagged snapshots | 35–55% token reduction | Yes | **Yes** | Medium |
| Graph-aware snapshot merge | Better long-session quality | Yes | **Yes** | Medium |

The two suggestions marked **Yes** under Unique to Mnemo — community-tagged snapshots and graph-aware snapshot merging — are the architectural features nobody else has shipped. They are the reason to build Mnemo rather than fork an existing tool.
