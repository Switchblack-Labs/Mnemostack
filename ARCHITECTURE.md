# Mnemo — Architecture

## System Overview

Mnemo is a local daemon exposing an MCP server. It maintains two memory tiers for any MCP-compatible coding assistant: a compressed session memory and a fast semantic codebase index. The client (Claude Code, Cursor, etc.) calls MCP tools instead of stuffing raw context into the prompt.

---

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S MACHINE                           │
│                                                                 │
│   ┌─────────────┐   MCP tools    ┌────────────────────────┐    │
│   │  MCP Client │ ◄────────────► │     MCP Server         │    │
│   │ (Claude Code│                │     (server.py)        │    │
│   │  Cursor etc)│                │                        │    │
│   └─────────────┘                │  - query_codebase()    │    │
│          │                       │  - get_session_context()│   │
│          │ writes session        │  - compress_session()  │    │
│          │ transcript            │  - log_relevance()     │    │
│          ▼                       └──────────┬─────────────┘    │
│   ┌─────────────┐                           │                  │
│   │  ~/.session │                           │ reads/writes     │
│   │  transcript │                           │                  │
│   └──────┬──────┘                           ▼                  │
│          │                       ┌────────────────────────┐    │
│          │ watches               │      Memory Store      │    │
│          ▼                       │                        │    │
│   ┌─────────────┐                │  ┌──────────────────┐  │    │
│   │  session_   │ ─── every N ──►│  │ Snapshot Stack   │  │    │
│   │  watcher.py │    turns       │  │ (versioned, not  │  │    │
│   └─────────────┘                │  │  merged blob)    │  │    │
│                                  │  │                  │  │    │
│   ┌─────────────┐                │  │ snap_001.json    │  │    │
│   │  code_      │ ─── on file ──►│  │ snap_002.json    │  │    │
│   │  watcher.py │    change      │  │ snap_003.json    │  │    │
│   └─────────────┘                │  └──────────────────┘  │    │
│                                  │                        │    │
│                                  │  ┌──────────────────┐  │    │
│                                  │  │  FAISS HNSW      │  │    │
│                                  │  │  Index           │  │    │
│                                  │  │  (index.faiss)   │  │    │
│                                  │  └──────────────────┘  │    │
│                                  │                        │    │
│                                  │  ┌──────────────────┐  │    │
│                                  │  │  Relevance Log   │  │    │
│                                  │  │  (feedback)      │  │    │
│                                  │  └──────────────────┘  │    │
│                                  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tier 1 — Compressed Session Memory

### Flow

```
session transcript (N new turns)
        │
        ▼
┌───────────────────┐
│  session_watcher  │  monitors transcript file for new turns
│  (watcher/        │  triggers compression when turn count
│   session_        │  crosses threshold (default: 25 turns)
│   watcher.py)     │
└────────┬──────────┘
         │ threshold crossed
         ▼
┌───────────────────┐
│  compression.py   │  sends transcript slice to LLM
│                   │  structured JSON output schema:
│  prompt: extract  │   {
│  - decisions made │     "decisions": [...],
│  - constraints    │     "constraints": [...],
│  - arch state     │     "open_questions": [...],
│  - open questions │     "file_relationships": [...]
│                   │   }
└────────┬──────────┘
         │ new snapshot
         ▼
┌───────────────────┐
│  consolidation.py │  appends snapshot to stack
│                   │  lazy merge: only merges snapshots
│  snapshot stack:  │  when total token count approaches
│  [s1, s2, s3...]  │  budget ceiling (default: 8k tokens)
│                   │
│  on merge:        │  merged snapshot replaces the two
│  s1 + s2 → s_m   │  oldest — never drops the latest
└────────┬──────────┘
         │ write
         ▼
┌───────────────────┐
│  memory store     │  versioned snapshot files
│  snapshots/       │  snap_001.json, snap_002.json ...
│                   │  latest pointer: latest.json (symlink)
└───────────────────┘
         │
         ▼
  served via get_session_context() MCP tool
  client decides whether to include in prompt
```

### Why versioned snapshots instead of a merged blob

Single merged blob: information loss compounds every cycle. By cycle 5 you've silently dropped nuance you didn't know you needed, with no way to recover it.

Versioned stack: each snapshot is immutable once written. Merge only happens when you're near the token budget, and only the two oldest snapshots merge — the recent ones stay intact. You can inspect history and roll back if something critical got dropped.

---

## Tier 2 — Semantic Codebase Retrieval

### Indexing Flow

```
codebase files
      │
      ▼
┌─────────────────┐
│  code_watcher   │  watches filesystem for file saves
│  (watcher/      │  debounces rapid saves (500ms window)
│   code_         │  queues changed files for re-indexing
│   watcher.py)   │
└────────┬────────┘
         │ file changed
         ▼
┌─────────────────┐
│  chunker.py     │  AST-aware chunking via code-chunk lib
│                 │  splits by function/class boundaries
│  per chunk:     │  not arbitrary token windows
│  - code text    │
│  - file path    │  also builds lightweight import graph:
│  - symbol name  │  for each chunk, tracks which files
│  - line range   │  it imports from
│  - imports[]    │
└────────┬────────┘
         │ chunks + import graph
         ▼
┌─────────────────┐
│  indexer.py     │  embeds each chunk (local model or API)
│                 │  upserts into FAISS HNSW index
│  FAISS HNSW:    │  stores chunk metadata in sqlite sidecar
│  O(log n)       │  (file path, symbol, line range, imports)
│  approximate    │
│  nearest        │
│  neighbor       │
└────────┬────────┘
         │
         ▼
   index.faiss + chunks.db
```

### Query Flow

```
user query (from MCP client)
        │
        ▼
┌───────────────────┐
│  query_codebase() │  MCP tool handler
│  (mcp/tools.py)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  indexer.py       │  embed query
│                   │  FAISS HNSW search → top-k chunks
│  O(log n)         │  look up each chunk's imports[]
└────────┬──────────┘
         │ top-k chunks + their imported files
         ▼
┌───────────────────┐
│  dependency       │  for each retrieved chunk:
│  expansion        │  fetch any directly imported chunks
│                   │  from chunks.db (one hop only)
│  auth.py chunk ──►│  this surfaces config.py constants
│  imports config   │  that auth.py depends on, which
│                   │  pure vector search would miss
└────────┬──────────┘
         │ expanded chunk set
         ▼
┌───────────────────┐
│  relevance log    │  records: query, chunks injected,
│  (feedback)       │  timestamp, session_id
│                   │  used later to tune top-k and
│                   │  chunking strategy
└────────┬──────────┘
         │
         ▼
  return chunks to MCP client
  client injects into prompt
```

---

## MCP Server — Tool Surface

```
┌────────────────────────────────────────────────────┐
│                   MCP Server                       │
│                  (mcp/server.py)                   │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │ query_codebase(query: str, top_k: int = 5)   │  │
│  │                                              │  │
│  │  → FAISS search + dependency expansion       │  │
│  │  → returns: [{code, file, symbol, lines}]    │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │ get_session_context()                        │  │
│  │                                              │  │
│  │  → reads latest snapshot from stack         │  │
│  │  → returns: structured JSON summary         │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │ compress_session(transcript: str)            │  │
│  │                                              │  │
│  │  → manual trigger for compression           │  │
│  │  → useful when user wants to force a        │  │
│  │    snapshot before a big refactor           │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │ log_relevance(query_id: str, helpful: bool)  │  │
│  │                                              │  │
│  │  → feedback signal from client              │  │
│  │  → written to relevance log                 │  │
│  │  → used to tune top-k and chunk strategy    │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  transport: stdio (local)                          │
└────────────────────────────────────────────────────┘
```

---

## File Structure

```
Mnemo/
├── core/
│   ├── compression.py      — LLM compression, structured JSON output
│   ├── consolidation.py    — versioned snapshot stack, lazy merge
│   ├── indexer.py          — FAISS HNSW index management + embedding
│   └── chunker.py          — AST-aware chunking + import graph
│
├── mcp/
│   ├── server.py           — MCP server init, transport (stdio)
│   └── tools.py            — tool handlers: query, context, compress, feedback
│
├── watcher/
│   ├── code_watcher.py     — filesystem watcher for code changes
│   └── session_watcher.py  — transcript watcher, triggers compression
│
├── store/
│   ├── snapshots/          — snap_001.json, snap_002.json ...
│   ├── index.faiss         — FAISS HNSW index
│   ├── chunks.db           — sqlite: chunk metadata + import graph
│   └── relevance.db        — sqlite: query/chunk feedback log
│
├── daemon/
│   └── lifecycle.py        — start/stop/restart, PID management
│
└── config/
    └── settings.py         — compression freq, token budgets, top-k, model
```

---

## Token Economics

```
WITHOUT Mnemo (turn 200):
  ┌─────────────────────────────────────────┐
  │ prompt = all files + full transcript    │
  │ ~50,000-200,000 tokens per turn         │
  │ grows linearly every turn               │
  │ O(n²) attention cost                    │
  └─────────────────────────────────────────┘

WITH Mnemo (turn 200):
  ┌─────────────────────────────────────────┐
  │ prompt = session summary (~3k tokens)   │
  │        + top-k chunks (~2-4k tokens)    │
  │ ~5,000-7,000 tokens per turn            │
  │ stays flat regardless of session length │
  │ O(k²) attention where k << n            │
  └─────────────────────────────────────────┘

  one-time compression cost: ~10k tokens per cycle
  saves: 40k-190k tokens on every subsequent turn
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Memory format | Versioned snapshot stack | Prevents silent info loss from recursive merge |
| Retrieval | FAISS HNSW | O(log n) vs O(n) for Chroma/naive cosine |
| Chunking | AST boundaries | Retrieves complete functions/classes, not arbitrary slices |
| Dependency expansion | One-hop import graph | Catches cross-file relationships without full call graph overhead |
| Compression output | Structured JSON schema | Prevents LLM from dropping fields, easier to inspect/debug |
| Feedback | Relevance log | Enables tuning top-k and compression frequency over time |
| Dev LLM | Ollama (Llama 3 8B) | Free, good enough for compression logic, no API credits burned |
| Transport | stdio | Microsecond overhead, no network stack needed for local use |
