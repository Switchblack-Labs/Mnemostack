"""FTS5 full-text keyword index for exact identifier search.

Uses SQLite FTS5 with BM25 scoring and Porter stemming. Complements the FAISS
semantic index — exact identifier names (e.g. `validate_token`, `AuthService`)
are best matched by BM25, not cosine similarity.

IMPORTANT: This module operates on the same chunks.db as faiss_index.py.
Both share the `chunks` table as the source of truth for IDs. The FTS5 virtual
table is a content-sync index over that shared table. Both MUST use the same
SQLite connection to avoid data visibility races.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mnemostack.config.settings import settings
from mnemostack.core.retrieval.faiss_index import _db_lock, create_chunks_db


class FTSResult:
    __slots__ = ("chunk_id", "bm25_score", "file_path", "symbol_name",
                 "qualified_name", "code", "line_start", "line_end",
                 "chunk_type", "last_modified", "dependencies")

    def __init__(
        self,
        chunk_id: int,
        bm25_score: float,
        file_path: str,
        symbol_name: str,
        qualified_name: str,
        code: str,
        line_start: int,
        line_end: int,
        chunk_type: str = "",
        last_modified: float = 0.0,
        dependencies: list[str] | None = None,
    ):
        self.chunk_id = chunk_id
        self.bm25_score = bm25_score
        self.file_path = file_path
        self.symbol_name = symbol_name
        self.qualified_name = qualified_name
        self.code = code
        self.line_start = line_start
        self.line_end = line_end
        self.chunk_type = chunk_type
        self.last_modified = last_modified
        self.dependencies = dependencies if dependencies is not None else []


class FTSIndex:
    """SQLite FTS5 keyword index operating on the shared chunks.db.

    Uses content-sync with the `chunks` table from faiss_index so both
    indexes share the same chunk IDs. RRF fusion in the ranker relies on
    matching IDs between FAISS and FTS results.

    Must receive the same SQLite connection as FaissIndex to guarantee
    data visibility after writes.
    """

    def __init__(self, store_dir: Path | None = None, db: sqlite3.Connection | None = None):
        self._store_dir = store_dir or settings.store.base_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._db = db
        self._owns_db = db is None
        self._init_schema()

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = create_chunks_db(self._store_dir)
            self._owns_db = True
            self._init_schema()
        return self._db

    def _init_schema(self) -> None:
        """Create FTS5 virtual table over the shared chunks table."""
        with _db_lock:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    symbol_name TEXT NOT NULL,
                    code TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    chunk_type TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    last_modified REAL NOT NULL,
                    dependencies TEXT NOT NULL DEFAULT '[]',
                    embedding BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_qname ON chunks(qualified_name);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    symbol_name,
                    qualified_name,
                    code,
                    content=chunks,
                    content_rowid=id,
                    tokenize='porter unicode61'
                );
            """)

    def sync_file(self, file_path: str) -> None:
        """Sync FTS index for all chunks belonging to a file.

        Call after FaissIndex.add() or remove_by_file() to keep FTS in sync.
        """
        self._rebuild_fts_for_file(file_path)

    def sync_added(self, chunk_ids: list[int]) -> None:
        """Add FTS entries for newly added chunks (by ID from shared chunks table)."""
        if not chunk_ids:
            return
        with _db_lock:
            placeholders = ",".join("?" * len(chunk_ids))
            rows = self.db.execute(
                f"""SELECT id, symbol_name, qualified_name, code
                    FROM chunks WHERE id IN ({placeholders})""",
                chunk_ids,
            ).fetchall()

            cursor = self.db.cursor()
            for row_id, symbol, qname, code in rows:
                cursor.execute(
                    """INSERT INTO chunks_fts(rowid, symbol_name, qualified_name, code)
                       VALUES (?, ?, ?, ?)""",
                    (row_id, symbol, qname, code),
                )
            self.db.commit()

    def sync_removed(self, file_path: str) -> None:
        """Remove FTS entries for chunks belonging to *file_path*.

        Preferred ordering:  call **before** FaissIndex.remove_by_file() so the
        chunk rows still exist and precise FTS delete commands can be issued.

        If the chunks have already been deleted (wrong ordering), this method
        falls back to a full FTS rebuild to guarantee consistency rather than
        silently leaving ghost entries in the inverted index.
        """
        with _db_lock:
            rows = self.db.execute(
                "SELECT id, symbol_name, qualified_name, code FROM chunks WHERE file_path = ?",
                (file_path,),
            ).fetchall()

            if not rows:
                # Chunks already gone — cannot issue precise FTS delete commands.
                # Fall back to a full rebuild so stale entries don't persist.
                self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
                self.db.commit()
                return

            cursor = self.db.cursor()
            for row_id, symbol, qname, code in rows:
                cursor.execute(
                    """INSERT INTO chunks_fts(chunks_fts, rowid, symbol_name, qualified_name, code)
                       VALUES ('delete', ?, ?, ?, ?)""",
                    (row_id, symbol, qname, code),
                )
            self.db.commit()

    def _rebuild_fts_for_file(self, file_path: str) -> None:
        """Rebuild FTS entries for a file by rebuilding the entire FTS index."""
        self.rebuild_all()

    def rebuild_all(self) -> None:
        """Rebuild the entire FTS index from the chunks table."""
        with _db_lock:
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            self.db.commit()

    def search(self, query: str, top_k: int = 10) -> list[FTSResult]:
        """Search using BM25 ranking."""
        if not query.strip():
            return []

        safe_query = _sanitize_fts_query(query)

        try:
            with _db_lock:
                rows = self.db.execute(
                    """SELECT chunks.id, bm25(chunks_fts) as score,
                              chunks.file_path, chunks.symbol_name,
                              chunks.qualified_name, chunks.code,
                              chunks.line_start, chunks.line_end,
                              chunks.chunk_type, chunks.last_modified,
                              chunks.dependencies
                       FROM chunks_fts
                       JOIN chunks ON chunks_fts.rowid = chunks.id
                       WHERE chunks_fts MATCH ?
                       ORDER BY score
                       LIMIT ?""",
                    (safe_query, top_k),
                ).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            FTSResult(
                chunk_id=row[0],
                bm25_score=-row[1],  # FTS5 bm25() returns negative (lower=better)
                file_path=row[2],
                symbol_name=row[3],
                qualified_name=row[4],
                code=row[5],
                line_start=row[6],
                line_end=row[7],
                chunk_type=row[8],
                last_modified=row[9],
                dependencies=json.loads(row[10]),
            )
            for row in rows
        ]

    @property
    def total_chunks(self) -> int:
        with _db_lock:
            row = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        if self._db and self._owns_db:
            self._db.close()
            self._db = None


def _sanitize_fts_query(query: str) -> str:
    """Convert a raw user query into a safe FTS5 query.

    Strips existing quotes, then wraps individual tokens in quotes to prevent
    FTS5 syntax errors from special characters in code identifiers.
    """
    # Strip existing quotes to avoid double-quoting
    cleaned = query.replace('"', " ")
    tokens = cleaned.split()
    if not tokens:
        return query
    return " ".join(f'"{t}"' for t in tokens)
