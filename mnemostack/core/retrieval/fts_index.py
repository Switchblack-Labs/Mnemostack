"""FTS5 full-text keyword index for exact identifier search.

Uses SQLite FTS5 with BM25 scoring and Porter stemming. Complements the FAISS
semantic index — exact identifier names (e.g. `validate_token`, `AuthService`)
are best matched by BM25, not cosine similarity.

IMPORTANT: This module operates on the same chunks.db as faiss_index.py.
Both share the `chunks` table as the source of truth for IDs. The FTS5 virtual
table is a content-sync index over that shared table.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mnemostack.config.settings import settings


class FTSResult:
    __slots__ = ("chunk_id", "bm25_score", "file_path", "symbol_name",
                 "qualified_name", "code", "line_start", "line_end",
                 "chunk_type", "last_modified")

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


class FTSIndex:
    """SQLite FTS5 keyword index operating on the shared chunks.db.

    Uses content-sync with the `chunks` table from faiss_index so both
    indexes share the same chunk IDs. RRF fusion in the ranker relies on
    matching IDs between FAISS and FTS results.
    """

    def __init__(self, store_dir: Path | None = None):
        self._store_dir = store_dir or settings.store.base_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._store_dir / "chunks.db"
        self._db: sqlite3.Connection | None = None

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(str(self._db_path))
            self._db.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
        return self._db

    def _init_schema(self) -> None:
        """Create FTS5 virtual table over the shared chunks table.

        The chunks table is created by FaissIndex. If it doesn't exist yet
        (FTS initialized first), we create it here too.
        """
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
                embedding BLOB
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
        """Remove FTS entries for chunks that were deleted from the shared table.

        Call BEFORE deleting from chunks table (need the data for FTS delete command).
        """
        rows = self.db.execute(
            "SELECT id, symbol_name, qualified_name, code FROM chunks WHERE file_path = ?",
            (file_path,),
        ).fetchall()
        if not rows:
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
        """Rebuild FTS entries for all chunks of a file."""
        rows = self.db.execute(
            "SELECT id, symbol_name, qualified_name, code FROM chunks WHERE file_path = ?",
            (file_path,),
        ).fetchall()
        cursor = self.db.cursor()
        for row_id, symbol, qname, code in rows:
            cursor.execute(
                """INSERT OR REPLACE INTO chunks_fts(rowid, symbol_name, qualified_name, code)
                   VALUES (?, ?, ?, ?)""",
                (row_id, symbol, qname, code),
            )
        self.db.commit()

    def rebuild_all(self) -> None:
        """Rebuild the entire FTS index from the chunks table."""
        self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        self.db.commit()

    def search(self, query: str, top_k: int = 10) -> list[FTSResult]:
        """Search using BM25 ranking.

        Args:
            query: Search query string.
            top_k: Maximum results to return.

        Returns:
            List of FTSResult ordered by BM25 relevance (best first).
        """
        if not query.strip():
            return []

        safe_query = _sanitize_fts_query(query)

        try:
            rows = self.db.execute(
                """SELECT chunks.id, bm25(chunks_fts) as score,
                          chunks.file_path, chunks.symbol_name,
                          chunks.qualified_name, chunks.code,
                          chunks.line_start, chunks.line_end,
                          chunks.chunk_type, chunks.last_modified
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
            )
            for row in rows
        ]

    @property
    def total_chunks(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None


def _sanitize_fts_query(query: str) -> str:
    """Convert a raw user query into a safe FTS5 query.

    Wraps individual tokens in quotes to prevent FTS5 syntax errors from
    special characters like colons, dots, or parentheses in code identifiers.
    """
    tokens = query.split()
    if not tokens:
        return query
    return " ".join(f'"{t}"' for t in tokens)
