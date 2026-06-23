"""FAISS HNSW index for semantic code search.

Manages embedding storage, upsert, deletion, and approximate nearest-neighbor queries.
Chunk metadata and embeddings are stored in SQLite (chunks.db). The HNSW index is
built in-memory from stored embeddings and rebuilt on removal (HNSW is append-only).

Design decision: HNSW rebuild on removal vs IVFFlat (supports removal natively).
HNSW has better recall at equivalent speed. Rebuild cost is ~1-2s for 10k chunks,
acceptable since removals are debounced by the file watcher (500ms). Alternatives
considered: IndexIVFFlat (needs training, lower recall), IndexFlatL2 (O(n) search).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from mnemostack.config.settings import settings
from mnemostack.core.retrieval.ast_chunker import Chunk

# --- Types ---


class SearchResult:
    """A single FAISS search result.

    Note: `score` is L2 distance (lower = more similar). Results are returned
    in ascending distance order (most similar first). The ranker uses rank
    position via RRF, not raw score values, so the L2 convention is safe.
    """

    __slots__ = ("chunk_id", "score", "file_path", "symbol_name", "code",
                 "line_start", "line_end", "chunk_type", "qualified_name",
                 "last_modified", "dependencies")

    def __init__(
        self,
        chunk_id: int,
        score: float,
        file_path: str,
        symbol_name: str,
        code: str,
        line_start: int,
        line_end: int,
        chunk_type: str,
        qualified_name: str,
        last_modified: float,
        dependencies: list[str],
    ):
        self.chunk_id = chunk_id
        self.score = score
        self.file_path = file_path
        self.symbol_name = symbol_name
        self.code = code
        self.line_start = line_start
        self.line_end = line_end
        self.chunk_type = chunk_type
        self.qualified_name = qualified_name
        self.last_modified = last_modified
        self.dependencies = dependencies


# --- Shared DB connection ---


def create_chunks_db(store_dir: Path) -> sqlite3.Connection:
    """Create or open the shared chunks.db with thread-safe settings.

    Both FaissIndex and FTSIndex must use the same connection to avoid
    data visibility races between separate WAL readers.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    db_path = store_dir / "chunks.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# Thread lock for all operations on the shared chunks.db connection AND
# the in-memory FAISS index. Required because check_same_thread=False
# disables SQLite's thread check but doesn't make concurrent access safe,
# and FAISS itself is not thread-safe for concurrent read/write.
_db_lock = threading.Lock()


# --- Index Manager ---


class FaissIndex:
    """Manages a FAISS HNSW index with a SQLite metadata store.

    Embeddings are persisted in SQLite so the HNSW index can be rebuilt on
    removal (HNSW doesn't support element deletion). The in-memory index is
    the search target; SQLite is the source of truth for persistence.
    """

    def __init__(
        self,
        store_dir: Path | None = None,
        dimension: int | None = None,
        db: sqlite3.Connection | None = None,
    ):
        self._store_dir = store_dir or settings.store.base_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._dimension = dimension
        self._index_path = self._store_dir / "index.faiss"
        self._db_path = self._store_dir / "chunks.db"
        self._index: faiss.IndexIDMap | None = None
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

    @property
    def index(self) -> faiss.IndexIDMap:
        if self._index is None:
            if self._index_path.exists():
                self._index = faiss.read_index(str(self._index_path))
                # Auto-detect dimension from loaded index
                if self._dimension is None:
                    self._dimension = self._index.d
            else:
                if self._dimension is None:
                    self._dimension = 768
                self._index = self._build_fresh_index()
        return self._index

    def _build_fresh_index(self) -> faiss.IndexIDMap:
        """Create a new empty HNSW index."""
        assert self._dimension is not None
        hnsw = faiss.IndexHNSWFlat(self._dimension, settings.retrieval.faiss_m)
        hnsw.hnsw.efConstruction = settings.retrieval.faiss_ef_construction
        hnsw.hnsw.efSearch = settings.retrieval.faiss_ef_search
        return faiss.IndexIDMap(hnsw)

    def _rebuild_index(self) -> None:
        """Rebuild the HNSW index from all embeddings stored in SQLite.

        Called after removal since HNSW is append-only and doesn't support deletion.
        Must be called while holding _db_lock.
        """
        rows = self.db.execute("SELECT id, embedding FROM chunks").fetchall()

        if not rows:
            if self._dimension is None:
                self._dimension = 768
            self._index = self._build_fresh_index()
            return

        ids = np.array([row[0] for row in rows], dtype=np.int64)
        embeddings = np.array(
            [np.frombuffer(row[1], dtype=np.float32) for row in rows],
            dtype=np.float32,
        )
        # Auto-detect dimension from stored embeddings
        self._dimension = embeddings.shape[1]
        self._index = self._build_fresh_index()
        self._index.add_with_ids(embeddings, ids)

    def _init_schema(self) -> None:
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
            """)

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> list[int]:
        """Add chunks and their embeddings to the index.

        Args:
            chunks: Code chunks to index.
            embeddings: numpy array of shape (len(chunks), dimension).

        Returns:
            List of assigned chunk IDs.
        """
        if len(chunks) == 0:
            return []

        # Auto-detect dimension from first embedding batch
        embed_dim = embeddings.shape[1]
        if self._dimension is None:
            self._dimension = embed_dim
            # Rebuild index with correct dimension if needed
            if self._index is not None and self._index.d != embed_dim:
                self._index = self._build_fresh_index()
            elif self._index is None:
                self._index = self._build_fresh_index()

        if embeddings.shape != (len(chunks), self._dimension):
            raise ValueError(
                f"Expected embeddings shape ({len(chunks)}, {self._dimension}), "
                f"got {embeddings.shape}"
            )

        embeddings = embeddings.astype(np.float32)
        ids: list[int] = []
        with _db_lock:
            cursor = self.db.cursor()
            for i, chunk in enumerate(chunks):
                cursor.execute(
                    """INSERT INTO chunks
                       (file_path, symbol_name, code, line_start, line_end,
                        chunk_type, qualified_name, last_modified, dependencies, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.file_path,
                        chunk.symbol_name,
                        chunk.code,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.chunk_type.value,
                        chunk.qualified_name,
                        chunk.last_modified,
                        json.dumps(chunk.dependencies),
                        embeddings[i].tobytes(),
                    ),
                )
                ids.append(cursor.lastrowid)
            self.db.commit()
            id_array = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(embeddings, id_array)
        return ids

    def remove_by_file(self, file_path: str) -> int:
        """Remove all chunks for a given file path. Rebuilds and persists HNSW index.

        Returns number of chunks removed.
        """
        with _db_lock:
            count = self.db.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_path = ?", (file_path,)
            ).fetchone()[0]
            if count == 0:
                return 0

            self.db.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            self.db.commit()
            # Rebuild while holding lock so no concurrent search/add races
            self._rebuild_index()

        self.save()
        return count

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Search for nearest neighbors.

        Args:
            query_embedding: numpy array of shape (dimension,) or (1, dimension).
            top_k: Number of results to return.

        Returns:
            List of SearchResult ordered by similarity (closest first).
        """
        with _db_lock:
            if self.index.ntotal == 0:
                return []

            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            k = min(top_k, self.index.ntotal)
            distances, ids = self.index.search(query_embedding.astype(np.float32), k)

            results: list[SearchResult] = []
            for dist, chunk_id in zip(distances[0], ids[0]):
                if chunk_id == -1:
                    continue
                row = self.db.execute(
                    """SELECT file_path, symbol_name, code, line_start, line_end,
                              chunk_type, qualified_name, last_modified, dependencies
                       FROM chunks WHERE id = ?""",
                    (int(chunk_id),),
                ).fetchone()
                if row is None:
                    continue

                results.append(SearchResult(
                    chunk_id=int(chunk_id),
                    score=float(dist),
                    file_path=row[0],
                    symbol_name=row[1],
                    code=row[2],
                    line_start=row[3],
                    line_end=row[4],
                    chunk_type=row[5],
                    qualified_name=row[6],
                    last_modified=row[7],
                    dependencies=json.loads(row[8]),
                ))
        return results

    def get_chunk_ids_by_qnames(self, qualified_names: list[str]) -> dict[str, int]:
        """Resolve qualified names to chunk IDs.

        Returns mapping of qualified_name -> chunk_id for found entries.
        """
        if not qualified_names:
            return {}
        with _db_lock:
            placeholders = ",".join("?" * len(qualified_names))
            rows = self.db.execute(
                f"SELECT qualified_name, id FROM chunks WHERE qualified_name IN ({placeholders})",
                qualified_names,
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_chunk_by_id(self, chunk_id: int) -> SearchResult | None:
        """Retrieve chunk metadata by ID."""
        with _db_lock:
            row = self.db.execute(
                """SELECT file_path, symbol_name, code, line_start, line_end,
                          chunk_type, qualified_name, last_modified, dependencies
                   FROM chunks WHERE id = ?""",
                (int(chunk_id),),
            ).fetchone()
        if row is None:
            return None
        return SearchResult(
            chunk_id=chunk_id,
            score=0.0,
            file_path=row[0],
            symbol_name=row[1],
            code=row[2],
            line_start=row[3],
            line_end=row[4],
            chunk_type=row[5],
            qualified_name=row[6],
            last_modified=row[7],
            dependencies=json.loads(row[8]),
        )

    def get_chunks_by_file(self, file_path: str) -> list[SearchResult]:
        """Get all chunks for a file path."""
        with _db_lock:
            rows = self.db.execute(
                """SELECT id, file_path, symbol_name, code, line_start, line_end,
                          chunk_type, qualified_name, last_modified, dependencies
                   FROM chunks WHERE file_path = ?""",
                (file_path,),
            ).fetchall()
        return [
            SearchResult(
                chunk_id=row[0],
                score=0.0,
                file_path=row[1],
                symbol_name=row[2],
                code=row[3],
                line_start=row[4],
                line_end=row[5],
                chunk_type=row[6],
                qualified_name=row[7],
                last_modified=row[8],
                dependencies=json.loads(row[9]),
            )
            for row in rows
        ]

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently indexed."""
        with _db_lock:
            row = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        with _db_lock:
            faiss.write_index(self.index, str(self._index_path))

    def close(self) -> None:
        """Close database connection and persist index."""
        if self._index is not None:
            self.save()
        if self._db and self._owns_db:
            self._db.close()
            self._db = None
