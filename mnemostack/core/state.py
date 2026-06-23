"""Shared runtime state for the MCP server.

Holds singleton instances of FAISS, FTS5, and CallGraph indexes.
Initialized lazily on first access so import doesn't trigger I/O.
FAISS and FTS share a single SQLite connection for data visibility.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from mnemostack.config.settings import settings
from mnemostack.core.retrieval.call_graph import CallGraph
from mnemostack.core.retrieval.faiss_index import FaissIndex, create_chunks_db
from mnemostack.core.retrieval.file_watcher import FileWatcher
from mnemostack.core.retrieval.fts_index import FTSIndex

_init_lock = threading.RLock()
log = logging.getLogger(__name__)


class _State:
    def __init__(self) -> None:
        self._chunks_db: sqlite3.Connection | None = None
        self._faiss: FaissIndex | None = None
        self._fts: FTSIndex | None = None
        self._graph: CallGraph | None = None
        self._watcher: FileWatcher | None = None

    @property
    def chunks_db(self) -> sqlite3.Connection:
        """Shared SQLite connection for FAISS and FTS indexes."""
        if self._chunks_db is None:
            with _init_lock:
                if self._chunks_db is None:
                    self._chunks_db = create_chunks_db(settings.store.base_dir)
        return self._chunks_db

    @property
    def faiss(self) -> FaissIndex:
        if self._faiss is None:
            with _init_lock:
                if self._faiss is None:
                    self._faiss = FaissIndex(db=self.chunks_db)
        return self._faiss

    @property
    def fts(self) -> FTSIndex:
        if self._fts is None:
            with _init_lock:
                if self._fts is None:
                    self._fts = FTSIndex(db=self.chunks_db)
        return self._fts

    @property
    def graph(self) -> CallGraph:
        if self._graph is None:
            with _init_lock:
                if self._graph is None:
                    self._graph = CallGraph()
        return self._graph

    def start_watching(self, root: Path) -> None:
        """Start the file watcher for incremental re-indexing."""
        self.stop_watching()

        from mnemostack.core.retrieval.indexer import reindex_file

        def _on_files_changed(paths: set[Path]) -> None:
            for path in paths:
                try:
                    reindex_file(path, self.faiss, self.fts, self.graph)
                except Exception:
                    log.exception("Failed to re-index %s", path)

        self._watcher = FileWatcher(root, _on_files_changed)
        self._watcher.start()
        log.info("File watcher started for %s", root)

    def stop_watching(self) -> None:
        """Stop the file watcher if running."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    @property
    def is_watching(self) -> bool:
        return self._watcher is not None and self._watcher.is_running

    def close(self) -> None:
        self.stop_watching()
        if self._faiss is not None:
            self._faiss.close()
            self._faiss = None
        if self._fts is not None:
            self._fts.close()
            self._fts = None
        if self._graph is not None:
            self._graph.close()
            self._graph = None
        if self._chunks_db is not None:
            self._chunks_db.close()
            self._chunks_db = None


state = _State()
