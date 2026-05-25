"""File system watcher for incremental index updates.

Monitors a project directory for file changes (create/modify/delete).
Debounces rapid saves and queues changed files for re-indexing.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from mnemostack.config.settings import settings
from mnemostack.core.retrieval.ast_chunker import SUPPORTED_EXTENSIONS

# File extensions we care about indexing (AST-supported + common text files)
_INDEXABLE_EXTENSIONS = SUPPORTED_EXTENSIONS | {
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".scala",
    ".yaml", ".yml", ".toml", ".json", ".md", ".txt",
}

# Directories to always skip
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "target", ".idea", ".vscode",
})


class _DebouncedHandler(FileSystemEventHandler):
    """Debounces rapid file events and calls the callback after quiet period."""

    def __init__(
        self,
        callback: Callable[[set[Path]], None],
        debounce_ms: int,
        watch_root: Path,
    ):
        super().__init__()
        self._callback = callback
        self._debounce_s = debounce_ms / 1000.0
        self._watch_root = watch_root
        self._pending: set[Path] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Skip non-indexable files
        if path.suffix.lower() not in _INDEXABLE_EXTENSIONS:
            return

        # Skip files in ignored directories (check relative to watch root only)
        try:
            rel_parts = path.relative_to(self._watch_root).parts
        except ValueError:
            return
        if any(part in _SKIP_DIRS for part in rel_parts):
            return

        with self._lock:
            self._pending.add(path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_s, self._flush)
            self._timer.start()

    def cancel(self) -> None:
        """Cancel any pending debounce timer and discard queued paths."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()

    def _flush(self) -> None:
        with self._lock:
            paths = self._pending.copy()
            self._pending.clear()
            self._timer = None

        if paths:
            self._callback(paths)


class FileWatcher:
    """Watches a directory tree for file changes and triggers re-indexing."""

    def __init__(
        self,
        watch_dir: Path,
        on_files_changed: Callable[[set[Path]], None],
        debounce_ms: int | None = None,
    ):
        self._watch_dir = watch_dir
        self._debounce_ms = (
            debounce_ms if debounce_ms is not None
            else settings.retrieval.file_watch_debounce_ms
        )
        self._handler = _DebouncedHandler(on_files_changed, self._debounce_ms, watch_dir)
        self._observer: Observer | None = None

    def start(self) -> None:
        """Start watching the directory (non-blocking, runs in background thread)."""
        if self._observer is not None:
            return
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self._watch_dir), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        """Stop watching and cancel any pending debounce timer."""
        # Cancel pending timer first to prevent callback firing after stop
        self._handler.cancel()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    @property
    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()
