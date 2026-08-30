"""Tests for the file watcher debouncing and integration."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from mnemostack.core.retrieval.file_watcher import FileWatcher


class TestFileWatcher:
    def test_start_and_stop(self, tmp_path):
        callback = MagicMock()
        watcher = FileWatcher(tmp_path, callback, debounce_ms=50)
        watcher.start()
        assert watcher.is_running
        watcher.stop()
        assert not watcher.is_running

    def test_detects_file_creation(self, tmp_path):
        received: list[set[Path]] = []
        event = threading.Event()

        def on_change(paths):
            received.append(paths)
            event.set()

        watcher = FileWatcher(tmp_path, on_change, debounce_ms=50)
        watcher.start()
        try:
            # Create a Python file
            (tmp_path / "new_file.py").write_text("x = 1\n")
            # Wait for debounce + callback
            assert event.wait(timeout=3), "Callback not fired within timeout"
            assert len(received) >= 1
            paths = received[0]
            assert any("new_file.py" in str(p) for p in paths)
        finally:
            watcher.stop()

    def test_ignores_non_indexable_extensions(self, tmp_path):
        received: list[set[Path]] = []
        event = threading.Event()

        def on_change(paths):
            received.append(paths)
            event.set()

        watcher = FileWatcher(tmp_path, on_change, debounce_ms=50)
        watcher.start()
        try:
            # Create a non-indexable file
            (tmp_path / "image.png").write_bytes(b"\x89PNG")
            # Should NOT trigger
            time.sleep(0.3)
            assert not event.is_set()
        finally:
            watcher.stop()

    def test_ignores_skip_dirs(self, tmp_path):
        received: list[set[Path]] = []
        event = threading.Event()

        def on_change(paths):
            received.append(paths)
            event.set()

        # Create node_modules before starting watcher
        nm = tmp_path / "node_modules"
        nm.mkdir()

        watcher = FileWatcher(tmp_path, on_change, debounce_ms=50)
        watcher.start()
        try:
            (nm / "pkg.js").write_text("module.exports = {}")
            time.sleep(0.3)
            assert not event.is_set()
        finally:
            watcher.stop()

    def test_debounces_rapid_saves(self, tmp_path):
        received: list[set[Path]] = []
        event = threading.Event()

        def on_change(paths):
            received.append(paths)
            event.set()

        watcher = FileWatcher(tmp_path, on_change, debounce_ms=200)
        watcher.start()
        try:
            f = tmp_path / "rapid.py"
            # Rapid saves
            for i in range(5):
                f.write_text(f"x = {i}\n")
                time.sleep(0.02)

            # Wait for single debounced callback
            assert event.wait(timeout=3)
            # Should have coalesced into 1 callback (or at most 2 if timing is unlucky)
            assert len(received) <= 2
        finally:
            watcher.stop()


class TestMultiRootWatching:
    """Several repos are indexed into one graph, so all of them stay watched."""

    def test_watching_a_second_root_keeps_the_first(self, tmp_path, monkeypatch):
        from mnemostack.core.state import _State

        repo_a, repo_b = tmp_path / "a", tmp_path / "b"
        repo_a.mkdir()
        repo_b.mkdir()
        state = _State()
        try:
            state.start_watching(repo_a)
            state.start_watching(repo_b)
            assert state.watched_roots == sorted([repo_a.resolve(), repo_b.resolve()])
            assert state.is_watching
        finally:
            state.stop_watching()
        assert state.watched_roots == []

    def test_nested_root_is_covered_by_its_parent(self, tmp_path):
        from mnemostack.core.state import _State

        parent = tmp_path / "repo"
        child = parent / "pkg"
        child.mkdir(parents=True)
        state = _State()
        try:
            state.start_watching(parent)
            state.start_watching(child)
            assert state.watched_roots == [parent.resolve()]
            # The reverse order collapses to the parent too, not two watchers.
            state.stop_watching()
            state.start_watching(child)
            state.start_watching(parent)
            assert state.watched_roots == [parent.resolve()]
        finally:
            state.stop_watching()
