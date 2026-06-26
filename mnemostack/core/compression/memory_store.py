"""Session memory store.

Holds the compressed session state on disk as a single JSON file:
  - snapshot: the latest LLM consolidation (SessionMemory-shaped dict)
  - pinned: manually-added facts (e.g. constraints) that never get evicted
  - pending: raw turns recorded since the last consolidation
  - last_consolidation_turn / turn_count / consolidation_count: bookkeeping

Design invariant: the snapshot is replaced wholesale by each consolidation and
pending is evicted at that point. We never re-compress a previous snapshot —
the consolidator carries it forward, pinned facts are always preserved.

ponytail: one JSON file, save-on-write. The memory budget is ~3k tokens, so this
stays tiny — swap to SQLite only if it measurably gets slow.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path

from mnemostack.config.settings import settings

log = logging.getLogger(__name__)

_SESSION_FIELDS = ("decisions", "constraints", "open_questions", "architecture_state", "resolved")


def _empty_snapshot() -> dict:
    return {
        "decisions": [],
        "constraints": [],
        "open_questions": [],
        "architecture_state": {},
        "resolved": [],
    }


def _empty_data() -> dict:
    return {
        "snapshot": _empty_snapshot(),
        "pinned": [],
        "pending": [],
        "last_consolidation_turn": None,
        "turn_count": 0,
        "consolidation_count": 0,
    }


class MemoryStore:
    def __init__(self, store_dir: Path | None = None) -> None:
        base = store_dir if store_dir is not None else settings.store.base_dir
        self._path = Path(base) / "memory.json"
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return _empty_data()
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt/partial file: don't brick the whole memory tier. Move it
            # aside for forensics and start fresh.
            bad = self._path.with_suffix(".json.corrupt")
            log.warning("memory.json unreadable; moving to %s and starting fresh", bad)
            self._path.replace(bad)
            return _empty_data()
        # Normalize missing keys/fields so older or hand-edited files don't KeyError.
        data = _empty_data()
        data.update(loaded)
        # Clamp to known fields so a hand-edited/stray key can't leak into the
        # SessionMemory wire model (extra="forbid").
        loaded_snap = data.get("snapshot", {})
        data["snapshot"] = {f: loaded_snap.get(f, _empty_snapshot()[f]) for f in _SESSION_FIELDS}
        return data

    def _save(self) -> None:
        # Atomic write: tmp file in same dir + os.replace, so a crash mid-write
        # can't truncate the live file.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def add_turn(self, text: str) -> int:
        """Append a raw conversation turn. Returns the new total turn count."""
        if not text:
            raise ValueError("turn text must not be empty")
        with self._lock:
            self._data["pending"].append(text)
            self._data["turn_count"] += 1
            self._save()
            return self._data["turn_count"]

    def add_pinned(self, kind: str, text: str) -> None:
        """Pin a fact (e.g. a constraint) so it survives every consolidation."""
        if not text:
            raise ValueError("pinned text must not be empty")
        with self._lock:
            self._data["pinned"].append({"kind": kind, "text": text})
            self._save()

    def pending(self) -> list[str]:
        with self._lock:
            return list(self._data["pending"])

    def pending_count(self) -> int:
        with self._lock:
            return len(self._data["pending"])

    def turn_count(self) -> int:
        with self._lock:
            return self._data["turn_count"]

    def snapshot(self) -> dict:
        # deepcopy: returned dict is the caller's; mutating it must not touch the store.
        with self._lock:
            return copy.deepcopy(self._data["snapshot"])

    def pinned(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._data["pinned"])

    def set_snapshot(self, snapshot: dict, turn: int, evict_count: int | None = None) -> None:
        """Replace the snapshot with a fresh consolidation and evict pending.

        evict_count: drop only the oldest N pending turns (those the caller
        consolidated), preserving any appended concurrently during the LLM call.
        None clears all pending.
        """
        with self._lock:
            self._data["snapshot"] = {
                f: snapshot.get(f, _empty_snapshot()[f]) for f in _SESSION_FIELDS
            }
            if evict_count is None:
                self._data["pending"] = []
            else:
                self._data["pending"] = self._data["pending"][evict_count:]
            self._data["last_consolidation_turn"] = turn
            self._data["consolidation_count"] += 1
            self._save()

    def session_view(self) -> dict:
        """Kwargs for a SessionMemory: snapshot + pinned constraints + pending count."""
        with self._lock:
            view = copy.deepcopy(self._data["snapshot"])
            pinned_constraints = [p for p in self._data["pinned"] if p.get("kind") == "constraint"]
            view["constraints"] = view["constraints"] + pinned_constraints
            view["local_extractions_pending"] = len(self._data["pending"])
            view["last_consolidation_turn"] = self._data["last_consolidation_turn"]
            return view

    def stats(self) -> dict:
        # ponytail: len/4 token estimate, good enough for a budget heuristic.
        with self._lock:
            blob = json.dumps({"snapshot": self._data["snapshot"], "pinned": self._data["pinned"]})
            return {
                "snapshot_count": self._data["consolidation_count"],
                "total_memory_tokens": len(blob) // 4,
                "last_consolidation_turn": self._data["last_consolidation_turn"],
            }


if __name__ == "__main__":  # smoke check
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        s = MemoryStore(store_dir=Path(d))
        assert s.add_turn("hello") == 1
        assert s.add_turn("world") == 2
        s.add_pinned("constraint", "use postgres")
        assert s.pending_count() == 2

        # reload from disk survives
        s2 = MemoryStore(store_dir=Path(d))
        assert s2.pending_count() == 2
        assert s2.session_view()["constraints"] == [{"kind": "constraint", "text": "use postgres"}]
        assert s2.session_view()["local_extractions_pending"] == 2

        # consolidation replaces snapshot, evicts pending, keeps pinned
        s2.set_snapshot({"decisions": [{"d": "x"}], "constraints": []}, turn=2)
        assert s2.pending_count() == 0
        assert s2.snapshot()["decisions"] == [{"d": "x"}]
        assert s2.session_view()["constraints"] == [{"kind": "constraint", "text": "use postgres"}]
        assert s2.stats()["snapshot_count"] == 1

        # corrupt file: recover, don't crash
        (Path(d) / "memory.json").write_text("{not json", encoding="utf-8")
        s3 = MemoryStore(store_dir=Path(d))
        assert s3.pending_count() == 0
        assert (Path(d) / "memory.json.corrupt").exists()
        print("ok")
