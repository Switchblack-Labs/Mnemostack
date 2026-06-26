"""LLM consolidation: compress pending turns + prior snapshot into a new snapshot.

One litellm.completion() call (any provider via settings.compression.consolidation_model).
Carries the prior snapshot forward and always preserves pinned facts — it never
re-compresses, matching the MemoryStore design invariant.

ponytail: len/4 token estimate for the budget; oldest pending turns are evicted
(not summarized) when the input would exceed memory_token_budget.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from litellm import completion

from mnemostack.config.settings import settings
from mnemostack.core.compression.memory_store import MemoryStore

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You compress a coding session's working memory. Merge the PRIOR MEMORY and the "
    "NEW TURNS into a single updated memory. Keep every pinned fact. Move settled "
    "questions from open_questions into resolved. Output ONLY a JSON object with keys: "
    "decisions (list), constraints (list), open_questions (list of strings), "
    "architecture_state (object), resolved (list of strings). No prose, no markdown."
)


class ConsolidationError(Exception):
    """Raised when the LLM call or its JSON parse fails."""


@dataclass
class Outcome:
    success: bool
    turns_consolidated: int
    snapshot_id: str
    token_count: int


def _est_tokens(obj: object) -> int:
    return len(json.dumps(obj)) // 4


def _extract_json(text: str) -> dict:
    """Parse the model output into a JSON object, tolerating fences/prose.

    Tries the whole text first, then the outermost {...}. Anything that isn't a
    JSON *object* (None, a list, junk) raises ConsolidationError, never a raw
    JSONDecodeError/TypeError — the caller's contract is ConsolidationError only.
    """
    if not text:
        raise ConsolidationError("model returned empty content")
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ConsolidationError(f"No JSON object in model output: {text[:200]!r}")


def _fit_budget(pending: list[str], snapshot: dict, pinned: list[dict], budget: int) -> list[str]:
    """Drop oldest pending turns until the input fits the token budget."""
    fixed = _est_tokens(snapshot) + _est_tokens(pinned)
    kept = list(pending)
    while kept and fixed + _est_tokens(kept) > budget:
        kept.pop(0)  # evict oldest, never re-compress
    return kept


def consolidate(store: MemoryStore, model: str | None = None) -> Outcome:
    """Run one consolidation cycle against the store. No-op if nothing is pending."""
    pending = store.pending()
    if not pending:
        return Outcome(success=True, turns_consolidated=0, snapshot_id="", token_count=0)

    model = model or settings.compression.consolidation_model
    snapshot = store.snapshot()
    pinned = store.pinned()
    kept = _fit_budget(pending, snapshot, pinned, settings.compression.memory_token_budget)

    user = json.dumps(
        {"prior_memory": snapshot, "pinned_facts": pinned, "new_turns": kept},
        indent=2,
    )
    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
    except Exception as exc:
        raise ConsolidationError(f"Consolidation LLM call failed with {model!r}: {exc}") from exc

    new_snapshot = _extract_json(content)

    if len(kept) < len(pending):
        log.info("Budget-evicted %d oldest pending turns", len(pending) - len(kept))

    turn = store.turn_count()
    # Evict exactly the turns we read; turns appended during the LLM call survive.
    store.set_snapshot(new_snapshot, turn=turn, evict_count=len(pending))
    stats = store.stats()
    return Outcome(
        success=True,
        turns_consolidated=len(kept),
        snapshot_id=str(stats["snapshot_count"]),
        token_count=stats["total_memory_tokens"],
    )


if __name__ == "__main__":  # smoke check: stub the LLM, verify the store transition
    import tempfile
    import types
    from pathlib import Path

    canned = '```json\n{"decisions": [{"d": "use postgres"}], "open_questions": []}\n```'
    # Rebind the module-global `completion` this file's consolidate() looks up.
    completion = lambda **kw: types.SimpleNamespace(  # noqa: E731
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=canned))]
    )

    with tempfile.TemporaryDirectory() as d:
        s = MemoryStore(store_dir=Path(d))
        assert consolidate(s).turns_consolidated == 0  # nothing pending -> no-op
        s.add_turn("we decided to use postgres")
        s.add_turn("dropped the redis idea")
        s.add_pinned("constraint", "no external services")

        out = consolidate(s)
        assert out.success and out.turns_consolidated == 2
        assert s.pending_count() == 0  # evicted
        assert s.snapshot()["decisions"] == [{"d": "use postgres"}]
        # pinned survives consolidation
        assert s.session_view()["constraints"] == [
            {"kind": "constraint", "text": "no external services"}
        ]
        assert out.snapshot_id == "1"

        # non-dict model output (a JSON list) -> ConsolidationError, not a raw exception
        try:
            _extract_json("[1, 2, 3]")
            raise AssertionError("expected ConsolidationError")
        except ConsolidationError:
            pass

        # a turn appended DURING the LLM call is not lost on eviction
        s.add_turn("first")

        def _slow_completion(**kw):
            s.add_turn("appended mid-call")  # simulates a concurrent writer
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=canned))]
            )

        globals()["completion"] = _slow_completion  # rebind the global consolidate() uses
        consolidate(s)
        assert s.pending() == ["appended mid-call"]  # the concurrent turn survived
        print("ok")
