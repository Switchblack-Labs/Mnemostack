"""Hostile input for the keyword search path.

A query reaches FTS5 from an agent, so it can contain anything: FTS5's own
operators, unbalanced quotes, code punctuation, other languages, or a deliberate
injection attempt. None of it may raise, and none of it may reach SQL.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mnemostack.core.retrieval.ast_chunker import Chunk, ChunkType
from mnemostack.core.retrieval.fts_index import _sanitize_fts_query

HOSTILE = [
    "", "   ", "\n\t",
    '"', '""', '"""', "'", "``", "\\", "\\\\",
    "*", "**", "^", "-", "--", ":", "::", "(", ")", "()", "[", "{", "}",
    "AND", "OR", "NOT", "NEAR", "and or not near",
    "a AND b", "x OR y", "NEAR(a b)", "col:value", "^prefix", "foo*",
    "parse-request", "parse.request", "a.b.c::d", "foo()", "self.method()",
    "<Widget />", "n+1", "x--y", "-leading", "trailing-",
    "café résumé", "日本語のクエリ", "emoji 🚀 query", "\x00null byte",
    "the a of is", "the", "?", "!?", "%", "%%", "_", "@", "#tag",
    "SELECT * FROM chunks; DROP TABLE chunks;--",
    "' OR 1=1 --", '" OR "" = "', '"; DELETE FROM chunks_fts; --',
    "x" * 5000,
    " ".join(f"term{i}" for i in range(500)),
]


@pytest.fixture
def populated(faiss_idx, fts_idx):
    chunk = Chunk(
        file_path="a.py",
        symbol_name="parse_request",
        code='def parse_request():\n    """Parse the request."""\n    return the_thing * 2',
        line_start=1,
        line_end=3,
        chunk_type=ChunkType.FUNCTION,
        last_modified=time.time(),
        qualified_name="a.py::parse_request",
        dependencies=[],
    )
    ids = faiss_idx.add([chunk], np.zeros((1, 4), dtype=np.float32))
    fts_idx.sync_added(ids)
    return fts_idx


@pytest.mark.parametrize("query", HOSTILE, ids=lambda q: repr(q[:24]))
def test_hostile_query_returns_a_list_without_raising(query, populated):
    assert isinstance(populated.search(query), list)


def test_injection_leaves_the_index_intact(populated, shared_db):
    for query in HOSTILE:
        populated.search(query)
    assert shared_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    assert populated.search("parse_request")[0].symbol_name == "parse_request"


def test_identifier_punctuation_is_not_treated_as_syntax(populated):
    """`-` and `*` are FTS5 operators; inside an identifier they are literal."""
    for query in ("parse_request", "parse-request", "parse.request", "parse_request*"):
        assert isinstance(populated.search(query), list)
    assert populated.search("parse_request")[0].symbol_name == "parse_request"


class TestSanitizer:
    def test_tokens_are_ored_not_anded(self):
        assert _sanitize_fts_query("debounce file events") == '"debounce" OR "file" OR "events"'

    def test_stopwords_are_dropped(self):
        assert _sanitize_fts_query("how does the watcher work") == '"watcher" OR "work"'

    def test_a_query_of_only_stopwords_produces_no_query(self):
        assert _sanitize_fts_query("what is the that") == ""
        assert _sanitize_fts_query("") == ""

    def test_embedded_quotes_cannot_escape_a_token(self):
        assert '""' not in _sanitize_fts_query('say "hello" now')
        assert _sanitize_fts_query('say "hello" now') == '"say" OR "hello" OR "now"'
