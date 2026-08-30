# Tests

Four layers, each answering a different question. All of them run offline: no
model server, no network, no fixed ports.

| Layer | Files | Question |
|---|---|---|
| Unit | `test_retrieval.py`, `test_js_graph.py`, `test_import_resolver.py`, `test_settings.py`, `test_file_watcher.py` | Does each piece behave? |
| Integration | `test_integration.py` | Do several repos, both languages, and the graph work together on real files? |
| Robustness | `test_fts_robustness.py` | Does hostile input break anything? |
| Properties | `test_ranking_invariants.py` | Does ranking hold its guarantees for any query? |

```sh
uv run pytest -q          # everything
uv run ruff check .
```

## Why embeddings are stubbed

`conftest.py` swaps the embedding call for a hashed bag-of-words vector. Real
embeddings need a running Ollama, which CI does not have, and random vectors
make a retrieval assertion meaningless: a test that passes on noise proves
nothing. The hash embedder is deterministic across processes and still places
text sharing vocabulary close together, so end-to-end ranking can be asserted.

It is a stand-in, not a measurement. Retrieval *quality* is measured separately.

## Measuring retrieval quality

Ranking regressions are invisible to unit tests: every test can pass while the
results get worse. `evals/run_eval.py` asks real questions of a real index with
real embeddings and reports where the right answer ranked.

```sh
python evals/run_eval.py --root ../service --root ../web-app --min-mrr 0.6
```

Run it before and after any ranking change. Keep the tuning and holdout split in
`cases.example.yaml`: if you tune against every case you have, the numbers only
prove the tuning fit itself. The parameters in `ranker.py` were chosen on the
tuning set and confirmed on the holdout set, which disagreed with it — the value
that scored best on the tuning set was worse on the holdout.

## Adding a test for a fix

Write the test first and watch it fail without the fix. A regression test that
passes against the bug is decoration. Each fix in the ranking and cross-repo
work was checked this way, by reintroducing the bug and confirming the suite
went red.
