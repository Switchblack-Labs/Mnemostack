#!/usr/bin/env python3
"""Measure retrieval quality: does a real question return the code that answers it?

Unit tests can say the pipeline returns rows. Only this can say the rows are the
right ones, so any ranking change should be run past it before and after.

    python evals/run_eval.py --root ../service --root ../web-app
    python evals/run_eval.py --cases evals/cases.example.yaml --min-mrr 0.6

Needs the configured embedding model to be reachable (Ollama by default), since
ranking measured on stand-in embeddings would measure the stand-in.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mnemostack.core.retrieval.call_graph import CallGraph  # noqa: E402
from mnemostack.core.retrieval.faiss_index import FaissIndex, create_chunks_db  # noqa: E402
from mnemostack.core.retrieval.fts_index import FTSIndex  # noqa: E402
from mnemostack.core.retrieval.indexer import index_directory  # noqa: E402
from mnemostack.core.retrieval.query import query_pipeline  # noqa: E402


def run_set(name, cases, faiss_idx, fts, graph, top_k):
    """Rank of the expected chunk per case, and the set's MRR."""
    ranks: list[int | None] = []
    print(f"\n{name}")
    for case in cases:
        results = query_pipeline(query=case["query"], faiss_idx=faiss_idx, fts_idx=fts,
                                 graph=graph, top_k=top_k)
        rank = next(
            (i for i, r in enumerate(results, 1) if case["expect"] in r.qualified_name), None
        )
        ranks.append(rank)
        top = results[0].qualified_name.rsplit("/", 1)[-1] if results else "-"
        label = f"rank {rank}" if rank else "MISS   "
        detail = case["expect"] if rank else f"top1={top}"
        print(f"  {label}  {case['query'][:56]:56}  {detail}")

    hits = [r for r in ranks if r]
    mrr = sum(1 / r for r in hits) / len(cases)
    print(f"  -> recall@{top_k} {len(hits)}/{len(cases)}   "
          f"top-1 {sum(1 for r in hits if r == 1)}/{len(cases)}   MRR {mrr:.3f}")
    return mrr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True,
                        help="repository to index (repeatable)")
    parser.add_argument("--cases", type=Path, default=Path(__file__).parent / "cases.example.yaml")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-mrr", type=float, default=None,
                        help="exit non-zero if any set scores below this")
    args = parser.parse_args()

    sets = yaml.safe_load(args.cases.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)
        db = create_chunks_db(store)
        faiss_idx, fts, graph = FaissIndex(store_dir=store, db=db), FTSIndex(
            store_dir=store, db=db), CallGraph(store_dir=store)
        for root in args.root:
            count = index_directory(root=root.resolve(), faiss_idx=faiss_idx, fts_idx=fts,
                                    graph=graph)
            print(f"indexed {count:4} chunks from {root}")

        scores = {name: run_set(name, cases, faiss_idx, fts, graph, args.top_k)
                  for name, cases in sets.items()}
        db.close()

    if args.min_mrr is not None:
        low = {n: s for n, s in scores.items() if s < args.min_mrr}
        if low:
            print(f"\nbelow --min-mrr {args.min_mrr}: " +
                  ", ".join(f"{n} {s:.3f}" for n, s in low.items()))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
