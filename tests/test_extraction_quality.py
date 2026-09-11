"""Recall and precision against Pyright, ratcheted so quality cannot silently rot.

test_cross_repo_recall.py pins four hand-written patterns and will happily stay
green while real-world reach collapses, because a synthetic fixture only tests
the mechanisms someone thought to write down. This measures the extractor
against an external oracle on real code: scip-python (Pyright) over this
package's own source.

The ground truth is committed as JSON rather than regenerated, so the check runs
offline with no Node.js and gives the same answer on every machine. Regenerate
it when the package layout changes substantially; the command is in the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemostack.core.impact.api_diff import fqn_index
from mnemostack.core.retrieval.call_graph import (
    CallGraph,
    EdgeType,
    build_nodes_for_python_file,
    link_python_file_imports,
)

GROUND_TRUTH = Path(__file__).parent.parent / "benchmarks" / "scip_ground_truth.json"
PACKAGE = "mnemostack"

# Measured against the committed ground truth. Raise these when an improvement
# lands, in the same commit, the way the cross-repo table works. Never lower one
# without saying in the commit message what was traded away and why.
#
# Current: 28.8% recall (21/73), 100% precision. Read the recall number as a
# floor on this tree, not as a general claim: the oracle is 73 pairs over one
# small package, so a few edges move it a percent either way. It exists to
# catch a collapse, not to certify an improvement. Published Python call-graph
# tools measured on real code sit at 23.3% (PyCG) and 60% (JARVIS) recall with
# libraries in scope, so this is a normal band and precision is the number we
# are actually beating them on.
MIN_RECALL = 0.28
MIN_PRECISION = 1.00


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    repo = Path(__file__).parent.parent
    truth = {
        (src, fqn) for src, fqn in (tuple(p) for p in json.loads(GROUND_TRUTH.read_text())["pairs"])
    }

    files = sorted((repo / PACKAGE).rglob("*.py"))
    graph = CallGraph(store_dir=tmp_path_factory.mktemp("store"))
    for f in files:
        build_nodes_for_python_file(f, graph=graph)
    for f in files:
        link_python_file_imports(f, graph=graph)

    by_node = {v: k for k, v in fqn_index(graph).items()}
    rows = graph.db.execute(
        "SELECT s.qualified_name, t.qualified_name FROM edges e "
        "JOIN nodes s ON s.id = e.source_id JOIN nodes t ON t.id = e.target_id "
        "WHERE e.edge_type IN (?, ?, ?)",
        (EdgeType.CALLS.value, EdgeType.INHERITS.value, EdgeType.REFERENCES.value),
    ).fetchall()

    ours = set()
    for src, tgt in rows:
        src_file, tgt_file = src.split("::")[0], tgt.split("::")[0]
        if src_file == tgt_file:
            continue  # same file is not a cross-file reference
        fqn = by_node.get(tgt)
        if fqn:
            rel = Path(src_file).relative_to(repo).as_posix().removeprefix(f"{PACKAGE}/")
            ours.add((rel, fqn))
    graph.close()
    yield truth, ours


def test_recall_against_pyright_does_not_regress(measured, capsys):
    truth, ours = measured
    hit = truth & ours
    recall = len(hit) / len(truth)
    with capsys.disabled():
        print(f"\n  recall vs pyright: {recall:.1%} ({len(hit)}/{len(truth)})")
    assert recall >= MIN_RECALL, (
        f"cross-file recall fell to {recall:.1%}, below the recorded "
        f"{MIN_RECALL:.0%}. Something stopped resolving."
    )


def test_precision_against_pyright_stays_total(measured, capsys):
    """Every edge we emit should be one Pyright also found.

    Precision is the invariant worth protecting hardest. A missed reference
    shows up as silence, which is honest; an invented one puts a name in an
    upgrade report that nobody actually uses, and one of those teaches a reader
    to distrust the whole thing.
    """
    truth, ours = measured
    if not ours:
        pytest.fail("extractor produced no cross-file references at all")
    hit = truth & ours
    precision = len(hit) / len(ours)
    with capsys.disabled():
        print(f"  precision vs pyright: {precision:.1%} ({len(hit)}/{len(ours)})")
    assert precision >= MIN_PRECISION, (
        f"precision fell to {precision:.1%}. These edges are not in Pyright's "
        f"result: {sorted(ours - truth)[:5]}"
    )


def test_ground_truth_is_still_about_this_tree(measured):
    """Fail loudly if the oracle has drifted from the code it describes.

    A stale ground truth quietly turns both numbers above into noise: recall
    measured against files that no longer exist is meaningless, and nothing
    else in the suite would notice.
    """
    truth, _ = measured
    repo = Path(__file__).parent.parent
    missing = {src for src, _ in truth if not (repo / PACKAGE / src).is_file()}
    assert not missing, (
        f"ground truth references {len(missing)} file(s) that no longer exist "
        f"({sorted(missing)[:3]}). Regenerate benchmarks/scip_ground_truth.json."
    )
