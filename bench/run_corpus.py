"""Noise benchmark: run upgrade-check across pinned repositories we did not write.

Every repository is fetched at the commit in corpus.json and every upgrade is
between exact versions, so a rerun sees the same code and the same libraries.

This measures what a maintainer would be told to look at, not whether it is
right. The BREAK list is short enough to check by hand, and should be.

    uv run python bench/run_corpus.py [--cache DIR] [--out bench/results/corpus.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from common import DEFAULT_CACHE, checkout

from mnemostack.core.impact.sweep import imported_roots
from mnemostack.core.impact.upgrade import UpgradeError, check_upgrade

HERE = Path(__file__).parent


def run_pair(repo: Path, name: str, pkg: str, dist: str, old: str, new: str) -> dict:
    row: dict = {"repo": name, "package": dist, "from": old, "to": new}
    start = time.time()
    try:
        report = check_upgrade(
            repo=repo, package=pkg, to_version=new, distribution=dist, from_version=old
        )
    except UpgradeError as exc:
        row["error"] = str(exc)[:160]
    except Exception as exc:  # noqa: BLE001 - one crash must not end the run
        row["crash"] = f"{type(exc).__name__}: {exc}"[:160]
    else:
        groups: dict[tuple[str, str, str], list[str]] = {}
        for impact in report.impacts:
            key = (impact.severity.value, impact.change.kind, impact.change.fqn.split(".")[-1])
            groups.setdefault(key, []).append(
                f"{impact.site.file}:{impact.site.line} | {impact.site.text}"
            )
        row["changes"] = report.verified_changes
        row["groups"] = [
            {"severity": s, "kind": k, "symbol": sym, "places": len(v), "sample": v[:3]}
            for (s, k, sym), v in sorted(groups.items(), key=lambda kv: (kv[0][0], len(kv[1])))
        ]
        row["deprecations"] = [
            {"symbol": d.symbol, "places": len(d.sites), "message": d.message[:100]}
            for d in report.deprecations
        ]
        row["unverified_removals"] = report.unwitnessed
    row["secs"] = round(time.time() - start, 1)
    return row


def summarize(results: list[dict]) -> str:
    counts = {"break": 0, "review": 0}
    places = 0
    breaks = []
    for row in results:
        for group in row.get("groups", []):
            counts[group["severity"]] = counts.get(group["severity"], 0) + 1
            places += group["places"]
            if group["severity"] == "break":
                breaks.append(
                    f"  BREAK  {row['repo']}/{row['package']}  {group['kind']} "
                    f"{group['symbol']}  x{group['places']}"
                )
    failed = [row for row in results if "error" in row or "crash" in row]
    lines = [
        f"{len(results)} (repo, package) pairs: {counts['break']} BREAK groups, "
        f"{counts['review']} REVIEW groups, {places} places, {len(failed)} failed",
        *breaks,
    ]
    lines += [
        f"  FAILED {r['repo']}/{r['package']}: {r.get('error') or r.get('crash')}" for r in failed
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "corpus.json")
    args = parser.parse_args()

    manifest = json.loads((HERE / "corpus.json").read_text())
    results = []
    for entry in manifest["repos"]:
        print(f"fetching {entry['name']} @ {entry['sha'][:10]}", file=sys.stderr)
        repo = checkout(entry["url"], entry["sha"], args.cache / "corpus" / entry["name"])
        roots = imported_roots(repo)
        for pkg, dist, old, new in manifest["cases"]:
            if pkg in roots:
                row = run_pair(repo, entry["name"], pkg, dist, old, new)
                results.append(row)
                print(f"  {entry['name']:<18} {dist:<12} {row['secs']}s", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
