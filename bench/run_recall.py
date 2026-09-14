"""Recall benchmark: did upgrade-check flag the lines a real migration had to change?

For each commit in migrations.json that moved a repository across a major
version of a library, upgrade-check runs at the commit's parent, between the
exact versions the migration moved between, and is scored against the lines
that commit changed.

- relevant lines: changed lines, as they were before the commit, that name
  something the file imports from the library. A migration commit also carries
  unrelated edits, and those are not counted.
- recall: the share of relevant lines that carry a finding. Reported for
  breaking findings alone, and with deprecations included.

A relevant line is not necessarily a break: a maintainer also edits import
lines to add names, and rewrites code that still worked. So recall here is a
floor on what matters and a ceiling on nothing.

    uv run python bench/run_recall.py [--cache DIR] [--out bench/results/recall.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from common import DEFAULT_CACHE, checkout, git

from mnemostack.core.impact.upgrade import UpgradeError, check_upgrade
from mnemostack.core.reach.static import code_lines, imported_names, parse_source

HERE = Path(__file__).parent


def changed_old_lines(repo: Path, sha: str) -> dict[str, set[int]]:
    """File -> line numbers, as they were before `sha`, that `sha` changed."""
    out: dict[str, set[int]] = {}
    current = None
    for line in git("diff", "-U0", f"{sha}^", sha, "--", "*.py", cwd=repo).splitlines():
        if line.startswith("--- "):
            current = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("@@") and current:
            match = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
            start, count = int(match.group(1)), int(match.group(2) or 1)
            out.setdefault(current, set()).update(range(start, start + count))
    return out


def relevant_lines(repo: Path, package: str, changed: dict[str, set[int]]) -> set[tuple[str, int]]:
    keep = set()
    for file, lines in changed.items():
        path = repo / file
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = parse_source(source)
        names = set(imported_names(tree, package)) if tree is not None else set()
        if not names:
            continue
        pattern = re.compile(rf"(?<![\w.])(?:{'|'.join(map(re.escape, sorted(names)))})\b")
        code = code_lines(source)
        keep |= {(file, n) for n in lines if n <= len(code) and pattern.search(code[n - 1])}
    return keep


def score(entry: dict, repo: Path) -> dict:
    sha = entry["sha"]
    changed = changed_old_lines(repo, sha)
    relevant = relevant_lines(repo, entry["package"], changed)
    row = {
        "repo": entry["name"],
        "package": entry["distribution"],
        "from": entry["from"],
        "to": entry["to"],
        "changed_lines": sum(map(len, changed.values())),
        "relevant_lines": len(relevant),
    }
    try:
        report = check_upgrade(
            repo,
            entry["package"],
            entry["to"],
            distribution=entry["distribution"],
            from_version=entry["from"],
        )
    except UpgradeError as exc:
        row["error"] = str(exc)[:200]
        return row

    def on_changed(sites: set[tuple[str, int]]) -> int:
        return sum(1 for file, n in sites if n in changed.get(file, ()))

    for severity in ("break", "review"):
        sites = {(i.site.file, i.site.line) for i in report.impacts if i.severity.value == severity}
        row[severity] = {"sites": len(sites), "on_changed_lines": on_changed(sites)}
    impact_sites = {(i.site.file, i.site.line) for i in report.impacts}
    dep_sites = {(s.file, s.line) for d in report.deprecations for s in d.sites}
    row["deprecation"] = {"sites": len(dep_sites), "on_changed_lines": on_changed(dep_sites)}
    if relevant:
        row["recall_findings"] = round(len(impact_sites & relevant) / len(relevant), 3)
        row["recall_with_deprecations"] = round(
            len((impact_sites | dep_sites) & relevant) / len(relevant), 3
        )
    source: dict[str, list[str]] = {}
    row["missed"] = []
    for file, n in sorted(relevant - impact_sites - dep_sites):
        if file not in source:
            source[file] = (repo / file).read_text(encoding="utf-8", errors="ignore").splitlines()
        row["missed"].append(f"{file}:{n} | {source[file][n - 1].strip()[:140]}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "recall.json")
    args = parser.parse_args()

    results = []
    for entry in json.loads((HERE / "migrations.json").read_text()):
        print(f"fetching {entry['name']} @ {entry['sha'][:10]}", file=sys.stderr)
        repo = checkout(entry["url"], entry["sha"], args.cache / "migrations" / entry["name"], 2)
        git("checkout", "-q", "--force", f"{entry['sha']}^", cwd=repo)
        results.append(score(entry, repo))
        git("checkout", "-q", "--force", entry["sha"], cwd=repo)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    for row in results:
        print(
            f"{row['repo']:<14} {row['package']} {row['from']} -> {row['to']}: "
            f"{row['relevant_lines']} relevant lines, "
            f"recall {row.get('recall_findings')} findings, "
            f"{row.get('recall_with_deprecations')} with deprecations"
            + (f", error: {row['error']}" if "error" in row else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
