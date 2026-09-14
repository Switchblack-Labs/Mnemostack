# Benchmarks

Two measurements, both reproducible from this directory. Every repository is
fetched at a pinned commit and every upgrade is between exact versions, so a
rerun sees the same code and the same libraries.

```console
uv run python bench/run_corpus.py    # noise: what a maintainer would be told
uv run python bench/run_recall.py    # recall: did it flag what a migration changed
```

Clones are cached in `~/.cache/mnemostack-bench` (`--cache` to move it), so a
rerun needs no network beyond the uv installs the verification step does.
Results are written to `bench/results/`.

## Noise: `run_corpus.py`

[`corpus.json`](corpus.json) pins 20 widely used Python repositories and 10
upgrades across major versions of libraries they depend on (pydantic 1 to 2,
click 7 to 8, SQLAlchemy 1.4 to 2.0, jinja2 2 to 3, and others). Each repository
is checked against every upgrade whose package it imports.

This counts what the report would put in front of a maintainer: BREAK groups,
REVIEW groups, and the places they reach. It does not say whether a finding is
right. Most of these repositories already run on the newer version, so almost
nothing should break, and the short BREAK list is meant to be read by hand.

## Recall: `run_recall.py`

[`migrations.json`](migrations.json) pins real commits that moved a repository
across a major version: two pydantic 1 to 2 migrations and one SQLAlchemy 1.4
to 2.0. `upgrade-check` runs at each commit's parent, between the versions the
commit moved between, and is scored against the lines the commit changed.

- **Relevant lines** are changed lines, as they were before the commit, that
  name something the file imports from the library. Unrelated edits in the same
  commit are not counted.
- **Recall** is the share of relevant lines that carry a finding, reported for
  breaking findings alone and with deprecations included.

Relevant is not the same as broken. Maintainers also edit import lines to add
names and rewrite code that still worked, so a miss here is not necessarily a
missed break. The SQLAlchemy migration in particular is mostly a rewrite to the
2.0 typing style, and its recall says little either way.

## Limits

Three migrations and twenty repositories are a small sample. The numbers show
whether a change made things better or worse on the same inputs; they do not
estimate accuracy on your code.

## Results

Measured on this repository at the commit these files were added with, and
recorded in [`results/`](results).

**Noise.** 52 (repository, package) pairs: 2 BREAK groups, 105 REVIEW groups,
2,631 places in total, no failures. Both BREAKs are real:

- instructor calls pydantic's `parse_file_as`, which pydantic 2 removed;
- pydantic-settings calls `PostgresDsn(...)`, which pydantic 2 turned from a
  class into an annotated type.

**Recall.**

| migration | relevant lines | breaking findings | with deprecations |
|---|---|---|---|
| distiller, pydantic 1.10.10 → 2.13.5 | 18 | 0.611 | 0.778 |
| dstack, pydantic 1.10.26 → 2.12.5 | 330 | 0.361 | 0.958 |
| onegov-cloud, SQLAlchemy 1.4.54 → 2.0.52 | 2,880 | 0.032 | 0.032 |

Most relevant lines missed on the pydantic migrations are import lines edited to
add names. The rest are behaviour the API diff cannot see: an `AnyHttpUrl = None`
default that v2 rejects, and `isinstance`/`issubclass` checks against
`BaseModel`. Each result file lists every missed line.
