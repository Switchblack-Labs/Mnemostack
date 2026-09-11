# mnemostack

See which symbols in *your* code an upgrade would break — before you upgrade.

```console
$ mnemostack upgrade-check requests 2.32

requests 2.28 -> 2.32: 12 breaking change(s)

  3 place(s) in your code touch what changed:

  [BREAK ] api/client.py::fetch_user
           PARAMETER_REMOVED  requests.Session.request
  [BREAK ] api/client.py::retry
           OBJECT_REMOVED  requests.Session.merge_hooks
  [REVIEW] jobs/sync.py::run
           RETURN_CHANGED_TYPE  requests.get
```

Exit code is 1 when anything `BREAK`-level is found, so it works as a CI gate.

## Why

`griffe check` will tell you a library changed 40 things. Published work on
Maven found only **7.9% of clients** are affected by a given breaking change,
and that for most libraries **88% of the API could be deleted** with three
quarters of clients never noticing. So a list of what changed in the library is
mostly noise. The useful question is which of those changes your own code
actually touches.

Dependabot's compatibility score answers a different question — whether *other
people's* CI passed — and cannot be computed at all for 83% of updates.
Commercial tools (Semgrep Upgrade Guidance, Endor Labs) do compute this, but
only for upgrades that fix a CVE, only in a web UI, and only on a paid plan.

## How it works

1. Fetch both versions of the library from PyPI with
   [griffe](https://mkdocstrings.github.io/griffe/). Nothing is installed.
2. Diff their public APIs into a classified breakage list. griffe is directional:
   adding a *required* parameter is breaking, adding an optional one is not.
3. Build a symbol graph of your code — calls, inheritance, and references such as
   type annotations, `isinstance` checks and decorators.
4. Intersect. Report only the call sites that actually touch a change.

Two filters keep it quiet. A claimed removal is re-checked against the new
version's source, because griffe reports `@overload` members as removed when
they are not. And a removed keyword-only parameter is only reported at call
sites that actually pass it.

## Install

```console
uv tool install mnemostack     # or: pip install mnemostack
```

## Usage

```console
mnemostack upgrade-check <package> <to-version> [--from-version X] [--repo .]
```

`--from-version` defaults to the version you have installed.

There is also an MCP server (`mnemostack` with no arguments) exposing the same
check as a tool, so a coding agent can ask the same question.

## What it does not do

It finds breakage you can see in a function's shape: removed symbols, changed
parameters, changed base classes. It does **not** find a library that quietly
changes what it *does* while the signature stays the same — published studies
put behavioural changes at the majority of breakage that actually manifests.
Tests catch those; nothing static can.

So this is not a replacement for a test suite. It is for the code your tests
do not cover, and for deciding whether an upgrade is worth starting.

## Accuracy

Measured against [scip-python](https://github.com/sourcegraph/scip-python)
(Pyright) ground truth on this repository: **100% precision, 32% recall** of
cross-file symbol references. For comparison, published Python call-graph tools
measured on real code rather than micro-benchmarks report 41.7%/23.3% (PyCG) and
35%/60% (JARVIS) precision/recall with libraries in scope.

Precision is the priority: a reported break is real, and a missed one shows up
as silence rather than as a wrong answer.

## Development

```console
uv sync
uv run pytest -q
uv run ruff check .
```
