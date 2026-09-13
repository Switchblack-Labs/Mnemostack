# mnemostack

Before you upgrade a Python dependency, see which lines of *your* code the new
version breaks.

```console
$ mnemostack upgrade-check mcp 2.2.0
mcp 1.28.1 -> 2.2.0: the library has 357 breaking API change(s). Only what reaches your code is listed.

  1 change(s) reach your code, across 2 place(s):

  [BREAK ] OBJECT_REMOVED  mcp.server.FastMCP
           mnemostack/mcp/tools.py:7
             from mcp.server.fastmcp import FastMCP
           mnemostack/mcp/tools.py:10
             mcp = FastMCP("mnemostack")
```

That is this repository, checked against a newer release of one of its own
dependencies.

## Commands

```console
mnemostack upgrade-check <package> <to-version> [--from-version X] [--distribution NAME] [--repo .]
mnemostack sweep [--repo .] [--quiet]
mnemostack                     # MCP server over stdio, with one tool: upgrade_check
```

`upgrade-check` exits 1 if anything is graded BREAK, 2 if the check could not
run, and 0 otherwise, so it can gate CI.

`--from-version` defaults to the version installed in the repo's own virtual
environment: `.venv` or `venv` in the repo, or an activated one. It is never
read from the environment mnemostack itself runs in. With no environment to
ask, pass it explicitly.

`sweep` runs the same check for every package the repo imports and has
installed in that environment, against its latest release on PyPI:

```console
$ mnemostack sweep
  breaks        mcp                    1.28.1 -> 2.2.0           [2 place(s)]
  safe          pydantic               2.13.4 -> 2.13.5
  current       platformdirs           4.11.8 -> 4.11.8
  ...

  1 of 2 available upgrade(s) can be taken with nothing to change; 4 already current. 1 need attention.
```

Statuses: `breaks`, `review`, `unverified` (possible removals nothing could
confirm), `deprecations`, `safe`, `current`, and `unknown` or `error` when PyPI
or the check itself failed. Packages imported but not installed are skipped.

## How it works

1. **Diff the library.** [griffe](https://mkdocstrings.github.io/griffe/) loads
   both versions from PyPI and lists breaking API changes: removed symbols,
   parameters added, removed, moved or made required, changed base classes and
   kinds.
2. **Find your uses.** Only files that import the package are scanned. A line
   matches when it names a changed symbol the way code reaches it: a name the
   file imported, an attribute path starting from an imported module, or
   `obj.member` where `obj` is the class, a subclass defined in the file, or a
   variable assigned or annotated as one.
3. **Grade each use.** A table maps the kind of change and how the line uses
   the symbol to BREAK, REVIEW or nothing. A removed base class breaks a
   subclass but only concerns a caller; a changed parameter matters to a call,
   not to an import line.
4. **Check against the real package.** The old and new versions are installed
   into throwaway uv environments and imported:
   - a removal is kept only if the import path your code uses fails in the new
     version; if it still imports but warns that it moved, it is listed as a
     deprecation instead;
   - a signature change is kept only if the signature really differs, and a
     call is cleared only if its arguments still bind to the new signature. The
     whole call is read from the file, so calls split across lines and calls
     passing `**kwargs` are decided where that is possible.
5. **Soften what your code already hedges.** A use inside a compatibility
   shim, a `try`/`except ImportError` fallback or an `if hasattr(...)` branch,
   is graded REVIEW rather than BREAK.
6. **List deprecations separately.** Things you use that the new version marks
   deprecated do not break yet, and a green test run hides their warnings.

## Before you rely on it

- **It runs the library's import code.** Step 4 installs the old and new
  versions from PyPI into temporary uv environments and imports them. Your own
  environment is not touched, but the package's code does execute, and `sweep`
  does this for every dependency it checks.
- **It needs [uv](https://docs.astral.sh/uv/) for step 4.** Without uv nothing
  is verified and findings are shown as the scan graded them, which is noisier.
  A removal reached through an object whose type the line does not show cannot
  be checked; it is counted in the output rather than shown, and `sweep` marks
  the package `unverified`, never `safe`.
- **It reads source lines, not types.** Receivers are matched by name across a
  whole file, not per scope, so a variable in one function named like a
  library object in another can produce a false finding. `getattr`, star
  imports and anything built at runtime are not followed.
- **It sees API shape, not behaviour.** A function that keeps its signature but
  changes what it does is invisible here. Tests catch those.
- **No accuracy figures are claimed.** There is no reproducible benchmark in
  this repository yet.

## Install

```console
uv tool install git+https://github.com/Switchblack-Labs/Mnemostack
```

Needs Python 3.11 or newer, and uv on `PATH`.

## Development

```console
uv sync
uv run pytest -q
uv run ruff check .
```
