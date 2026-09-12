"""Find where a repo touches a package, by reading imports and scanning lines.

This replaces a 1,165-line symbol graph that resolved names itself. That graph
reached 29% of the references Pyright finds, because two thirds of what a real
codebase references are attributes, enum members and module-level values it had
no way to represent, and because it only ever saw top-level definitions.

Scanning does not have those blind spots. It finds the call inside a nested
function, the annotation under `if TYPE_CHECKING`, the attribute read, the
constant. It pays for that with precision, which is why the import table gates
it: a file is only scanned for names it actually imported from the package in
question, so an unrelated local `get` is never mistaken for the library's.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
import warnings
from pathlib import Path

from mnemostack.core.reach import RefKind, Site


def parse_source(source: str) -> ast.Module | None:
    """Parse foreign source quietly, or None if it is not valid Python here.

    Code we did not write routinely contains invalid escape sequences, and
    ast.parse reports each one as a SyntaxWarning on stderr. Across twenty real
    repositories that was dozens of lines of noise interleaved with the report.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ast.parse(source)
        except (SyntaxError, ValueError):
            return None


def code_lines(source: str) -> list[str]:
    """Source lines with string literals and comments blanked to spaces.

    Matching used to run against raw text, so `render_template('index.html')`
    was credited to a removed `werkzeug.html`. Blanking keeps columns and line
    numbers intact, so a match still points at the right place in the real line.

    Quoted forward references (`s: 'Session'`) are lost along with the noise.
    They are rarer than what they would keep in, and unquoted annotations still
    match normally.
    """
    lines = source.splitlines()
    blank = {tokenize.STRING, tokenize.COMMENT}
    if hasattr(tokenize, "FSTRING_MIDDLE"):
        blank.add(tokenize.FSTRING_MIDDLE)  # literal parts of an f-string only
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return lines

    rows = [list(line) for line in lines]
    for tok in tokens:
        if tok.type not in blank:
            continue
        (start_row, start_col), (end_row, end_col) = tok.start, tok.end
        for row_no in range(start_row, end_row + 1):
            if row_no - 1 >= len(rows):
                break
            row = rows[row_no - 1]
            first = start_col if row_no == start_row else 0
            last = end_col if row_no == end_row else len(row)
            for col in range(first, min(last, len(row))):
                row[col] = " "
    return ["".join(row) for row in rows]


SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".eggs",
        ".ruff_cache",
    }
)


def imported_names(tree: ast.AST, package: str) -> dict[str, str]:
    """Local name -> dotted path within `package`, for one module's imports.

    `from requests import Session as S` gives {"S": "Session"}; `import requests`
    gives {"requests": ""}, so `requests.get` is matched by attribute later.
    Only imports of the package in question are returned, which is what keeps
    the scan from matching same-named symbols that have nothing to do with it.
    """
    found: dict[str, str] = {}
    root = package.split(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == root or alias.name.startswith(f"{root}."):
                    found[alias.asname or alias.name.split(".")[0]] = ""
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module != root and not module.startswith(f"{root}."):
                continue
            prefix = module[len(root) :].lstrip(".")
            for alias in node.names:
                if alias.name == "*":
                    continue
                dotted = f"{prefix}.{alias.name}" if prefix else alias.name
                found[alias.asname or alias.name] = dotted
    return found


def _kind(line: str, name: str) -> RefKind:
    """How this line uses `name`. Cheap, and only ever narrows the severity."""
    if re.search(rf"\bclass\s+\w+\s*\([^)]*\b{re.escape(name)}\b", line):
        return RefKind.SUBCLASS
    if re.search(rf"\b{re.escape(name)}\s*\(", line):
        return RefKind.CALL
    if re.search(rf"(:\s*|->\s*)\[?\b{re.escape(name)}\b", line):
        return RefKind.ANNOTATION
    return RefKind.MENTION


def _source_files(root: Path) -> list[Path]:
    return [
        p
        for p in sorted(root.rglob("*.py"))
        if not any(part in SKIP_DIRS for part in p.relative_to(root).parts)
    ]


def via_path(
    package: str, symbol: str, imports: dict[str, str], code: str, leaf: str
) -> str | None:
    """The dotted path, package included, that this line uses to reach `symbol`.

    griffe names where a symbol is defined and code names where it imports it
    from; a witness has to test the second. For a from-import the path is what
    the file imported plus whatever of `symbol` lies beyond it:
    `from paylib import Session` reaching `Session.drain` gives
    `paylib.Session.drain`, and `from pydantic import ValidationError` reaching
    `error_wrappers.ValidationError` gives `pydantic.ValidationError`.

    For a module import it is read off the line, as `alias.chain.leaf`. Anything
    else, a method on an object whose type the line does not show, has no path a
    single import can test and returns None.
    """
    root = package.split(".")[0]
    parts = symbol.split(".")
    for target in imports.values():
        if not target:
            continue
        target_leaf = target.split(".")[-1]
        if target_leaf in parts:
            rest = parts[parts.index(target_leaf) + 1 :]
            return ".".join([root, target, *rest])
    for local, target in imports.items():
        if target:
            continue
        found = re.search(rf"\b{re.escape(local)}\.((?:\w+\.)*{re.escape(leaf)})\b", code)
        if found:
            return f"{root}.{found.group(1)}"
    return None


def find_sites(repo: Path, package: str, symbols: set[str]) -> list[Site]:
    """Every place in `repo` that touches one of `symbols` from `package`.

    `symbols` are dotted paths below the package, e.g. {"Session.request",
    "get", "adapters.HTTPAdapter"}. A site matches when the file imports
    something that leads to the symbol and the line names its last component.

    The last component is what the source actually says: code that imported
    `Session` writes `s.request(...)`, never `Session.request(...)`. Matching on
    the leaf is what makes a method on an imported class findable at all.
    """
    if not symbols:
        return []

    sites: list[Site] = []
    for path in _source_files(repo):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tree = parse_source(source)
        if tree is None:
            continue  # not valid Python for this interpreter: skip, quietly

        imports = imported_names(tree, package)
        if not imports:
            continue

        # A symbol is reachable if the file imported any component of its path.
        # Matching components rather than prefixes is what bridges re-exports:
        # code writes `from pydantic import validator`, while the change is
        # reported against `deprecated.class_validators.validator`, so neither
        # name is a prefix of the other but they share a component.
        module_imported = any(target == "" for target in imports.values())
        bound = set(imports) | {t.split(".")[-1] for t in imports.values() if t}
        reachable = {
            symbol for symbol in symbols if module_imported or (set(symbol.split(".")) & bound)
        }
        if not reachable:
            continue

        # Group by the name the source will actually say, and by how it will say
        # it. A directly imported name is written bare; anything reached through
        # an object is written as an attribute. Requiring the dot for the second
        # case is what stops a change to `BaseModel.dict` matching every call to
        # the `dict` builtin in a file that happens to import BaseModel.
        wanted: dict[tuple[str, bool], set[str]] = {}
        for symbol in reachable:
            parts = symbol.split(".")
            leaf = parts[-1]
            if leaf == "__init__" and len(parts) > 1:
                # Calling `Session(...)` invokes `Session.__init__`, but the
                # source never writes the constructor's name. A constructor
                # gaining a required argument is the most common breakage shape
                # there is, so searching for `__init__` would miss all of them.
                leaf = parts[-2]
            wanted.setdefault((leaf, leaf in bound), set()).add(symbol)

        relative = path.relative_to(repo).as_posix()
        # Match against code only, but report the real line: `text` keeps what
        # was actually written, for the reader and for narrow()'s keyword check.
        raw_lines = source.splitlines()
        for lineno, line in enumerate(code_lines(source), start=1):
            stripped = raw_lines[lineno - 1].strip() if lineno <= len(raw_lines) else ""
            if not line.strip():
                continue  # blank, or nothing left once strings and comments go
            for (leaf, bare), owners in wanted.items():
                pattern = rf"\b{re.escape(leaf)}\b" if bare else rf"\.{re.escape(leaf)}\b"
                if not re.search(pattern, line):
                    continue
                kind = _kind(line, leaf)
                for symbol in owners:
                    sites.append(
                        Site(
                            file=relative,
                            line=lineno,
                            symbol=symbol,
                            kind=kind,
                            text=stripped[:200],
                            via=via_path(package, symbol, imports, line, leaf),
                        )
                    )
    return sites
