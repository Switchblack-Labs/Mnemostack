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
import re
from pathlib import Path

from mnemostack.core.reach import RefKind, Site

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
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue  # unreadable or not valid for this interpreter: skip, quietly

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
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
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
                        )
                    )
    return sites
