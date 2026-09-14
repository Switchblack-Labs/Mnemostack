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
import functools
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
    # Prefer the import of the class a member belongs to. Any other import whose
    # name merely appears in the path builds a path to something the line never
    # touched, and the witness then faithfully confirms that the wrong thing is
    # gone: `Table.exists` tested through an unrelated `from sqlalchemy.sql
    # import exists` did exactly that.
    owner = parts[-2] if len(parts) > 1 else None
    targets = sorted((t for t in imports.values() if t), key=lambda t: t.split(".")[-1] != owner)
    for target in targets:
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


_FALLBACK_ERRORS = frozenset({"ImportError", "ModuleNotFoundError", "AttributeError"})


def _import_bindings(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names |= {a.asname or a.name.split(".")[0] for a in node.names}
    return names


def compat_guards(tree: ast.AST) -> tuple[set[int], set[str]]:
    """Lines inside compatibility shims, and names the shims bind either way.

    Two shapes, both found guarding removals across twenty repositories we did
    not write: mkdocs's `try: from jinja2 import pass_context as contextfilter /
    except ImportError: from jinja2 import contextfilter`, and starlette's
    `if hasattr(jinja2, "pass_context"): ... else: jinja2.contextfunction`.

    A try only counts when a handler offers an alternative. `except ImportError:
    raise ImportError("install jinja2")` is an optional dependency, not a shim,
    and counting it would hedge every use of the package in the module.

    A name imported in both the try and a handler is hedged wherever it is used,
    which is how mkdocs's `@contextfilter` decorators work on either version. A
    handler that sets `np = None` binds by assignment and hedges nothing.
    """
    lines: set[int] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            fallbacks = []
            for handler in node.handlers:
                types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
                caught = {getattr(t, "id", None) or getattr(t, "attr", None) for t in types}
                offers_alternative = not all(isinstance(s, ast.Raise) for s in handler.body)
                if caught & _FALLBACK_ERRORS and offers_alternative:
                    fallbacks.append(handler)
            if not fallbacks:
                continue
            region = [*node.body, *(s for h in fallbacks for s in h.body)]
            tried = _import_bindings(node.body)
            names |= {n for h in fallbacks for n in _import_bindings(h.body) & tried}
        elif isinstance(node, ast.If) and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hasattr"
            for n in ast.walk(node.test)
        ):
            # ponytail: hasattr only; version comparisons (sys.version_info,
            # __version__ >= ...) are not recognised, add them when a real shim needs it
            region = [*node.body, *node.orelse]
        else:
            continue
        for statement in region:
            lines.update(range(statement.lineno, (statement.end_lineno or statement.lineno) + 1))
    return lines, names


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_EVERYWHERE = 10**9


def _dotted(node: ast.AST) -> str | None:
    """`a.b.C` for a name or an attribute chain, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


# Subscripts whose arguments are what the value is, rather than what it holds.
_UNWRAP = {"Optional", "Union", "Annotated", "Final", "ClassVar", "type", "Type"}


def _annotated_types(annotation: ast.AST) -> set[str]:
    """The types a value annotated this way can be.

    `User`, `"User"`, `User | None`, `Optional[User]`, `Annotated[User, ...]` and
    `type[User]` can all be a User. `dict[User, int]` cannot: it is a dict.
    Counting every name an annotation mentions made onegov-cloud's
    `_DATAMANAGERS: WeakKeyDictionary[Session, ...]` a Session, so its dict
    `.get` answered for a change to `Session.get`.
    """
    node = annotation
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value.strip(), mode="eval").body
        except SyntaxError:
            return set()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotated_types(node.left) | _annotated_types(node.right)
    if isinstance(node, ast.Subscript):
        outer = _dotted(node.value)
        if outer is None:
            return set()
        if outer.rsplit(".", 1)[-1] not in _UNWRAP:
            return {outer}
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if outer.rsplit(".", 1)[-1] == "Annotated":
            args = args[:1]
        return set().union(*(_annotated_types(arg) for arg in args))
    dotted = _dotted(node)
    return {dotted} if dotted else set()


def _scoped(tree: ast.AST):
    """Every node, with the (first, last) lines of the scope a name bound there lives in."""
    stack = [(tree, (1, _EVERYWHERE))]
    while stack:
        node, span = stack.pop()
        for child in ast.iter_child_nodes(node):
            yield child, span
            inner = span
            if isinstance(child, _SCOPES):
                inner = (child.lineno, child.end_lineno or child.lineno)
            stack.append((child, inner))


def _receivers(owner: str, imports: dict[str, str], tree: ast.AST) -> list[tuple[str, int, int]]:
    """(name, first line, last line) for names that can refer to class `owner` or hold one.

    A member such as `Table.exists` is only reachable through a receiver the file
    grounds itself: the name the class was imported as, a class here that
    subclasses it, or a variable assigned from, annotated as, or opened with one
    of those. Inside the body of such a subclass, `self` and `cls` count too.

    Accepting any `.exists` let `super().execute(...)` in a Session subclass
    answer for a removed `Executable.execute`, and a bare `exists` imported from
    elsewhere answer for `Table.exists`. Across twenty repositories we did not
    write, those were most of the removals the witness kept.

    Each name counts only on the lines of the scope that binds it. Matched across
    the whole file instead, a `session: Session` parameter in one function made
    `session.get(url)` on a requests session in another a SQLAlchemy call, and
    a payment model's `self.transaction` a removed Session attribute.
    """
    base = {local for local, target in imports.items() if target.split(".")[-1] == owner}
    base |= {f"{local}.{owner}" for local, target in imports.items() if not target}
    if not base:
        return []

    found = {(name, 1, _EVERYWHERE) for name in base}

    def known(name: str | None, line: int) -> bool:
        return any(n == name and first <= line <= last for n, first, last in found)

    nodes = list(_scoped(tree))
    for _ in range(3):  # a subclass of a subclass, a variable of a subclass
        grown = set(found)
        for node, (first, last) in nodes:
            if isinstance(node, ast.ClassDef):
                bases = (b.value if isinstance(b, ast.Subscript) else b for b in node.bases)
                if any(known(_dotted(b), node.lineno) for b in bases):
                    end = node.end_lineno or node.lineno
                    grown |= {(node.name, first, last), ("self", node.lineno, end)}
                    grown.add(("cls", node.lineno, end))
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if known(_dotted(node.value.func), node.lineno):
                    grown |= {(t.id, first, last) for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if any(known(n, node.lineno) for n in _annotated_types(node.annotation)):
                    grown.add((node.target.id, first, last))
            elif isinstance(node, ast.arg) and node.annotation is not None:
                if any(known(n, node.lineno) for n in _annotated_types(node.annotation)):
                    grown.add((node.arg, first, last))
            elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                call = node.context_expr
                if isinstance(call, ast.Call) and known(_dotted(call.func), call.lineno):
                    grown.add((node.optional_vars.id, first, last))
        if grown == found:
            break
        found = grown
    return sorted(found)


@functools.lru_cache(maxsize=4096)
def _member_pattern(names: frozenset[str], leaf: str) -> re.Pattern[str]:
    alternation = "|".join(sorted(map(re.escape, names)))
    return re.compile(rf"(?<![\w.])(?:{alternation})\.{re.escape(leaf)}\b")


def find_sites(
    repo: Path, package: str, symbols: set[str], members: set[str] | None = None
) -> list[Site]:
    """Every place in `repo` that touches one of `symbols` from `package`.

    `members` are the symbols whose owner is a class, as the library's API says.
    Without it, a capitalised owner is taken to be a class, which is wrong for
    sqlalchemy's `array` and `func`: spelled lowercase, `array.append` fell to
    matching any `.append`, and on onegov-cloud that was 46 BREAKs on list appends.

    `symbols` are dotted paths below the package, e.g. {"Session.request",
    "get", "adapters.HTTPAdapter"}. A site matches when the file imports
    something that leads to the symbol and the line names it the way code does:
    bare if it was imported directly, as its class if it is a constructor, and as
    `receiver.member` if it belongs to a class, where the receiver has to be
    grounded in this file rather than be any object with a same-named attribute.
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

        code = code_lines(source)
        receiver_cache: dict[str, list[tuple[str, int, int]]] = {}

        # One pattern per way the source can name a symbol. A directly imported
        # name is written bare. A member of a class is written `receiver.member`
        # with a grounded receiver. Anything else is an attribute path starting
        # from a name this file imported: `requests.get`, `sa.orm.relationship`.
        wanted: dict[str, tuple[str, bool, set[str]]] = {}
        # Members are matched per line, against the receivers in scope there.
        member_wanted: dict[tuple[str, str], set[str]] = {}
        for symbol in reachable:
            parts = symbol.split(".")
            leaf = parts[-1]
            if leaf == "__init__" and len(parts) > 1:
                # Calling `Session(...)` invokes `Session.__init__`, but the
                # source never writes the constructor's name, so search for the
                # class being called. Calls only: importing BaseModel or
                # declaring `class User(BaseModel)` does not run BaseModel's
                # constructor, and matching those hung a constructor change on
                # every model declaration and import line in a pydantic codebase.
                leaf = parts[-2]
                names = {
                    local
                    for local, target in imports.items()
                    if target and target.split(".")[-1] == leaf
                } | {f"{local}.{leaf}" for local, target in imports.items() if not target}
                if not names:
                    continue
                bare = False
                alternation = "|".join(sorted(map(re.escape, names)))
                pattern = rf"(?<![\w.])(?:{alternation})\s*\("
            elif len(parts) > 1 and (
                symbol in members if members is not None else parts[-2][:1].isupper()
            ):
                owner = parts[-2]
                if owner not in receiver_cache:
                    receiver_cache[owner] = _receivers(owner, imports, tree)
                if receiver_cache[owner]:  # otherwise nothing here can hold one
                    member_wanted.setdefault((owner, leaf), set()).add(symbol)
                continue
            else:
                bare = leaf in bound
                if bare:
                    pattern = rf"\b{re.escape(leaf)}\b"
                else:
                    # Any `.get` at all matched `cfg.get("token")` for a change to
                    # requests.get. The path has to start from an imported name.
                    alternation = "|".join(sorted(map(re.escape, imports)))
                    pattern = rf"(?<![\w.])(?:{alternation})(?:\.\w+)*\.{re.escape(leaf)}\b"
            wanted.setdefault(pattern, (leaf, bare, set()))[2].add(symbol)

        relative = path.relative_to(repo).as_posix()
        # From the parse, so every line of `from x import (\n a,\n b,\n)` counts.
        import_lines = {
            n
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for n in range(node.lineno, (node.end_lineno or node.lineno) + 1)
        }
        guard_lines, guard_names = compat_guards(tree)
        hedged = (
            re.compile(rf"(?<![\w.])(?:{'|'.join(sorted(map(re.escape, guard_names)))})\b")
            if guard_names
            else None
        )
        # Match against code only, but report the real line: `text` keeps what
        # was actually written, for the reader.
        raw_lines = source.splitlines()
        for lineno, line in enumerate(code, start=1):
            stripped = raw_lines[lineno - 1].strip() if lineno <= len(raw_lines) else ""
            if not line.strip():
                continue  # blank, or nothing left once strings and comments go
            is_import = lineno in import_lines
            guarded = lineno in guard_lines or bool(hedged and hedged.search(line))
            # A dotted module path in an import names no attribute, so on import
            # lines only bare names can match.
            matched = [
                (leaf, owners, False)
                for pattern, (leaf, bare, owners) in wanted.items()
                if (bare or not is_import) and re.search(pattern, line)
            ]
            if not is_import:
                for (owner, leaf), owners in member_wanted.items():
                    receivers = receiver_cache[owner]
                    scoped = frozenset(n for n, first, last in receivers if first <= lineno <= last)
                    if scoped and _member_pattern(scoped, leaf).search(line):
                        matched.append((leaf, owners, False))
                        continue
                    # A name bound as the class in another scope is a guess. Not
                    # self or cls: in another class those are that class.
                    elsewhere = frozenset(n for n, _, _ in receivers) - scoped - {"self", "cls"}
                    if elsewhere and _member_pattern(elsewhere, leaf).search(line):
                        matched.append((leaf, owners, True))
            for leaf, owners, unscoped in matched:
                kind = RefKind.IMPORT if is_import else _kind(line, leaf)
                for symbol in owners:
                    sites.append(
                        Site(
                            file=relative,
                            line=lineno,
                            symbol=symbol,
                            kind=kind,
                            text=stripped[:200],
                            via=via_path(package, symbol, imports, line, leaf),
                            guarded=guarded,
                            unscoped=unscoped,
                        )
                    )
    return sites
