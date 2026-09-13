"""Would this call still bind against the new signature?

A witnessed signature change is not yet a broken call. click.Option's
constructor really did change between click 7 and 8, and flask's
`click.Option(["--version"], is_flag=True, ...)` works on both, because the
arguments it passes are ones the new signature accepts. Across twenty
repositories we did not write, every signature BREAK left after witnessing was
exactly this: a real change that the call in question did not care about.

So the call's literal arguments are bound against the new parameters, following
the rules Python applies: too many positional arguments, a keyword the function
does not take, a value given twice, or a required parameter left unfilled.

The call is read from the parsed file when there is one, so a call spread over
several lines binds like any other: click's own `click.Option(` calls put every
argument on a line of its own. Only a call passing *args or **kwargs, whose
contents the source does not show, is left undecided.
"""

from __future__ import annotations

import ast

POSITIONAL = frozenset({"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"})
KEYWORDABLE = frozenset({"POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"})

# Changes a successful bind settles. A moved parameter can still bind while
# handing a positional argument to a different parameter, and a changed default
# alters behaviour without touching the call, so neither is cleared by binding.
BIND_DECIDES = frozenset(
    {
        "PARAMETER_ADDED_REQUIRED",
        "PARAMETER_REMOVED",
        "PARAMETER_CHANGED_REQUIRED",
        "PARAMETER_CHANGED_KIND",
    }
)


def call_shapes(text: str, name: str) -> list[tuple[int, list[str]]] | None:
    """(positional count, keyword names) for each call to `name` on this line.

    None when a line cannot decide it: it does not parse on its own, nothing on
    it calls `name`, or a call passes *args or **kwargs whose contents the line
    does not show.
    """
    tree = None
    # Header lines need a body to parse: `if check(x):`, `with open(p) as f:`,
    # and a decorator needs something to decorate.
    for candidate in (text, f"{text} pass", f"{text}\ndef _(): pass"):
        try:
            tree = ast.parse(candidate)
            break
        except SyntaxError:
            continue
    if tree is None:
        return None
    return tree_call_shapes(tree, name)


def tree_call_shapes(
    tree: ast.AST, name: str, line: int | None = None
) -> list[tuple[int, list[str]]] | None:
    """Like call_shapes, over a parsed tree, limited to calls whose name ends on `line`.

    A site's line is where the called name appears, which for a call split over
    several lines is the line with the opening parenthesis, not where it ends.
    """
    shapes: list[tuple[int, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called = func.id
        elif isinstance(func, ast.Attribute):
            called = func.attr
        else:
            continue
        if called != name or (line is not None and func.end_lineno != line):
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            kw.arg is None for kw in node.keywords
        ):
            return None
        shapes.append((len(node.args), [kw.arg for kw in node.keywords if kw.arg]))
    return shapes or None


def binds(params: list, positional: int, keywords: list[str]) -> bool:
    """Whether a call passing this many positionals and these keywords binds.

    `params` are (name, kind, default) as the probe reports them, where default
    is None when the parameter has none. A leading `self` is skipped: the probe
    describes a method as it sits on the class, and the call passes it implicitly.
    """
    params = [tuple(p) for p in params]
    if params and params[0][0] == "self":
        params = params[1:]

    slots = [p for p in params if p[1] in POSITIONAL]
    var_positional = any(p[1] == "VAR_POSITIONAL" for p in params)
    var_keyword = any(p[1] == "VAR_KEYWORD" for p in params)
    if positional > len(slots) and not var_positional:
        return False

    filled = {p[0] for p in slots[:positional]}
    by_name = {p[0]: p for p in params}
    for keyword in keywords:
        param = by_name.get(keyword)
        if param is None or param[1] not in KEYWORDABLE:
            if not var_keyword:
                return False
            continue
        if keyword in filled:
            return False  # given positionally and again by keyword
        filled.add(keyword)

    required = POSITIONAL | {"KEYWORD_ONLY"}
    return all(p[0] in filled for p in params if p[1] in required and p[2] is None)


def still_binds(
    text: str, fqn: str, params: list, tree: ast.AST | None = None, line: int | None = None
) -> bool | None:
    """Whether every call to the changed symbol on this line fits `params`.

    With the file's `tree` and the site's `line`, the calls are read from the
    tree; otherwise from the line's text alone.

    A constructor change is reported against `X.__init__`, but the line calls
    `X(...)`, so the class name is what gets searched for.
    """
    parts = fqn.split(".")
    name = parts[-2] if parts[-1] == "__init__" and len(parts) > 1 else parts[-1]
    if tree is not None and line is not None:
        shapes = tree_call_shapes(tree, name, line)
    else:
        shapes = call_shapes(text, name)
    if not shapes:
        return None
    return all(binds(params, positional, keywords) for positional, keywords in shapes)
