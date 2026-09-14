"""Where a codebase touches a library's symbols.

Sites are found by reading the source; see static.py.

The interface is deliberately a list of sites rather than a graph. Every version
of this that tried to model the code as a graph spent its budget on resolution
and its accuracy on Python's dynamism. A site is a file, a line, and how the
symbol was used, which is all the severity table needs and all a reader wants.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RefKind(str, Enum):
    """How a symbol is used at one site.

    This is the axis severity depends on: removing a base class breaks a
    subclass and merely concerns a caller, and a moved parameter breaks a
    positional call and does nothing to an annotation.
    """

    CALL = "call"  # name(...)
    SUBCLASS = "subclass"  # class X(name)
    ANNOTATION = "annotation"  # x: name, -> name
    IMPORT = "import"  # from x import name: only a removal fails on this line
    MENTION = "mention"  # anything else: isinstance, a value, a decorator


@dataclass(frozen=True)
class Site:
    """One place in the user's code that touches one library symbol."""

    file: str  # relative to the repo root
    line: int  # 1-based
    symbol: str  # the library symbol, as the user's code names it
    kind: RefKind
    text: str  # the source line, so the reader can judge without opening it
    via: str | None = None
    """The dotted path, package included, that this code reaches the symbol by.

    griffe names where a symbol is defined; code names where it imports it from.
    Those differ whenever a package re-exports, which is most of the time, and a
    witness has to test the path the code actually uses:
    `pydantic.error_wrappers.ValidationError` is gone in v2, while
    `from pydantic import ValidationError` works fine. None when the line reaches
    the symbol through an object whose type a line of source does not reveal.
    """
    guarded: bool = False
    """Whether the code hedges on this symbol being there.

    A line inside a compatibility shim, `try: from x import new / except
    ImportError: from x import old` or `if hasattr(x, "new"): ... else: x.old`,
    was written expecting the symbol to disappear. Its removal is a line to clean
    up, not a break.
    """
    unscoped: bool = False
    """Whether the receiver is known to be the class only somewhere else in the file.

    click's `ctx.invoke(...)` gets `ctx` from a decorator, and matches only
    because another function binds `ctx = click.Context(...)`. The same name is
    usually the same type, but not always: onegov-cloud's `session` is a requests
    session in one function and a SQLAlchemy one in another. Such a line is
    worth a look, and never shown to break.
    """
