"""Where a codebase touches a library's symbols.

Two providers answer the same question differently. Static reads the source and
infers; runtime watches a test run and observes. Both return the same shape, so
the report does not care which one produced it, and a project without a usable
test suite still gets an answer.

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
    MENTION = "mention"  # anything else: isinstance, a value, a decorator


@dataclass(frozen=True)
class Site:
    """One place in the user's code that touches one library symbol."""

    file: str  # relative to the repo root
    line: int  # 1-based
    symbol: str  # the library symbol, as the user's code names it
    kind: RefKind
    text: str  # the source line, so the reader can judge without opening it
    covered: bool | None = None
    """Whether the test suite exercises this site.

    None when nothing measured it. The runtime provider sets it; that split,
    between sites your tests will catch and sites they will not, is the one
    thing a test suite cannot tell you about itself.
    """
    via: str | None = None
    """The dotted path, package included, that this code reaches the symbol by.

    griffe names where a symbol is defined; code names where it imports it from.
    Those differ whenever a package re-exports, which is most of the time, and a
    witness has to test the path the code actually uses:
    `pydantic.error_wrappers.ValidationError` is gone in v2, while
    `from pydantic import ValidationError` works fine. None when the line reaches
    the symbol through an object whose type a line of source does not reveal.
    """
