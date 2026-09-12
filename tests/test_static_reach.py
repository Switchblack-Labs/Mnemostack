"""Finding where a repo touches a package's symbols.

The tests that matter here are the ones the deleted symbol graph could not
pass: a call inside a nested function, an annotation under `if TYPE_CHECKING`,
an attribute read, a conditional definition. Two thirds of its recall gap was
shapes it had no way to represent, and those shapes are the point of scanning.

The precision tests matter just as much. Scanning trades precision for reach,
and the import table is what stops that trade from being a bad one.
"""

from __future__ import annotations

import ast
from pathlib import Path

from mnemostack.core.reach import RefKind
from mnemostack.core.reach.static import find_sites, imported_names


def _symbols(sites) -> set[str]:
    return {s.symbol for s in sites}


def _kinds(sites, symbol: str) -> set[RefKind]:
    return {s.kind for s in sites if s.symbol == symbol}


# --- the import table gates everything -------------------------------------


def test_imported_names_maps_aliases_to_package_paths():
    tree = ast.parse(
        "import requests\n"
        "import requests.adapters as ad\n"
        "from requests import Session as S, get\n"
        "from requests.adapters import HTTPAdapter\n"
        "from flask import Flask\n"
    )
    table = imported_names(tree, "requests")

    assert table["S"] == "Session"
    assert table["get"] == "get"
    assert table["HTTPAdapter"] == "adapters.HTTPAdapter"
    assert table["requests"] == ""  # module itself: reachable by attribute
    assert "Flask" not in table, "imports of other packages must not appear"


def test_a_file_that_does_not_import_the_package_is_never_scanned(tree):
    """The guard that makes scanning viable.

    `get` is an extremely common name. Without the import table, every dict
    access in the repo would match a change to `requests.get`.
    """
    repo = tree(
        {
            "unrelated.py": "config = {}\n\n\ndef read():\n    return config.get('key')\n",
        }
    )
    assert find_sites(repo, "requests", {"get"}) == []


# --- shapes the symbol graph could not see ---------------------------------


def test_call_inside_a_nested_function_is_found(tree):
    repo = tree(
        {
            "app.py": (
                "from requests import get\n"
                "\n"
                "\n"
                "def outer():\n"
                "    def inner():\n"
                "        return get('http://x')\n"
                "    return inner\n"
            )
        }
    )
    sites = find_sites(repo, "requests", {"get"})
    assert "get" in _symbols(sites)
    assert RefKind.CALL in _kinds(sites, "get")


def test_annotation_under_type_checking_is_found(tree):
    """Where third-party types are named, and invisible to a top-level walk."""
    repo = tree(
        {
            "app.py": (
                "from typing import TYPE_CHECKING\n"
                "\n"
                "if TYPE_CHECKING:\n"
                "    from requests import Session\n"
                "\n"
                "\n"
                "def use(s: 'Session') -> None:\n"
                "    return None\n"
                "\n"
                "\n"
                "def use2(s: Session) -> None:\n"
                "    return None\n"
            )
        }
    )
    sites = find_sites(repo, "requests", {"Session"})
    assert RefKind.ANNOTATION in _kinds(sites, "Session")


def test_conditional_definition_is_found(tree):
    """A compat shim is the likeliest place a version-sensitive call lives."""
    repo = tree(
        {
            "app.py": (
                "from requests import get\n"
                "\n"
                "try:\n"
                "    def fetch():\n"
                "        return get('http://x')\n"
                "except ImportError:\n"
                "    fetch = None\n"
            )
        }
    )
    assert "get" in _symbols(find_sites(repo, "requests", {"get"}))


def test_method_on_an_imported_class_is_found_by_leaf_name(tree):
    """Code writes `s.request(...)`, never `Session.request(...)`.

    Matching the last component is the only way a changed method on an imported
    class is findable at all.
    """
    repo = tree(
        {
            "app.py": (
                "from requests import Session\n"
                "\n"
                "\n"
                "def go():\n"
                "    s = Session()\n"
                "    return s.request('GET', 'http://x')\n"
            )
        }
    )
    sites = find_sites(repo, "requests", {"Session.request"})
    assert "Session.request" in _symbols(sites)


def test_module_attribute_access_is_found(tree):
    repo = tree({"app.py": "import requests\n\n\ndef go():\n    return requests.get('http://x')\n"})
    assert "get" in _symbols(find_sites(repo, "requests", {"get"}))


# --- reference kinds drive severity, so they have to be right ---------------


def test_subclass_is_distinguished_from_call(tree):
    repo = tree({"app.py": ("from requests import Session\n\n\nclass Mine(Session):\n    pass\n")})
    assert RefKind.SUBCLASS in _kinds(find_sites(repo, "requests", {"Session"}), "Session")


def test_mention_is_the_fallback(tree):
    repo = tree(
        {
            "app.py": (
                "from requests import Session\n"
                "\n"
                "\n"
                "def check(x):\n"
                "    return isinstance(x, Session)\n"
            )
        }
    )
    kinds = _kinds(find_sites(repo, "requests", {"Session"}), "Session")
    assert RefKind.MENTION in kinds or RefKind.CALL in kinds


def test_comments_and_blank_lines_are_skipped(tree):
    repo = tree({"app.py": "from requests import get\n\n# get is mentioned here only\n"})
    sites = [s for s in find_sites(repo, "requests", {"get"}) if s.line > 1]
    assert sites == []


def test_unparseable_file_does_not_stop_the_scan(tree):
    """One bad file must not cost the whole report."""
    repo = tree(
        {
            "broken.py": "def (:\n",
            "good.py": "from requests import get\n\n\ndef go():\n    return get('http://x')\n",
        }
    )
    assert "get" in _symbols(find_sites(repo, "requests", {"get"}))


def test_sites_carry_their_own_line_text(tree):
    """The report shows the line, and narrow() re-reads it instead of the file."""
    repo = tree({"app.py": "from requests import get\n\n\ndef go():\n    return get(timeout=5)\n"})
    site = next(s for s in find_sites(repo, "requests", {"get"}) if s.line == 5)
    assert "timeout=5" in site.text
    assert site.file == "app.py"


def test_no_symbols_means_no_scan(tree):
    repo = tree({"app.py": "from requests import get\n"})
    assert find_sites(repo, "requests", set()) == []


def test_virtualenvs_and_caches_are_not_scanned(tree):
    repo = tree(
        {
            ".venv/lib/site.py": "from requests import get\n\n\ndef x():\n    return get('u')\n",
            "app.py": "print('hi')\n",
        }
    )
    assert find_sites(repo, "requests", {"get"}) == []


def test_scanning_a_real_package_terminates(tmp_path: Path):
    """The regression that killed the previous design.

    Its predecessor walked griffe's module tree with no cycle guard and wrote
    3.1 GB into the user's repository in 45 seconds on pydantic. Nothing here
    recurses over the package at all, so the property to pin is simply that a
    scan against a large real dependency returns.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        "from pydantic import BaseModel\n\n\nclass M(BaseModel):\n    x: int\n"
    )
    sites = find_sites(repo, "pydantic", {"BaseModel", "BaseModel.dict"})
    assert "BaseModel" in _symbols(sites)


# --- false-positive classes found on twenty repos we did not write ----------


def test_strings_and_comments_are_not_matched(tree):
    """`render('index.html')` is not a use of a removed `werkzeug.html`.

    Matching raw text credited a template filename to werkzeug 44 times in flask
    alone. Strings and comments are blanked before matching.
    """
    repo = tree(
        {
            "app.py": (
                "import werkzeug\n"
                "\n"
                "\n"
                "def page():\n"
                "    return render('index.html')  # see werkzeug.html\n"
            )
        }
    )
    assert [s for s in find_sites(repo, "werkzeug", {"html"}) if s.line == 5] == []


def test_reported_text_is_the_original_line_not_the_blanked_one(tree):
    """Blanking is for matching only. The reader sees what was written."""
    repo = tree({"app.py": "from requests import get\n\n\ndef go():\n    return get('u')  # hi\n"})
    site = next(s for s in find_sites(repo, "requests", {"get"}) if s.line == 5)
    assert site.text == "return get('u')  # hi"


def test_foreign_source_with_bad_escapes_parses_quietly(tree):
    """Code we did not write often has invalid escapes; parsing must stay quiet.

    ast.parse reports each one as a SyntaxWarning, which across twenty real
    repos was dozens of warning lines interleaved with the report.
    """
    import warnings

    repo = tree(
        {
            "app.py": (
                'from requests import get\n\nPATTERN = "\\d+"\n\n\ndef go():\n    return get("u")\n'
            )
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sites = find_sites(repo, "requests", {"get"})
    assert "get" in _symbols(sites)
    assert not [w for w in caught if issubclass(w.category, SyntaxWarning)]
