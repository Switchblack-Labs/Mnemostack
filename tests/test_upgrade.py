"""End to end, and the two filters that decide whether the report is trusted.

A tool that flags every use of a changed function trains you to ignore it. Both
filters below were wrong in the previous design and neither failure was visible
from the tests that existed then: one deleted true positives, the other never
fired at all. So each is now tested in both directions.
"""

from __future__ import annotations

from pathlib import Path

import griffe
import pytest

from mnemostack.core.impact.api_diff import ApiChange
from mnemostack.core.impact.propagate import impact_report
from mnemostack.core.impact.upgrade import (
    UpgradeError,
    changed_symbols,
    check_upgrade,
    verify,
)
from mnemostack.core.reach.static import find_sites

LIB_V1 = """
class Session:
    def __init__(self, url, *, timeout=None, retries=0):
        self.url = url

    def send(self, body):
        return body

    def drain(self):
        return None


def connect(url):
    return Session(url)
"""

LIB_V2 = """
class Session:
    # `timeout` is gone: breaks only callers that passed it.
    def __init__(self, url, *, retries=0):
        self.url = url

    def send(self, body):
        return body

    # `drain` is gone: breaks every caller.


def connect(url, token):
    # `token` is now required: breaks every caller.
    return Session(url)
"""

CONSUMER = """
from paylib import Session, connect


def passes_the_removed_kwarg():
    return Session("http://x", timeout=5)


def does_not_pass_it():
    return Session("http://x", retries=2)


def calls_removed_method():
    s = Session("http://x")
    return s.drain()


def calls_changed_function():
    return connect("http://x")
"""


def _write_pkg(root: Path, body: str) -> Path:
    pkg = root / "paylib"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(body)
    return root


def _changes(old, new) -> list[ApiChange]:
    return [
        ApiChange(
            fqn=str(d.get("object_path")),
            kind=getattr(d.get("kind"), "name", str(d.get("kind"))),
            old=None if d.get("old_value") is None else str(d.get("old_value")),
            new=None if d.get("new_value") is None else str(d.get("new_value")),
        )
        for d in (b.as_dict() for b in griffe.find_breaking_changes(old, new))
    ]


@pytest.fixture
def scenario(tmp_path: Path):
    old = griffe.load("paylib", search_paths=[_write_pkg(tmp_path / "v1", LIB_V1)])
    new = griffe.load("paylib", search_paths=[_write_pkg(tmp_path / "v2", LIB_V2)])

    consumer = tmp_path / "app"
    consumer.mkdir()
    (consumer / "handlers.py").write_text(CONSUMER)

    changes = verify(_changes(old, new), new)
    sites = find_sites(consumer, "paylib", changed_symbols(changes, "paylib"))
    yield changes, sites


def _report(scenario):
    changes, sites = scenario
    return impact_report(sites, changes)


def _texts(report) -> str:
    return " | ".join(i.site.text for i in report)


# --- the filters ------------------------------------------------------------


def test_verify_keeps_a_real_removal(scenario):
    changes, _ = scenario
    assert ("paylib.Session.drain", "OBJECT_REMOVED") in {(c.fqn, c.kind) for c in changes}


def test_verify_drops_a_removal_the_new_source_contradicts(tmp_path: Path):
    """griffe drops @overload members, reporting present methods as removed.

    The previous implementation checked the package's top-level __init__, which
    for any real package is a re-export facade containing no definitions, so it
    always returned False and the filter silently never fired on anything.
    """
    src = (
        "from typing import overload\n"
        "\n"
        "\n"
        "class Thing:\n"
        "    @overload\n"
        "    def go(self, x: int) -> None: ...\n"
        "    @overload\n"
        "    def go(self, x: str) -> None: ...\n"
        "    def go(self, x): ...\n"
    )
    new = griffe.load("paylib", search_paths=[_write_pkg(tmp_path / "v2", src)])
    invented = ApiChange(fqn="paylib.Thing.go", kind="OBJECT_REMOVED", old=None, new=None)

    assert verify([invented], new) == [], "a method still in the source is not removed"


def test_verify_keeps_a_removal_when_only_another_class_has_the_name(tmp_path: Path):
    """`def close` anywhere in the file used to count as `A.close` still existing."""
    v1 = "class A:\n    def close(self): ...\n\n\nclass B:\n    def close(self): ...\n"
    v2 = "class A:\n    pass\n\n\nclass B:\n    def close(self): ...\n"
    old = griffe.load("paylib", search_paths=[_write_pkg(tmp_path / "v1", v1)])
    new = griffe.load("paylib", search_paths=[_write_pkg(tmp_path / "v2", v2)])

    kept = {(c.fqn, c.kind) for c in verify(_changes(old, new), new)}
    assert ("paylib.A.close", "OBJECT_REMOVED") in kept


# --- end to end -------------------------------------------------------------


def test_removed_method_and_new_required_parameter_are_reported(scenario):
    texts = _texts(_report(scenario))
    assert "s.drain()" in texts
    assert 'connect("http://x")' in texts


def test_changed_symbols_are_relative_to_the_package():
    changes = [ApiChange(fqn="paylib.core.Session.send", kind="OBJECT_REMOVED", old=None, new=None)]
    assert changed_symbols(changes, "paylib") == {"core.Session.send"}


def test_a_bad_version_is_an_error_not_a_traceback(tmp_path: Path):
    """This is pitched as a CI gate, so every failure must be actionable."""
    with pytest.raises(UpgradeError) as exc:
        check_upgrade(
            repo=tmp_path, package="griffe", to_version="999.999.999", from_version="1.4.0"
        )
    assert "griffe" in str(exc.value)


def test_no_installed_version_is_an_error_naming_where_it_looked(tmp_path: Path, monkeypatch):
    """Never a fallback to the environment mnemostack itself runs in."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    with pytest.raises(UpgradeError) as exc:
        check_upgrade(repo=tmp_path, package="griffe", to_version="1.5.0")
    assert "--from-version" in str(exc.value)
    assert "griffe" in str(exc.value)


def test_check_upgrade_accepts_externally_measured_sites(tmp_path: Path):
    """A caller may supply sites found some other way instead of the static scan."""
    (tmp_path / "app.py").write_text("import griffe\n")
    report = check_upgrade(
        repo=tmp_path,
        package="griffe",
        to_version="1.5.0",
        from_version="1.4.0",
        sites=[],
    )
    assert report.impacts == []
    assert report.to_version == "1.5.0"


def test_pypi_cache_directory_is_created_if_missing(tmp_path, monkeypatch):
    """The first-run failure that CI caught.

    griffe's load_pypi opens a TemporaryDirectory inside its cache directory
    but never creates it, so on a machine that has never used griffe the first
    run dies with a FileNotFoundError naming a temp path the user has never
    seen. That is every new user.
    """
    import platformdirs

    from mnemostack.core.impact.upgrade import _ensure_pypi_cache

    cache = tmp_path / "nested" / "griffe"
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *a, **k: str(cache))

    assert not cache.exists()
    _ensure_pypi_cache()
    assert cache.is_dir()


# --- deprecations: the thing a green test run hides -------------------------

DEPRECATED_LIB = """
from typing_extensions import deprecated


class Model:
    @deprecated("The `dict` method is deprecated; use `model_dump` instead.")
    def dict(self):
        return {}

    def model_dump(self):
        return {}
"""


def test_deprecated_symbols_reads_the_decorator_not_griffes_flag(tmp_path: Path):
    """griffe's own is_deprecated does not see the decorator real packages use.

    It returns False for `@typing_extensions.deprecated`, which is what pydantic
    marks `dict`, `copy` and `parse_obj` with. The decorator text is populated
    though, so the message is read from there.
    """
    from mnemostack.core.impact.upgrade import deprecated_symbols

    mod = griffe.load("paylib", search_paths=[_write_pkg(tmp_path / "v", DEPRECATED_LIB)])
    assert mod["Model.dict"].is_deprecated is False, "the flag griffe offers is not usable"

    found = deprecated_symbols(mod)
    assert "Model.dict" in found
    assert "model_dump" in found["Model.dict"]
    assert "Model.model_dump" not in found


def test_deprecations_are_reported_once_per_object_at_its_public_path():
    """griffe reaches a class through every module that imports it.

    Unfiltered, three deprecated methods became 105 findings for the same three
    lines. Folding them by name instead merged different deprecated methods that
    happen to share one, so they are folded by what object they are.
    """
    from mnemostack.core.impact.upgrade import deprecated_symbols

    inner = (
        "from typing_extensions import deprecated\n\n\n"
        "class Model:\n    @deprecated('use model_dump')\n    def dict(self): ...\n\n\n"
        "class Other:\n    @deprecated('')\n    def dict(self): ...\n"
    )
    files = {"__init__.py": "from paylib.inner import Model\n", "inner.py": inner}
    with griffe.temporary_visited_package("paylib", files) as pkg:
        found = deprecated_symbols(pkg)
    assert found == {"Model.dict": "use model_dump", "inner.Other.dict": ""}


def test_a_bounded_walk_survives_an_import_cycle(tmp_path: Path):
    """The regression that wrote 3.1 GB into a user's repository.

    A module that re-exports its own parent used to generate infinitely
    lengthening paths. The walk is bounded and tracks what it has seen.
    """
    from mnemostack.core.impact.upgrade import deprecated_symbols

    root = tmp_path / "v"
    pkg = root / "paylib"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("from paylib.sub import thing\n")
    (pkg / "sub" / "__init__.py").write_text("import paylib\n\n\ndef thing():\n    return 1\n")

    assert deprecated_symbols(griffe.load("paylib", search_paths=[root])) == {}


def test_class_members_are_read_from_the_api():
    import griffe

    from mnemostack.core.impact.upgrade import class_members

    code = "class array:\n    def append(self): ...\n\ndef helper(): ...\n"
    with griffe.temporary_visited_module(code) as module:
        assert class_members(module, {"array.append", "helper", "gone.thing"}) == {"array.append"}
