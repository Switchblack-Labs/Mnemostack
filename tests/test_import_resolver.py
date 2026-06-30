"""Tests for Python import parsing and module->file resolution.

Pure logic: no embeddings, no graph, no network.
"""

from __future__ import annotations

from mnemostack.core.retrieval.import_resolver import (
    ImportRecord,
    build_import_table,
    extract_imports,
    find_import_root,
    resolve_module_file,
)


def _rec(records: list[ImportRecord], local_name: str) -> ImportRecord:
    matches = [r for r in records if r.local_name == local_name]
    assert matches, f"no record bound as {local_name!r} in {records}"
    return matches[0]


class TestExtractImports:
    def test_from_import_single(self):
        recs = extract_imports(b"from a.b import c\n")
        r = _rec(recs, "c")
        assert (r.module, r.symbol, r.level, r.is_wildcard) == ("a.b", "c", 0, False)

    def test_from_import_alias(self):
        recs = extract_imports(b"from a.b import c as d\n")
        r = _rec(recs, "d")
        assert r.module == "a.b"
        assert r.symbol == "c"

    def test_from_import_multiple(self):
        recs = extract_imports(b"from pkg.mod import alpha, beta\n")
        assert _rec(recs, "alpha").symbol == "alpha"
        assert _rec(recs, "beta").symbol == "beta"

    def test_plain_import(self):
        recs = extract_imports(b"import a.b.c\n")
        r = _rec(recs, "a.b.c")
        assert r.module == "a.b.c"
        assert r.symbol is None
        assert r.level == 0

    def test_plain_import_alias(self):
        recs = extract_imports(b"import a.b as ab\n")
        r = _rec(recs, "ab")
        assert r.module == "a.b"
        assert r.symbol is None

    def test_plain_import_multiple(self):
        recs = extract_imports(b"import os, sys\n")
        assert _rec(recs, "os").module == "os"
        assert _rec(recs, "sys").module == "sys"

    def test_relative_import_current_package(self):
        recs = extract_imports(b"from . import g\n")
        r = _rec(recs, "g")
        assert r.level == 1
        assert r.module == ""
        assert r.symbol == "g"

    def test_relative_import_submodule(self):
        recs = extract_imports(b"from .mod import h\n")
        r = _rec(recs, "h")
        assert r.level == 1
        assert r.module == "mod"

    def test_relative_import_parent(self):
        recs = extract_imports(b"from ..pkg.mod import i\n")
        r = _rec(recs, "i")
        assert r.level == 2
        assert r.module == "pkg.mod"

    def test_wildcard_import_excluded_from_table(self):
        table = build_import_table(b"from a.b import *\n", importing_file=None)  # type: ignore[arg-type]
        assert table == {}

    def test_module_name_not_treated_as_symbol(self):
        # 'a' and 'b' are the module; only 'c' is imported.
        recs = extract_imports(b"from a.b import c\n")
        assert {r.local_name for r in recs} == {"c"}


class TestResolveModuleFile:
    def _make_pkg(self, tmp_path):
        # root/
        #   topkg/__init__.py
        #     util.py
        #     sub/__init__.py
        #       deep.py
        root = tmp_path / "root"
        pkg = root / "topkg"
        sub = pkg / "sub"
        sub.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (sub / "__init__.py").write_text("")
        (pkg / "util.py").write_text("def helper():\n    pass\n")
        (sub / "deep.py").write_text("def deep_fn():\n    pass\n")
        return root, pkg, sub

    def test_find_import_root_walks_out_of_package(self, tmp_path):
        root, pkg, sub = self._make_pkg(tmp_path)
        assert find_import_root(sub / "deep.py") == root
        assert find_import_root(pkg / "util.py") == root

    def test_find_import_root_loose_script(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("x = 1\n")
        assert find_import_root(f) == tmp_path

    def test_absolute_import_resolves_to_module(self, tmp_path):
        root, pkg, sub = self._make_pkg(tmp_path)
        resolved = resolve_module_file(
            importing_file=sub / "deep.py", module="topkg.util", level=0
        )
        assert resolved == pkg / "util.py"

    def test_absolute_import_resolves_to_package_init(self, tmp_path):
        root, pkg, sub = self._make_pkg(tmp_path)
        resolved = resolve_module_file(
            importing_file=pkg / "util.py", module="topkg.sub", level=0
        )
        assert resolved == sub / "__init__.py"

    def test_relative_import_submodule(self, tmp_path):
        root, pkg, sub = self._make_pkg(tmp_path)
        # In topkg/util.py: `from .sub import deep`  (level 1, module "sub")
        resolved = resolve_module_file(
            importing_file=pkg / "util.py", module="sub.deep", level=1
        )
        assert resolved == sub / "deep.py"

    def test_relative_import_parent_package(self, tmp_path):
        root, pkg, sub = self._make_pkg(tmp_path)
        # In topkg/sub/deep.py: `from ..util import helper` (level 2, module "util")
        resolved = resolve_module_file(
            importing_file=sub / "deep.py", module="util", level=2
        )
        assert resolved == pkg / "util.py"

    def test_relative_bare_dot_resolves_package_init(self, tmp_path):
        root, pkg, sub = self._make_pkg(tmp_path)
        # `from . import deep` in topkg/sub/deep.py -> the sub package's __init__.
        resolved = resolve_module_file(
            importing_file=sub / "deep.py", module="", level=1
        )
        assert resolved == sub / "__init__.py"

    def test_unresolvable_external_module_returns_none(self, tmp_path):
        root, pkg, sub = self._make_pkg(tmp_path)
        assert (
            resolve_module_file(
                importing_file=pkg / "util.py", module="numpy", level=0
            )
            is None
        )


class TestBuildImportTable:
    def test_table_binds_local_names(self):
        src = b"from a.b import c as d\nimport e.f as g\nfrom h import i\n"
        table = build_import_table(src, importing_file=None)  # type: ignore[arg-type]
        assert set(table) == {"d", "g", "i"}
        assert table["d"].symbol == "c"
        assert table["g"].symbol is None and table["g"].module == "e.f"
        assert table["i"].symbol == "i"

    def test_later_binding_wins(self):
        src = b"from a import x\nfrom b import x\n"
        table = build_import_table(src, importing_file=None)  # type: ignore[arg-type]
        assert table["x"].module == "b"
