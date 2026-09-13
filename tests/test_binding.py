"""Binding a call's arguments against the new signature.

A witnessed signature change is only a broken call if the call no longer binds.
Every signature BREAK left after witnessing on twenty repos we did not write was
a real change that the call itself did not care about, so these tests pin both
sides: calls that still fit are cleared, calls that do not are kept.
"""

from __future__ import annotations

from mnemostack.core.impact.binding import binds, call_shapes, still_binds

PK = "POSITIONAL_OR_KEYWORD"
KO = "KEYWORD_ONLY"


# --- binds ------------------------------------------------------------------


def test_missing_required_parameter_does_not_bind():
    assert binds([["x", PK, None], ["token", PK, None]], positional=1, keywords=[]) is False


def test_too_many_positionals_do_not_bind():
    assert binds([["x", PK, None]], positional=2, keywords=[]) is False


def test_unknown_keyword_does_not_bind_without_var_keyword():
    assert binds([["x", PK, None]], positional=1, keywords=["timeout"]) is False
    assert binds([["x", PK, None], ["kw", "VAR_KEYWORD", None]], 1, ["timeout"]) is True


def test_value_given_twice_does_not_bind():
    assert binds([["x", PK, None]], positional=1, keywords=["x"]) is False


def test_required_keyword_only_must_be_passed():
    params = [["x", PK, None], ["mode", KO, None]]
    assert binds(params, positional=1, keywords=[]) is False
    assert binds(params, positional=1, keywords=["mode"]) is True


def test_leading_self_is_implicit():
    assert binds([["self", PK, None], ["x", PK, None]], positional=1, keywords=[]) is True


def test_defaults_fill_what_the_call_leaves_out():
    assert binds([["x", PK, None], ["retries", PK, "0"]], positional=1, keywords=[]) is True


# --- call_shapes ------------------------------------------------------------


def test_call_shapes_reads_positionals_and_keywords():
    assert call_shapes("x = Session('u', timeout=5)", "Session") == [(1, ["timeout"])]


def test_call_shapes_reads_attribute_calls_and_header_lines():
    assert call_shapes("if client.fetch(url):", "fetch") == [(1, [])]


def test_star_arguments_are_undecidable():
    assert call_shapes("return ctx.invoke(f, *args, **kwargs)", "invoke") is None


def test_a_call_split_across_lines_is_undecidable():
    assert call_shapes("return await super().execute(", "execute") is None


# --- still_binds ------------------------------------------------------------


def test_constructor_change_binds_against_the_class_call():
    """flask's click.Option(...) fits click 8's changed constructor."""
    new = [
        ["self", PK, None],
        ["param_decls", PK, "None"],
        ["show_default", PK, "None"],
        ["is_flag", PK, "None"],
        ["attrs", "VAR_KEYWORD", None],
    ]
    text = 'version_option = click.Option(["--version"], is_flag=True, expose_value=False)'
    assert still_binds(text, "click.core.Option.__init__", new) is True


def test_call_that_no_longer_fits_is_reported_as_not_binding():
    new = [["url", PK, None], ["token", PK, None]]
    assert still_binds("return connect('http://x')", "paylib.connect", new) is False


def test_a_call_split_across_lines_binds_from_the_file():
    """click's own click.Option( calls put each argument on its own line."""
    import ast

    source = 'opt = click.Option(\n    ["--name"],\n    is_flag=True,\n)\n'
    new = [["self", PK, None], ["param_decls", PK, "None"], ["attrs", "VAR_KEYWORD", None]]
    tree = ast.parse(source)
    assert still_binds("opt = click.Option(", "click.core.Option.__init__", new) is None
    assert still_binds("", "click.core.Option.__init__", new, tree=tree, line=1) is True
    assert still_binds("", "click.core.Option.__init__", new, tree=tree, line=2) is None


def test_tree_calls_that_no_longer_fit_are_reported():
    import ast

    tree = ast.parse("connect(\n    'http://x',\n)\n")
    new = [["url", PK, None], ["token", PK, None]]
    assert still_binds("", "paylib.connect", new, tree=tree, line=1) is False
