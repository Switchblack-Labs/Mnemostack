"""Installed versions come from the project's environment, never mnemostack's own."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mnemostack.core.impact.environment import (
    Environment,
    canonical,
    project_environment,
    project_python,
)


def test_names_are_compared_the_way_pypi_compares_them():
    env = Environment(
        python=Path("python"), imports={"yaml": ["PyYAML"]}, versions={"pyyaml": "6.0.1"}
    )
    assert env.distribution_for("yaml") == "PyYAML"
    assert env.version_of("PyYAML") == env.version_of("pyyaml") == "6.0.1"
    assert canonical("typing_extensions") == canonical("Typing.Extensions") == "typing-extensions"


def test_mnemostacks_own_environment_is_never_taken_for_the_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    assert project_python(tmp_path) is None


@pytest.mark.skipif(sys.platform == "win32", reason="posix virtualenv layout")
def test_versions_are_read_by_running_the_repos_interpreter(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/sh\necho \'{"imports": {"yaml": ["PyYAML"]}, "versions": {"PyYAML": "6.0.1"}}\'\n'
    )
    python.chmod(0o755)

    env = project_environment(tmp_path)
    assert env is not None and env.python == python
    assert env.distribution_for("yaml") == "PyYAML"
    assert env.version_of("pyyaml") == "6.0.1"
