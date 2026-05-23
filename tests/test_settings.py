from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mnemostack.config.settings import (
    _DEFAULTS_PATH,
    _PROJECT_ROOT,
    RankingWeights,
    Settings,
    StoreConfig,
    _deep_merge,
    load_settings,
)


def _write_yaml(tmp_path: Path, data: object, name: str = "user.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# --- Default load ---


def test_load_no_override_matches_pure_defaults():
    """defaults.yaml must declare the same values as the Pydantic defaults.
    If they diverge, configuration behavior changes silently between
    YAML-loaded and direct-instantiation paths."""
    loaded = load_settings()
    pure = Settings()
    assert loaded == pure


def test_settings_constructable_without_args():
    s = Settings()
    assert s.compression.consolidation_interval == 25
    assert s.retrieval.top_k == 5
    assert s.retrieval.ranking_weights.semantic == 0.6


# --- Path resolution (H1 + M2) ---


def test_relative_base_dir_resolves_against_project_root(tmp_path: Path):
    cfg = _write_yaml(tmp_path, {"store": {"base_dir": "custom"}})
    s = load_settings(cfg)
    assert s.store.base_dir == _PROJECT_ROOT / "custom"


def test_absolute_base_dir_preserved(tmp_path: Path):
    cfg = _write_yaml(tmp_path, {"store": {"base_dir": str(tmp_path / "store")}})
    s = load_settings(cfg)
    assert s.store.base_dir == tmp_path / "store"


def test_snapshots_dir_tracks_base_dir_override(tmp_path: Path):
    """Regression: snapshots_dir used to stay rooted at _PROJECT_ROOT
    when base_dir was overridden."""
    cfg = _write_yaml(tmp_path, {"store": {"base_dir": str(tmp_path / "store")}})
    s = load_settings(cfg)
    assert s.store.snapshots_dir == tmp_path / "store" / "snapshots"


def test_base_dir_expanduser(tmp_path: Path):
    cfg = _write_yaml(tmp_path, {"store": {"base_dir": "~/mnemo-test"}})
    s = load_settings(cfg)
    assert "~" not in str(s.store.base_dir)
    assert s.store.base_dir.is_absolute()


def test_empty_base_dir_rejected(tmp_path: Path):
    """Regression: empty base_dir used to silently resolve to _PROJECT_ROOT."""
    cfg = _write_yaml(tmp_path, {"store": {"base_dir": ""}})
    with pytest.raises(ValidationError, match="base_dir"):
        load_settings(cfg)


def test_dot_base_dir_rejected(tmp_path: Path):
    """Path('.') has no parts and would also collapse to _PROJECT_ROOT."""
    cfg = _write_yaml(tmp_path, {"store": {"base_dir": "."}})
    with pytest.raises(ValidationError, match="base_dir"):
        load_settings(cfg)


def test_snapshots_dir_not_user_configurable(tmp_path: Path):
    """snapshots_dir is computed from base_dir — supplying it is a typo per extra='forbid'."""
    cfg = _write_yaml(tmp_path, {"store": {"snapshots_dir": "elsewhere"}})
    with pytest.raises(ValidationError):
        load_settings(cfg)


# --- Strict schema (H3) ---


def test_unknown_top_level_key_rejected(tmp_path: Path):
    cfg = _write_yaml(tmp_path, {"unknown_section": {"foo": "bar"}})
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_typo_in_nested_key_rejected(tmp_path: Path):
    cfg = _write_yaml(tmp_path, {"retrieval": {"top_k_typo": 999}})
    with pytest.raises(ValidationError):
        load_settings(cfg)


# --- Numeric validation (M1) ---


def test_negative_consolidation_interval_rejected(tmp_path: Path):
    cfg = _write_yaml(tmp_path, {"compression": {"consolidation_interval": -5}})
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_zero_top_k_rejected(tmp_path: Path):
    cfg = _write_yaml(tmp_path, {"retrieval": {"top_k": 0}})
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_zero_faiss_m_rejected(tmp_path: Path):
    cfg = _write_yaml(tmp_path, {"retrieval": {"faiss_m": 0}})
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_negative_ranking_weight_rejected():
    with pytest.raises(ValidationError):
        RankingWeights(semantic=-1.0)


def test_zero_weights_accepted():
    # All-zero weights are degenerate but not invalid — ranker should normalize.
    RankingWeights(semantic=0.0, recency=0.0, dependency=0.0)


# --- Server transport (M8) ---


def test_unknown_transport_rejected(tmp_path: Path):
    """Regression: server.py used to hardcode 'stdio' regardless of config; once it
    started reading the config, an invalid value would crash inside the server loop.
    Reject at validation time instead."""
    cfg = _write_yaml(tmp_path, {"server": {"transport": "carrier-pigeon"}})
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_known_transports_accepted(tmp_path: Path):
    for value in ("stdio", "sse", "streamable-http"):
        cfg = _write_yaml(tmp_path, {"server": {"transport": value}}, name=f"{value}.yaml")
        s = load_settings(cfg)
        assert s.server.transport == value


def test_server_run_uses_configured_transport(monkeypatch):
    """server.run() must dispatch the transport from settings, not hardcode it."""
    import mnemostack.mcp.server as srv

    captured = {}

    def fake_run(self, transport="stdio", mount_path=None):
        captured["transport"] = transport

    monkeypatch.setattr(type(srv.mcp), "run", fake_run)
    srv.run()
    assert captured["transport"] == srv.settings.server.transport


# --- Frozen (M3) ---


def test_settings_frozen():
    s = Settings()
    with pytest.raises(ValidationError):
        s.retrieval.top_k = 99


def test_store_frozen():
    s = StoreConfig()
    with pytest.raises(ValidationError):
        s.base_dir = Path("/tmp/whatever")


# --- YAML loader error paths (L1, L2, M5, M6) ---


def test_yaml_root_list_raises_clean_error(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n- 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected YAML mapping"):
        load_settings(p)


def test_yaml_root_string_raises_clean_error(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected YAML mapping"):
        load_settings(p)


def test_yaml_empty_file_is_no_op(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    s = load_settings(p)
    assert s == Settings()


def test_yaml_null_root_is_no_op(tmp_path: Path):
    p = tmp_path / "null.yaml"
    p.write_text("~\n", encoding="utf-8")
    s = load_settings(p)
    assert s == Settings()


def test_missing_config_path_raises(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_settings(nonexistent)


def test_dict_clobbered_by_scalar_caught_by_validation(tmp_path: Path):
    """If the user writes `store: oops` (string instead of dict), the merge no
    longer crashes opaquely — Pydantic surfaces a typed validation error."""
    cfg = _write_yaml(tmp_path, {"store": "oops"})
    with pytest.raises(ValidationError):
        load_settings(cfg)


# --- _deep_merge ---


def test_deep_merge_returns_new_dict_does_not_mutate_inputs():
    base = {"a": {"x": 1}, "b": 2}
    overrides = {"a": {"y": 3}, "c": 4}
    out = _deep_merge(base, overrides)
    assert out == {"a": {"x": 1, "y": 3}, "b": 2, "c": 4}
    # Inputs untouched
    assert base == {"a": {"x": 1}, "b": 2}
    assert overrides == {"a": {"y": 3}, "c": 4}


def test_deep_merge_scalar_overwrites_dict():
    """Scalar overrides a dict at the same key — the dict is replaced entirely."""
    base = {"a": {"x": 1}}
    overrides = {"a": "scalar"}
    out = _deep_merge(base, overrides)
    assert out == {"a": "scalar"}


def test_deep_merge_dict_overwrites_scalar():
    base = {"a": "scalar"}
    overrides = {"a": {"x": 1}}
    out = _deep_merge(base, overrides)
    assert out == {"a": {"x": 1}}


# --- defaults.yaml self-consistency ---


def test_defaults_yaml_exists():
    assert _DEFAULTS_PATH.exists()


def test_defaults_yaml_is_a_mapping():
    with open(_DEFAULTS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)


# --- Round-trip ---


def test_settings_model_dump_round_trips():
    """Regression: computed_field + extra='forbid' used to break model_validate(model_dump())
    because the computed key appeared in the dump but was rejected as an extra on input."""
    s = Settings()
    d = s.model_dump()
    s2 = Settings.model_validate(d)
    assert s == s2


def test_snapshots_dir_not_in_model_dump():
    """snapshots_dir is a plain @property (not a stored field), so it should
    not leak into serialized output."""
    s = Settings()
    d = s.model_dump()
    assert "snapshots_dir" not in d["store"]
