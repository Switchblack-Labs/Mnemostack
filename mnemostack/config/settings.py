from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Must match the transports FastMCP.run() accepts; constraining here gives a clean
# validation error instead of a runtime crash inside the server loop.
Transport = Literal["stdio", "sse", "streamable-http"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.yaml"

# Frozen so a shared settings singleton can't be mutated by a stray caller;
# extra="forbid" so typo'd keys raise instead of silently doing nothing;
# validate_default=True so field validators (e.g. path resolution) also run
# when a field falls back to its declared default.
_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class RankingWeights(BaseModel):
    model_config = _MODEL_CONFIG

    semantic: float = Field(default=0.6, ge=0.0)
    recency: float = Field(default=0.25, ge=0.0)
    dependency: float = Field(default=0.15, ge=0.0)


class CompressionConfig(BaseModel):
    model_config = _MODEL_CONFIG

    consolidation_interval: int = Field(default=25, gt=0)
    memory_token_budget: int = Field(default=3000, gt=0)
    consolidation_model: str = Field(default="ollama/llama3:8b", min_length=1)
    local_extraction_mode: str = Field(default="all", min_length=1)


class RetrievalConfig(BaseModel):
    model_config = _MODEL_CONFIG

    embedding_model: str = Field(default="ollama/nomic-embed-text", min_length=1)
    top_k: int = Field(default=5, gt=0)
    faiss_m: int = Field(default=32, gt=0)
    faiss_ef_construction: int = Field(default=200, gt=0)
    faiss_ef_search: int = Field(default=128, gt=0)
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)
    recency_half_life_minutes: int = Field(default=60, gt=0)
    dependency_hops: int = Field(default=2, ge=0)
    file_watch_debounce_ms: int = Field(default=500, ge=0)


class StoreConfig(BaseModel):
    model_config = _MODEL_CONFIG

    base_dir: Path = Path("store")

    @field_validator("base_dir", mode="after")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        # Path("") and Path(".") both have empty parts and would silently
        # resolve to the project root, which is almost never intended.
        if not v.parts:
            raise ValueError("base_dir must be a non-empty path")
        v = v.expanduser()
        return v if v.is_absolute() else _PROJECT_ROOT / v

    @property
    def snapshots_dir(self) -> Path:
        """Derived from base_dir — not user-configurable."""
        return self.base_dir / "snapshots"


class ServerConfig(BaseModel):
    model_config = _MODEL_CONFIG

    transport: Transport = "stdio"


class Settings(BaseModel):
    model_config = _MODEL_CONFIG

    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: nested dicts merge recursively, other types overwrite."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected YAML mapping at root, got {type(data).__name__}"
        )
    return data


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from defaults.yaml, optionally overridden by a user config file.

    Raises:
        FileNotFoundError: if config_path is provided but doesn't exist.
        ValueError: if either YAML file's root is not a mapping.
        pydantic.ValidationError: if the merged config violates schema constraints.
    """
    merged: dict[str, Any] = {}
    if _DEFAULTS_PATH.exists():
        merged = _load_yaml(_DEFAULTS_PATH)
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        merged = _deep_merge(merged, _load_yaml(config_path))
    return Settings.model_validate(merged)


settings = load_settings()
