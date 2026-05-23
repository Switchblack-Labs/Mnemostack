from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.yaml"


class RankingWeights(BaseModel):
    semantic: float = 0.6
    recency: float = 0.25
    dependency: float = 0.15


class CompressionConfig(BaseModel):
    consolidation_interval: int = 25
    memory_token_budget: int = 3000
    consolidation_model: str = "ollama/llama3:8b"
    local_extraction_mode: str = "all"


class RetrievalConfig(BaseModel):
    embedding_model: str = "ollama/nomic-embed-text"
    top_k: int = 5
    faiss_m: int = 32
    faiss_ef_construction: int = 200
    faiss_ef_search: int = 128
    ranking_weights: RankingWeights = RankingWeights()
    recency_half_life_minutes: int = 60
    dependency_hops: int = 2
    file_watch_debounce_ms: int = 500


class StoreConfig(BaseModel):
    base_dir: Path = Path("store")
    snapshots_dir: Path = Path("store/snapshots")


class ServerConfig(BaseModel):
    transport: str = "stdio"


class Settings(BaseModel):
    compression: CompressionConfig = CompressionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    store: StoreConfig = StoreConfig()
    server: ServerConfig = ServerConfig()


def _resolve_store_paths(settings: Settings) -> None:
    """Resolve store paths relative to project root if they are relative."""
    if not settings.store.base_dir.is_absolute():
        settings.store.base_dir = _PROJECT_ROOT / settings.store.base_dir
    if not settings.store.snapshots_dir.is_absolute():
        settings.store.snapshots_dir = _PROJECT_ROOT / settings.store.snapshots_dir


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from defaults.yaml, optionally overridden by a user config file."""
    merged: dict = {}

    if _DEFAULTS_PATH.exists():
        with open(_DEFAULTS_PATH) as f:
            merged = yaml.safe_load(f) or {}

    if config_path and config_path.exists():
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        _deep_merge(merged, overrides)

    settings = Settings.model_validate(merged)
    _resolve_store_paths(settings)
    return settings


def _deep_merge(base: dict, overrides: dict) -> None:
    """Merge overrides into base dict, recursively for nested dicts."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


settings = load_settings()
