"""Embedding function for code chunks.

Thin wrapper around litellm.embedding() — works with any provider (Ollama local,
OpenAI, Anthropic, etc.) via the model string in settings.retrieval.embedding_model.
"""

from __future__ import annotations

import logging

import numpy as np
from litellm import embedding

from mnemostack.config.settings import settings

log = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding fails after retries."""


def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    """Embed a list of text strings into vectors.

    Args:
        texts: Strings to embed (code chunks, queries, etc.).
        model: Override embedding model. Defaults to settings.retrieval.embedding_model.

    Returns:
        numpy array of shape (len(texts), dimension).

    Raises:
        EmbeddingError: If the embedding API call fails.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = model or settings.retrieval.embedding_model

    try:
        response = embedding(model=model, input=texts)
    except Exception as exc:
        raise EmbeddingError(
            f"Embedding failed for {len(texts)} texts with model {model!r}: {exc}"
        ) from exc

    vectors = [item["embedding"] for item in response.data]
    return np.array(vectors, dtype=np.float32)


def embed_query(query: str, model: str | None = None) -> np.ndarray:
    """Embed a single query string. Returns shape (dimension,)."""
    result = embed_texts([query], model=model)
    return result[0]
