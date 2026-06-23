"""Indexing orchestrator.

Takes a directory, chunks all files, embeds them, and populates FAISS + FTS5 + call graph
in a single pass. Also handles incremental re-indexing for individual files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mnemostack.core.retrieval.ast_chunker import Chunk, chunk_file_auto
from mnemostack.core.retrieval.call_graph import CallGraph, build_graph_for_python_file
from mnemostack.core.retrieval.constants import INDEXABLE_EXTENSIONS, SKIP_DIRS
from mnemostack.core.retrieval.embed import EmbeddingError, embed_texts
from mnemostack.core.retrieval.faiss_index import FaissIndex
from mnemostack.core.retrieval.fts_index import FTSIndex

log = logging.getLogger(__name__)


def _collect_files(root: Path) -> list[Path]:
    """Recursively collect indexable files, skipping ignored directories."""
    files: list[Path] = []
    for path in root.rglob("*"):
        # Use relative path to avoid false positives from parent dirs named "build" etc.
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if path.is_file() and path.suffix.lower() in INDEXABLE_EXTENSIONS:
            files.append(path)
    return files


def index_directory(
    root: Path,
    faiss_idx: FaissIndex,
    fts_idx: FTSIndex,
    graph: CallGraph,
    batch_size: int = 64,
) -> int:
    """Index an entire directory from scratch.

    Idempotent: removes existing data for files under `root` before re-indexing.
    Chunks all files, embeds them, populates FAISS + FTS5, and builds the call graph.
    Embedding failures on individual batches are logged and skipped, not fatal.

    Returns:
        Total number of chunks indexed.
    """
    files = _collect_files(root)

    # Clear existing data for files under this root (idempotent re-indexing)
    for f in files:
        fpath_str = str(f)
        fts_idx.sync_removed(fpath_str)
        faiss_idx.remove_by_file(fpath_str)
        graph.remove_file(fpath_str)
    all_chunks: list[Chunk] = []

    for f in files:
        chunks = chunk_file_auto(f)
        all_chunks.extend(chunks)

        # Build call graph for Python files
        if f.suffix.lower() == ".py":
            build_graph_for_python_file(f, graph=graph)

    if not all_chunks:
        return 0

    # Embed and index in batches
    total_indexed = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c.code for c in batch]
        try:
            embeddings = embed_texts(texts)
        except EmbeddingError:
            log.exception("Embedding failed for batch %d-%d, skipping", i, i + len(batch))
            continue
        chunk_ids = faiss_idx.add(batch, embeddings)
        fts_idx.sync_added(chunk_ids)
        total_indexed += len(batch)

    faiss_idx.save()
    return total_indexed


def reindex_file(
    file_path: Path,
    faiss_idx: FaissIndex,
    fts_idx: FTSIndex,
    graph: CallGraph,
) -> int:
    """Re-index a single file (incremental update).

    Removes old data for the file, then re-chunks, re-embeds, and re-indexes.

    Returns:
        Number of new chunks indexed for this file.
    """
    fpath_str = str(file_path)

    # Remove old data (FTS first, then FAISS — correct ordering)
    fts_idx.sync_removed(fpath_str)
    faiss_idx.remove_by_file(fpath_str)
    graph.remove_file(fpath_str)

    # If file was deleted, we're done
    if not file_path.exists():
        return 0

    # Re-chunk
    chunks = chunk_file_auto(file_path)
    if not chunks:
        return 0

    # Re-embed and add
    texts = [c.code for c in chunks]
    try:
        embeddings = embed_texts(texts)
    except EmbeddingError:
        log.exception("Embedding failed for %s, skipping", file_path)
        return 0
    chunk_ids = faiss_idx.add(chunks, embeddings)
    fts_idx.sync_added(chunk_ids)

    # Rebuild call graph for Python files
    if file_path.suffix.lower() == ".py":
        build_graph_for_python_file(file_path, graph=graph)

    faiss_idx.save()
    return len(chunks)
