"""Offline ingestion pipeline: load -> chunk -> enrich -> dedupe -> embed -> store.

The ordering of those stages is the whole design. Two rules drive it:

1. **Deduplicate before you embed.** Embedding is the only step that costs money
   and wall-clock time. Every chunk we can recognise as already-stored is an API
   call we never make. Re-running ingestion on an unchanged corpus should cost
   approximately zero.

2. **Content addressing, not sequential IDs.** A chunk's ID *is* the SHA-256 of
   (source file, page, text). Identical content therefore always lands on the
   same ID, which makes "have I seen this?" a primary-key lookup instead of a
   similarity search, and makes the whole pipeline safely re-runnable and
   crash-resumable: an ingestion killed halfway through simply resumes.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document, MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.models import build_embed_model
from app.db.vector_store import VectorStoreError, VectorStoreManager, get_vector_store

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = [".txt", ".pdf"]

# Metadata keys that must never influence the embedding vector or reach the LLM.
# A hash contributes nothing but noise to a 1536-dim vector; a file size
# actively pollutes it. This is a small, real quality lever that most tutorials
# get wrong -- LlamaIndex embeds metadata alongside content by default.
_EXCLUDED_FROM_EMBEDDING = ["content_hash", "ingested_at", "file_path", "doc_type"]
_EXCLUDED_FROM_LLM = ["content_hash", "ingested_at", "doc_type"]


class IngestionError(RuntimeError):
    """Raised when the pipeline cannot complete."""


@dataclass
class IngestionStats:
    documents_loaded: int = 0
    chunks_produced: int = 0
    chunks_new: int = 0
    chunks_skipped: int = 0
    chunks_pruned: int = 0
    files: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Stage 1: load
# ----------------------------------------------------------------------
def load_documents(data_dir: Path) -> list[Document]:
    """Read every .txt and .pdf under ``data_dir``.

    ``SimpleDirectoryReader`` emits one ``Document`` per *page* for PDFs and one
    per *file* for text, which is exactly what we want -- page identity survives
    into the chunk metadata and therefore into citations.
    """
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise IngestionError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise IngestionError(f"Data path is not a directory: {data_dir}")

    logger.info("Loading %s from %s", "/".join(SUPPORTED_EXTENSIONS), data_dir)
    try:
        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            required_exts=SUPPORTED_EXTENSIONS,
            recursive=True,
            filename_as_id=True,
        )
        documents: list[Document] = reader.load_data(show_progress=False)
    except ValueError as exc:
        # Reader raises ValueError on an empty directory -- not an error state.
        logger.warning("No matching files found in %s (%s)", data_dir, exc)
        return []
    except Exception as exc:
        raise IngestionError(f"Failed to read documents from {data_dir}: {exc}") from exc

    logger.info("Loaded %d document(s)", len(documents))
    return documents


# ----------------------------------------------------------------------
# Stage 2: chunk
# ----------------------------------------------------------------------
def build_splitter(settings: Settings) -> SentenceSplitter:
    """Sentence-aware token splitter.

    NOTE ON TERMINOLOGY: this is *not* semantic chunking. ``SentenceSplitter``
    packs whole sentences into ``chunk_size``-token windows and never cuts
    mid-sentence. Genuine semantic chunking is
    ``SemanticSplitterNodeParser``, which embeds every sentence and splits where
    the cosine distance between consecutive sentences jumps -- it produces more
    coherent chunks at the cost of an extra full embedding pass at ingest time
    and unbounded chunk sizes. Swap here if you want to A/B it; nothing else in
    the pipeline depends on which parser you choose.
    """
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def chunk_documents(documents: Sequence[Document], settings: Settings) -> list[BaseNode]:
    if not documents:
        return []
    splitter = build_splitter(settings)
    nodes: list[BaseNode] = splitter.get_nodes_from_documents(
        list(documents), show_progress=False
    )
    logger.info(
        "Split %d document(s) into %d chunk(s) [size=%d, overlap=%d]",
        len(documents),
        len(nodes),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return nodes


# ----------------------------------------------------------------------
# Stage 3: enrich metadata + content-addressed IDs
# ----------------------------------------------------------------------
def _content_hash(file_path: str, page_number: int, text: str) -> str:
    digest = hashlib.sha256()
    for part in (file_path, str(page_number), text):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")  # domain separator: prevents field-boundary collisions
    return digest.hexdigest()


def _page_number(raw: object) -> int:
    """PDF readers give a ``page_label`` string; .txt files give nothing."""
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return 1


def enrich_nodes(nodes: Sequence[BaseNode], data_dir: Path) -> list[BaseNode]:
    """Replace reader metadata with a clean, flat, store-safe schema.

    Chroma only accepts str/int/float/bool metadata values -- no None, no lists,
    no nested dicts -- and ``SimpleDirectoryReader`` hands us several fields that
    violate that. So we rebuild the dict from scratch rather than filtering it;
    an explicit schema is also what the query pipeline will filter on later.
    """
    data_dir = data_dir.resolve()
    ingested_at = datetime.now(UTC).isoformat()
    enriched: list[BaseNode] = []

    for node in nodes:
        source = node.metadata or {}
        absolute = source.get("file_path") or source.get("file_name") or "unknown"

        # Store paths RELATIVE to the data directory. Absolute paths differ
        # between your laptop and the container, which would make identical
        # content hash differently depending on where ingestion ran.
        try:
            relative = str(Path(absolute).resolve().relative_to(data_dir))
        except (ValueError, OSError):
            relative = str(source.get("file_name") or absolute)

        page = _page_number(source.get("page_label"))
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        digest = _content_hash(relative, page, text)

        node.metadata = {
            "file_name": str(source.get("file_name") or Path(relative).name),
            "file_path": relative,
            "page_number": page,
            "doc_type": Path(relative).suffix.lstrip(".").lower() or "unknown",
            "content_hash": digest,
            "ingested_at": ingested_at,
        }
        node.excluded_embed_metadata_keys = list(_EXCLUDED_FROM_EMBEDDING)
        node.excluded_llm_metadata_keys = list(_EXCLUDED_FROM_LLM)

        # The content hash *is* the primary key.
        node.id_ = digest
        enriched.append(node)

    return enriched


# ----------------------------------------------------------------------
# Stage 4: dedupe
# ----------------------------------------------------------------------
def partition_new(
    nodes: Sequence[BaseNode], store: VectorStoreManager
) -> tuple[list[BaseNode], int]:
    """Split chunks into (not-yet-stored, count-already-stored).

    Also collapses duplicates *within* this run -- the same boilerplate
    paragraph repeated across pages of one file hashes identically and should be
    stored once.
    """
    if not nodes:
        return [], 0

    seen_in_batch: set[str] = set()
    unique: list[BaseNode] = []
    intra_batch_dupes = 0
    for node in nodes:
        if node.node_id in seen_in_batch:
            intra_batch_dupes += 1
            continue
        seen_in_batch.add(node.node_id)
        unique.append(node)

    already_stored = store.existing_ids([n.node_id for n in unique])
    new_nodes = [n for n in unique if n.node_id not in already_stored]

    skipped = len(already_stored) + intra_batch_dupes
    logger.info(
        "Dedupe: %d new, %d already stored, %d duplicate within batch",
        len(new_nodes),
        len(already_stored),
        intra_batch_dupes,
    )
    return new_nodes, skipped


def prune_stale_chunks(
    all_nodes: Sequence[BaseNode], store: VectorStoreManager
) -> int:
    """Delete chunks belonging to re-ingested files that no longer exist there.

    Content-hash dedup alone handles *additions* but not *edits*: change one
    paragraph and you get a new chunk ID, while the old chunk lingers forever
    and keeps surfacing in retrieval as a ghost. This closes that hole by
    reconciling, per file, what is stored against what we just produced.

    Files absent from this run are left untouched -- ingesting one new document
    must never delete the rest of the corpus.
    """
    if not all_nodes:
        return 0

    fresh_by_file: dict[str, set[str]] = {}
    for node in all_nodes:
        fresh_by_file.setdefault(node.metadata["file_path"], set()).add(node.node_id)

    stale: list[str] = []
    for file_path, fresh_ids in fresh_by_file.items():
        stored_ids = store.ids_for_source(file_path)
        stale.extend(stored_ids - fresh_ids)

    if stale:
        logger.info("Pruning %d stale chunk(s) from re-ingested file(s)", len(stale))
        store.delete_ids(stale)
    return len(stale)


# ----------------------------------------------------------------------
# Stage 5: embed
# ----------------------------------------------------------------------
def embed_nodes(nodes: Sequence[BaseNode], embed_model: OpenAIEmbedding) -> list[BaseNode]:
    """Attach embeddings in place.

    Done explicitly rather than via ``VectorStoreIndex(nodes)`` because the
    index constructor would happily re-embed everything, including the chunks we
    just spent a round-trip proving we already have.
    """
    if not nodes:
        return []

    texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
    logger.info("Embedding %d chunk(s) with %s", len(texts), embed_model.model_name)
    try:
        vectors = embed_model.get_text_embedding_batch(texts, show_progress=False)
    except Exception as exc:
        raise IngestionError(f"Embedding failed: {exc}") from exc

    if len(vectors) != len(nodes):
        raise IngestionError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(nodes)} chunks"
        )

    for node, vector in zip(nodes, vectors, strict=True):
        node.embedding = vector
    return list(nodes)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_ingestion(
    data_dir: Path | None = None,
    collection_name: str | None = None,
    rebuild: bool = False,
    prune: bool = True,
    dry_run: bool = False,
    settings: Settings | None = None,
) -> IngestionStats:
    """Execute the full offline pipeline. Safe to run repeatedly."""
    settings = settings or get_settings()
    data_dir = Path(data_dir) if data_dir else settings.data_dir
    started = time.perf_counter()
    stats = IngestionStats()

    store = get_vector_store(collection_name)

    if rebuild and not dry_run:
        store.reset_collection()

    documents = load_documents(data_dir)
    stats.documents_loaded = len(documents)
    if not documents:
        logger.warning("Nothing to ingest.")
        stats.duration_seconds = time.perf_counter() - started
        return stats

    nodes = enrich_nodes(chunk_documents(documents, settings), data_dir)
    stats.chunks_produced = len(nodes)
    stats.files = sorted({n.metadata["file_path"] for n in nodes})

    new_nodes, skipped = partition_new(nodes, store)
    stats.chunks_new = len(new_nodes)
    stats.chunks_skipped = skipped

    if dry_run:
        logger.info("[dry-run] Would embed and store %d chunk(s)", len(new_nodes))
        stats.duration_seconds = time.perf_counter() - started
        return stats

    if new_nodes:
        embed_model = build_embed_model(settings)
        embedded = embed_nodes(new_nodes, embed_model)
        try:
            store.add_nodes(embedded)
        except VectorStoreError as exc:
            raise IngestionError(str(exc)) from exc
    else:
        logger.info("No new chunks -- corpus already up to date.")

    # Prune after writing: if the process dies mid-run we would rather have
    # duplicates (harmless, deduped next run) than a gap in the index.
    if prune and not rebuild:
        stats.chunks_pruned = prune_stale_chunks(nodes, store)

    stats.duration_seconds = time.perf_counter() - started
    logger.info(
        "Ingestion complete in %.2fs | docs=%d chunks=%d new=%d skipped=%d pruned=%d | collection size=%d",
        stats.duration_seconds,
        stats.documents_loaded,
        stats.chunks_produced,
        stats.chunks_new,
        stats.chunks_skipped,
        stats.chunks_pruned,
        store.count(),
    )
    return stats
