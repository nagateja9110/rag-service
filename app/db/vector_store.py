"""ChromaDB wrapper.

Why a wrapper at all, when LlamaIndex already ships ``ChromaVectorStore``?

Because ``ChromaVectorStore`` is an *index-facing* abstraction: it knows how to
add and query nodes, and nothing else. The ingestion pipeline needs operations
that live below that line -- "which of these 4,000 chunk IDs do you already
have?", "drop the stale chunks for this one file", "how many vectors are in
here?". Those are raw-collection concerns.

So this module owns the split:

  * ``VectorStoreManager.collection``    -> raw Chroma, for lifecycle/dedup work
  * ``VectorStoreManager.llama_store``   -> LlamaIndex adapter, for the query path

When we migrate to Qdrant, this file is the only one that changes: the ingestion
pipeline talks to the six methods below, and the query pipeline talks to
``llama_store``. Neither imports ``chromadb``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaClientSettings
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class VectorStoreError(RuntimeError):
    """Raised when the vector store is unreachable or rejects an operation."""


class EmbeddingMismatchError(VectorStoreError):
    """The collection was embedded with a different model than is configured."""


# Collection-metadata key recording which vector space the data lives in.
EMBEDDING_SIGNATURE_KEY = "embedding_signature"


def is_stale_collection_error(exc: BaseException) -> bool:
    """True if ``exc`` means "your cached collection handle is dead".

    Chroma resolves a collection to a UUID once and caches it. Drop and recreate
    that collection -- which is exactly what ``ingest --rebuild`` does -- and
    every holder of the old handle starts raising ``InvalidCollectionException``
    against a UUID that no longer exists. A long-running API process will keep
    doing that until it is restarted.

    This predicate lives here, not in the query engine, so that chromadb's
    exception types stay behind the storage boundary. The Qdrant port replaces
    the body and nothing upstream changes.
    """
    try:
        from chromadb.errors import InvalidCollectionException

        if isinstance(exc, InvalidCollectionException):
            return True
    except ImportError:  # pragma: no cover
        pass
    return "does not exist" in str(exc).lower()


class VectorStoreManager:
    """Owns the Chroma client, collection and the LlamaIndex adapter over it."""

    def __init__(self, settings: Settings | None = None, collection_name: str | None = None):
        self._settings = settings or get_settings()
        self._collection_name = collection_name or self._settings.chroma_collection
        self._client: ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._llama_store: ChromaVectorStore | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    @property
    def client(self) -> ClientAPI:
        if self._client is not None:
            return self._client

        # Telemetry off by default: this is someone else's data leaving the
        # network boundary, and in prod that is a compliance conversation.
        chroma_settings = ChromaClientSettings(anonymized_telemetry=False)

        try:
            if self._settings.chroma_mode == "http":
                logger.info(
                    "Connecting to Chroma over HTTP at %s:%s",
                    self._settings.chroma_host,
                    self._settings.chroma_port,
                )
                self._client = chromadb.HttpClient(
                    host=self._settings.chroma_host,
                    port=self._settings.chroma_port,
                    settings=chroma_settings,
                )
            else:
                path = self._settings.chroma_persist_dir.resolve()
                path.mkdir(parents=True, exist_ok=True)
                logger.info("Opening embedded Chroma at %s", path)
                self._client = chromadb.PersistentClient(
                    path=str(path), settings=chroma_settings
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not connect to Chroma (mode={self._settings.chroma_mode}): {exc}"
            ) from exc

        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """The raw Chroma collection -- ingestion/lifecycle operations only."""
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={
                        # Cosine is the right metric for normalised embeddings.
                        # Chroma defaults to L2 (squared euclidean); on
                        # normalised vectors the ranking is equivalent, but the
                        # *scores* are not, and relevance thresholds care.
                        "hnsw:space": "cosine",
                        # Stamped at creation so a collection always knows which
                        # embedding model produced it.
                        EMBEDDING_SIGNATURE_KEY: self._settings.embedding_signature,
                    },
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Could not open collection '{self._collection_name}': {exc}"
                ) from exc
        return self._collection

    @property
    def llama_store(self) -> ChromaVectorStore:
        """The LlamaIndex-facing adapter -- query path only."""
        if self._llama_store is None:
            self._llama_store = ChromaVectorStore(chroma_collection=self.collection)
        return self._llama_store

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def count(self) -> int:
        return int(self.collection.count())

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Return the subset of ``ids`` already present in the collection.

        This is the primitive that makes ingestion idempotent, and it is
        deliberately called *before* embedding: an ID round-trip is free, an
        embedding call is not.
        """
        found: set[str] = set()
        for batch in _batched(list(ids), self._settings.store_batch_size):
            result = self.collection.get(ids=list(batch), include=[])
            found.update(result.get("ids") or [])
        return found

    def ids_for_source(self, file_path: str) -> set[str]:
        """Every chunk ID currently stored for one source file."""
        result = self.collection.get(where={"file_path": file_path}, include=[])
        return set(result.get("ids") or [])

    def stored_embedding_signature(self) -> str | None:
        """Which provider:model built this collection, if it says."""
        metadata = self.collection.metadata or {}
        value = metadata.get(EMBEDDING_SIGNATURE_KEY)
        return str(value) if value else None

    def verify_embedding_signature(self) -> None:
        """Refuse to mix vector spaces.

        Switching ``embedding_provider`` changes dimensionality (OpenAI
        text-embedding-3-small is 1536-d, all-MiniLM-L6-v2 is 384-d). Chroma
        does raise on a dimension mismatch, but the message names raw numbers
        and gives no hint that a config switch caused it -- and two *same-sized*
        models from different providers would not error at all: you would just
        get silently meaningless similarity scores, which is far worse than a
        crash.

        An empty or unstamped collection is (re)stamped rather than rejected --
        there is no data to invalidate, so switching providers before ingesting
        anything is a legitimate thing to do.
        """
        expected = self._settings.embedding_signature
        stored = self.stored_embedding_signature()

        if stored == expected:
            return

        if stored is None or self.count() == 0:
            self._stamp_embedding_signature(expected)
            return

        raise EmbeddingMismatchError(
            f"Collection '{self._collection_name}' was built with "
            f"'{stored}' but the app is configured for '{expected}'. "
            f"These embeddings are not comparable. Either restore the previous "
            f"embedding_provider/model, point at a different CHROMA_COLLECTION, "
            f"or re-ingest from scratch: "
            f"python -m app.ingestion.cli ingest --rebuild"
        )

    def _stamp_embedding_signature(self, signature: str) -> None:
        metadata = dict(self.collection.metadata or {})
        metadata[EMBEDDING_SIGNATURE_KEY] = signature
        try:
            # Chroma rejects updates to its reserved hnsw:* keys, and they are
            # immutable after creation anyway.
            self.collection.modify(
                metadata={k: v for k, v in metadata.items() if not k.startswith("hnsw:")}
            )
            logger.info("Stamped collection with embedding signature %s", signature)
        except Exception as exc:  # noqa: BLE001 - stamping is best-effort
            logger.warning("Could not stamp embedding signature: %s", exc)

    def load_all_nodes(self, limit: int | None = None) -> list[BaseNode]:
        """Materialise the whole corpus as LlamaIndex nodes.

        Needed because BM25 is a keyword index over raw text and Chroma is not
        one -- there is no server-side inverted index to query, so the sparse
        retriever has to hold the corpus itself. Paginated so we never ask
        Chroma for a million rows in a single response.

        This is the scaling ceiling of Chroma-backed hybrid search. See
        app/query/retriever.py for what to do about it.
        """
        nodes: list[BaseNode] = []
        offset = 0
        page = max(self._settings.store_batch_size, 256)

        while True:
            batch = self.collection.get(
                limit=page, offset=offset, include=["metadatas", "documents"]
            )
            ids = batch.get("ids") or []
            if not ids:
                break

            metadatas = batch.get("metadatas") or [{}] * len(ids)
            documents = batch.get("documents") or [""] * len(ids)

            for node_id, metadata, text in zip(ids, metadatas, documents, strict=True):
                nodes.append(self._to_node(node_id, metadata or {}, text or ""))

            offset += len(ids)
            if limit is not None and len(nodes) >= limit:
                logger.warning(
                    "Corpus exceeds bm25_max_nodes (%d); BM25 will only cover the "
                    "first %d chunks and recall will be incomplete.",
                    limit,
                    limit,
                )
                return nodes[:limit]
            if len(ids) < page:
                break

        return nodes

    @staticmethod
    def _to_node(node_id: str, metadata: dict, text: str) -> BaseNode:
        """Rebuild a node from a Chroma row.

        LlamaIndex stashes a serialised copy of the node under ``_node_content``
        when it writes, so the fast path is to deserialise that and restore
        relationships/excluded-metadata exactly. The fallback covers rows written
        by anything other than LlamaIndex.
        """
        try:
            node = metadata_dict_to_node(metadata)
            if not node.get_content() and text:
                node.set_content(text)
            return node
        except Exception:  # noqa: BLE001 - fall back to a plain text node
            clean = {k: v for k, v in metadata.items() if not k.startswith("_")}
            return TextNode(id_=node_id, text=text, metadata=clean)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add_nodes(self, nodes: Sequence[BaseNode]) -> int:
        """Insert pre-embedded nodes. Caller guarantees ``node.embedding`` is set."""
        if not nodes:
            return 0
        written = 0
        for batch in _batched(list(nodes), self._settings.store_batch_size):
            try:
                self.llama_store.add(list(batch))
            except Exception as exc:
                raise VectorStoreError(f"Failed writing {len(batch)} nodes: {exc}") from exc
            written += len(batch)
            logger.debug("Wrote %d/%d nodes", written, len(nodes))
        return written

    def delete_ids(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        for batch in _batched(ids, self._settings.store_batch_size):
            self.collection.delete(ids=list(batch))
        return len(ids)

    def refresh(self) -> None:
        """Drop cached handles so the next access re-resolves by name.

        Cheap, and the recovery path for a collection recreated underneath us.
        """
        self._collection = None
        self._llama_store = None

    def reset_collection(self) -> None:
        """Drop and recreate. The blunt instrument behind ``ingest --rebuild``."""
        logger.warning("Dropping collection '%s'", self._collection_name)
        try:
            self.client.delete_collection(self._collection_name)
        except Exception as exc:  # noqa: BLE001 - absent collection is not an error
            logger.debug("delete_collection was a no-op: %s", exc)
        self._collection = None
        self._llama_store = None
        _ = self.collection  # force recreation now, so failures surface here

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def healthy(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Vector store health check failed: %s", exc)
            return False


def get_vector_store(
    collection_name: str | None = None, settings: Settings | None = None
) -> VectorStoreManager:
    return VectorStoreManager(settings=settings, collection_name=collection_name)
