"""Online query pipeline: cache -> retrieve -> fuse -> rerank -> synthesise.

This is the counterpart to app/ingestion/pipeline.py, and the two never import
each other. Everything here is latency-sensitive and runs per request; nothing
here parses a PDF.

CONCURRENCY
-----------
Two of the four stages are blocking and CPU-bound (BM25 scoring, cross-encoder
inference) and one is blocking I/O (Chroma's client is synchronous). Running any
of them directly in an ``async def`` handler would block the event loop and
stall every other in-flight request -- the classic way an async service ends up
slower than a threaded one. They are offloaded with ``anyio.to_thread``.

Note also that dense and sparse retrieval are independent, so they are wrapped
in a ``gather``: total retrieval latency is max(dense, sparse) rather than
their sum.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import anyio.to_thread
from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.schema import MetadataMode, NodeWithScore

from app.core.cache import CacheBackend, build_cache, make_cache_key
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import record_error, record_query
from app.core.models import build_embed_model, build_llm
from app.db.vector_store import (
    VectorStoreManager,
    get_vector_store,
    is_stale_collection_error,
)
from app.query.prompts import NO_ANSWER_RESPONSE, QA_TEMPLATE, REFINE_TEMPLATE
from app.query.reranker import ResilientReranker, build_reranker
from app.query.retriever import FusionDebug, HybridRetriever

logger = get_logger(__name__)

_PREVIEW_CHARS = 320


@dataclass
class Source:
    """A cited chunk. Every answer must be traceable to one of these."""

    chunk_id: str
    file_name: str
    page_number: int
    score: float
    preview: str


@dataclass
class QueryResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    cached: bool = False
    grounded: bool = True  # False when we returned the refusal
    reranker: str = "none"
    reranker_degraded: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    # FULL text of every chunk sent to the LLM, in rank order. Deliberately not
    # in QueryResponse -- the HTTP API returns 320-char previews, because
    # shipping whole chunks over the wire is wasteful and leaks more of the
    # corpus than a citation needs to. But RAGAS scores context precision and
    # recall against the actual retrieved text, so truncated previews would
    # produce quietly wrong evaluation numbers. Internal consumers get the real
    # thing; HTTP clients get the preview.
    contexts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [asdict(s) for s in self.sources]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QueryResult:
        """Rebuild from a cached dict.

        Unknown keys are dropped rather than raising: a cache entry written by
        an older deploy must not crash a newer one mid-rollout. Missing keys
        fall back to the dataclass defaults for the same reason.
        """
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in payload.items() if k in known}
        data["sources"] = [Source(**s) for s in payload.get("sources", [])]
        return cls(**data)


class QueryEngine:
    """Holds the expensive, long-lived objects: index, BM25, cross-encoder, cache.

    Built once at application startup, not per request. Loading a cross-encoder
    takes seconds and building BM25 scales with the corpus; doing either inside
    a request handler would put that cost on a user.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        # Pass settings through explicitly: an engine built with an override
        # must not end up talking to the globally-configured collection.
        self._store: VectorStoreManager = get_vector_store(settings=self._settings)

        self._embed_model = build_embed_model(self._settings)
        self._llm = build_llm(self._settings)

        self._reranker: ResilientReranker = build_reranker(self._settings)
        self._build_index()

        self._synthesizer = get_response_synthesizer(
            llm=self._llm,
            text_qa_template=QA_TEMPLATE,
            refine_template=REFINE_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
        )

        self._cache: CacheBackend | None = build_cache(self._settings)

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        # from_vector_store() does NOT read the corpus -- it wraps the store and
        # queries it lazily. Cheap to construct, so cheap to rebuild.
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._store.llama_store,
            embed_model=self._embed_model,
        )
        self._retriever = HybridRetriever(self._store, self._index, self._settings)

    async def _reconnect(self) -> None:
        """Re-resolve the collection after it was recreated underneath us.

        Triggered when ``ingest --rebuild`` drops and recreates the collection
        while this process is serving. Without it, a long-running API keeps
        querying a dead UUID and every request 500s until someone restarts the
        pod -- a genuinely nasty failure because ingestion looks like it
        succeeded and the API looks like it crashed for no reason.
        """
        logger.warning("Collection was recreated; re-resolving handles.")
        self._store.refresh()
        self._build_index()
        if self._cache is not None:
            # Answers derived from the old corpus are not valid for the new one.
            await self._cache.clear()

    async def warmup(self) -> None:
        """Pay the expensive startup costs before the first user arrives.

        Deliberately ordered: the embedding-space check runs FIRST and is the
        one thing here allowed to abort startup. Serving queries against a
        collection embedded by a different model produces confident nonsense,
        which is far worse than failing to boot -- so this is a crash, not a
        warning.
        """
        self._store.verify_embedding_signature()

        if self._cache is not None:
            ping = getattr(self._cache, "ping", None)
            if ping is not None:
                await ping()

        try:
            self._retriever.ensure_bm25(force=True)
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            logger.error("BM25 warmup failed (will retry on first query): %s", exc)

    async def aclose(self) -> None:
        """Release the cache connection pool on shutdown."""
        close = getattr(self._cache, "close", None)
        if close is not None:
            await close()

    async def invalidate(self) -> int:
        """Drop cached answers and the sparse index. Call after re-ingestion."""
        self._retriever.invalidate()
        return await self._cache.clear() if self._cache else 0

    @property
    def cache_stats(self) -> dict[str, Any]:
        return self._cache.stats() if self._cache else {"enabled": False}

    @property
    def reranker_name(self) -> str:
        return self._reranker.name

    # ------------------------------------------------------------------
    async def answer(self, question: str, top_k: int | None = None) -> QueryResult:
        top_k = top_k or self._settings.rerank_top_n
        started = time.perf_counter()

        cache_key = make_cache_key(
            question,
            top_k=top_k,
            llm=self._settings.llm_model,
            reranker=self._reranker.name,
            embedding=self._settings.embedding_signature,
            collection=self._settings.chroma_collection,
        )
        if self._cache is not None:
            hit = await self._cache.get(cache_key)
            if hit is not None:
                logger.info("Cache hit for question: %.60s", question)
                # Rebuilt from the dict, so a caller mutating the result cannot
                # reach back into an in-memory cache entry.
                cached = QueryResult.from_dict(hit)
                cached.cached = True
                cached.timings_ms = {
                    "total": round((time.perf_counter() - started) * 1000, 2)
                }
                record_query(
                    cached.timings_ms,
                    None,
                    grounded=cached.grounded,
                    cached=True,
                )
                return cached

        # --- Retrieve (blocking: Chroma client + BM25 scoring) --------------
        t0 = time.perf_counter()
        try:
            fused, debug = await anyio.to_thread.run_sync(
                self._retriever.retrieve, question
            )
        except Exception as exc:
            if not is_stale_collection_error(exc):
                raise
            # Recover once, in-process, rather than serving 500s until restart.
            await self._reconnect()
            fused, debug = await anyio.to_thread.run_sync(
                self._retriever.retrieve, question
            )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not fused:
            logger.info("No candidates retrieved; refusing without an LLM call.")
            return self._refusal(started, retrieval_ms, debug)

        # --- Re-rank (blocking: cross-encoder forward pass) -----------------
        t0 = time.perf_counter()
        top_nodes = await anyio.to_thread.run_sync(
            self._reranker.rerank, question, fused, top_k
        )
        rerank_ms = (time.perf_counter() - t0) * 1000

        threshold = self._settings.min_rerank_score
        if threshold is not None:
            kept = [n for n in top_nodes if (n.score or 0.0) >= threshold]
            if not kept:
                logger.info(
                    "All %d candidates scored below min_rerank_score=%.3f; refusing.",
                    len(top_nodes),
                    threshold,
                )
                return self._refusal(started, retrieval_ms, debug, rerank_ms)
            top_nodes = kept

        # --- Synthesise (async I/O: OpenAI) ---------------------------------
        t0 = time.perf_counter()
        try:
            response = await self._synthesizer.asynthesize(
                query=question, nodes=top_nodes
            )
            answer_text = str(response).strip()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Surface as a typed failure; the route turns it into a 502 so the
            # client can distinguish "model is down" from "no answer exists".
            record_error()
            raise SynthesisError(str(exc)) from exc
        synthesis_ms = (time.perf_counter() - t0) * 1000

        grounded = not is_refusal(answer_text)

        result = QueryResult(
            answer=answer_text,
            sources=[_to_source(n) for n in top_nodes] if grounded else [],
            cached=False,
            grounded=grounded,
            reranker=self._reranker.name,
            reranker_degraded=self._reranker.degraded,
            timings_ms={
                "retrieval": round(retrieval_ms, 2),
                "rerank": round(rerank_ms, 2),
                "synthesis": round(synthesis_ms, 2),
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
            retrieval=asdict(debug),
            contexts=[
                n.node.get_content(metadata_mode=MetadataMode.NONE) for n in top_nodes
            ],
        )
        record_query(
            result.timings_ms,
            result.retrieval,
            grounded=result.grounded,
            cached=False,
            reranker_degraded=result.reranker_degraded,
        )

        if self._cache is not None:
            # Refusals are cached too: a question the corpus cannot answer will
            # not become answerable within the TTL, and re-running the full
            # pipeline to produce the same refusal is pure waste.
            await self._cache.set(cache_key, result.as_dict())

        logger.info(
            "Answered in %.0fms (retrieve %.0f / rerank %.0f / synth %.0f) "
            "grounded=%s sources=%d",
            result.timings_ms["total"],
            retrieval_ms,
            rerank_ms,
            synthesis_ms,
            grounded,
            len(result.sources),
        )
        return result

    # ------------------------------------------------------------------
    def _refusal(
        self,
        started: float,
        retrieval_ms: float,
        debug: FusionDebug,
        rerank_ms: float = 0.0,
    ) -> QueryResult:
        """Return the guardrail response without spending an LLM call.

        The prompt already instructs the model to refuse on empty context, but
        relying on that means paying for a token round-trip to be told what we
        already know. Deterministic beats probabilistic when the answer is
        knowable locally.
        """
        result = QueryResult(
            answer=NO_ANSWER_RESPONSE,
            sources=[],
            cached=False,
            grounded=False,
            reranker=self._reranker.name,
            reranker_degraded=self._reranker.degraded,
            timings_ms={
                "retrieval": round(retrieval_ms, 2),
                "rerank": round(rerank_ms, 2),
                "synthesis": 0.0,
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
            retrieval=asdict(debug),
        )
        record_query(
            result.timings_ms,
            result.retrieval,
            grounded=False,
            cached=False,
            reranker_degraded=result.reranker_degraded,
        )
        return result


class SynthesisError(RuntimeError):
    """The LLM call failed -- distinct from 'the corpus has no answer'."""


def _normalise_for_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.casefold()).strip()


_REFUSAL_NORMALISED = _normalise_for_compare(NO_ANSWER_RESPONSE)


def is_refusal(answer: str) -> bool:
    """True only when the answer IS the guardrail response.

    Deliberately an equality test, not a substring test. Substring matching
    looks equivalent and is not:

      * Rule 4 of the prompt explicitly invites partial answers ("here is what
        the context supports, here is what is missing"). Those are grounded
        answers that must keep their citations, and they very naturally contain
        the phrase "I don't have enough information to answer this ... fully".
      * A retrieved chunk could quote the phrase, and the model could quote the
        chunk.

    Either would strip every source off a perfectly good answer -- a citation
    bug that only shows up on the hard queries, which is the worst place for it.
    """
    return _normalise_for_compare(answer) == _REFUSAL_NORMALISED


def _to_source(hit: NodeWithScore) -> Source:
    metadata = hit.node.metadata or {}
    text = hit.node.get_content(metadata_mode=MetadataMode.NONE)
    preview = text[:_PREVIEW_CHARS].strip()
    if len(text) > _PREVIEW_CHARS:
        preview += "..."
    return Source(
        chunk_id=hit.node.node_id,
        file_name=str(metadata.get("file_name", "unknown")),
        page_number=int(metadata.get("page_number", 1) or 1),
        score=round(float(hit.score or 0.0), 6),
        preview=preview,
    )
