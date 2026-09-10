"""Hybrid retrieval: dense vectors + BM25 keywords, merged with RRF.

WHY HYBRID
----------
The two retrievers fail in opposite directions, which is exactly what you want
from an ensemble. Dense retrieval understands that "how do I stop duplicate
ingestion?" and "idempotency" are the same question, but it is hopeless at exact
tokens -- part numbers, error codes, function names, rare acronyms all get
smeared into semantic neighbourhoods. BM25 nails those and completely misses
paraphrase. Union of the two beats either alone on essentially every public
benchmark, and the gap is widest on precisely the queries users actually type.

WHY RRF AND NOT SCORE-WEIGHTED BLENDING
---------------------------------------
Cosine similarity lives in roughly [0, 1] with a long tail of "sort of related"
around 0.7. BM25 scores are unbounded, corpus-dependent, and vary by query
length. There is no principled constant that puts them on the same scale, and
any weighted sum you tune today breaks when the corpus grows. Reciprocal Rank
Fusion sidesteps the whole problem by throwing scores away and using only
*rank*:

    RRF(d) = sum over retrievers of  1 / (k + rank(d))

k (default 60, from Cormack et al. 2009) flattens the curve near the top so that
rank 1 does not dominate rank 2 outright -- it encodes "being found by both
retrievers matters more than being found first by one." A document at rank 3 in
both lists outranks a document at rank 1 in one list and absent from the other.
That is the behaviour you want, and it needs no tuning.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.vector_store import VectorStoreManager

logger = get_logger(__name__)


@dataclass
class FusionDebug:
    """Per-query retrieval telemetry. Cheap to collect, invaluable when tuning."""

    vector_hits: int = 0
    bm25_hits: int = 0
    fused_candidates: int = 0
    overlap: int = 0  # documents both retrievers found -- the ensemble's value
    bm25_available: bool = True


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[NodeWithScore]],
    k: int = 60,
    top_n: int = 20,
) -> tuple[list[NodeWithScore], int]:
    """Merge ranked lists by RRF. Returns (fused, overlap_count).

    Nodes are deduplicated by ``node_id`` -- which, thanks to the content-hash
    IDs from the ingestion pipeline, means identical chunks collapse correctly
    even if both retrievers surface them.
    """
    scores: dict[str, float] = {}
    nodes: dict[str, NodeWithScore] = {}
    appearances: dict[str, int] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            node_id = hit.node.node_id
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
            appearances[node_id] = appearances.get(node_id, 0) + 1
            # Keep the first NodeWithScore we saw; we overwrite .score below.
            nodes.setdefault(node_id, hit)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    overlap = sum(1 for count in appearances.values() if count > 1)

    fused: list[NodeWithScore] = []
    for node_id, score in ordered:
        hit = nodes[node_id]
        # Replace the retriever-native score with the fusion score. Downstream
        # consumers must not compare a cosine value against a BM25 value.
        fused.append(NodeWithScore(node=hit.node, score=float(score)))
    return fused, overlap


class HybridRetriever:
    """Dense + sparse retrieval over one Chroma collection.

    THE SCALING CAVEAT, STATED PLAINLY
    ----------------------------------
    Chroma has no inverted index, so BM25 runs in this process over a full
    in-memory copy of the corpus. Consequences:

      * Memory grows linearly with the corpus (~1-2 KB/chunk).
      * Every replica holds its own copy and rebuilds it independently.
      * The copy goes stale the moment ingestion writes.

    Mitigated below by rebuilding when the collection's vector count changes,
    checked at most every ``bm25_refresh_seconds``. That is adequate to roughly
    the 100k-chunk range. Past that, the fix is not a better cache -- it is
    moving sparse retrieval into the store. Qdrant supports native sparse
    vectors and server-side fusion, which is the strongest single argument for
    the migration you already have planned: this class collapses to one
    ``query_points`` call with a prefetch, and the RAM problem disappears.
    """

    def __init__(
        self,
        store: VectorStoreManager,
        index: VectorStoreIndex,
        settings: Settings,
    ):
        self._store = store
        self._index = index
        self._settings = settings
        # Typed Any: BM25Retriever is an optional import, so the attribute
        # cannot reference the class at module scope.
        self._bm25: Any = None
        self._bm25_node_count = -1
        self._bm25_checked_at = 0.0

    # ------------------------------------------------------------------
    # BM25 lifecycle
    # ------------------------------------------------------------------
    def _build_bm25(self, nodes: Sequence[BaseNode]) -> Any:
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
        except ImportError:
            logger.warning(
                "llama-index-retrievers-bm25 is not installed; running dense-only."
            )
            return None

        if not nodes:
            return None

        started = time.perf_counter()
        try:
            retriever = BM25Retriever.from_defaults(
                nodes=list(nodes),
                similarity_top_k=self._settings.bm25_top_k,
            )
        except Exception as exc:  # noqa: BLE001
            # A broken sparse index must degrade to dense-only, never 500. Half a
            # good answer beats an error page.
            logger.error("BM25 index build failed, falling back to dense-only: %s", exc)
            return None

        logger.info(
            "Built BM25 index over %d chunk(s) in %.2fs",
            len(nodes),
            time.perf_counter() - started,
        )
        return retriever

    def ensure_bm25(self, force: bool = False) -> None:
        """Rebuild the sparse index if the corpus has drifted.

        Uses ``collection.count()`` as a cheap change detector rather than
        hashing the corpus. It misses same-count edits (delete one chunk, add
        one chunk in the same window) -- acceptable, because the pruning logic
        in ingestion means that window is seconds long and the consequence is
        one stale keyword hit, not a wrong answer.
        """
        now = time.monotonic()
        if (
            not force
            and self._bm25 is not None
            and now - self._bm25_checked_at < self._settings.bm25_refresh_seconds
        ):
            return

        self._bm25_checked_at = now
        try:
            current_count = self._store.count()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not read collection count: %s", exc)
            return

        if not force and current_count == self._bm25_node_count:
            return

        logger.info(
            "Corpus changed (%d -> %d chunks); rebuilding BM25 index",
            self._bm25_node_count,
            current_count,
        )
        nodes = self._store.load_all_nodes(limit=self._settings.bm25_max_nodes)
        self._bm25 = self._build_bm25(nodes)
        self._bm25_node_count = current_count

    def invalidate(self) -> None:
        self._bm25 = None
        self._bm25_node_count = -1
        self._bm25_checked_at = 0.0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> tuple[list[NodeWithScore], FusionDebug]:
        """Blocking. Callers on the event loop must offload this to a thread."""
        bundle = QueryBundle(query_str=question)
        debug = FusionDebug()

        self.ensure_bm25()

        dense = self._index.as_retriever(
            similarity_top_k=self._settings.vector_top_k
        ).retrieve(bundle)
        debug.vector_hits = len(dense)

        sparse: list[NodeWithScore] = []
        if self._bm25 is not None:
            try:
                sparse = self._bm25.retrieve(bundle)
            except Exception as exc:  # noqa: BLE001
                logger.error("BM25 retrieval failed, using dense results only: %s", exc)
                sparse = []
        else:
            debug.bm25_available = False
        debug.bm25_hits = len(sparse)

        fused, overlap = reciprocal_rank_fusion(
            [dense, sparse],
            k=self._settings.rrf_k,
            top_n=self._settings.fusion_top_n,
        )
        debug.fused_candidates = len(fused)
        debug.overlap = overlap

        logger.debug(
            "Retrieval: dense=%d sparse=%d fused=%d overlap=%d",
            debug.vector_hits,
            debug.bm25_hits,
            debug.fused_candidates,
            debug.overlap,
        )
        return fused, debug
