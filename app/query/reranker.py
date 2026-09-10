"""Cross-encoder re-ranking, with a hosted fallback.

WHY RE-RANK AT ALL
------------------
Retrieval and re-ranking optimise different things, and the difference is
architectural rather than a matter of model quality.

A bi-encoder (``text-embedding-3-small``) embeds the question and the chunk
*independently*. That is what makes vector search possible -- chunk vectors are
computed once at ingest time -- but it also means the model never sees the query
and the document together. It has to compress a chunk into 1536 numbers without
knowing what will be asked of it.

A cross-encoder scores the pair jointly: query and chunk go into the model in
one forward pass, with full attention between them. Far more accurate, and
completely unusable for search -- it cannot precompute anything, so scoring a
100k-chunk corpus means 100k forward passes per query.

Hence the funnel: cheap recall-oriented retrieval to 20 candidates, expensive
precision-oriented scoring on just those 20. Two orders of magnitude less
compute than scoring the corpus, and it fixes the failure that hurts RAG most --
a marginally-relevant chunk crowding out the right one in the LLM's context.

BACKEND SELECTION
-----------------
``cross-encoder`` needs sentence-transformers, which pulls torch: roughly 2.5GB
of image for a 90MB model. That is a real deployment cost, so it is NOT in the
base requirements -- see requirements-local.txt and the INSTALL_LOCAL_MODELS
build arg. ``cohere`` is in the base image and needs no torch. ``none`` is an
honest passthrough for development where you want neither.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from llama_index.core.schema import MetadataMode, NodeWithScore

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Reranker(Protocol):
    """The contract. Swapping backends must not touch the query engine."""

    name: str

    def rerank(
        self, question: str, nodes: Sequence[NodeWithScore], top_n: int
    ) -> list[NodeWithScore]: ...


class PassthroughReranker:
    """No re-ranking; preserve fusion order. Honest about doing nothing."""

    name = "none"

    def rerank(
        self, question: str, nodes: Sequence[NodeWithScore], top_n: int
    ) -> list[NodeWithScore]:
        return list(nodes[:top_n])


class LocalCrossEncoderReranker:
    """sentence-transformers cross-encoder, run in-process on CPU.

    ~15-40ms for 20 candidates on a modern CPU. Blocking and CPU-bound, so the
    caller is responsible for keeping it off the event loop.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str, max_length: int = 512):
        # Imported lazily and inside __init__ so that torch is only required
        # when this backend is actually selected.
        from sentence_transformers import CrossEncoder

        started = time.perf_counter()
        self._model = CrossEncoder(model_name, max_length=max_length)
        self._model_name = model_name
        logger.info(
            "Loaded cross-encoder '%s' in %.2fs",
            model_name,
            time.perf_counter() - started,
        )

    def rerank(
        self, question: str, nodes: Sequence[NodeWithScore], top_n: int
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        pairs = [
            (question, n.node.get_content(metadata_mode=MetadataMode.NONE))
            for n in nodes
        ]
        scores = self._model.predict(pairs)

        scored = [
            NodeWithScore(node=n.node, score=float(s))
            # strict: predict() must return exactly one score per pair. A silent
            # length mismatch would misalign every score with the wrong chunk.
            for n, s in zip(nodes, scores, strict=True)
        ]
        scored.sort(key=lambda n: n.score or 0.0, reverse=True)
        return scored[:top_n]


class CohereReranker:
    """Hosted rerank API. No torch, but a network hop and a per-call cost."""

    name = "cohere"

    def __init__(self, api_key: str, model: str):
        import cohere

        self._client = cohere.Client(api_key)
        self._model = model

    def rerank(
        self, question: str, nodes: Sequence[NodeWithScore], top_n: int
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        documents = [
            n.node.get_content(metadata_mode=MetadataMode.NONE) for n in nodes
        ]
        response = self._client.rerank(
            model=self._model,
            query=question,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
        return [
            NodeWithScore(
                node=nodes[result.index].node, score=float(result.relevance_score)
            )
            for result in response.results
        ]


class ResilientReranker:
    """Wraps a backend so a re-ranking failure degrades instead of 500-ing.

    Re-ranking is a *quality* improvement over an already-relevant candidate
    set. If the model OOMs or Cohere times out, returning the top-N fusion
    results is a slightly worse answer; returning an error is no answer. Only
    one of those is acceptable in production.
    """

    def __init__(self, inner: Reranker):
        self._inner = inner
        self.name = inner.name
        self.degraded = False

    def rerank(
        self, question: str, nodes: Sequence[NodeWithScore], top_n: int
    ) -> list[NodeWithScore]:
        try:
            self.degraded = False
            return self._inner.rerank(question, nodes, top_n)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Reranker '%s' failed; falling back to fusion order: %s",
                self._inner.name,
                exc,
            )
            self.degraded = True
            return list(nodes[:top_n])


def build_reranker(settings: Settings) -> ResilientReranker:
    """Construct the configured backend, degrading gracefully if unavailable.

    Selection is explicit via `reranker` in config.yaml (or RERANKER), but a
    missing dependency must
    not stop the service from booting -- it falls through to the next best
    option and says so loudly in the logs.
    """
    backend = settings.reranker

    if backend == "cross-encoder":
        try:
            return ResilientReranker(
                LocalCrossEncoderReranker(settings.local_reranker_model)
            )
        except ImportError:
            logger.warning(
                "reranker='cross-encoder' but sentence-transformers is not installed "
                "(pip install -r requirements.txt -r requirements-local.txt)."
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not load local cross-encoder: %s", exc)

        if settings.cohere_api_key is not None:
            logger.warning("Falling back to Cohere rerank.")
            backend = "cohere"
        else:
            logger.warning("Falling back to no re-ranking; answer quality will drop.")
            return ResilientReranker(PassthroughReranker())

    if backend == "cohere":
        try:
            return ResilientReranker(
                CohereReranker(
                    api_key=settings.cohere_api_key.get_secret_value(),  # type: ignore[union-attr]
                    model=settings.cohere_rerank_model,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not initialise Cohere reranker: %s", exc)
            return ResilientReranker(PassthroughReranker())

    return ResilientReranker(PassthroughReranker())
