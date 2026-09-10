"""HTTP surface.

Health endpoints come in two flavours, and the distinction matters for any real
deployment:

  * ``/health``  -- liveness. "Is this process alive?" Never touches a
                    dependency. If it did, a Chroma blip would make Kubernetes
                    kill perfectly healthy API pods and turn a degraded system
                    into an outage.
  * ``/ready``   -- readiness. "Can this process serve traffic?" Checks the
                    vector store and returns 503 when it cannot, so the load
                    balancer stops routing to it without restarting it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.schemas import (
    CacheStatsResponse,
    HealthResponse,
    InvalidateResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
)
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import render_latest
from app.db.vector_store import VectorStoreError, VectorStoreManager, get_vector_store
from app.query.engine import QueryEngine, SynthesisError

logger = get_logger(__name__)

router = APIRouter()


def get_engine(request: Request) -> QueryEngine:
    """Pull the singleton built during startup.

    Deliberately NOT a module-level global and NOT constructed per request:
    the engine owns a cross-encoder and a BM25 index, so building one costs
    seconds and hundreds of megabytes. app.state is where FastAPI expects
    process-scoped resources to live.
    """
    engine: QueryEngine | None = getattr(request.app.state, "query_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="query engine is not initialised",
        )
    return engine


def get_store() -> VectorStoreManager:
    return get_vector_store()


# ----------------------------------------------------------------------
# Ops
# ----------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness probe. Intentionally dependency-free."""
    return HealthResponse(status="ok", environment=settings.app_env)


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(
    settings: Settings = Depends(get_settings),
    store: VectorStoreManager = Depends(get_store),
    engine: QueryEngine = Depends(get_engine),
) -> ReadyResponse:
    """Readiness probe. Fails closed if the vector store is unreachable."""
    try:
        if not store.healthy():
            raise VectorStoreError("heartbeat failed")
        vectors = store.count()
    except VectorStoreError as exc:
        logger.error("Readiness check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vector store unavailable",
        ) from exc

    return ReadyResponse(
        status="ready",
        vector_store=settings.chroma_mode,
        collection=settings.chroma_collection,
        vectors=vectors,
        reranker=engine.reranker_name,
    )


@router.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape endpoint.

    Excluded from the OpenAPI schema because it is not part of the product API
    -- it is operational plumbing with a text/plain exposition format that
    OpenAPI describes badly.

    NOTE: this is unauthenticated and it leaks operational shape (query volume,
    latency distribution, corpus activity). Standard practice is to bind it to
    an internal-only port or restrict it by network policy so only your
    Prometheus can reach it.
    """
    payload, content_type = render_latest()
    return Response(content=payload, media_type=content_type)


@router.get("/cache", response_model=CacheStatsResponse, tags=["ops"])
def cache_stats(engine: QueryEngine = Depends(get_engine)) -> CacheStatsResponse:
    """Hit rate, size, evictions. Watch this before tuning TTL or max size."""
    return CacheStatsResponse(cache=engine.cache_stats)


@router.post("/cache/invalidate", response_model=InvalidateResponse, tags=["ops"])
async def invalidate(engine: QueryEngine = Depends(get_engine)) -> InvalidateResponse:
    """Drop cached answers and the sparse index -- call after re-ingesting.

    Without this, a 5-minute TTL means up to 5 minutes of answers derived from
    a corpus that no longer exists. In a real deployment the ingestion job
    should call this (or publish an event) on completion.

    NOTE: this mutates server state and is unauthenticated. Put it behind auth
    or a network policy before this service is reachable from anywhere but your
    own cluster.
    """
    cleared = await engine.invalidate()
    return InvalidateResponse(
        cleared_entries=cleared, detail="query cache cleared; BM25 index will rebuild"
    )


# ----------------------------------------------------------------------
# Query
# ----------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query(
    payload: QueryRequest,
    engine: QueryEngine = Depends(get_engine),
) -> QueryResponse:
    """Answer a question from the indexed corpus.

    Returns 200 with ``grounded: false`` when the corpus cannot answer -- that
    is a valid, correct answer, not an error. 502 is reserved for the LLM
    itself failing, so clients can retry on one and not the other.
    """
    try:
        result = await engine.answer(payload.question, top_k=payload.top_k)
    except SynthesisError as exc:
        logger.error("Synthesis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="language model request failed",
        ) from exc
    except VectorStoreError as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vector store unavailable",
        ) from exc

    return QueryResponse(**result.as_dict())
