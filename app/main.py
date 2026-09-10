"""FastAPI application factory -- the ONLINE entrypoint.

Architectural rule enforced here: this module, and everything it imports, must
never import ``app.ingestion``. The serving path has no business carrying PDF
parsers or bulk-embedding code, and keeping the boundary physical is what stops
someone adding an "just ingest it inline" endpoint six months from now. Shared
concerns (the embedding model, config, logging) live in ``app.core``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.metrics import multiprocess_enabled
from app.db.vector_store import EmbeddingMismatchError
from app.query.engine import QueryEngine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build expensive singletons once, before the first request arrives.

    The cross-encoder takes seconds to load and BM25 scales with the corpus.
    Doing either lazily would make some unlucky user pay for it, and would make
    your p99 latency graph a mystery.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    if multiprocess_enabled():
        logger.info("Prometheus multiprocess mode active")
    logger.info(
        "Starting API | env=%s chroma=%s collection=%s llm=%s embed=%s reranker=%s",
        settings.app_env,
        settings.chroma_mode,
        settings.chroma_collection,
        settings.llm_model,
        settings.embedding_model,
        settings.reranker,
    )

    engine = QueryEngine(settings)
    try:
        await engine.warmup()
    except EmbeddingMismatchError as exc:
        # Refusing to start is the correct behaviour: querying a collection
        # embedded by a different model returns confident nonsense, and a crash
        # loop with this message in the logs is far easier to diagnose than
        # silently bad answers in production.
        logger.error("FATAL: %s", exc)
        raise

    app.state.query_engine = engine
    logger.info(
        "Query engine ready (embeddings=%s, reranker=%s, cache=%s)",
        settings.embedding_signature,
        engine.reranker_name,
        settings.cache_backend if settings.query_cache_enabled else "disabled",
    )

    yield

    await engine.aclose()
    app.state.query_engine = None
    logger.info("Shutting down API")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Service",
        version="0.2.0",
        description="Hybrid-retrieval RAG API (dense + BM25, RRF fusion, cross-encoder re-ranking).",
        # Never expose interactive docs in production without auth in front.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # CORS. Origins are an explicit allowlist from config -- "*" is fine for a
    # public read-only API but becomes a real vulnerability the moment you add
    # cookie auth, which is why Settings rejects wildcard + credentials.
    # Methods and headers are narrowed to what this API actually uses; a
    # reflexive allow_methods=["*"] advertises a surface you do not have.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=600,
    )

    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Log the detail, return a generic message.

        Stack traces and dependency error strings routinely leak file paths,
        connection strings and occasionally API keys; the client gets none of
        that.
        """
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    return app


app = create_app()
