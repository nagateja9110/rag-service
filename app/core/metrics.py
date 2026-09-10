"""Prometheus instrumentation.

NAMING NOTE
-----------
``prometheus_client`` appends ``_total`` to every Counter, per OpenMetrics
convention. So the counter declared here as ``rag_retrieval_hit_count`` is
scraped as **rag_retrieval_hit_count_total** -- that is the series name to use
in PromQL. There is no way to suppress the suffix for a counter, and you should
not want to: the suffix is what tells a reader (and Grafana's autocomplete) that
``rate()`` is the sensible thing to apply. The Histogram keeps its exact name and
exposes ``rag_query_latency_seconds_bucket`` / ``_sum`` / ``_count``.

LABEL CARDINALITY
-----------------
Every distinct label-value combination is a separate time series held in
Prometheus's memory forever. The single most destructive mistake in
instrumenting a RAG service is labelling by question text, user ID, or document
name -- that is unbounded cardinality and it will take down your Prometheus, not
your app. Every label below has a small, fixed, enumerable domain:
``stage`` (4), ``retriever`` (3), ``outcome`` (3), ``result`` (2).

MULTIPROCESS
------------
Under ``uvicorn --workers 4`` each worker holds its own in-memory registry. A
scrape hits ONE worker, so counters appear to jump around and reset as the load
balancer moves the scrape between workers -- graphs become nonsense and
``rate()`` produces garbage. The fix is ``PROMETHEUS_MULTIPROC_DIR``: workers
write to shared mmap files and the scrape aggregates across all of them. See
``render_latest()``.
"""

from __future__ import annotations

import contextlib
import os
import shutil

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

from app.core.logging import get_logger

logger = get_logger(__name__)

# MUST run before any metric is constructed. In multiprocess mode
# prometheus_client opens an mmap file the moment a Counter/Histogram is
# defined -- i.e. at import time -- and raises FileNotFoundError if the
# directory does not exist yet. Creating it here (rather than in a startup hook)
# is what makes `import app.core.metrics` safe regardless of who imports first.
# This only ensures the directory exists; clearing it is a separate, deliberate
# step -- see reset_multiprocess_dir().
_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
if _MULTIPROC_DIR:
    os.makedirs(_MULTIPROC_DIR, exist_ok=True)

# Default prometheus_client buckets top out at 10s and waste half their
# resolution below 100ms -- wrong shape for RAG, where a cache hit is ~5ms and a
# cold query with re-ranking and generation is comfortably 2-15s. These are
# roughly Fibonacci-spaced across the range that actually occurs.
_LATENCY_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, float("inf"),
)

# --- The two metrics requested, plus the minimum needed to interpret them ----

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "Query latency by pipeline stage.",
    labelnames=("stage",),
    buckets=_LATENCY_BUCKETS,
)

RETRIEVAL_HITS = Counter(
    "rag_retrieval_hit_count",
    "Chunks returned, by retriever. Scraped as rag_retrieval_hit_count_total.",
    labelnames=("retriever",),
)

QUERIES = Counter(
    "rag_queries",
    "Queries answered, by outcome.",
    labelnames=("outcome",),  # grounded | refused | error
)

CACHE_EVENTS = Counter(
    "rag_cache_events",
    "Query cache lookups.",
    labelnames=("result",),  # hit | miss
)

RERANKER_DEGRADED = Counter(
    "rag_reranker_degraded",
    "Times the reranker failed and the pipeline fell back to fusion order.",
)


def record_query(
    timings_ms: dict[str, float],
    retrieval: dict[str, int] | None,
    grounded: bool,
    cached: bool,
    reranker_degraded: bool = False,
) -> None:
    """Emit metrics for one answered query. Never raises.

    Instrumentation must not be able to break the request it is measuring, so
    this swallows its own errors. A missing metric is an annoyance; a 500
    because a label was malformed is an outage.
    """
    try:
        CACHE_EVENTS.labels(result="hit" if cached else "miss").inc()
        QUERIES.labels(outcome="grounded" if grounded else "refused").inc()

        for stage, millis in timings_ms.items():
            QUERY_LATENCY.labels(stage=stage).observe(millis / 1000.0)

        # Cache hits skip retrieval entirely; recording zeros would dilute the
        # hit-rate signal with meaningless samples.
        if retrieval and not cached:
            RETRIEVAL_HITS.labels(retriever="dense").inc(retrieval.get("vector_hits", 0))
            RETRIEVAL_HITS.labels(retriever="sparse").inc(retrieval.get("bm25_hits", 0))
            RETRIEVAL_HITS.labels(retriever="fused").inc(
                retrieval.get("fused_candidates", 0)
            )

        if reranker_degraded:
            RERANKER_DEGRADED.inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metric recording failed (ignored): %s", exc)


def record_error() -> None:
    with contextlib.suppress(Exception):
        QUERIES.labels(outcome="error").inc()


# ---------------------------------------------------------------------------
# Exposition
# ---------------------------------------------------------------------------
def multiprocess_enabled() -> bool:
    return bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))


def reset_multiprocess_dir() -> None:
    """Clear stale mmap files. Run ONCE, in the entrypoint, BEFORE workers fork.

    Two reasons this is not called from the FastAPI lifespan:

    1. Files from a previous run survive a restart, so without this the process
       comes back up already reporting the dead run's counters.
    2. With ``--workers N`` the lifespan runs in EVERY worker. A worker wiping
       the directory on startup would delete its siblings' live mmap files and
       silently zero their metrics -- a worse bug than the stale data it was
       meant to fix.

    Hence: ``python -m app.core.metrics`` in the entrypoint, then exec uvicorn.
    Safe to run unconditionally; it no-ops when multiprocess mode is off.
    """
    path = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not path:
        return
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    logger.info("Prometheus multiprocess dir ready at %s", path)


def render_latest() -> tuple[bytes, str]:
    """Produce the scrape payload.

    In multiprocess mode a fresh registry aggregates every worker's mmap files;
    otherwise the default in-process registry is used.
    """
    if multiprocess_enabled():
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST


if __name__ == "__main__":  # pragma: no cover - entrypoint helper
    reset_multiprocess_dir()
