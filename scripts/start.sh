#!/usr/bin/env bash
#
# Container entrypoint: prepare -> (optionally) ingest -> serve.
#
# Runs as PID 1 in the api container. Three responsibilities, in order:
#
#   1. Clear stale Prometheus mmap files -- ONCE, before uvicorn forks workers.
#   2. Wait for the vector store, then ingest ./data if anything is mounted.
#   3. exec uvicorn, so it inherits PID 1 and receives SIGTERM directly.
#
# That last point matters more than it looks: without `exec`, bash stays PID 1,
# does not forward signals to its child by default, and `docker stop` waits the
# full 10s grace period before SIGKILLing your API mid-request. With `exec`,
# uvicorn shuts down gracefully.
set -euo pipefail

log() { printf '%s | %-7s | start.sh | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S%z)" "$1" "$2"; }

DATA_DIR="${DATA_DIR:-/app/data}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8080}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
SKIP_INGESTION="${SKIP_INGESTION:-false}"
# If ingestion fails: false (default) logs and serves anyway -- the collection
# may already be populated from a previous run, and a grounded "I don't have
# enough information" beats a crash loop. Set true in environments where an
# incomplete index is worse than being down.
INGEST_STRICT="${INGEST_STRICT:-false}"
WAIT_FOR_STORE_SECONDS="${WAIT_FOR_STORE_SECONDS:-60}"

# --------------------------------------------------------------------------
# 1. Prometheus multiprocess directory
# --------------------------------------------------------------------------
# No-op unless PROMETHEUS_MULTIPROC_DIR is set. Must happen here rather than in
# the FastAPI lifespan: the lifespan runs in EVERY worker, so a worker clearing
# the directory would delete its siblings' live metrics.
python -m app.core.metrics

# --------------------------------------------------------------------------
# 2. Wait for the vector store
# --------------------------------------------------------------------------
# Compose's `depends_on: service_healthy` already gates this, but the wait is
# kept for environments without that guarantee (Kubernetes, plain docker run),
# where the API otherwise crash-loops until Chroma finishes booting.
if [ "${CHROMA_MODE:-persistent}" = "http" ]; then
  log INFO "Waiting up to ${WAIT_FOR_STORE_SECONDS}s for Chroma at ${CHROMA_HOST:-chroma}:${CHROMA_PORT:-8000}"
  deadline=$(( $(date +%s) + WAIT_FOR_STORE_SECONDS ))
  until python - <<'PYCHECK'
import os, sys, urllib.request
host = os.environ.get("CHROMA_HOST", "chroma")
port = os.environ.get("CHROMA_PORT", "8000")
for path in ("/api/v2/heartbeat", "/api/v1/heartbeat"):
    try:
        with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=3) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception:
        continue
sys.exit(1)
PYCHECK
  do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log ERROR "Chroma did not become ready in ${WAIT_FOR_STORE_SECONDS}s"
      exit 1
    fi
    sleep 2
  done
  log INFO "Chroma is ready"
fi

# --------------------------------------------------------------------------
# 3. Ingest, if a corpus is actually mounted
# --------------------------------------------------------------------------
# "Mounted" means: the directory exists AND contains at least one supported
# file. An empty ./data is the normal case for an API-only deployment where a
# separate job owns ingestion, so it is not an error -- just skip.
#
# Running this on every start is safe because ingestion is idempotent: chunk IDs
# are content hashes, so an unchanged corpus costs one ID lookup per chunk and
# zero embedding calls. Edited files are re-embedded and their stale chunks
# pruned. That is what makes "ingest on boot" a reasonable default rather than a
# way to burn money on every restart.
if [ "$SKIP_INGESTION" = "true" ]; then
  log INFO "SKIP_INGESTION=true -- not ingesting"
elif [ ! -d "$DATA_DIR" ]; then
  log INFO "No data directory at ${DATA_DIR} -- skipping ingestion"
elif [ -z "$(find "$DATA_DIR" -type f \( -name '*.txt' -o -name '*.pdf' \) -print -quit 2>/dev/null)" ]; then
  log INFO "No .txt or .pdf files under ${DATA_DIR} -- skipping ingestion"
else
  log INFO "Ingesting from ${DATA_DIR}"
  if python -m app.ingestion.cli ingest --data-dir "$DATA_DIR"; then
    log INFO "Ingestion complete"
  elif [ "$INGEST_STRICT" = "true" ]; then
    log ERROR "Ingestion failed and INGEST_STRICT=true -- refusing to start"
    exit 1
  else
    log WARNING "Ingestion failed; starting the API anyway (set INGEST_STRICT=true to fail hard)"
  fi
fi

# --------------------------------------------------------------------------
# 4. Serve
# --------------------------------------------------------------------------
# Reminder: >1 worker REQUIRES PROMETHEUS_MULTIPROC_DIR, or /metrics reports
# whichever worker happened to answer the scrape.
if [ "$UVICORN_WORKERS" -gt 1 ] && [ -z "${PROMETHEUS_MULTIPROC_DIR:-}" ]; then
  log WARNING "UVICORN_WORKERS=${UVICORN_WORKERS} without PROMETHEUS_MULTIPROC_DIR -- /metrics will be per-worker and misleading"
fi

log INFO "Starting uvicorn on ${API_HOST}:${API_PORT} with ${UVICORN_WORKERS} worker(s)"
exec uvicorn app.main:app \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --workers "$UVICORN_WORKERS" \
  --no-server-header \
  --timeout-keep-alive 30
