# syntax=docker/dockerfile:1.7
#
# Multi-stage build. The builder carries compilers and headers; the runtime
# carries neither -- roughly a 400MB saving and, more importantly, a large chunk
# of attack surface that never ships.
#
# One image serves the API, the ingestion CLI and the start script: same code,
# same dependency set, different entrypoint. Splitting them is a legitimate
# later optimisation (the API never needs pypdf), but two images means two build
# pipelines and two chances to ship a version skew between the embedding model
# used at ingest and the one used at query. Not worth it yet.

# ---------------------------------------------------------------------------
# Stage 1: build dependencies into a venv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Requirements copied alone so this layer caches on dependency changes, not on
# every source edit.
COPY requirements.txt requirements-local.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Local (self-hosted) models are opt-in at build time, because torch adds ~2.5GB.
# Covers BOTH embedding_provider=sentence-transformers and reranker=cross-encoder.
#   docker build --build-arg INSTALL_LOCAL_MODELS=true .
ARG INSTALL_LOCAL_MODELS=false
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# HF_HOME must point somewhere the runtime stage copies from. The default
# (~/.cache/huggingface) belongs to root in THIS stage and would be discarded,
# silently reintroducing the cold-start download we are trying to remove.
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p /opt/hf-cache && \
    if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then \
        pip install -r requirements-local.txt && \
        # Bake the weights in. Otherwise the first request after every deploy
        # blocks on a HuggingFace download, and an HF outage becomes your outage.
        python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
CrossEncoder('${RERANK_MODEL}'); SentenceTransformer('${EMBED_MODEL}')" ; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

# `tini` reaps zombies. uvicorn's worker supervisor forks children, and without
# a real init those become defunct processes that accumulate in long-lived pods.
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root. A container escape should not start from uid 0.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home app

COPY --from=builder /opt/venv /opt/venv
# Empty when INSTALL_LOCAL_MODELS=false; carries baked weights when true.
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

WORKDIR /app
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
# Baseline config. Compose can mount a different profile over it, or set
# CONFIG_FILE to select another path.
COPY --chown=app:app config.yaml ./config.yaml

RUN chmod +x scripts/start.sh \
    && mkdir -p /app/data /app/chroma_db /app/logs /tmp/prometheus \
    && chown -R app:app /app/data /app/chroma_db /app/logs /tmp/prometheus

USER app

EXPOSE 8080

# Liveness only -- /health never touches Chroma or Redis, so a dependency blip
# cannot cause Docker to kill a healthy container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./scripts/start.sh"]
