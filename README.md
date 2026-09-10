# Production RAG Service

Retrieval-augmented generation over your own documents: hybrid retrieval
(dense + BM25 fused with RRF), cross-encoder re-ranking, grounded synthesis with
a refusal guardrail, a shared query cache, Prometheus metrics, and an automated
RAGAS evaluation harness.

FastAPI · LlamaIndex · ChromaDB · Redis · Groq/OpenAI · Docker

---

## Architecture

The central constraint is a hard separation between the **offline** ingestion
pipeline and the **online** query pipeline. They share `app/core/` and nothing
else — `app/api/` may never import `app/ingestion/`, which is asserted in the
test suite. The serving image has no business carrying PDF parsers, and the
boundary being physical is what stops someone adding an "just ingest it inline"
endpoint six months from now.

```
app/
├── core/          config, logging, model factories, cache, metrics  (shared)
├── ingestion/     load → chunk → enrich → dedupe → embed → store    (offline)
├── query/         retrieve → fuse → rerank → synthesise             (online)
├── db/            vector store wrapper (the only module importing chromadb)
├── api/           FastAPI routes + schemas
└── main.py        app factory
scripts/
├── start.sh       container entrypoint: prepare → ingest → serve
└── evaluate.py    RAGAS harness, runs the REAL QueryEngine
```

### Ingestion (offline)

```
python -m app.ingestion.cli ingest
```

**Idempotent by construction.** A chunk's ID *is* the SHA-256 of
(relative path, page, text), so identical content always lands on the same ID.
"Have I seen this?" becomes a primary-key lookup rather than a similarity
search, which makes re-runs free and the pipeline crash-resumable.

Deduplication happens **before** embedding — embedding is the only step that
costs money, so re-ingesting an unchanged corpus costs approximately nothing.
Editing a file re-embeds it and prunes its stale chunks; without that, changed
paragraphs linger forever as ghosts in retrieval.

### Query (online)

```
POST /query  {"question": "..."}
```

1. **Cache** — exact-match on normalised question + the parameters that shaped
   the answer. Redis (shared across workers/replicas) or in-process.
2. **Hybrid retrieval** — dense top-10 and BM25 top-10, run concurrently.
   The two fail in opposite directions: dense understands paraphrase and is
   hopeless at exact tokens (error codes, function names, rare acronyms); BM25
   is the reverse.
3. **RRF fusion** — `score = Σ 1/(k + rank)`, k=60. Ranks, not scores: cosine
   lives in ~[0,1] while BM25 is unbounded and corpus-dependent, so no
   weighted sum survives corpus growth. RRF makes *agreement* beat confidence.
4. **Cross-encoder re-rank** — 20 candidates → 5. A bi-encoder embeds question
   and chunk independently; a cross-encoder scores them jointly with full
   attention. Far more accurate, and unusable for search since nothing can be
   precomputed — hence the funnel.
5. **Grounded synthesis** — strict prompt; returns a fixed refusal string when
   the context cannot answer. Empty retrieval short-circuits without an LLM
   call at all.

Every answer carries citations (`file_name`, `page_number`, score).

---

## Quickstart

```bash
cp .env.example .env          # add your API key
docker compose up -d --build  # chroma + redis + api, ingests ./data on boot
curl localhost:8080/health
curl -X POST localhost:8080/query -H 'Content-Type: application/json' \
     -d '{"question":"..."}'
```

Put your documents (`.txt`, `.pdf`) in `./data/`.

Running outside Docker:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-local.txt
CONFIG_FILE=config.local.yaml .venv/bin/python -m app.ingestion.cli ingest
CONFIG_FILE=config.local.yaml .venv/bin/uvicorn app.main:app --port 8080
```

---

## Frontend

A React (Vite) single-page app lives in `frontend/`. It is a thin client over
the API: ask a question, read the answer, inspect the citations.

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

It expects the API at `http://localhost:8080` (override with `VITE_API_URL`).
Vite's default port 5173 is already in the backend's CORS allowlist — change one
and you must change the other, or the browser blocks every request.

What it surfaces that a plain chat box would not:

- **Citations per answer** — file name, page, and a relative confidence bar,
  with the underlying chunk text expandable.
- **Refusals as a first-class state.** "Not found in your documents" gets its
  own treatment rather than looking like a failed request, because it is a
  correct outcome.
- **Pipeline internals** — how many chunks each retriever found, how many *both*
  found (the agreement signal RRF rewards), and per-stage timings.
- **Backend health**, polled independently, so the UI can say the API is down
  before you type a question rather than after.

## Configuration

Precedence, highest first:

```
constructor args  >  environment  >  .env  >  config.yaml  >  defaults
```

`config.yaml` is the committed, reviewable description of how the system is
wired. `.env` is **secrets only** — because it outranks the YAML, duplicating
tuning keys there silently shadows the config file.

Two profiles ship: `config.yaml` (production) and `config.local.yaml`
(embedded Chroma, in-process cache, no external services). Select with
`CONFIG_FILE`.

### Swappable backends

| Knob | Options | Notes |
|---|---|---|
| `llm_provider` | `openai`, `groq` | Groq is generation-only — no embeddings endpoint |
| `embedding_provider` | `openai`, `sentence-transformers` | 1536-d vs 384-d — **not interchangeable** |
| `reranker` | `cross-encoder`, `cohere`, `none` | cross-encoder needs `requirements-local.txt` (torch) |
| `cache_backend` | `redis`, `memory` | memory is per-process; hit rate divides by worker count |

**Switching `embedding_provider` requires a re-ingest.** The two produce vectors
of different dimensionality in different spaces. Collections are stamped with
`provider:model` and the app *refuses to start* on a mismatch rather than
serving confident nonsense — Chroma would catch a dimension change, but two
same-sized models from different providers would silently produce meaningless
similarity scores.

`sentence-transformers` embeddings + `cross-encoder` re-ranking run entirely
locally, so only generation needs an API key.

---

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /query` | Answer a question. 200 + `grounded:false` when the corpus can't answer — that's a correct answer, not an error. 502 is reserved for the LLM failing. |
| `GET /health` | Liveness. Touches no dependency, so a Chroma blip can't get healthy pods killed. |
| `GET /ready` | Readiness. 503 when the vector store is unreachable. |
| `GET /metrics` | Prometheus scrape. |
| `GET /cache` | Cache hit rate and size. |
| `POST /cache/invalidate` | Drop cached answers — call after re-ingesting. |

`/metrics` and `/cache/invalidate` are unauthenticated: bind them to an
internal port or restrict by network policy.

### Metrics

`rag_query_latency_seconds` (histogram, labelled by pipeline stage) and
`rag_retrieval_hit_count_total` (counter, by retriever), plus query outcomes and
cache events. Labels are deliberately low-cardinality — never question text.

Set `PROMETHEUS_MULTIPROC_DIR` when running more than one worker, or a scrape
reports whichever worker happened to answer it.

---

## Evaluation

```bash
python scripts/evaluate.py
python scripts/evaluate.py --fail-under context_recall=0.7,faithfulness=0.8
```

Runs the **real** `QueryEngine` — the same code that serves `/query`. A harness
with its own retrieval path measures a system no user talks to, and the two
drift silently.

Three RAGAS metrics, which together localise a regression:

- `context_precision` — junk in the retrieved set → the **reranker**
- `context_recall` — the answer was never retrieved → the **retriever**
- `faithfulness` — unsupported claims in the answer → the **generator**

Unanswerable questions are scored *separately* as `refusal_accuracy`. RAGAS
metrics are undefined for a correct refusal, and averaging them in punishes the
system for behaving correctly.

Results append to `logs/evaluation_results.json` with a config snapshot — a
score without its configuration is uninterpretable — and each metric records
`scored/total` coverage so a partially-scored run can't masquerade as a
complete one.

Judge calls are LLM-based and rate-limit sensitive: use `--judge-workers 1` and
a fast non-reasoning `--judge-model` on a throttled provider.

---

## Development

```bash
pip install -r requirements.txt -r requirements-local.txt -r requirements-dev.txt
pytest          # 67 tests, no network, no API key
ruff check app scripts tests
mypy
```

The suite stubs the embedding model and the LLM, so it runs offline and
deterministically. Notable cases: RRF ranking behaviour, ingestion idempotency,
the embedding-space guard, the refusal path (asserting no LLM call is made),
metric label cardinality, and an assertion that the serving path never imports
`app.ingestion`.

> Install requirement files **together in one pip invocation**. pip does not
> backtrack already-installed packages, so adding them one at a time can
> resolve a dependency set that imports but is subtly wrong.

## Known limitations

- **BM25 is in-process.** Chroma has no inverted index, so the sparse retriever
  holds the corpus in RAM per replica. Fine to ~100k chunks; past that the fix
  is Qdrant's native sparse vectors, not a bigger cache.
- **Deleting a source file leaves orphaned chunks.** Ingestion reconciles files
  it sees; a file removed from `./data` is never visited. Needs a full sweep.
- **Nothing invalidates the cache after ingestion** — wire the ingest job to
  `POST /cache/invalidate`.
- **No dependency lockfile.** Ranges only; `pip freeze` before deploying.
