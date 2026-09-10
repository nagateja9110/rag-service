"""Offline RAG evaluation with RAGAS.

    python scripts/evaluate.py
    python scripts/evaluate.py --fail-under context_recall=0.7,faithfulness=0.8

WHY THIS SCRIPT IMPORTS THE REAL QueryEngine
--------------------------------------------
It would be easier to reimplement retrieval here -- build an index, pull top-k,
score it. That is also the single most common way an evaluation harness starts
lying to you. The moment the harness has its own retrieval path, you are
measuring a system that no user ever talks to, and the two drift silently: you
change ``fusion_top_n`` in the app, the eval keeps reporting last month's
architecture, and your dashboard says everything is fine.

So this runs the exact ``QueryEngine`` that serves ``POST /query``, including
BM25, RRF and the cross-encoder. Same code, same config object.

WHAT THE METRICS ACTUALLY MEASURE
---------------------------------
The three together triangulate *where* a regression is, which is the only
reason to track more than one:

  * ``context_precision`` -- of the chunks retrieved, how many were relevant,
    weighted toward the top ranks. Low => the RERANKER is letting junk through,
    or fusion is over-recalling.
  * ``context_recall``    -- of the claims in the ground truth, how many are
    supported by the retrieved chunks. Low => the RETRIEVER never found the
    answer, and no prompt or model change will save you.
  * ``faithfulness``      -- of the claims in the answer, how many are supported
    by the retrieved chunks. Low => the GENERATOR is hallucinating; retrieval
    is fine and the prompt is not holding.

The classic misread is a low faithfulness score sending you to prompt-tune when
context_recall is the number that is actually broken. Precision and recall grade
the retriever; faithfulness grades the generator. Always read them together.

NOTE ON NAMING: RAGAS has no metric called "answer faithfulness" -- it is
``faithfulness``. The answer-side counterpart you may also want later is
``answer_relevancy`` (does the answer address the question at all).

COST
----
Every metric is LLM-judged. Budget roughly 3-6 extra model calls per question on
top of your own pipeline's. A 10-question suite is cents; do not point this at a
1,000-question set without checking the bill first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make `python scripts/evaluate.py` work without PYTHONPATH gymnastics.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.query.engine import QueryEngine, is_refusal

logger = get_logger("evaluate")

DEFAULT_OUTPUT = Path("logs/evaluation_results.json")

# The three requested metrics. Keys are RAGAS's own metric names, which is what
# lands in the JSON and what --fail-under matches on.
METRIC_NAMES = ("context_precision", "context_recall", "faithfulness")


# ----------------------------------------------------------------------
# Test set
# ----------------------------------------------------------------------
# Written against the sample corpus in ./data. Replace with questions from your
# own documents -- a golden set is only useful if it reflects real user queries.
#
# `unanswerable: True` marks questions the corpus genuinely cannot answer. These
# are scored SEPARATELY (see below) and excluded from the RAGAS averages, for a
# specific reason: RAGAS metrics are undefined for a correct refusal. There are
# no claims to verify, no ground-truth claims to recall, and a correct "I don't
# know" would score 0.0 on faithfulness and drag your averages down for doing
# exactly the right thing. Mixing them in is how teams end up optimising their
# guardrail away.
DEFAULT_DATASET: list[dict[str, Any]] = [
    {
        "question": "What makes the ingestion pipeline idempotent?",
        "ground_truth": (
            "Idempotent ingestion means running the pipeline twice produces the same "
            "stored state as running it once. It is achieved by content addressing: "
            "deriving each record's primary key from a hash of its content."
        ),
    },
    {
        "question": "Why does chunk overlap exist?",
        "ground_truth": (
            "Chunk overlap exists so that a sentence spanning a chunk boundary still "
            "appears intact in at least one chunk."
        ),
    },
    {
        "question": "What sets the ceiling on retrieval quality?",
        "ground_truth": (
            "The retrieval quality ceiling is set by the chunking strategy, not by the "
            "generation model."
        ),
    },
    {
        "question": "How many dimensions does text-embedding-3-small return by default?",
        "ground_truth": "text-embedding-3-small returns 1536 dimensions by default.",
    },
    {
        "question": "Which similarity metric suits normalised OpenAI embeddings?",
        "ground_truth": (
            "Cosine similarity is the correct metric for normalised OpenAI embeddings."
        ),
    },
    {
        "question": "What index does Chroma use for approximate search?",
        "ground_truth": "Chroma uses HNSW, a graph index, for approximate search.",
    },
    {
        "question": "How does Qdrant differ from Chroma?",
        "ground_truth": (
            "Chroma is an embedded-first vector database suited to development "
            "workloads and small corpora, while Qdrant is a Rust-based vector database "
            "designed for production workloads with payload indexing and filtering."
        ),
    },
    {
        "question": "What does retrieval-augmented generation do?",
        "ground_truth": (
            "Retrieval-augmented generation grounds a language model's answer in "
            "retrieved source documents."
        ),
    },
    # --- Refusal set: the corpus cannot answer these -----------------------
    {
        "question": "What is the CEO of Anthropic's home address?",
        "ground_truth": "",
        "unanswerable": True,
    },
    {
        "question": "What were this project's Q3 revenue figures?",
        "ground_truth": "",
        "unanswerable": True,
    },
]


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class Prediction:
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    unanswerable: bool
    refused: bool
    latency_ms: float
    retrieved: int


@dataclass
class EvaluationRun:
    timestamp: str
    duration_seconds: float
    dataset_size: int
    scored_questions: int
    refusal_questions: int
    averages: dict[str, float] = field(default_factory=dict)
    # How many samples each average was actually computed from. A metric whose
    # judge calls failed is reported over a SMALLER sample, and an average of
    # three is not the same claim as an average of eight -- recording it is the
    # difference between a score and an honest score.
    coverage: dict[str, str] = field(default_factory=dict)
    refusal_accuracy: float | None = None
    per_question: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    thresholds: dict[str, float] = field(default_factory=dict)


def config_snapshot(settings: Settings) -> dict[str, Any]:
    """Record what produced these scores.

    A score without its configuration is uninterpretable. "context_recall 0.62"
    means nothing six weeks later; "0.62 at chunk_size=1024, fusion_top_n=20,
    reranker=local" is a data point you can act on and compare against.
    """
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.active_llm_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "vector_top_k": settings.vector_top_k,
        "bm25_top_k": settings.bm25_top_k,
        "rrf_k": settings.rrf_k,
        "fusion_top_n": settings.fusion_top_n,
        "rerank_top_n": settings.rerank_top_n,
        "reranker": settings.reranker,
        "embedding_provider": settings.embedding_provider,
        "collection": settings.chroma_collection,
    }


# ----------------------------------------------------------------------
# Stage 1: run the real pipeline
# ----------------------------------------------------------------------
async def collect_predictions(
    engine: QueryEngine, dataset: list[dict[str, Any]]
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for i, row in enumerate(dataset, start=1):
        question = row["question"]
        logger.info("[%d/%d] %s", i, len(dataset), question)
        result = await engine.answer(question)
        predictions.append(
            Prediction(
                question=question,
                ground_truth=row.get("ground_truth", ""),
                answer=result.answer,
                # Full chunk text, not the API's truncated previews -- scoring
                # against previews would silently deflate context recall.
                contexts=list(result.contexts),
                unanswerable=bool(row.get("unanswerable", False)),
                refused=is_refusal(result.answer),
                latency_ms=result.timings_ms.get("total", 0.0),
                retrieved=len(result.contexts),
            )
        )
    return predictions


# ----------------------------------------------------------------------
# Stage 2: score with RAGAS
# ----------------------------------------------------------------------
def score_with_ragas(
    predictions: list[Prediction],
    settings: Settings,
    judge_timeout: int = 600,
    judge_workers: int = 4,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Run RAGAS over the answerable subset. Returns (averages, per-sample)."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig

    scored = [p for p in predictions if not p.unanswerable]
    if not scored:
        return {}, []

    samples = [
        SingleTurnSample(
            user_input=p.question,
            retrieved_contexts=p.contexts,
            response=p.answer,
            reference=p.ground_truth,
        )
        for p in scored
    ]

    # The judge is deliberately its own model instance at temperature 0. Reusing
    # the app's LLM object would inherit its temperature and make scores jitter
    # between runs, which destroys your ability to see a real regression.
    judge_llm = LangchainLLMWrapper(_build_judge_llm(settings))
    judge_embeddings = LangchainEmbeddingsWrapper(_build_judge_embeddings(settings))

    # RAGAS defaults to 16 concurrent jobs and a 180s per-call timeout. Against
    # a rate-limited hosted provider those 16 requests queue behind each other,
    # so the tail calls sit waiting and trip the timeout -- and a timed-out job
    # becomes NaN, silently shrinking the sample your averages are computed
    # from. Fewer workers means less self-inflicted queueing; the longer timeout
    # absorbs a slow verdict rather than discarding it.
    run_config = RunConfig(
        timeout=judge_timeout,
        max_workers=judge_workers,
        max_retries=3,
    )

    logger.info(
        "Scoring %d question(s) with judge=%s (workers=%d, timeout=%ds)",
        len(samples),
        settings.active_judge_model,
        judge_workers,
        judge_timeout,
    )

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    per_sample = _normalise_scores(result, len(scored))
    averages = _average(per_sample)
    return averages, per_sample


def _build_judge_llm(settings: Settings) -> Any:
    """The grader. Must be independent of the app's own LLM settings.

    Groq is reached through langchain's OpenAI client rather than a dedicated
    integration: Groq exposes an OpenAI-compatible API, so pointing `base_url`
    at it works and avoids adding langchain-groq to an already delicate
    dependency graph (ragas + langchain + openai all have to co-resolve).
    """
    from langchain_openai import ChatOpenAI

    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is required to run LLM-judged metrics.")
        return ChatOpenAI(
            model=settings.active_judge_model,
            base_url=settings.groq_base_url,
            api_key=settings.groq_api_key.get_secret_value(),
            temperature=0.0,
            # Reasoning models need headroom or the judge returns an empty
            # verdict, which RAGAS reports as NaN.
            max_tokens=settings.llm_max_tokens,
        )

    if settings.openai_api_key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is required: RAGAS metrics are LLM-judged. Use "
            "--no-ragas to run the pipeline without scoring."
        )
    return ChatOpenAI(
        model=settings.active_judge_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.0,
    )


def _build_judge_embeddings(settings: Settings) -> Any:
    """Embeddings for the metrics that need them.

    Mirrors the app's embedding provider so the judge measures similarity in
    the same space the retriever searched. Groq has no embeddings endpoint, so
    a Groq deployment always lands on the local model here.
    """
    if settings.embedding_provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)

    from langchain_openai import OpenAIEmbeddings

    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI judge embeddings.")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )


def _normalise_scores(result: Any, expected: int) -> list[dict[str, float]]:
    """Flatten a RAGAS result into per-sample dicts.

    RAGAS's result object has changed shape repeatedly across releases, so this
    tries the modern ``.scores`` list first and falls back to a DataFrame. Worth
    the defensiveness: a version bump should not silently produce empty scores.
    """
    raw = getattr(result, "scores", None)
    if raw:
        rows = [dict(r) for r in raw]
    else:  # pragma: no cover - older ragas
        frame = result.to_pandas()
        keep = [c for c in frame.columns if c in METRIC_NAMES]
        rows = frame[keep].to_dict(orient="records")

    normalised = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            name = _canonical_metric_name(key)
            if name and isinstance(value, (int, float)):
                clean[name] = float(value)
        normalised.append(clean)

    if len(normalised) != expected:
        logger.warning(
            "RAGAS returned %d score rows for %d samples", len(normalised), expected
        )
    return normalised


def _canonical_metric_name(key: str) -> str | None:
    """Map RAGAS's assorted metric keys onto our three stable names."""
    k = key.lower()
    if "precision" in k:
        return "context_precision"
    if "recall" in k:
        return "context_recall"
    if "faithful" in k:
        return "faithfulness"
    return None


def _coverage(per_sample: list[dict[str, float]], scored: int) -> dict[str, str]:
    """Report scored/total per metric so partial results cannot masquerade."""
    out: dict[str, str] = {}
    for name in METRIC_NAMES:
        have = sum(
            1 for row in per_sample if name in row and row[name] == row[name]
        )
        out[name] = f"{have}/{scored}"
    return out


def _warn_on_partial_coverage(coverage: dict[str, str]) -> None:
    for name, ratio in coverage.items():
        have, total = (int(x) for x in ratio.split("/"))
        if total and have < total:
            logger.warning(
                "%s was scored on only %d of %d questions -- the average is "
                "computed from that subset. Judge calls failed or timed out; "
                "raise --judge-timeout, lower --judge-workers, or pick a "
                "faster --judge-model.",
                name,
                have,
                total,
            )


def _average(per_sample: list[dict[str, float]]) -> dict[str, float]:
    """Mean per metric, ignoring NaN.

    RAGAS emits NaN when a judge call fails or a sample has nothing to score.
    Treating NaN as zero would punish the pipeline for the judge's failure, so
    those samples are excluded from that metric's mean instead.
    """
    averages: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = [
            row[name]
            for row in per_sample
            if name in row and row[name] == row[name]  # NaN != NaN
        ]
        if values:
            averages[name] = round(sum(values) / len(values), 4)
    return averages


# ----------------------------------------------------------------------
# Stage 3: refusal accuracy
# ----------------------------------------------------------------------
def refusal_accuracy(predictions: list[Prediction]) -> float | None:
    """Fraction of unanswerable questions the pipeline correctly refused.

    Tracked separately from RAGAS because it measures the opposite failure mode.
    RAGAS asks "was the answer grounded?"; this asks "did it correctly decline
    to answer?" A system that hallucinates confidently and one that refuses
    everything can post identical faithfulness scores -- this number
    distinguishes them.
    """
    unanswerable = [p for p in predictions if p.unanswerable]
    if not unanswerable:
        return None
    correct = sum(1 for p in unanswerable if p.refused)
    return round(correct / len(unanswerable), 4)


# ----------------------------------------------------------------------
# Stage 4: persist
# ----------------------------------------------------------------------
def write_results(run: EvaluationRun, output: Path) -> None:
    """Append this run to the history file.

    Append, not overwrite: a single score is nearly useless, while a series
    tells you whether last week's chunking change helped. The file is a JSON
    array of runs, newest last.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    if output.exists():
        try:
            existing = json.loads(output.read_text())
            history = existing if isinstance(existing, list) else [existing]
        except json.JSONDecodeError:
            # Never lose a run to a corrupt history file.
            backup = output.with_suffix(".corrupt.json")
            output.replace(backup)
            logger.warning("Existing results were unreadable; moved to %s", backup)

    history.append(run.__dict__)
    output.write_text(json.dumps(history, indent=2, default=str))
    logger.info("Wrote results to %s (%d run(s) in history)", output, len(history))


# ----------------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------------
def parse_thresholds(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    thresholds = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"--fail-under expects metric=value pairs, got {part!r}"
            )
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in METRIC_NAMES and name != "refusal_accuracy":
            raise argparse.ArgumentTypeError(
                f"unknown metric {name!r}; choose from "
                f"{', '.join(METRIC_NAMES)}, refusal_accuracy"
            )
        thresholds[name] = float(value)
    return thresholds


def check_thresholds(run: EvaluationRun) -> bool:
    """Gate CI on quality. Returns True if every threshold is met."""
    ok = True
    for name, minimum in run.thresholds.items():
        actual = (
            run.refusal_accuracy
            if name == "refusal_accuracy"
            else run.averages.get(name)
        )
        if actual is None:
            logger.error("Threshold set for %s but no score was produced", name)
            ok = False
        elif actual < minimum:
            logger.error("%s = %.4f is below threshold %.4f", name, actual, minimum)
            ok = False
        else:
            logger.info("%s = %.4f meets threshold %.4f", name, actual, minimum)
    return ok


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def load_dataset(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_DATASET
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array of objects")
    for row in rows:
        if "question" not in row:
            raise ValueError(f"dataset row missing 'question': {row}")
    return rows


def build_eval_settings() -> Settings:
    """App settings with the query cache disabled.

    Non-negotiable for evaluation: with the cache on, a repeated question would
    return a stored answer and you would be scoring the cache rather than the
    pipeline. It also makes latency numbers meaningless.
    """
    base = get_settings()
    return base.model_copy(update={"query_cache_enabled": False})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument("--dataset", type=Path, help="JSON array of test cases.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Only evaluate the first N cases.")
    parser.add_argument("--collection", help="Override the Chroma collection.")
    parser.add_argument(
        "--fail-under",
        help="Exit non-zero if a metric is below threshold, e.g. "
        "'context_recall=0.7,faithfulness=0.8'. Use this to gate CI.",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Run the pipeline and record latency/refusals, but skip LLM-judged "
        "scoring. Useful for a smoke test that costs nothing.",
    )
    parser.add_argument(
        "--judge-model",
        help="Model used to GRADE answers, independent of the model that writes "
        "them. Prefer a fast non-reasoning model here (e.g. qwen/qwen3.8-27b): "
        "every reasoning verdict costs a hidden reasoning pass, which is what "
        "trips RAGAS's per-call timeout and turns scores into NaN.",
    )
    parser.add_argument(
        "--judge-timeout", type=int, default=600,
        help="Per-judge-call timeout in seconds (default 600).",
    )
    parser.add_argument(
        "--judge-workers", type=int, default=4,
        help="Concurrent judge calls (default 4). Lower this on a rate-limited "
        "provider -- excess concurrency just queues and then times out.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    settings = build_eval_settings()
    configure_logging(level="DEBUG" if args.verbose else "INFO", fmt=settings.log_format)
    logging.getLogger("ragas").setLevel(logging.WARNING)

    if args.collection:
        settings = settings.model_copy(update={"chroma_collection": args.collection})
    if args.judge_model:
        settings = settings.model_copy(update={"judge_model": args.judge_model})

    thresholds = parse_thresholds(args.fail_under)
    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    started = time.perf_counter()
    engine = QueryEngine(settings)

    async def _run() -> list[Prediction]:
        await engine.warmup()
        try:
            return await collect_predictions(engine, dataset)
        finally:
            await engine.aclose()

    predictions = asyncio.run(_run())

    averages: dict[str, float] = {}
    per_sample: list[dict[str, float]] = []
    if not args.no_ragas:
        try:
            averages, per_sample = score_with_ragas(
                predictions,
                settings,
                judge_timeout=args.judge_timeout,
                judge_workers=args.judge_workers,
            )
        except ImportError as exc:
            logger.error(
                "RAGAS is not installed: %s\n"
                "Install it TOGETHER with the app requirements so pip can resolve "
                "a compatible openai version:\n"
                "  pip install -r requirements.txt -r requirements-eval.txt",
                exc,
            )
            return 2
        except Exception as exc:
            logger.exception("RAGAS evaluation failed: %s", exc)
            return 2

    scored = [p for p in predictions if not p.unanswerable]
    per_question = []
    for i, pred in enumerate(scored):
        row = {
            "question": pred.question,
            "answer": pred.answer,
            "retrieved_chunks": pred.retrieved,
            "latency_ms": pred.latency_ms,
        }
        if i < len(per_sample):
            row.update(per_sample[i])
        per_question.append(row)
    for pred in (p for p in predictions if p.unanswerable):
        per_question.append(
            {
                "question": pred.question,
                "answer": pred.answer,
                "unanswerable": True,
                "correctly_refused": pred.refused,
                "latency_ms": pred.latency_ms,
            }
        )

    run = EvaluationRun(
        timestamp=datetime.now(UTC).isoformat(),
        duration_seconds=round(time.perf_counter() - started, 2),
        dataset_size=len(dataset),
        scored_questions=len(scored),
        refusal_questions=len(dataset) - len(scored),
        averages=averages,
        coverage=_coverage(per_sample, len(scored)) if per_sample else {},
        refusal_accuracy=refusal_accuracy(predictions),
        per_question=per_question,
        config=config_snapshot(settings),
        thresholds=thresholds,
    )
    _warn_on_partial_coverage(run.coverage)
    run.passed = check_thresholds(run)
    write_results(run, args.output)

    print("\n" + "=" * 62)
    print(f"RAG EVALUATION  {run.timestamp}")
    print(f"  judge: {settings.active_judge_model}")
    print("-" * 62)
    for name in METRIC_NAMES:
        value = run.averages.get(name)
        cov = run.coverage.get(name, "")
        flag = ""
        if cov:
            have, total = (int(x) for x in cov.split("/"))
            flag = f"   [{cov} scored]" + ("  <-- PARTIAL" if have < total else "")
        print(f"  {name:<20} {value if value is not None else 'n/a'}{flag}")
    if run.refusal_accuracy is not None:
        print(f"  {'refusal_accuracy':<20} {run.refusal_accuracy}"
              f"  ({run.refusal_questions} unanswerable, scored separately)")
    latencies = [p.latency_ms for p in predictions]
    if latencies:
        print(f"  {'mean_latency_ms':<20} {round(sum(latencies) / len(latencies), 1)}")
    print(f"\n  {len(dataset)} question(s) in {run.duration_seconds}s")
    if run.thresholds:
        print(f"  gate: {'PASS' if run.passed else 'FAIL'}")
    print("=" * 62)

    return 0 if run.passed else 1


if __name__ == "__main__":
    sys.exit(main())
