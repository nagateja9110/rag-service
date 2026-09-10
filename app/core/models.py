"""Model factories shared by both pipelines.

These live in ``core`` rather than in either pipeline for a specific reason: the
embedding model is the one thing ingestion and query MUST agree on exactly.
Embed the corpus with one provider and query it with another and you do not get
an error -- you get silently terrible retrieval, because the vectors are in
incomparable spaces and cosine similarity still returns a number.

Defining it once, from one Settings object, makes that class of bug
unrepresentable. It also keeps the ingestion/query import boundary intact: the
API needs an embedder, and without this module the only place to get one would
have been ``app.ingestion.pipeline``.
"""

from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelUnavailableError(RuntimeError):
    """A configured model backend cannot be constructed."""


def build_embed_model(settings: Settings) -> BaseEmbedding:
    """Construct the embedder for the configured provider.

    Return type is the ``BaseEmbedding`` interface, not a concrete class, so
    every caller is structurally prevented from depending on provider-specific
    behaviour. Swapping providers is a config change, not a code change.
    """
    if settings.embedding_provider == "sentence-transformers":
        return _build_local_embed_model(settings)
    return _build_openai_embed_model(settings)


def _build_openai_embed_model(settings: Settings) -> BaseEmbedding:
    if settings.openai_api_key is None:  # pragma: no cover - blocked by validation
        raise ModelUnavailableError("OPENAI_API_KEY is required for OpenAI embeddings.")
    return OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
        # Matryoshka truncation -- OpenAI-only. Pinning it explicitly (rather
        # than relying on the model default) gives the Qdrant migration an
        # unambiguous vector size to declare up front.
        dimensions=settings.embedding_dimension,
        embed_batch_size=settings.embed_batch_size,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout_seconds,
    )


def _build_local_embed_model(settings: Settings) -> BaseEmbedding:
    """Self-hosted embeddings: no API calls, no per-token cost, no rate limits.

    The trade is latency and quality -- all-MiniLM-L6-v2 is 384-dimensional and
    measurably weaker than text-embedding-3-small on retrieval benchmarks, but
    it runs offline and costs nothing, which is what you want for local
    development and for CI that must not depend on a third party being up.

    Note there is no ``dimensions`` parameter: local models emit whatever
    dimensionality they were trained with. Passing OpenAI's would be silently
    ignored, which is exactly why the two paths are separate functions.
    """
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise ModelUnavailableError(
            "embedding_provider='sentence-transformers' requires the local model "
            "extras: pip install -r requirements.txt -r requirements-local.txt"
        ) from exc

    logger.info("Loading local embedding model %s", settings.local_embedding_model)
    return HuggingFaceEmbedding(
        model_name=settings.local_embedding_model,
        embed_batch_size=settings.embed_batch_size,
    )


def build_llm(settings: Settings) -> LLM:
    """Query path only -- ingestion never calls a generative model.

    Returns the ``LLM`` interface rather than a concrete class so the response
    synthesiser cannot grow a dependency on one provider's quirks.
    """
    if settings.llm_provider == "groq":
        return _build_groq_llm(settings)
    return _build_openai_llm(settings)


def _build_openai_llm(settings: Settings) -> LLM:
    if settings.openai_api_key is None:  # pragma: no cover - blocked by validation
        raise ModelUnavailableError("OPENAI_API_KEY is required for generation.")
    return OpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout_seconds,
    )


def _build_groq_llm(settings: Settings) -> LLM:
    """Groq: open models on LPU hardware behind an OpenAI-compatible API.

    Cheap and very fast, but it serves GENERATION ONLY -- there is no
    embeddings endpoint, so pair it with embedding_provider='sentence-transformers'
    (Settings enforces this).

    max_tokens matters more here than with gpt-4o-mini: the gpt-oss-* models
    emit hidden reasoning tokens before the answer, and a tight cap yields an
    EMPTY response rather than a truncated one.
    """
    try:
        from llama_index.llms.groq import Groq
    except ImportError as exc:
        raise ModelUnavailableError(
            "llm_provider='groq' requires: pip install llama-index-llms-groq"
        ) from exc

    if settings.groq_api_key is None:  # pragma: no cover - blocked by validation
        raise ModelUnavailableError("GROQ_API_KEY is required for generation.")

    logger.info("Using Groq LLM %s", settings.groq_model)
    return Groq(
        model=settings.groq_model,
        api_key=settings.groq_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout_seconds,
    )
