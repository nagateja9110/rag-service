"""Shared fixtures.

Test doubles rather than live APIs: the suite must run offline, cost nothing,
and give the same answer every time. The embedding stub is a hashed
bag-of-words, which keeps retrieval *semantically meaningful* (shared
vocabulary raises cosine similarity) so retrieval tests are real tests rather
than assertions about random numbers.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from collections.abc import Iterator
from typing import Any

import pytest

# Must be set before app.core.config is imported anywhere.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.update(
    {
        # A profile that does not exist, so the shipped production config.yaml
        # cannot leak into tests and change their meaning.
        "CONFIG_FILE": "config.test-nonexistent.yaml",
        "CHROMA_MODE": "persistent",
        "CHROMA_PERSIST_DIR": "./.pytest_chroma",
        "DATA_DIR": "./data",
        "RERANKER": "none",
        "CACHE_BACKEND": "memory",
        "APP_ENV": "local",
        "LLM_PROVIDER": "openai",
    }
)

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import (
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

REFUSAL = "I don't have enough information to answer this."
DIM = 1536

# CustomLLM is a pydantic model, so mutable call-tracking state lives outside it.
LLM_CALLS: dict[str, Any] = {"n": 0, "last_prompt": ""}


class BagOfWordsEmbedding(BaseEmbedding):
    """Deterministic hashed bag-of-words in the same dimensionality as OpenAI's."""

    @classmethod
    def class_name(cls) -> str:
        return "bow"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    def get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kw: Any
    ) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class StubLLM(CustomLLM):
    """Answers from context, and refuses on questions the corpus cannot cover."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=128000, num_output=512, model_name="stub")

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kw: Any) -> CompletionResponse:
        LLM_CALLS["n"] += 1
        LLM_CALLS["last_prompt"] = prompt
        question = prompt.rsplit("QUESTION:", 1)[-1].lower()
        if "revenue" in question or "home address" in question:
            return CompletionResponse(text=REFUSAL)
        return CompletionResponse(text="Content addressing via a hash of the chunk text.")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kw: Any) -> Any:
        yield CompletionResponse(text="x", delta="x")


@pytest.fixture(autouse=True)
def _stub_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the factories where they are USED, not only where defined.

    `from x import y` binds a new name in the importing module, so patching
    app.core.models alone would leave app.query.engine holding the original.
    """
    from app.core import models as core_models
    from app.ingestion import pipeline
    from app.query import engine as engine_mod

    monkeypatch.setattr(core_models, "build_embed_model", lambda s: BagOfWordsEmbedding())
    monkeypatch.setattr(core_models, "build_llm", lambda s: StubLLM())
    monkeypatch.setattr(engine_mod, "build_embed_model", lambda s: BagOfWordsEmbedding())
    monkeypatch.setattr(engine_mod, "build_llm", lambda s: StubLLM())
    monkeypatch.setattr(pipeline, "build_embed_model", lambda s: BagOfWordsEmbedding())


@pytest.fixture
def llm_calls() -> dict[str, Any]:
    LLM_CALLS["n"] = 0
    LLM_CALLS["last_prompt"] = ""
    return LLM_CALLS


@pytest.fixture
def corpus() -> Iterator[Any]:
    """A freshly ingested collection, torn down afterwards."""
    from app.db.vector_store import get_vector_store
    from app.ingestion import pipeline

    name = "pytest_corpus"
    store = get_vector_store(name)
    store.reset_collection()
    pipeline.run_ingestion(collection_name=name)
    yield store
    # Teardown must not fail the test that just passed.
    with contextlib.suppress(Exception):
        store.client.delete_collection(name)
