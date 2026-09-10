"""Request/response contracts.

Kept separate from routes.py because these are the public API surface -- the
thing clients generate types from and the thing you cannot change without a
version bump. Route handlers are implementation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language question to answer from the indexed corpus.",
        examples=["What makes the ingestion pipeline idempotent?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Chunks passed to the LLM after re-ranking. Defaults to RERANK_TOP_N.",
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # min_length alone would happily accept "   ".
        cleaned = " ".join(v.split())
        if len(cleaned) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return cleaned


class SourceModel(BaseModel):
    chunk_id: str
    file_name: str
    page_number: int
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    # Always returned, even when empty. A client rendering citations should not
    # have to branch on key presence.
    sources: list[SourceModel] = Field(default_factory=list)
    # False when the service returned the "not enough information" guardrail.
    # Lets a UI render that case differently instead of string-matching.
    grounded: bool = True
    cached: bool = False
    reranker: str = "none"
    reranker_degraded: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    environment: str


class ReadyResponse(BaseModel):
    status: str
    vector_store: str
    collection: str
    vectors: int
    reranker: str


class CacheStatsResponse(BaseModel):
    cache: dict[str, Any]


class InvalidateResponse(BaseModel):
    cleared_entries: int
    detail: str
