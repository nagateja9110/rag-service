"""Centralised, validated application configuration.

Everything the app needs to know about its environment lives here and nowhere
else. No module may call ``os.getenv`` directly -- if it needs a knob, it gets
one on ``Settings``. That gives us three things a scattered-getenv codebase
never has:

1. Fail-fast: a missing or malformed variable blows up at process start with a
   readable pydantic error, not three minutes into an ingestion run.
2. A single, greppable inventory of every input the system takes.
3. Type coercion and cross-field validation for free.

PRECEDENCE
----------
Highest wins::

    constructor args  >  environment  >  .env  >  config.yaml  >  field defaults

That ordering is the whole point of having both a YAML file and env vars:
``config.yaml`` is the committed, reviewable description of how the system is
wired (which embedding provider, which reranker, chunk sizes), while the
environment carries secrets and per-deployment overrides that must never be
committed. A staging box changes one env var; it does not fork the config file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

EmbeddingProvider = Literal["openai", "sentence-transformers"]
LLMProvider = Literal["openai", "groq"]
RerankerBackend = Literal["cross-encoder", "cohere", "none"]
CacheBackendName = Literal["memory", "redis"]

DEFAULT_CONFIG_FILE = "config.yaml"

# Providers that must never appear in a model-identifier field.
_PROVIDER_NAMES = {"openai", "sentence-transformers", "huggingface", "hf"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Default profile. The actual path is resolved per-instantiation in
        # settings_customise_sources() so CONFIG_FILE is honoured whenever
        # Settings is constructed, not just at import time.
        yaml_file=DEFAULT_CONFIG_FILE,
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML profile BELOW the environment in priority.

        CONFIG_FILE is read HERE rather than in ``model_config`` because
        ``model_config`` is evaluated once, when the class object is created at
        import time. Reading the environment there would mean CONFIG_FILE only
        took effect if it was set before the first import of this module --
        true inside the container, quietly false in tests, notebooks and
        anything that sets it programmatically.

        A missing file is not an error: the source yields an empty mapping, so
        the app runs on env vars and defaults exactly as it would without it.
        """
        yaml_path = os.environ.get("CONFIG_FILE", DEFAULT_CONFIG_FILE)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),
            file_secret_settings,
        )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    app_env: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    # THE provider switch. Changing this changes vector dimensionality (OpenAI
    # text-embedding-3-small = 1536, all-MiniLM-L6-v2 = 384), so an existing
    # collection built with the other provider is unusable -- see
    # `embedding_signature` and the guard in app/db/vector_store.py.
    embedding_provider: EmbeddingProvider = Field(
        default="openai",
        validation_alias=AliasChoices("embedding_provider", "embedding_backend"),
    )
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Only meaningful for OpenAI (Matryoshka truncation). Local models emit
    # whatever dimensionality they were trained with.
    embedding_dimension: int = 1536

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    # openai -> gpt-4o-mini et al.
    # groq   -> OpenAI-compatible endpoint serving open models on LPUs. Much
    #           cheaper and faster, but NOTE: Groq has no embeddings endpoint,
    #           so embedding_provider must be 'sentence-transformers' (or you
    #           must keep an OpenAI key just for embeddings).
    llm_provider: LLMProvider = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    # Reasoning models (gpt-oss-*) spend part of their budget on hidden
    # reasoning tokens before emitting an answer. Too small a cap and the
    # answer comes back EMPTY rather than truncated, which looks like a bug in
    # the pipeline rather than a budget problem.
    llm_max_tokens: int = 1024

    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # The model that GRADES answers during evaluation, kept separate from the
    # model that WRITES them. The judge does not need to be eloquent, it needs
    # to be fast and consistent -- and a reasoning model makes a poor judge in
    # practice because every verdict costs a hidden reasoning pass, which is
    # what pushes RAGAS past its per-call timeout. Empty = reuse the app's LLM.
    judge_model: str = ""
    openai_max_retries: int = 3
    openai_timeout_seconds: float = 60.0

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    data_dir: Path = Path("./data")
    embed_batch_size: int = 64
    store_batch_size: int = 256

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_persist_dir: Path = Path("./chroma_db")
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "documents"

    # ------------------------------------------------------------------
    # Retrieval (online path)
    # ------------------------------------------------------------------
    vector_top_k: int = 10
    bm25_top_k: int = 10
    rrf_k: int = 60
    fusion_top_n: int = 20
    rerank_top_n: int = 5
    min_rerank_score: float | None = None

    bm25_refresh_seconds: int = 60
    bm25_max_nodes: int = 50_000

    # ------------------------------------------------------------------
    # Re-ranking
    # ------------------------------------------------------------------
    # cross-encoder -> in-process sentence-transformers (needs requirements-local.txt)
    # cohere        -> hosted rerank API, no torch in the image
    # none          -> passthrough, keeps RRF order
    reranker: RerankerBackend = Field(
        default="cross-encoder",
        validation_alias=AliasChoices("reranker", "reranker_backend"),
    )
    local_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cohere_api_key: SecretStr | None = None
    cohere_rerank_model: str = "rerank-english-v3.0"

    # ------------------------------------------------------------------
    # Query cache
    # ------------------------------------------------------------------
    cache_backend: CacheBackendName = "memory"
    query_cache_enabled: bool = True
    query_cache_max_size: int = 512
    query_cache_ttl_seconds: float = 300.0  # 5 minutes
    redis_url: str = "redis://redis:6379/0"
    redis_prefix: str = "rag:q:"
    redis_timeout_seconds: float = 2.0

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    cors_allow_origins: str = "http://localhost:3000,http://localhost:5173"
    cors_allow_credentials: bool = False

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------
    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def active_llm_model(self) -> str:
        """The model identifier for whichever LLM provider is selected."""
        if self.llm_provider == "groq":
            return self.groq_model
        return self.llm_model

    @property
    def active_judge_model(self) -> str:
        """Model used for LLM-judged metrics; falls back to the app's LLM."""
        return self.judge_model or self.active_llm_model

    @property
    def needs_openai_key(self) -> bool:
        """OpenAI is only required for the parts actually pointed at it."""
        return self.llm_provider == "openai" or self.embedding_provider == "openai"

    @property
    def active_embedding_model(self) -> str:
        """The model identifier for whichever provider is selected."""
        if self.embedding_provider == "sentence-transformers":
            return self.local_embedding_model
        return self.embedding_model

    @property
    def embedding_signature(self) -> str:
        """Identity of the vector space a collection was built in.

        Stamped into the Chroma collection at ingestion and checked before
        querying. Provider + model is sufficient: the dimensionality and the
        semantics both follow from it, and comparing two strings is cheaper and
        clearer than probing the model for its output size.
        """
        return f"{self.embedding_provider}:{self.active_embedding_model}"

    @property
    def is_production(self) -> bool:
        return self.app_env in ("staging", "prod")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @field_validator("reranker", mode="before")
    @classmethod
    def _normalise_reranker(cls, v: object) -> object:
        # "local" was the name used before the config file existed; accept it
        # so an older .env keeps working instead of failing at startup.
        if isinstance(v, str) and v.strip().lower() in ("local", "cross_encoder"):
            return "cross-encoder"
        return v

    @field_validator("embedding_model", "local_embedding_model", "llm_model")
    @classmethod
    def _model_field_is_not_a_provider(cls, v: str) -> str:
        if v.strip().lower() in _PROVIDER_NAMES:
            raise ValueError(
                f"{v!r} is a provider name, not a model identifier. Set "
                f"EMBEDDING_PROVIDER (or `embedding_provider:` in config.yaml) to "
                f"choose the provider, and leave this field as the model name "
                f"(e.g. 'text-embedding-3-small')."
            )
        return v

    @field_validator(
        "chunk_size", "chunk_overlap", "embed_batch_size", "store_batch_size"
    )
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @model_validator(mode="after")
    def _required_keys_present(self) -> Settings:
        """Require exactly the credentials the chosen providers actually use.

        Previously this demanded an OpenAI key unconditionally. That was wrong
        the moment a second LLM provider existed: a fully-Groq + local-embedding
        deployment needs no OpenAI account at all, and forcing a dummy key just
        to boot teaches people to put fake secrets in .env.
        """
        if self.needs_openai_key and self.openai_api_key is None:
            used_for = []
            if self.llm_provider == "openai":
                used_for.append("generation")
            if self.embedding_provider == "openai":
                used_for.append("embeddings")
            raise ValueError(
                f"OPENAI_API_KEY is required because it is used for "
                f"{' and '.join(used_for)}."
            )
        if self.llm_provider == "groq" and self.groq_api_key is None:
            raise ValueError("llm_provider='groq' requires GROQ_API_KEY to be set.")
        return self

    @model_validator(mode="after")
    def _groq_has_no_embeddings(self) -> Settings:
        """Groq serves LLM inference only -- there is no embeddings endpoint.

        Caught here with an actionable message, because the alternative is a
        confusing 404 from the embeddings call midway through an ingestion run.
        """
        if self.llm_provider == "groq" and self.embedding_provider == "openai" \
                and self.openai_api_key is None:
            raise ValueError(
                "llm_provider='groq' cannot supply embeddings (Groq has no "
                "embeddings endpoint). Either set embedding_provider="
                "'sentence-transformers' for local embeddings, or provide "
                "OPENAI_API_KEY to keep using OpenAI embeddings."
            )
        return self

    @model_validator(mode="after")
    def _cohere_backend_needs_a_key(self) -> Settings:
        if self.reranker == "cohere" and self.cohere_api_key is None:
            raise ValueError("reranker='cohere' requires COHERE_API_KEY to be set.")
        return self

    @model_validator(mode="after")
    def _cors_wildcard_conflicts_with_credentials(self) -> Settings:
        # The CORS spec forbids this combination and browsers reject it at
        # runtime. Failing at startup beats debugging it from the console.
        if self.cors_allow_credentials and "*" in self.cors_origins:
            raise ValueError(
                "CORS_ALLOW_CREDENTIALS=true cannot be combined with a '*' origin; "
                "list the origins explicitly."
            )
        return self

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_chunk(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise the splitter cannot "
                f"make forward progress."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton.

    Cached so that FastAPI dependencies, the CLI and background workers all see
    the same object, and so that .env/config.yaml parsing happens exactly once.
    """
    return Settings()
