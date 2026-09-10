"""Configuration layering, provider selection and the guards around them."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings


def build(**kw: Any) -> Settings:
    """Construct Settings from loosely-typed kwargs.

    Deliberately `**Any`: these tests exercise behaviour that lives OUTSIDE the
    static signature on purpose -- pydantic-settings' `_env_file` runtime
    kwarg, str -> SecretStr coercion, validation aliases like
    `reranker_backend`, and the legacy `reranker="local"` value that a
    before-validator maps onto a Literal it is not itself a member of. Routing
    through one `Any` seam keeps the type checker meaningful everywhere else
    instead of sprinkling per-line ignores.
    """
    return Settings(**kw)


def _settings(tmp_path: Path, yaml: str, **kw: Any) -> Settings:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(textwrap.dedent(yaml))
    old = os.environ.get("CONFIG_FILE")
    os.environ["CONFIG_FILE"] = str(cfg)
    try:
        return build(_env_file=None, **kw)
    finally:
        if old is not None:
            os.environ["CONFIG_FILE"] = old


class TestPrecedence:
    def test_yaml_is_applied(self, tmp_path: Path) -> None:
        s = _settings(tmp_path, "chunk_size: 512\nvector_top_k: 7\n")
        assert s.chunk_size == 512
        assert s.vector_top_k == 7

    def test_constructor_beats_yaml(self, tmp_path: Path) -> None:
        assert _settings(tmp_path, "chunk_size: 512\n", chunk_size=900).chunk_size == 900

    def test_missing_yaml_is_not_an_error(self) -> None:
        os.environ["CONFIG_FILE"] = "definitely-not-here.yaml"
        assert build(_env_file=None).chunk_size == 1024


class TestShippedProfiles:
    @pytest.mark.parametrize("name", ["config.yaml", "config.local.yaml"])
    def test_profile_is_valid(self, name: str) -> None:
        """A profile that does not parse breaks every deployment using it."""
        old = os.environ.get("CONFIG_FILE")
        os.environ["CONFIG_FILE"] = name
        try:
            s = build(_env_file=None, groq_api_key="gsk-test", cohere_api_key="ck-test")
            assert s.embedding_signature
        finally:
            if old is not None:
                os.environ["CONFIG_FILE"] = old


class TestGuards:
    def test_reranker_legacy_alias(self) -> None:
        assert build(_env_file=None, reranker="local").reranker == "cross-encoder"

    def test_reranker_backend_alias(self) -> None:
        assert build(_env_file=None, reranker_backend="none").reranker == "none"

    @pytest.mark.parametrize("bad", ["openai", "sentence-transformers"])
    def test_provider_name_rejected_in_model_field(self, bad: str) -> None:
        with pytest.raises(ValueError, match="provider name"):
            build(_env_file=None, embedding_model=bad)

    def test_cohere_requires_a_key(self) -> None:
        with pytest.raises(ValueError, match="COHERE_API_KEY"):
            build(_env_file=None, reranker="cohere", cohere_api_key=None)

    def test_groq_requires_a_key(self) -> None:
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            build(_env_file=None, llm_provider="groq", groq_api_key=None)

    def test_wildcard_cors_with_credentials_rejected(self) -> None:
        with pytest.raises(ValueError, match="credentials"):
            build(_env_file=None, cors_allow_credentials=True, cors_allow_origins="*")

    def test_overlap_must_be_smaller_than_chunk(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            build(_env_file=None, chunk_size=100, chunk_overlap=200)

    def test_groq_without_openai_needs_no_openai_key(self) -> None:
        """A Groq + local-embeddings deployment needs no OpenAI account at all."""
        s = build(
            _env_file=None,
            llm_provider="groq",
            groq_api_key="gsk-test",
            embedding_provider="sentence-transformers",
            openai_api_key=None,
        )
        assert s.needs_openai_key is False


class TestEmbeddingSignature:
    def test_openai(self) -> None:
        assert (
            build(_env_file=None).embedding_signature
            == "openai:text-embedding-3-small"
        )

    def test_local(self) -> None:
        s = build(_env_file=None, embedding_provider="sentence-transformers")
        assert s.embedding_signature.startswith("sentence-transformers:")

    def test_judge_falls_back_to_app_llm(self) -> None:
        s = build(_env_file=None, llm_provider="groq", groq_api_key="gsk-test")
        assert s.active_judge_model == s.groq_model
