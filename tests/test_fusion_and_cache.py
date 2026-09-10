"""Pure units: RRF fusion, the TTL cache, and cache-key construction."""

from __future__ import annotations

import time

from llama_index.core.schema import NodeWithScore, TextNode

from app.core.cache import TTLCache, make_cache_key
from app.query.retriever import reciprocal_rank_fusion


def _n(node_id: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=node_id), score=1.0)


class TestReciprocalRankFusion:
    def test_agreement_beats_confidence(self) -> None:
        """A doc both retrievers rank 3rd must outrank one only seen by one.

        This is the whole reason RRF exists rather than a weighted score sum.
        """
        dense = [_n("A"), _n("X"), _n("B")]
        sparse = [_n("Y"), _n("Z"), _n("B")]
        fused, overlap = reciprocal_rank_fusion([dense, sparse], k=60, top_n=10)
        assert fused[0].node.node_id == "B"
        assert overlap == 1

    def test_scores_follow_the_formula(self) -> None:
        dense, sparse = [_n("A")], [_n("A")]
        fused, _ = reciprocal_rank_fusion([dense, sparse], k=60)
        assert abs(fused[0].score - 2 * (1 / 61)) < 1e-9

    def test_deduplicates_by_node_id(self) -> None:
        fused, _ = reciprocal_rank_fusion([[_n("A"), _n("B")], [_n("B"), _n("C")]])
        assert len(fused) == 3

    def test_descending_order(self) -> None:
        fused, _ = reciprocal_rank_fusion([[_n("A"), _n("X"), _n("B")], [_n("B")]])
        scores = [f.score or 0 for f in fused]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_truncates(self) -> None:
        fused, _ = reciprocal_rank_fusion([[_n("A"), _n("B"), _n("C")]], top_n=2)
        assert len(fused) == 2

    def test_empty_input_is_safe(self) -> None:
        assert reciprocal_rank_fusion([[], []])[0] == []

    def test_survives_one_retriever_returning_nothing(self) -> None:
        """BM25 can fail or be unavailable; fusion must degrade to dense-only."""
        dense = [_n("A"), _n("X"), _n("B")]
        fused, _ = reciprocal_rank_fusion([dense, []])
        assert [f.node.node_id for f in fused] == ["A", "X", "B"]


class TestTTLCache:
    def test_get_and_set(self) -> None:
        c: TTLCache[dict] = TTLCache(max_size=4, ttl_seconds=60)
        c.set("k", {"v": 1})
        assert c.get("k") == {"v": 1}
        assert c.get("absent") is None

    def test_evicts_least_recently_used(self) -> None:
        c: TTLCache[str] = TTLCache(max_size=2, ttl_seconds=60)
        c.set("a", "1")
        c.set("b", "2")
        c.get("a")           # 'a' is now the most recent, so 'b' should go
        c.set("c", "3")
        assert c.get("a") == "1"
        assert c.get("b") is None

    def test_entries_expire(self) -> None:
        c: TTLCache[str] = TTLCache(max_size=4, ttl_seconds=0.3)
        c.set("k", "v")
        time.sleep(0.4)
        assert c.get("k") is None
        assert c.stats().expirations >= 1

    def test_clear_reports_count(self) -> None:
        c: TTLCache[str] = TTLCache(max_size=4, ttl_seconds=60)
        c.set("a", "1")
        c.set("b", "2")
        assert c.clear() == 2
        assert c.stats().size == 0


class TestCacheKey:
    def test_normalises_case_and_whitespace(self) -> None:
        assert make_cache_key("What is RRF?", top_k=5) == make_cache_key(
            "what is  rrf?", top_k=5
        )

    def test_parameters_change_the_key(self) -> None:
        """Serving a top_k=5 answer to a top_k=20 request is a correctness bug."""
        assert make_cache_key("q", top_k=5) != make_cache_key("q", top_k=20)

    def test_different_questions_differ(self) -> None:
        assert make_cache_key("a", top_k=5) != make_cache_key("b", top_k=5)
