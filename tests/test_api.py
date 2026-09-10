"""HTTP contract: routes, validation, CORS, metrics, and the module boundary."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


@pytest.fixture
def client(corpus: Any) -> Iterator[TestClient]:
    get_settings.cache_clear()
    import os

    os.environ["CHROMA_COLLECTION"] = "pytest_corpus"
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as c:
        yield c


class TestOpsEndpoints:
    def test_health_is_dependency_free(self, client: TestClient) -> None:
        """Liveness must not fail when Chroma blips, or K8s kills healthy pods."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ready_reports_corpus_size(self, client: TestClient) -> None:
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["vectors"] > 0

    def test_cache_stats(self, client: TestClient) -> None:
        assert client.get("/cache").status_code == 200

    def test_invalidate(self, client: TestClient) -> None:
        assert client.post("/cache/invalidate").status_code == 200


class TestQueryEndpoint:
    def test_answers_with_sources(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "What makes ingestion idempotent?"})
        assert r.status_code == 200
        body = r.json()
        assert body["sources"]
        assert body["grounded"] is True

    def test_full_contexts_are_not_exposed_over_http(self, client: TestClient) -> None:
        """Previews are cited; whole chunks stay server-side."""
        body = client.post("/query", json={"question": "What makes ingestion idempotent?"}).json()
        assert "contexts" not in body

    def test_unanswerable_is_200_not_an_error(self, client: TestClient) -> None:
        """'I don't know' is a correct answer, not a failure."""
        r = client.post("/query", json={"question": "What were the Q3 revenue figures?"})
        assert r.status_code == 200
        assert r.json()["grounded"] is False

    @pytest.mark.parametrize(
        "payload", [{"question": "   "}, {"question": "hi"}, {}, {"question": "ok q", "top_k": 99}]
    )
    def test_invalid_requests_rejected(self, client: TestClient, payload: dict) -> None:
        assert client.post("/query", json=payload).status_code == 422


class TestCORS:
    def test_listed_origin_allowed(self, client: TestClient) -> None:
        r = client.options(
            "/query",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_unlisted_origin_gets_no_header(self, client: TestClient) -> None:
        r = client.options(
            "/query",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert r.headers.get("access-control-allow-origin") is None


class TestMetrics:
    def test_exposes_the_documented_series(self, client: TestClient) -> None:
        client.post("/query", json={"question": "What makes ingestion idempotent?"})
        body = client.get("/metrics").text
        assert "rag_query_latency_seconds_bucket" in body
        # prometheus_client appends _total to every counter.
        assert "rag_retrieval_hit_count_total" in body

    def test_labels_stay_low_cardinality(self, client: TestClient) -> None:
        """Labelling by question text would take down Prometheus, not the app."""
        client.post("/query", json={"question": "What makes ingestion idempotent?"})
        body = client.get("/metrics").text
        assert "idempotent" not in body
        # Only our own series -- prometheus_client's platform collectors
        # (python_info, gc stats) carry labels we neither set nor control.
        rag_lines = [ln for ln in body.splitlines() if ln.startswith("rag_")]
        labels = set(re.findall(r'(\w+)="', "\n".join(rag_lines)))
        # "le" is the histogram bucket boundary -- intrinsic to Prometheus, not
        # a label we choose. Everything else must be one of ours.
        assert labels <= {"stage", "retriever", "outcome", "result", "le"}, labels

    def test_excluded_from_openapi(self, client: TestClient) -> None:
        assert "/metrics" not in client.get("/openapi.json").json()["paths"]


def test_serving_path_never_imports_ingestion() -> None:
    """The offline/online boundary, enforced rather than documented.

    The API must not drag PDF parsers into the serving image, and this is what
    stops someone adding an "just ingest it inline" endpoint later.

    Run in a FRESH interpreter rather than by clearing sys.modules: re-importing
    inside this process would re-execute module-level code (including metric
    registration) and test an import graph already polluted by other tests.
    """
    probe = (
        "import sys; import app.main; "
        "print(','.join(m for m in sys.modules if m.startswith('app.ingestion')))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    leaked = proc.stdout.strip()
    assert leaked == "", f"serving path imported: {leaked}"
