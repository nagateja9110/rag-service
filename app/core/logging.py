"""Logging setup.

Deliberately stdlib-only. Structured logging libraries (structlog, loguru) are
nice, but a RAG service's logging needs are modest and every dependency you add
to the serving image is a dependency you have to patch. What matters is that
configuration happens exactly once, at process entry, and that third-party
libraries do not spam INFO into our output.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Libraries that log far too enthusiastically at INFO.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "chromadb",
)

# Chroma's posthog telemetry shim logs its own failures at ERROR even when
# telemetry is disabled (chromadb 0.6.x). It is not actionable and it is not
# ours, so it does not get to pollute the operational log.
_SILENCED_LOGGERS = ("chromadb.telemetry",)

_configured = False


class JsonFormatter(logging.Formatter):
    """One JSON object per line -- what log aggregators want in production."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything attached via logger.info(..., extra={...}) rides along.
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__ and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Idempotent root-logger configuration. Safe to call from CLI and API."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
