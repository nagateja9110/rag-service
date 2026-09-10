"""Bounded TTL cache for exact-duplicate questions.

WHY NOT ``functools.lru_cache``
-------------------------------
It cannot expire entries. There is no TTL parameter, no timer, no eviction
callback -- an entry lives until it is pushed out by ``maxsize`` pressure. Three
consequences that matter for a RAG service:

1. **Stale answers are permanent.** Re-ingest the corpus and every cached answer
   is now derived from documents that may no longer exist. ``lru_cache`` has no
   way to know and no way to be told, short of ``cache_clear()``.
2. **The usual TTL hack is worse than it looks.** Folding a time bucket into the
   key -- ``_answer(question, int(time.time() // 300))`` -- does technically
   expire things, but every entry shares the same bucket boundary, so the entire
   cache dies at once every 5 minutes and each boundary produces a thundering
   herd of identical cache misses. Per-entry expiry spreads that load out.
3. **No observability.** You get ``cache_info()`` and nothing else; you cannot
   see or evict a single key.

So: an ``OrderedDict`` in LRU configuration, with a monotonic deadline per
entry. Small enough to read in one sitting, correct enough to keep.

BACKENDS
--------
``TTLCache`` below is the in-process implementation. It is correct but
process-local: under ``uvicorn --workers 4`` you have four independent caches
and your hit rate divides by roughly four; across replicas, more still.
``RedisCache`` shares one cache across every worker and every replica, which is
what you want in production and what ``cache_backend: redis`` selects.

Both satisfy the same async ``CacheBackend`` protocol, so the query engine does
not know or care which one it holds.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.config import Settings as SettingsLike
else:  # pragma: no cover - runtime alias for the annotation below
    SettingsLike = Any

logger = get_logger(__name__)

V = TypeVar("V")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0
    ttl_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hit_rate": self.hit_rate,
        }


class TTLCache(Generic[V]):
    """Thread-safe LRU cache with per-entry expiry.

    Thread safety is not optional here: FastAPI runs sync work in a threadpool
    and async work across tasks, so several requests can touch this object at
    once. ``OrderedDict`` mutation from multiple threads without a lock will
    eventually corrupt its linked list.
    """

    def __init__(self, max_size: int = 512, ttl_seconds: float = 300.0):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._data: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ------------------------------------------------------------------
    def get(self, key: str) -> V | None:
        """Return the cached value, or None on miss/expiry."""
        now = time.monotonic()
        with self._lock:
            # Membership test rather than a sentinel default: a sentinel widens
            # the value type to `object` and defeats narrowing, and `V` may
            # legitimately be a type that includes None.
            if key not in self._data:
                self._misses += 1
                return None

            deadline, value = self._data[key]
            if deadline <= now:
                # Lazy expiry: cheaper than a background sweeper, and an expired
                # entry costs nothing until someone asks for it.
                del self._data[key]
                self._expirations += 1
                self._misses += 1
                return None

            self._data.move_to_end(key)  # mark as recently used
            self._hits += 1
            return value

    def set(self, key: str, value: V) -> None:
        now = time.monotonic()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (now + self._ttl, value)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)  # drop least-recently-used
                self._evictions += 1

    def clear(self) -> int:
        """Drop everything. Call this after re-ingestion."""
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
                size=len(self._data),
                max_size=self._max_size,
                ttl_seconds=self._ttl,
            )


def make_cache_key(question: str, **params: Any) -> str:
    """Stable key for a question plus the parameters that shaped its answer.

    The parameters are not decoration. Cache a question asked with top_k=5 and
    then serve that same entry to a request asking for top_k=20 and you have a
    correctness bug that looks like a flaky model. Anything that changes the
    output belongs in the key.

    Normalisation is casefold + whitespace collapse: "What is RRF?" and
    "what is  rrf?" are the same question. This is deliberately *exact-match*
    only -- semantic caching (embed the question, serve on cosine > 0.97) is a
    different feature with a different failure mode, namely confidently
    answering a question nobody asked.
    """
    normalised = " ".join(question.split()).casefold()
    payload = normalised + "\x00" + "\x00".join(
        f"{k}={params[k]!r}" for k in sorted(params)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
# Cached values are plain JSON-serialisable dicts, never domain objects. That is
# a deliberate boundary: Redis can only store bytes, and pickling a QueryResult
# would couple the cache's wire format to a dataclass definition -- a field
# rename would then deserialise into a crash, or worse, silently drop data. The
# engine owns the dict<->QueryResult conversion; the cache just moves dicts.


class CacheBackend(Protocol):
    """Async because Redis is network I/O and must not block the event loop."""

    name: str

    async def get(self, key: str) -> dict[str, Any] | None: ...

    async def set(self, key: str, value: dict[str, Any]) -> None: ...

    async def clear(self) -> int: ...

    def stats(self) -> dict[str, Any]: ...


class InMemoryCache:
    """Process-local backend. Zero infrastructure, zero sharing."""

    name = "memory"

    def __init__(self, max_size: int = 512, ttl_seconds: float = 300.0) -> None:
        self._cache: TTLCache[dict[str, Any]] = TTLCache(max_size, ttl_seconds)

    async def get(self, key: str) -> dict[str, Any] | None:
        return self._cache.get(key)

    async def set(self, key: str, value: dict[str, Any]) -> None:
        self._cache.set(key, value)

    async def clear(self) -> int:
        return self._cache.clear()

    def stats(self) -> dict[str, Any]:
        payload = self._cache.stats().as_dict()
        payload["backend"] = self.name
        return payload


class RedisCache:
    """Shared backend: one cache for every worker and every replica.

    Failure policy: a cache is an optimisation, never a dependency. Every
    operation degrades to "miss" if Redis is unreachable, so a Redis outage
    makes the service slower and more expensive -- not broken. The alternative
    (propagating the error) would turn a cache blip into a full outage, which is
    precisely backwards.
    """

    name = "redis"

    def __init__(
        self,
        url: str,
        prefix: str = "rag:q:",
        ttl_seconds: float = 300.0,
        timeout_seconds: float = 2.0,
    ) -> None:
        import redis.asyncio as aioredis

        self._prefix = prefix
        self._ttl = int(ttl_seconds)
        # Short timeouts on purpose: a slow cache lookup must never dominate the
        # latency of the pipeline it is meant to accelerate.
        self._client = aioredis.from_url(
            url,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
            decode_responses=True,
        )
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._degraded = False

    def _key(self, key: str) -> str:
        # Namespaced so this cache can share a Redis instance with anything else.
        return f"{self._prefix}{key}"

    async def ping(self) -> bool:
        try:
            await self._client.ping()
            self._degraded = False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unreachable at startup: %s", exc)
            self._degraded = True
            return False

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = await self._client.get(self._key(key))
        except Exception as exc:  # noqa: BLE001
            self._errors += 1
            self._degraded = True
            logger.warning("Redis GET failed, treating as miss: %s", exc)
            return None

        if raw is None:
            self._misses += 1
            return None
        try:
            value: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            # A poisoned or format-changed entry must not break the request.
            self._misses += 1
            logger.warning("Discarding unreadable cache entry for %s", key)
            return None
        self._hits += 1
        self._degraded = False
        return value

    async def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            # SETEX: expiry is set atomically with the value, so a crash between
            # write and expire cannot leave an immortal entry behind.
            await self._client.setex(self._key(key), self._ttl, json.dumps(value))
        except Exception as exc:  # noqa: BLE001
            self._errors += 1
            logger.warning("Redis SET failed, continuing uncached: %s", exc)

    async def clear(self) -> int:
        """Delete every key under our prefix.

        Uses SCAN, never KEYS: KEYS is O(n) over the entire keyspace and blocks
        the single-threaded Redis server, which on a shared instance means
        stalling every other client on the box.
        """
        removed = 0
        try:
            async for key in self._client.scan_iter(match=f"{self._prefix}*", count=500):
                await self._client.delete(key)
                removed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis clear failed: %s", exc)
        return removed

    async def close(self) -> None:
        # Shutdown must not fail because a connection was already gone.
        with contextlib.suppress(Exception):
            await self._client.aclose()

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": self.name,
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "degraded": self._degraded,
            "ttl_seconds": self._ttl,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }


def build_cache(settings: SettingsLike) -> CacheBackend | None:
    """Construct the configured backend, or None when caching is disabled.

    Falls back to the in-memory backend if Redis is selected but the client
    library is missing -- a misconfigured cache should degrade the service, not
    prevent it from starting.
    """
    if not settings.query_cache_enabled:
        logger.info("Query cache disabled")
        return None

    if settings.cache_backend == "redis":
        try:
            return RedisCache(
                url=settings.redis_url,
                prefix=settings.redis_prefix,
                ttl_seconds=settings.query_cache_ttl_seconds,
                timeout_seconds=settings.redis_timeout_seconds,
            )
        except ImportError as exc:
            logger.error("redis package unavailable (%s); using in-memory cache", exc)

    return InMemoryCache(
        max_size=settings.query_cache_max_size,
        ttl_seconds=settings.query_cache_ttl_seconds,
    )
