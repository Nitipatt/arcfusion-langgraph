"""Schema cache — PostgreSQL-backed, avoids redundant DB introspection.

The database schema rarely changes during normal operation, so we cache
the schema_extractor result with a configurable TTL (default 30 min).
Persisted in the `schema_cache` table so it survives restarts.
"""

import hashlib
import logging
import threading
from typing import Any

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class SchemaCache:
    """PostgreSQL-backed TTL cache for database schema, keyed by db_url hash."""

    def __init__(self, db_url: str, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._engine = create_engine(db_url, pool_pre_ping=True, pool_size=2)
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0}

    @staticmethod
    def _hash(db_url: str) -> str:
        return hashlib.sha256(db_url.encode()).hexdigest()

    def get(self, db_url: str) -> str | None:
        """Return cached schema or None if expired/missing."""
        url_hash = self._hash(db_url)
        with self._lock:
            with self._engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT schema_text FROM schema_cache "
                        "WHERE db_url_hash = :h "
                        "AND created_at >= now() - make_interval(secs => :ttl)"
                    ),
                    {"h": url_hash, "ttl": self.ttl_seconds},
                ).fetchone()

                if row:
                    self._stats["hits"] += 1
                    logger.info("Schema cache HIT (skipping DB introspection)")
                    return row[0]

                # Cleanup expired entry if exists
                conn.execute(
                    text("DELETE FROM schema_cache WHERE db_url_hash = :h"),
                    {"h": url_hash},
                )
                conn.commit()

            self._stats["misses"] += 1
            return None

    def put(self, db_url: str, schema: str) -> None:
        """Store schema in cache (upsert)."""
        url_hash = self._hash(db_url)
        with self._lock:
            with self._engine.connect() as conn:
                conn.execute(
                    text(
                        "INSERT INTO schema_cache (db_url_hash, schema_text) "
                        "VALUES (:h, :s) "
                        "ON CONFLICT (db_url_hash) DO UPDATE SET "
                        "  schema_text = EXCLUDED.schema_text, "
                        "  created_at = now()"
                    ),
                    {"h": url_hash, "s": schema},
                )
                conn.commit()
            logger.info("Schema cache STORED")

    def invalidate(self, db_url: str | None = None) -> None:
        """Clear schema cache for a specific db_url or all."""
        with self._lock:
            with self._engine.connect() as conn:
                if db_url:
                    conn.execute(
                        text("DELETE FROM schema_cache WHERE db_url_hash = :h"),
                        {"h": self._hash(db_url)},
                    )
                else:
                    conn.execute(text("DELETE FROM schema_cache"))
                conn.commit()
            logger.info("Schema cache invalidated")

    def stats(self) -> dict[str, Any]:
        """Return cache stats."""
        with self._engine.connect() as conn:
            count = conn.execute(text("SELECT count(*) FROM schema_cache")).scalar()

        total = self._stats["hits"] + self._stats["misses"]
        return {
            "cached_schemas": count,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(self._stats["hits"] / total, 3) if total > 0 else 0.0,
            "ttl_seconds": self.ttl_seconds,
        }
