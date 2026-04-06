"""Semantic cache for analytics query results — PostgreSQL-backed.

Uses TF-IDF vectorization + cosine similarity to detect semantically
similar queries and return cached results, saving LLM calls.

Cache layers:
  1. Exact-match (SHA-256 hash of normalized query) — O(1) DB lookup
  2. Semantic match (TF-IDF cosine similarity against all cached queries)

Results are persisted in the `query_cache` table so they survive restarts.
"""

import hashlib
import json
import logging
import re
import threading
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class SemanticCache:
    """PostgreSQL-backed semantic cache with TF-IDF similarity matching.

    Features:
        - Exact-match fast path via normalized query hash (DB index)
        - Semantic similarity matching via TF-IDF + cosine similarity
        - TTL-based expiration (checked on read)
        - Max size with oldest-entry eviction
        - Thread-safe operations
        - Persisted across restarts
    """

    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "so", "if", "than", "that", "this", "these", "those",
        "what", "which", "who", "whom", "how", "when", "where", "why",
        "me", "my", "i", "we", "our", "you", "your", "it", "its",
        "show", "tell", "give", "get", "find", "list", "display",
        "please", "also", "just",
    })

    def __init__(
        self,
        db_url: str,
        similarity_threshold: float = 0.75,
        ttl_seconds: int = 3600,
        max_size: int = 500,
    ):
        self.db_url = db_url
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._engine = create_engine(db_url, pool_pre_ping=True, pool_size=3)
        self._lock = threading.Lock()

        # In-memory stats (reset on restart, that's fine for monitoring)
        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
        }

    @staticmethod
    def _normalize(query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q)
        return q.strip()

    @staticmethod
    def _hash(normalized: str) -> str:
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _cleanup_expired(self) -> int:
        """Delete expired entries from DB. Returns number deleted."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    "DELETE FROM query_cache "
                    "WHERE created_at < now() - make_interval(secs => :ttl) "
                    "RETURNING id"
                ),
                {"ttl": self.ttl_seconds},
            )
            deleted = result.rowcount
            conn.commit()
            if deleted > 0:
                logger.info(f"Cache cleanup: removed {deleted} expired entries")
            return deleted

    def _enforce_max_size(self) -> None:
        """Evict oldest entries if cache exceeds max_size."""
        with self._engine.connect() as conn:
            count = conn.execute(text("SELECT count(*) FROM query_cache")).scalar()
            if count >= self.max_size:
                to_delete = count - self.max_size + 1
                conn.execute(
                    text(
                        "DELETE FROM query_cache WHERE id IN ("
                        "  SELECT id FROM query_cache ORDER BY last_hit_at ASC LIMIT :n"
                        ")"
                    ),
                    {"n": to_delete},
                )
                conn.commit()
                logger.info(f"Cache eviction: removed {to_delete} oldest entries")

    def get(self, query: str, user_db_url: str) -> dict[str, Any] | None:
        """Look up a cached result for the given query."""
        db_url_hash = self._hash(user_db_url)
        normalized = self._normalize(query)
        query_hash = self._hash(normalized + "|" + db_url_hash)

        with self._lock:
            # Cleanup expired first
            self._cleanup_expired()

            with self._engine.connect() as conn:
                # Layer 1: Exact match by hash
                row = conn.execute(
                    text(
                        "SELECT id, query, result, hit_count FROM query_cache "
                        "WHERE query_hash = :h AND db_url_hash = :dh "
                        "AND created_at >= now() - make_interval(secs => :ttl)"
                    ),
                    {"h": query_hash, "dh": db_url_hash, "ttl": self.ttl_seconds},
                ).fetchone()

                if row:
                    # Update hit count and last_hit_at
                    conn.execute(
                        text(
                            "UPDATE query_cache SET hit_count = hit_count + 1, "
                            "last_hit_at = now() WHERE id = :id"
                        ),
                        {"id": row[0]},
                    )
                    conn.commit()
                    self._stats["exact_hits"] += 1
                    logger.info(
                        f"Cache EXACT HIT for query: '{query[:80]}' "
                        f"(hits: {row[3] + 1})"
                    )
                    return row[2]  # result is jsonb, auto-deserialized

                # Layer 2: Semantic match
                all_rows = conn.execute(
                    text(
                        "SELECT id, query, normalized_query, result, hit_count "
                        "FROM query_cache "
                        "WHERE db_url_hash = :dh "
                        "AND created_at >= now() - make_interval(secs => :ttl)"
                    ),
                    {"dh": db_url_hash, "ttl": self.ttl_seconds},
                ).fetchall()

            if not all_rows:
                self._stats["misses"] += 1
                return None

            try:
                cached_norms = [r[2] for r in all_rows]
                vectorizer = TfidfVectorizer(
                    analyzer="word",
                    stop_words=list(self._STOP_WORDS),
                    ngram_range=(1, 2),
                    max_features=5000,
                )
                tfidf_matrix = vectorizer.fit_transform(cached_norms)
                query_vec = vectorizer.transform([normalized])
                similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])

                if best_score >= self.similarity_threshold:
                    matched = all_rows[best_idx]
                    # Update hit count
                    with self._engine.connect() as conn:
                        conn.execute(
                            text(
                                "UPDATE query_cache SET hit_count = hit_count + 1, "
                                "last_hit_at = now() WHERE id = :id"
                            ),
                            {"id": matched[0]},
                        )
                        conn.commit()
                    self._stats["semantic_hits"] += 1
                    logger.info(
                        f"Cache SEMANTIC HIT for query: '{query[:80]}' "
                        f"→ matched '{matched[1][:80]}' "
                        f"(similarity: {best_score:.3f}, hits: {matched[4] + 1})"
                    )
                    return matched[3]  # result jsonb
                else:
                    logger.debug(
                        f"Best semantic match score {best_score:.3f} "
                        f"below threshold {self.similarity_threshold}"
                    )
            except Exception as e:
                logger.warning(f"Semantic similarity search failed: {e}")

            self._stats["misses"] += 1
            return None

    def put(self, query: str, user_db_url: str, result: dict[str, Any]) -> None:
        """Store a query result in the cache."""
        db_url_hash = self._hash(user_db_url)
        normalized = self._normalize(query)
        query_hash = self._hash(normalized + "|" + db_url_hash)

        with self._lock:
            self._enforce_max_size()

            with self._engine.connect() as conn:
                # Upsert: if same hash exists, update it
                conn.execute(
                    text(
                        "INSERT INTO query_cache (query, normalized_query, query_hash, db_url_hash, result) "
                        "VALUES (:q, :nq, :h, :dh, CAST(:r AS jsonb)) "
                        "ON CONFLICT (query_hash) DO UPDATE SET "
                        "  result = EXCLUDED.result, "
                        "  created_at = now(), "
                        "  last_hit_at = now(), "
                        "  hit_count = 0"
                    ),
                    {"q": query, "nq": normalized, "h": query_hash, "dh": db_url_hash, "r": json.dumps(result)},
                )
                conn.commit()

            logger.info(f"Cache STORED result for query: '{query[:80]}'")

    def invalidate_all(self) -> None:
        """Clear the entire query cache."""
        with self._lock:
            with self._engine.connect() as conn:
                conn.execute(text("DELETE FROM query_cache"))
                conn.commit()
            logger.info("Cache invalidated (all entries cleared)")

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._engine.connect() as conn:
            size = conn.execute(text("SELECT count(*) FROM query_cache")).scalar()
            total_hits_db = conn.execute(
                text("SELECT COALESCE(SUM(hit_count), 0) FROM query_cache")
            ).scalar()

        total_hits = self._stats["exact_hits"] + self._stats["semantic_hits"]
        total_requests = total_hits + self._stats["misses"]
        return {
            "size": size,
            "max_size": self.max_size,
            "total_requests_this_session": total_requests,
            "exact_hits": self._stats["exact_hits"],
            "semantic_hits": self._stats["semantic_hits"],
            "total_hits_this_session": total_hits,
            "misses": self._stats["misses"],
            "hit_rate": (
                round(total_hits / total_requests, 3)
                if total_requests > 0
                else 0.0
            ),
            "total_hits_all_time": total_hits_db,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds,
        }
