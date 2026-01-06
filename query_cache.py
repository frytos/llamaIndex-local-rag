"""
Query embedding cache to avoid recomputation.

Usage:
    from query_cache import cache

    # Check cache first
    embedding = cache.get(query, model_name)
    if embedding is None:
        embedding = model.encode(query)
        cache.set(query, model_name, embedding)
"""

import hashlib
import json
import os
from typing import List, Optional
from pathlib import Path
import logging

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/query_embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class QueryCache:
    """
    Disk-backed cache for query embeddings.

    Avoids recomputing embeddings for repeated queries.
    Thread-safe for reads, writes use atomic file operations.
    """

    def __init__(self, cache_dir: str = str(CACHE_DIR)):
        """
        Initialize query cache.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Query cache initialized: {self.cache_dir}")

    def _hash(self, query: str, model: str) -> str:
        """
        Generate cache key from query and model name.

        Args:
            query: Query text
            model: Model name/identifier

        Returns:
            MD5 hash as hex string
        """
        key = f"{model}:{query}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, query: str, model: str) -> Optional[List[float]]:
        """
        Get cached embedding if available.

        Args:
            query: Query text
            model: Model name

        Returns:
            Cached embedding vector or None if not found
        """
        cache_file = self.cache_dir / f"{self._hash(query, model)}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    embedding = json.load(f)
                log.debug(f"Cache hit: {query[:50]}...")
                return embedding
            except (json.JSONDecodeError, IOError) as e:
                log.warning(f"Cache read error: {e}")
                return None

        log.debug(f"Cache miss: {query[:50]}...")
        return None

    def set(self, query: str, model: str, embedding: List[float]):
        """
        Cache an embedding.

        Args:
            query: Query text
            model: Model name
            embedding: Embedding vector to cache
        """
        cache_file = self.cache_dir / f"{self._hash(query, model)}.json"

        try:
            # Atomic write: write to temp file, then rename
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, "w") as f:
                json.dump(embedding, f)
            temp_file.rename(cache_file)
            log.debug(f"Cached: {query[:50]}...")
        except (IOError, OSError) as e:
            log.warning(f"Cache write error: {e}")

    def clear(self):
        """Clear all cached embeddings"""
        count = 0
        for file in self.cache_dir.glob("*.json"):
            try:
                file.unlink()
                count += 1
            except OSError as e:
                log.warning(f"Error deleting {file}: {e}")
        log.info(f"Cleared {count} cached embeddings")

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (count, size)
        """
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "count": len(files),
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }


# Singleton instance for easy import
cache = QueryCache()


if __name__ == "__main__":
    # Test cache
    print("="*70)
    print("Query Cache Test")
    print("="*70)

    test_query = "What did Elena say about Morocco?"
    test_model = "bge-large-en-v1.5"
    test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"\n1. Testing cache miss...")
    result = cache.get(test_query, test_model)
    assert result is None, "Expected cache miss"
    print("   ✓ Cache miss works")

    print(f"\n2. Caching embedding...")
    cache.set(test_query, test_model, test_embedding)
    print("   ✓ Embedding cached")

    print(f"\n3. Testing cache hit...")
    result = cache.get(test_query, test_model)
    assert result == test_embedding, "Cached embedding doesn't match"
    print("   ✓ Cache hit works")

    print(f"\n4. Cache stats:")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print(f"\n5. Clearing cache...")
    cache.clear()
    result = cache.get(test_query, test_model)
    assert result is None, "Cache should be empty"
    print("   ✓ Cache cleared")

    print("\n" + "="*70)
    print("✓ All tests passed")
    print("="*70)
