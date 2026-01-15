"""Tests for query caching system.

This module tests the query_cache module, including:
- Exact match query caching
- Semantic similarity-based caching
- LRU eviction
- TTL expiration
- Cache statistics

Week 1 - Day 4: Query cache basic tests (10 tests)
"""

import pytest
import time
import numpy as np
from pathlib import Path
from utils.query_cache import QueryCache, SemanticQueryCache


# ============================================================================
# Day 4: Basic QueryCache Tests (2 tests)
# ============================================================================


class TestQueryCache:
    """Test exact match query cache.

    These tests validate:
    - Cache get/set operations
    - MD5-based cache key generation
    - Cache miss/hit detection
    """

    @pytest.mark.unit
    def test_cache_set_and_get(self, query_cache_dir):
        """Test setting and retrieving cached embedding.

        Given: An empty cache
        When: Embedding is cached and retrieved
        Then: Retrieved embedding matches stored embedding
        """
        cache = QueryCache(cache_dir=str(query_cache_dir))
        query = "What is machine learning?"
        model = "BAAI/bge-small-en"
        embedding = [0.1, 0.2, 0.3] * 128  # 384-dim vector

        # Cache miss on first get
        result = cache.get(query, model)
        assert result is None

        # Set cache
        cache.set(query, model, embedding)

        # Cache hit on second get
        result = cache.get(query, model)
        assert result is not None
        assert result == embedding

    @pytest.mark.unit
    def test_cache_different_models(self, query_cache_dir):
        """Test cache keys are model-specific.

        Given: Same query cached for two different models
        When: get() is called with different model names
        Then: Returns model-specific embeddings
        """
        cache = QueryCache(cache_dir=str(query_cache_dir))
        query = "test query"
        model1 = "model-a"
        model2 = "model-b"
        embedding1 = [1.0] * 384
        embedding2 = [2.0] * 384

        # Cache for both models
        cache.set(query, model1, embedding1)
        cache.set(query, model2, embedding2)

        # Retrieve model-specific embeddings
        result1 = cache.get(query, model1)
        result2 = cache.get(query, model2)

        assert result1 == embedding1
        assert result2 == embedding2
        assert result1 != result2


# ============================================================================
# Day 4: Semantic Query Cache Tests (3 tests)
# ============================================================================


class TestSemanticQueryCache:
    """Test semantic similarity-based cache.

    These tests validate:
    - Exact match detection
    - Similar query detection (>threshold)
    - Dissimilar query rejection (<threshold)
    """

    @pytest.mark.unit
    def test_semantic_cache_exact_match(self, semantic_cache_dir, sample_embeddings):
        """Test cache hit for exact query.

        Given: A query cached in semantic cache
        When: Same query is requested
        Then: Returns cached response
        """
        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            cache_dir=str(semantic_cache_dir),
        )

        query = "What is RAG?"
        embedding = sample_embeddings['query1']
        response = {"answer": "RAG is Retrieval-Augmented Generation"}

        # Cache the query
        cache.set_semantic(query, embedding, response)

        # Retrieve exact same query
        result = cache.get_semantic(query, embedding)

        assert result is not None
        assert result == response
        assert cache.hits == 1
        assert cache.misses == 0

    @pytest.mark.unit
    def test_semantic_cache_similar_query(self, semantic_cache_dir, sample_embeddings):
        """Test cache hit for similar query (>0.92 similarity).

        Given: A query cached in semantic cache
        When: Similar query is requested (similarity > threshold)
        Then: Returns cached response
        """
        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            cache_dir=str(semantic_cache_dir),
        )

        # Cache original query
        original_query = "What is RAG?"
        original_embedding = sample_embeddings['query1']
        response = {"answer": "RAG is Retrieval-Augmented Generation"}
        cache.set_semantic(original_query, original_embedding, response)

        # Request similar query
        similar_query = "What is Retrieval-Augmented Generation?"
        similar_embedding = sample_embeddings['similar']  # 99% similar

        result = cache.get_semantic(similar_query, similar_embedding)

        assert result is not None
        assert result == response
        assert cache.hits == 1

    @pytest.mark.unit
    def test_semantic_cache_dissimilar_query(self, semantic_cache_dir, sample_embeddings):
        """Test cache miss for dissimilar query (<0.92 similarity).

        Given: A query cached in semantic cache
        When: Dissimilar query is requested (similarity < threshold)
        Then: Returns None (cache miss)
        """
        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            cache_dir=str(semantic_cache_dir),
        )

        # Cache original query
        original_query = "What is RAG?"
        original_embedding = sample_embeddings['query1']
        response = {"answer": "RAG is Retrieval-Augmented Generation"}
        cache.set_semantic(original_query, original_embedding, response)

        # Request completely different query
        different_query = "What is the weather today?"
        different_embedding = sample_embeddings['different']

        result = cache.get_semantic(different_query, different_embedding)

        assert result is None
        assert cache.misses == 1


# ============================================================================
# Day 4: LRU Eviction Tests (2 tests)
# ============================================================================


class TestSemanticCacheLRU:
    """Test LRU eviction behavior.

    These tests validate:
    - Eviction when cache reaches max_size
    - Oldest entry is evicted first
    """

    @pytest.mark.unit
    def test_lru_eviction_at_max_size(self, semantic_cache_dir, mock_embedding):
        """Test LRU eviction when cache is full.

        Given: Cache with max_size=5 containing 5 entries
        When: A 6th entry is added
        Then: Oldest entry is evicted
        """
        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            max_size=5,
            cache_dir=str(semantic_cache_dir),
        )

        # Add 5 entries (fill to max)
        for i in range(5):
            query = f"Query {i}"
            embedding = mock_embedding(seed=i)
            response = {"answer": f"Answer {i}"}
            cache.set_semantic(query, embedding, response)
            time.sleep(0.01)  # Ensure different timestamps

        assert len(cache.cache) == 5
        assert cache.evictions == 0

        # Add 6th entry (triggers eviction)
        cache.set_semantic("Query 5", mock_embedding(seed=5), {"answer": "Answer 5"})

        # Verify eviction occurred
        assert len(cache.cache) == 5  # Still at max
        assert cache.evictions == 1

        # Verify oldest entry (Query 0) was evicted
        query0_id = cache._generate_id("Query 0")
        assert query0_id not in cache.cache

    @pytest.mark.unit
    def test_lru_access_time_update(self, semantic_cache_dir, mock_embedding):
        """Test access time updates on cache hit.

        Given: Cache with multiple entries
        When: An entry is accessed (cache hit)
        Then: Access time is updated (prevents premature eviction)
        """
        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            max_size=3,
            cache_dir=str(semantic_cache_dir),
        )

        # Add 2 entries
        query1 = "First query"
        query2 = "Second query"
        emb1 = mock_embedding(seed=1)
        emb2 = mock_embedding(seed=2)

        cache.set_semantic(query1, emb1, {"answer": "Answer 1"})
        time.sleep(0.01)
        cache.set_semantic(query2, emb2, {"answer": "Answer 2"})

        # Access first query (updates access time)
        time.sleep(0.01)
        result = cache.get_semantic(query1, emb1)
        assert result is not None

        # Add 2 more entries (should evict query2, not query1)
        time.sleep(0.01)
        cache.set_semantic("Third query", mock_embedding(seed=3), {"answer": "Answer 3"})

        # query1 should still be in cache (was accessed recently)
        query1_id = cache._generate_id(query1)
        assert query1_id in cache.cache


# ============================================================================
# Day 4: TTL Expiration Tests (2 tests)
# ============================================================================


class TestSemanticCacheTTL:
    """Test time-to-live expiration.

    These tests validate:
    - Expired entries return cache miss
    - Non-expired entries return cache hit
    """

    @pytest.mark.unit
    def test_ttl_expiration(self, semantic_cache_dir, mock_embedding):
        """Test cache entries expire after TTL.

        Given: Cache with TTL=1 second
        When: Entry is accessed after TTL expires
        Then: Returns None (cache miss) and entry is removed
        """
        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            ttl=1,  # 1 second TTL
            cache_dir=str(semantic_cache_dir),
        )

        query = "Test query"
        embedding = mock_embedding(seed=42)
        response = {"answer": "Test answer"}

        # Cache the query
        cache.set_semantic(query, embedding, response)
        assert len(cache.cache) == 1

        # Immediately retrieve (should hit)
        result = cache.get_semantic(query, embedding)
        assert result is not None
        assert cache.hits == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Try to retrieve again (should miss and remove)
        result = cache.get_semantic(query, embedding)
        assert result is None
        assert cache.misses == 1
        assert len(cache.cache) == 0  # Expired entry removed

    @pytest.mark.unit
    def test_ttl_disabled(self, semantic_cache_dir, mock_embedding):
        """Test cache without TTL (entries never expire).

        Given: Cache with ttl=None (no expiration)
        When: Entry is accessed long after creation
        Then: Returns cached response (no expiration)
        """
        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            ttl=None,  # No TTL
            cache_dir=str(semantic_cache_dir),
        )

        query = "Test query"
        embedding = mock_embedding(seed=42)
        response = {"answer": "Test answer"}

        # Cache the query
        cache.set_semantic(query, embedding, response)

        # Immediately retrieve (should hit)
        result = cache.get_semantic(query, embedding)
        assert result is not None

        # Entry should still be valid (no TTL)
        assert len(cache.cache) == 1


# ============================================================================
# Day 4: Cache Statistics Test (1 test)
# ============================================================================


class TestSemanticCacheStatistics:
    """Test cache statistics tracking.

    These tests validate:
    - Hit/miss counting
    - Hit rate calculation
    - Eviction tracking
    """

    @pytest.mark.unit
    def test_cache_statistics(self, semantic_cache_dir, mock_embedding):
        """Test cache statistics tracking.

        Given: Cache with multiple operations
        When: stats() is called
        Then: Returns accurate statistics
        """
        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            max_size=3,
            cache_dir=str(semantic_cache_dir),
        )

        # Add 2 entries
        cache.set_semantic("Query 1", mock_embedding(seed=1), {"answer": "A1"})
        cache.set_semantic("Query 2", mock_embedding(seed=2), {"answer": "A2"})

        # Cache hit
        result = cache.get_semantic("Query 1", mock_embedding(seed=1))
        assert result is not None

        # Cache miss
        result = cache.get_semantic("Query 3", mock_embedding(seed=3))
        assert result is None

        # Add entries to trigger eviction
        cache.set_semantic("Query 4", mock_embedding(seed=4), {"answer": "A4"})
        cache.set_semantic("Query 5", mock_embedding(seed=5), {"answer": "A5"})  # Evicts oldest

        # Get statistics
        stats = cache.stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5  # 1 hit / 2 total
        assert stats["evictions"] == 1
        assert stats["count"] == 3  # Max size maintained
        assert stats["threshold"] == 0.92
        assert stats["enabled"] is True


# ============================================================================
# Test Discovery and Execution Notes
# ============================================================================
"""
Run Day 4 tests:
    pytest tests/test_query_cache.py -v

Run with coverage:
    pytest tests/test_query_cache.py \
        --cov=utils.query_cache \
        --cov-report=term-missing

Expected results:
    - 10 tests passing
    - Coverage increase: +50-60% for utils/query_cache.py
    - Validates semantic caching and performance optimization
"""
