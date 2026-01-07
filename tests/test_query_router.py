"""
Unit tests for query router module.

Tests classification accuracy, routing logic, caching, and performance.
"""

import os
import pytest
import numpy as np
from pathlib import Path

from utils.query_router import (
    QueryRouter,
    QueryType,
    RetrievalConfig,
    RoutingResult,
    is_enabled,
    ROUTING_CACHE_DIR,
)


class TestQueryTypeClassification:
    """Test query type classification accuracy"""

    @pytest.fixture
    def router_pattern(self):
        """Pattern-based router for testing"""
        return QueryRouter(method="pattern", cache_decisions=False)

    def test_factual_queries(self, router_pattern):
        """Test factual query classification"""
        factual_queries = [
            "What is machine learning?",
            "Who invented Python?",
            "When did World War 2 end?",
            "Where is the Eiffel Tower?",
            "How many planets are there?",
            "Define artificial intelligence",
        ]

        for query in factual_queries:
            query_type, confidence, _ = router_pattern.classify_query(query)
            assert query_type == QueryType.FACTUAL, f"Failed to classify as FACTUAL: {query}"
            assert confidence >= 0.8, f"Low confidence for factual query: {query}"

    def test_conceptual_queries(self, router_pattern):
        """Test conceptual query classification"""
        conceptual_queries = [
            "How does photosynthesis work?",
            "Explain neural networks",
            "Why do we dream?",
            "Describe the water cycle",
            "What causes earthquakes?",
        ]

        for query in conceptual_queries:
            query_type, confidence, _ = router_pattern.classify_query(query)
            assert query_type == QueryType.CONCEPTUAL, f"Failed to classify as CONCEPTUAL: {query}"
            assert confidence >= 0.8, f"Low confidence for conceptual query: {query}"

    def test_procedural_queries(self, router_pattern):
        """Test procedural query classification"""
        procedural_queries = [
            "How to bake a cake?",
            "Steps to install Docker",
            "How can I fix this bug?",
            "Guide to setting up PostgreSQL",
            "How do I configure the server?",
        ]

        for query in procedural_queries:
            query_type, confidence, _ = router_pattern.classify_query(query)
            assert query_type == QueryType.PROCEDURAL, f"Failed to classify as PROCEDURAL: {query}"
            assert confidence >= 0.8, f"Low confidence for procedural query: {query}"

    def test_conversational_queries(self, router_pattern):
        """Test conversational query classification"""
        conversational_queries = [
            "What about it?",
            "Tell me more",
            "And then?",
            "Why is that?",
        ]

        for query in conversational_queries:
            query_type, confidence, _ = router_pattern.classify_query(query)
            assert query_type == QueryType.CONVERSATIONAL, f"Failed to classify as CONVERSATIONAL: {query}"

    def test_comparative_queries(self, router_pattern):
        """Test comparative query classification"""
        comparative_queries = [
            "Python vs JavaScript",
            "Compare React and Vue",
            "What's the difference between RAM and ROM?",
            "Is A better than B?",
            "Alternatives to MySQL",
        ]

        for query in comparative_queries:
            query_type, confidence, _ = router_pattern.classify_query(query)
            assert query_type == QueryType.COMPARATIVE, f"Failed to classify as COMPARATIVE: {query}"
            assert confidence >= 0.8, f"Low confidence for comparative query: {query}"

    def test_empty_query(self, router_pattern):
        """Test empty query handling"""
        query_type, confidence, metadata = router_pattern.classify_query("")
        assert query_type == QueryType.UNKNOWN
        assert confidence == 0.0
        assert "error" in metadata


class TestRetrievalConfig:
    """Test retrieval configuration generation"""

    @pytest.fixture
    def router(self):
        return QueryRouter(method="pattern", cache_decisions=False)

    def test_factual_config(self, router):
        """Test factual query config"""
        config = router._get_config_for_type(QueryType.FACTUAL)
        assert config.chunk_size == 200  # Small chunks for precision
        assert config.top_k == 3
        assert config.hybrid_alpha == 0.3  # More BM25
        assert config.enable_reranking is True
        assert config.enable_query_expansion is False
        assert config.temperature == 0.1  # Deterministic

    def test_conceptual_config(self, router):
        """Test conceptual query config"""
        config = router._get_config_for_type(QueryType.CONCEPTUAL)
        assert config.chunk_size == 800  # Large chunks for context
        assert config.top_k == 5
        assert config.hybrid_alpha == 0.7  # More semantic
        assert config.enable_reranking is True
        assert config.enable_query_expansion is True
        assert config.temperature == 0.3

    def test_procedural_config(self, router):
        """Test procedural query config"""
        config = router._get_config_for_type(QueryType.PROCEDURAL)
        assert config.chunk_size == 400  # Medium chunks
        assert config.top_k == 6
        assert config.preserve_order is True  # Maintain step order
        assert config.enable_reranking is True
        assert config.temperature == 0.2

    def test_comparative_config(self, router):
        """Test comparative query config"""
        config = router._get_config_for_type(QueryType.COMPARATIVE)
        assert config.chunk_size == 600
        assert config.top_k == 8  # Higher for multiple subjects
        assert config.enable_reranking is True
        assert config.enable_query_expansion is True

    def test_conversational_config(self, router):
        """Test conversational query config"""
        config = router._get_config_for_type(QueryType.CONVERSATIONAL)
        assert config.chunk_size == 500
        assert config.enable_reranking is False  # Speed over precision
        assert config.temperature == 0.4

    def test_config_serialization(self, router):
        """Test config can be serialized/deserialized"""
        config = router._get_config_for_type(QueryType.FACTUAL)
        config_dict = config.to_dict()

        # Verify all fields present
        assert "chunk_size" in config_dict
        assert "top_k" in config_dict
        assert "hybrid_alpha" in config_dict

        # Verify can reconstruct
        new_config = RetrievalConfig(**config_dict)
        assert new_config.chunk_size == config.chunk_size
        assert new_config.top_k == config.top_k


class TestRouting:
    """Test end-to-end routing"""

    @pytest.fixture
    def router(self):
        return QueryRouter(method="pattern", cache_decisions=False, log_decisions=False)

    def test_route_factual(self, router):
        """Test routing factual query"""
        result = router.route("What is Python?")

        assert isinstance(result, RoutingResult)
        assert result.query_type == QueryType.FACTUAL
        assert result.confidence > 0.8
        assert isinstance(result.config, RetrievalConfig)
        assert result.config.chunk_size == 200
        assert "elapsed_ms" in result.metadata

    def test_route_conceptual(self, router):
        """Test routing conceptual query"""
        result = router.route("How does Python garbage collection work?")

        assert result.query_type == QueryType.CONCEPTUAL
        assert result.config.chunk_size == 800
        assert result.config.enable_query_expansion is True

    def test_route_procedural(self, router):
        """Test routing procedural query"""
        result = router.route("How to install numpy?")

        assert result.query_type == QueryType.PROCEDURAL
        assert result.config.preserve_order is True
        assert result.config.chunk_size == 400

    def test_route_comparative(self, router):
        """Test routing comparative query"""
        result = router.route("Python vs Java for beginners")

        assert result.query_type == QueryType.COMPARATIVE
        assert result.config.top_k == 8  # Higher k for comparisons
        assert result.config.enable_query_expansion is True

    def test_routing_performance(self, router):
        """Test routing is fast"""
        import time

        query = "What is machine learning?"
        start = time.time()
        result = router.route(query)
        elapsed = (time.time() - start) * 1000  # ms

        # Pattern-based routing should be < 2ms
        assert elapsed < 2.0, f"Routing too slow: {elapsed:.2f}ms"
        assert result.metadata["elapsed_ms"] < 2.0


class TestCaching:
    """Test routing decision caching"""

    @pytest.fixture
    def router_with_cache(self):
        """Router with caching enabled"""
        router = QueryRouter(method="pattern", cache_decisions=True, log_decisions=False)
        # Clear cache before test
        router.clear_cache()
        yield router
        # Cleanup after test
        router.clear_cache()

    def test_cache_miss_then_hit(self, router_with_cache):
        """Test cache miss followed by cache hit"""
        query = "What is caching?"

        # First call - cache miss
        result1 = router_with_cache.route(query)
        assert router_with_cache.stats["cache_misses"] == 1
        assert router_with_cache.stats["cache_hits"] == 0

        # Second call - cache hit
        result2 = router_with_cache.route(query)
        assert router_with_cache.stats["cache_hits"] == 1

        # Results should be identical
        assert result1.query_type == result2.query_type
        assert result1.confidence == result2.confidence
        assert result1.config.chunk_size == result2.config.chunk_size

    def test_cache_persistence(self, router_with_cache):
        """Test cache persists across router instances"""
        query = "What is persistence?"

        # Route with first instance
        result1 = router_with_cache.route(query)

        # Create new instance (should load from disk)
        new_router = QueryRouter(method="pattern", cache_decisions=True, log_decisions=False)
        result2 = new_router.route(query)

        # Should be cache hit on second instance
        assert new_router.stats["cache_hits"] == 1

        # Results should match
        assert result1.query_type == result2.query_type

        # Cleanup
        new_router.clear_cache()

    def test_clear_cache(self, router_with_cache):
        """Test cache clearing"""
        # Add some cached entries
        router_with_cache.route("Query 1")
        router_with_cache.route("Query 2")
        router_with_cache.route("Query 3")

        # Clear cache
        router_with_cache.clear_cache()

        # Should have no cache files
        cache_files = list(ROUTING_CACHE_DIR.glob("*.json"))
        assert len(cache_files) == 0


class TestStatistics:
    """Test statistics tracking"""

    @pytest.fixture
    def router(self):
        router = QueryRouter(method="pattern", cache_decisions=False, log_decisions=False)
        router.reset_stats()
        return router

    def test_stats_tracking(self, router):
        """Test statistics are tracked correctly"""
        # Route different query types
        router.route("What is X?")  # Factual
        router.route("How does X work?")  # Conceptual
        router.route("How to do X?")  # Procedural

        stats = router.get_stats()

        assert stats["total_queries"] == 3
        assert stats["classifications"]["factual"] == 1
        assert stats["classifications"]["conceptual"] == 1
        assert stats["classifications"]["procedural"] == 1
        assert stats["avg_routing_time_ms"] > 0

    def test_cache_hit_rate(self, router):
        """Test cache hit rate calculation"""
        router_cached = QueryRouter(method="pattern", cache_decisions=True, log_decisions=False)
        router_cached.clear_cache()
        router_cached.reset_stats()

        # Route same query twice
        query = "What is cache hit rate?"
        router_cached.route(query)  # Miss
        router_cached.route(query)  # Hit

        stats = router_cached.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate"] == 0.5

        router_cached.clear_cache()

    def test_reset_stats(self, router):
        """Test statistics reset"""
        # Generate some stats
        router.route("Test query")

        # Reset
        router.reset_stats()

        stats = router.get_stats()
        assert stats["total_queries"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0


class TestUtilityFunctions:
    """Test utility functions"""

    def test_is_enabled_default(self):
        """Test is_enabled defaults to False"""
        # Ensure env var not set
        if "ENABLE_QUERY_ROUTING" in os.environ:
            del os.environ["ENABLE_QUERY_ROUTING"]

        assert is_enabled() is False

    def test_is_enabled_when_set(self):
        """Test is_enabled returns True when set"""
        os.environ["ENABLE_QUERY_ROUTING"] = "1"
        assert is_enabled() is True

        # Cleanup
        del os.environ["ENABLE_QUERY_ROUTING"]


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def router(self):
        return QueryRouter(method="pattern", cache_decisions=False, log_decisions=False)

    def test_empty_query(self, router):
        """Test empty query handling"""
        result = router.route("")
        assert result.query_type == QueryType.UNKNOWN
        assert result.confidence == 0.0

    def test_very_long_query(self, router):
        """Test very long query"""
        long_query = "What is " + " ".join(["machine learning"] * 100) + "?"
        result = router.route(long_query)
        # Should still classify as factual
        assert result.query_type == QueryType.FACTUAL

    def test_special_characters(self, router):
        """Test query with special characters"""
        query = "What is @#$%^&*() in Python?"
        result = router.route(query)
        # Should still classify (factual)
        assert result.query_type == QueryType.FACTUAL

    def test_non_english_query(self, router):
        """Test non-English query (should fallback gracefully)"""
        query = "¿Qué es machine learning?"
        result = router.route(query)
        # May classify as UNKNOWN or fall back to pattern matching
        assert isinstance(result.query_type, QueryType)

    def test_invalid_method(self):
        """Test invalid routing method"""
        # Should default to 'pattern' with warning
        router = QueryRouter(method="invalid_method", log_decisions=False)
        assert router.method == "pattern"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
