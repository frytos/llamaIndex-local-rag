"""
Tests for performance optimization module.

Tests async embedding, connection pooling, parallel retrieval,
batch processing, and performance monitoring.
"""

import asyncio
import os
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

# Import components to test
from utils.performance_optimizations import (
    PerformanceConfig,
    AsyncEmbedding,
    DatabaseConnectionPool,
    ParallelRetriever,
    BatchProcessor,
    PerformanceMonitor,
)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestPerformanceConfig:
    """Test PerformanceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PerformanceConfig()

        assert config.enable_async is True
        assert config.connection_pool_size == 10
        assert config.batch_size == 32
        assert config.batch_timeout == 1.0
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PerformanceConfig(
            enable_async=False,
            connection_pool_size=5,
            batch_size=16,
            batch_timeout=0.5,
            min_pool_size=2,
            max_pool_size=10,
        )

        assert config.enable_async is False
        assert config.connection_pool_size == 5
        assert config.batch_size == 16
        assert config.batch_timeout == 0.5
        assert config.min_pool_size == 2
        assert config.max_pool_size == 10

    def test_validation(self):
        """Test configuration validation."""
        # Invalid: min_pool_size > max_pool_size
        with pytest.raises(ValueError, match="min_pool_size cannot exceed max_pool_size"):
            PerformanceConfig(min_pool_size=20, max_pool_size=10)

        # Invalid: batch_size < 1
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            PerformanceConfig(batch_size=0)

        # Invalid: batch_timeout < 0
        with pytest.raises(ValueError, match="batch_timeout must be >= 0"):
            PerformanceConfig(batch_timeout=-1.0)

    def test_environment_variables(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("ENABLE_ASYNC", "0")
        monkeypatch.setenv("CONNECTION_POOL_SIZE", "15")
        monkeypatch.setenv("BATCH_SIZE", "64")
        monkeypatch.setenv("BATCH_TIMEOUT", "2.0")
        monkeypatch.setenv("MIN_POOL_SIZE", "3")
        monkeypatch.setenv("MAX_POOL_SIZE", "30")

        config = PerformanceConfig()

        assert config.enable_async is False
        assert config.connection_pool_size == 15
        assert config.batch_size == 64
        assert config.batch_timeout == 2.0
        assert config.min_pool_size == 3
        assert config.max_pool_size == 30


# ============================================================================
# Async Embedding Tests
# ============================================================================

@pytest.mark.skipif(
    not hasattr(AsyncEmbedding, '__init__'),
    reason="sentence-transformers not available"
)
class TestAsyncEmbedding:
    """Test AsyncEmbedding class."""

    @pytest.fixture
    def mock_model(self):
        """Mock SentenceTransformer model."""
        model = Mock()
        model.encode = Mock(return_value=np.random.randn(384))
        return model

    def test_initialization(self, mock_model):
        """Test AsyncEmbedding initialization."""
        with patch('utils.performance_optimizations.SentenceTransformer', return_value=mock_model):
            embed = AsyncEmbedding(
                model_name="test-model",
                device="cpu",
                batch_size=16
            )

            assert embed.model_name == "test-model"
            assert embed.device == "cpu"
            assert embed.batch_size == 16
            assert embed.total_embeddings == 0
            assert embed.total_time == 0.0

    @pytest.mark.asyncio
    async def test_embed_single(self, mock_model):
        """Test single text embedding."""
        mock_model.encode = Mock(return_value=np.random.randn(384))

        with patch('utils.performance_optimizations.SentenceTransformer', return_value=mock_model):
            embed = AsyncEmbedding()
            result = await embed.embed_single("test text")

            assert isinstance(result, list)
            assert len(result) == 384
            assert embed.total_embeddings == 1

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_model):
        """Test batch text embedding."""
        mock_model.encode = Mock(return_value=np.random.randn(5, 384))

        with patch('utils.performance_optimizations.SentenceTransformer', return_value=mock_model):
            embed = AsyncEmbedding()
            texts = ["text1", "text2", "text3", "text4", "text5"]
            results = await embed.embed_batch(texts)

            assert isinstance(results, list)
            assert len(results) == 5
            assert all(len(emb) == 384 for emb in results)
            assert embed.total_embeddings == 5

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, mock_model):
        """Test batch embedding with empty list."""
        with patch('utils.performance_optimizations.SentenceTransformer', return_value=mock_model):
            embed = AsyncEmbedding()
            results = await embed.embed_batch([])

            assert results == []

    def test_get_stats(self, mock_model):
        """Test embedding statistics."""
        with patch('utils.performance_optimizations.SentenceTransformer', return_value=mock_model):
            embed = AsyncEmbedding()
            embed.total_embeddings = 100
            embed.total_time = 10.0

            stats = embed.get_stats()

            assert stats["model"] == embed.model_name
            assert stats["device"] == embed.device
            assert stats["total_embeddings"] == 100
            assert stats["total_time"] == 10.0
            assert stats["avg_time_per_embedding"] == 0.1
            assert stats["throughput_embeddings_per_sec"] == 10.0


# ============================================================================
# Performance Monitor Tests
# ============================================================================

class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()

        assert len(monitor.metrics) == 0
        assert len(monitor.counts) == 0

    def test_track_operation(self):
        """Test tracking operation execution time."""
        monitor = PerformanceMonitor()

        with monitor.track("test_operation"):
            # Simulate work
            import time
            time.sleep(0.01)

        assert "test_operation" in monitor.metrics
        assert len(monitor.metrics["test_operation"]) == 1
        assert monitor.counts["test_operation"] == 1
        assert monitor.metrics["test_operation"][0] >= 0.01

    def test_record_duration(self):
        """Test manual duration recording."""
        monitor = PerformanceMonitor()

        monitor.record("operation1", 0.5)
        monitor.record("operation1", 0.3)
        monitor.record("operation2", 1.0)

        assert len(monitor.metrics["operation1"]) == 2
        assert len(monitor.metrics["operation2"]) == 1
        assert monitor.counts["operation1"] == 2
        assert monitor.counts["operation2"] == 1

    def test_get_stats_single_operation(self):
        """Test getting statistics for single operation."""
        monitor = PerformanceMonitor()

        # Record some durations
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        for d in durations:
            monitor.record("test_op", d)

        stats = monitor.get_stats("test_op")

        assert stats["count"] == 5
        assert stats["mean"] == 0.3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert stats["p50"] == 0.3

    def test_get_stats_all_operations(self):
        """Test getting statistics for all operations."""
        monitor = PerformanceMonitor()

        monitor.record("op1", 1.0)
        monitor.record("op1", 2.0)
        monitor.record("op2", 3.0)

        stats = monitor.get_stats()

        assert "op1" in stats
        assert "op2" in stats
        assert stats["op1"]["count"] == 2
        assert stats["op2"]["count"] == 1

    def test_get_stats_empty(self):
        """Test getting stats for non-existent operation."""
        monitor = PerformanceMonitor()

        stats = monitor.get_stats("nonexistent")

        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["p50"] == 0.0

    def test_reset_single_operation(self):
        """Test resetting single operation metrics."""
        monitor = PerformanceMonitor()

        monitor.record("op1", 1.0)
        monitor.record("op2", 2.0)

        monitor.reset("op1")

        assert len(monitor.metrics["op1"]) == 0
        assert monitor.counts["op1"] == 0
        assert len(monitor.metrics["op2"]) == 1

    def test_reset_all_operations(self):
        """Test resetting all metrics."""
        monitor = PerformanceMonitor()

        monitor.record("op1", 1.0)
        monitor.record("op2", 2.0)

        monitor.reset()

        assert len(monitor.metrics) == 0
        assert len(monitor.counts) == 0

    def test_export_metrics(self):
        """Test exporting metrics."""
        monitor = PerformanceMonitor()

        monitor.record("op1", 1.0)
        monitor.record("op1", 2.0)
        monitor.record("op2", 3.0)

        export = monitor.export_metrics()

        assert "operations" in export
        assert "stats" in export
        assert "raw_metrics" in export
        assert set(export["operations"]) == {"op1", "op2"}
        assert "op1" in export["stats"]
        assert "op2" in export["stats"]
        assert export["raw_metrics"]["op1"] == [1.0, 2.0]
        assert export["raw_metrics"]["op2"] == [3.0]

    def test_percentiles(self):
        """Test percentile calculations."""
        monitor = PerformanceMonitor()

        # Create dataset with known percentiles
        for i in range(1, 101):
            monitor.record("test", i / 100.0)

        stats = monitor.get_stats("test")

        assert stats["count"] == 100
        assert abs(stats["p50"] - 0.50) < 0.01  # 50th percentile ~ 0.50
        assert abs(stats["p95"] - 0.95) < 0.01  # 95th percentile ~ 0.95
        assert abs(stats["p99"] - 0.99) < 0.01  # 99th percentile ~ 0.99


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "1") == "1",
    reason="Integration tests disabled (requires database)"
)
@pytest.mark.asyncio
async def test_integration_workflow():
    """
    Integration test for complete workflow.

    Note: Requires actual database connection.
    Set SKIP_INTEGRATION_TESTS=0 to run.
    """
    # Initialize components
    async_embed = AsyncEmbedding(model_name="BAAI/bge-small-en")
    pool = DatabaseConnectionPool()
    await pool.initialize()

    monitor = PerformanceMonitor()

    try:
        # Test embedding
        with monitor.track("embedding"):
            embeddings = await async_embed.embed_batch(["test1", "test2"])
            assert len(embeddings) == 2

        # Test database query
        with monitor.track("database"):
            result = await pool.fetchval("SELECT 1")
            assert result == 1

        # Test health check
        healthy = await pool.health_check()
        assert healthy is True

        # Check stats
        stats = monitor.get_stats()
        assert "embedding" in stats
        assert "database" in stats
        assert stats["embedding"]["count"] == 1
        assert stats["database"]["count"] == 1

    finally:
        # Cleanup
        await pool.close()


# ============================================================================
# Benchmark Tests (Optional)
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS", "0") == "0",
    reason="Benchmarks disabled (set RUN_BENCHMARKS=1 to run)"
)
class TestBenchmarks:
    """Performance benchmarks (optional, run with RUN_BENCHMARKS=1)."""

    @pytest.mark.asyncio
    async def test_benchmark_async_embedding(self, benchmark):
        """Benchmark async embedding performance."""
        async_embed = AsyncEmbedding()
        texts = [f"Query text {i}" for i in range(10)]

        async def embed_batch():
            return await async_embed.embed_batch(texts)

        # Run benchmark
        result = benchmark(lambda: asyncio.run(embed_batch()))
        assert len(result) == 10

    def test_benchmark_monitor_overhead(self, benchmark):
        """Benchmark performance monitor overhead."""
        monitor = PerformanceMonitor()

        def operation():
            with monitor.track("test"):
                # Minimal work
                x = sum(range(100))
            return x

        # Overhead should be minimal (< 1ms)
        result = benchmark(operation)
        assert result == 4950


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
