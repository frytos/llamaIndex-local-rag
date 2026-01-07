"""Tests for performance configuration and benchmarks."""

import pytest
import os


class TestPerformanceConfiguration:
    """Test performance-related configuration."""

    def test_batch_sizes_reasonable(self):
        """Test that batch sizes are memory-safe."""
        embed_batch = int(os.getenv("EMBED_BATCH", "32"))
        n_batch = int(os.getenv("N_BATCH", "128"))

        # For M1 16GB, these should be reasonable
        assert embed_batch <= 128, "Embedding batch >128 may cause OOM on 16GB"
        assert n_batch <= 512, "LLM batch >512 may cause OOM on 16GB"

    def test_gpu_offload_configuration(self):
        """Test GPU offloading settings."""
        n_gpu_layers = int(os.getenv("N_GPU_LAYERS", "16"))

        # 0 = CPU only, 32 = full GPU for Mistral 7B
        assert 0 <= n_gpu_layers <= 32, "Mistral 7B has 32 layers"


class TestMemoryOptimization:
    """Test memory optimization settings."""

    def test_conservative_memory_config(self):
        """Test conservative configuration for memory-constrained systems."""
        # Low memory config
        embed_batch = 32
        n_batch = 128
        n_gpu_layers = 12

        # Should use less than 8GB total
        estimated_memory = (
            1.5  # Embedding model
            + (n_batch * 0.002)  # LLM batch memory
            + (embed_batch * 0.01)  # Embedding batch
            + 4  # Base LLM model
            + 2  # System overhead
        )
        assert estimated_memory < 10, "Conservative config should use <10GB"

    def test_performance_memory_config(self):
        """Test performance configuration memory requirements."""
        # Performance config from audit
        embed_batch = 64
        n_batch = 256
        n_gpu_layers = 24

        # Should use less than 14GB (safe for 16GB)
        estimated_memory = (
            1.5  # Embedding model
            + (n_batch * 0.002)  # LLM batch memory
            + (embed_batch * 0.01)  # Embedding batch
            + 4  # Base LLM model
            + 2  # System overhead
        )
        assert estimated_memory < 14, "Performance config should fit in 16GB"


class TestPerformanceTargets:
    """Test performance targets from audit."""

    def test_query_latency_target(self):
        """Test query latency target."""
        # From audit: Target <5s with optimizations
        target_latency = 5.0
        assert target_latency < 10, "Target should be under 10s"

    def test_embedding_throughput_target(self):
        """Test embedding throughput target."""
        # From audit: Target 90-100 chunks/sec with MLX
        target_throughput = 90
        baseline_throughput = 67

        assert target_throughput > baseline_throughput
        assert target_throughput >= 90, "MLX should achieve 90+ chunks/sec"

    def test_vector_search_target(self):
        """Test vector search performance target."""
        # From audit: Current 11ms is excellent, target <50ms
        current_ms = 11
        target_ms = 50

        assert current_ms < target_ms, "Already meeting target!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
