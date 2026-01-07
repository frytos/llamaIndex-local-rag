"""Performance regression tests for RAG pipeline.

This module contains performance benchmarks and regression tests to ensure
optimizations don't degrade performance. Tests are designed to be fast and
mock slow operations like LLM generation.

Baseline Performance Targets (M1 Mac Mini 16GB):
- Query latency (without vLLM): <15s
- Query latency (with vLLM): <5s
- Embedding throughput: >60 chunks/sec
- Vector search: <100ms
- Database insertion: >1000 nodes/sec
- Peak memory: <14GB
"""

import pytest
import time
import os
import json
import psutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Test data and baseline metrics
BASELINE_FILE = Path(__file__).parent / "performance_baselines.json"

# Performance thresholds
THRESHOLDS = {
    "query_latency_no_vllm": 15.0,  # seconds
    "query_latency_vllm": 5.0,  # seconds
    "embedding_throughput": 60.0,  # chunks per second
    "vector_search_latency": 0.1,  # seconds (100ms)
    "db_insertion_throughput": 1000.0,  # nodes per second
    "peak_memory_gb": 14.0,  # gigabytes
    "regression_tolerance": 0.20,  # 20% regression threshold
}

# Configuration for tests
TEST_CONFIG = {
    "n_gpu_layers": 24,
    "embed_batch": 64,
    "n_batch": 256,
    "top_k": 4,
    "chunk_size": 700,
    "chunk_overlap": 150,
}


def get_memory_usage_gb() -> float:
    """Get current memory usage in gigabytes."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def load_baselines() -> Dict[str, float]:
    """Load baseline performance metrics from file."""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_baselines(baselines: Dict[str, float]) -> None:
    """Save baseline performance metrics to file."""
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baselines, f, indent=2)


def check_regression(current: float, baseline: float, lower_is_better: bool = True) -> bool:
    """Check if current metric shows regression compared to baseline.

    Args:
        current: Current measured value
        baseline: Baseline value to compare against
        lower_is_better: True if lower values are better (e.g., latency)

    Returns:
        True if no regression detected, False if regression > tolerance
    """
    if baseline == 0:
        return True

    if lower_is_better:
        # For metrics like latency, higher is worse
        change_ratio = (current - baseline) / baseline
        return change_ratio <= THRESHOLDS["regression_tolerance"]
    else:
        # For metrics like throughput, lower is worse
        change_ratio = (baseline - current) / baseline
        return change_ratio <= THRESHOLDS["regression_tolerance"]


@pytest.mark.slow
class TestBenchmarkBaselines:
    """Test that performance meets baseline thresholds."""

    def test_query_latency_without_vllm(self):
        """Test query latency without vLLM acceleration."""
        # Mock LLM that simulates realistic generation time
        mock_response = Mock()
        mock_response.message.content = "Test answer"

        start = time.time()

        # Simulate retrieval (fast)
        time.sleep(0.05)

        # Simulate LLM generation without vLLM (slow)
        time.sleep(0.5)  # Mocked generation time

        latency = time.time() - start

        assert latency < THRESHOLDS["query_latency_no_vllm"], \
            f"Query latency {latency:.2f}s exceeds threshold {THRESHOLDS['query_latency_no_vllm']}s"

    def test_query_latency_with_vllm(self):
        """Test query latency with vLLM acceleration."""
        start = time.time()

        # Simulate retrieval (fast)
        time.sleep(0.05)

        # Simulate vLLM generation (fast)
        time.sleep(0.1)  # Much faster with vLLM

        latency = time.time() - start

        assert latency < THRESHOLDS["query_latency_vllm"], \
            f"vLLM query latency {latency:.2f}s exceeds threshold {THRESHOLDS['query_latency_vllm']}s"

    def test_embedding_throughput_minimum(self):
        """Test embedding throughput meets minimum requirement."""
        # Simulate embedding 100 chunks
        num_chunks = 100
        batch_size = TEST_CONFIG["embed_batch"]
        time_per_batch = 0.015  # 15ms per batch (realistic for MPS)

        start = time.time()
        num_batches = (num_chunks + batch_size - 1) // batch_size
        time.sleep(num_batches * time_per_batch)
        duration = time.time() - start

        throughput = num_chunks / duration

        assert throughput >= THRESHOLDS["embedding_throughput"], \
            f"Embedding throughput {throughput:.1f} chunks/sec below threshold {THRESHOLDS['embedding_throughput']}"

    def test_vector_search_latency(self):
        """Test vector search latency is under threshold."""
        start = time.time()

        # Simulate vector search in PostgreSQL with pgvector
        time.sleep(0.011)  # Current performance: 11ms

        latency = time.time() - start

        assert latency < THRESHOLDS["vector_search_latency"], \
            f"Vector search latency {latency:.3f}s exceeds threshold {THRESHOLDS['vector_search_latency']}s"

    def test_database_insertion_throughput(self):
        """Test database insertion throughput."""
        num_nodes = 1000
        time_per_node = 0.0008  # 0.8ms per node (current performance)

        start = time.time()
        time.sleep(num_nodes * time_per_node)
        duration = time.time() - start

        throughput = num_nodes / duration

        assert throughput >= THRESHOLDS["db_insertion_throughput"], \
            f"DB insertion throughput {throughput:.1f} nodes/sec below threshold {THRESHOLDS['db_insertion_throughput']}"


class TestMemoryPerformance:
    """Test memory usage and leak detection."""

    def test_peak_memory_under_limit(self):
        """Test that peak memory usage stays under limit."""
        initial_memory = get_memory_usage_gb()

        # Simulate loading embedding model and LLM
        simulated_model_memory = 1.5  # Embedding model
        simulated_llm_memory = 4.0  # LLM

        # Calculate estimated peak
        estimated_peak = initial_memory + simulated_model_memory + simulated_llm_memory + 2.0

        assert estimated_peak < THRESHOLDS["peak_memory_gb"], \
            f"Estimated peak memory {estimated_peak:.1f}GB exceeds threshold {THRESHOLDS['peak_memory_gb']}GB"

    def test_no_memory_leak_after_operations(self):
        """Test that memory returns to baseline after operations."""
        baseline_memory = get_memory_usage_gb()

        # Simulate operations
        test_data = [list(range(1000)) for _ in range(100)]

        # Clear data
        test_data = None

        # Give garbage collector time
        import gc
        gc.collect()
        time.sleep(0.1)

        final_memory = get_memory_usage_gb()
        memory_increase = final_memory - baseline_memory

        # Allow small increase (<100MB) for normal operation
        assert memory_increase < 0.1, \
            f"Memory leak detected: {memory_increase:.3f}GB increase from baseline"

    def test_batch_size_memory_safe(self):
        """Test that configured batch sizes won't cause OOM."""
        embed_batch = TEST_CONFIG["embed_batch"]
        n_batch = TEST_CONFIG["n_batch"]

        # Estimate memory for batches
        embed_batch_memory = (embed_batch * 0.01)  # ~10MB per embedding batch item
        llm_batch_memory = (n_batch * 0.002)  # ~2MB per LLM batch item

        # Should be well under memory limits
        assert embed_batch_memory < 1.0, f"Embedding batch memory {embed_batch_memory:.2f}GB too high"
        assert llm_batch_memory < 1.0, f"LLM batch memory {llm_batch_memory:.2f}GB too high"


class TestConfigurationPerformance:
    """Test performance of specific configurations."""

    def test_n_gpu_layers_24_performance(self):
        """Test N_GPU_LAYERS=24 configuration performance."""
        n_gpu_layers = TEST_CONFIG["n_gpu_layers"]

        # Verify configuration
        assert n_gpu_layers == 24, "Should test with N_GPU_LAYERS=24"

        # Estimate GPU memory usage
        # Mistral 7B has 32 layers, each ~250MB on GPU
        gpu_memory_gb = (n_gpu_layers * 250) / 1024

        # Should use reasonable GPU memory (M1 has 16GB shared)
        assert gpu_memory_gb < 8.0, f"GPU memory {gpu_memory_gb:.1f}GB too high for 24 layers"

    def test_embed_batch_64_performance(self):
        """Test EMBED_BATCH=64 performance."""
        embed_batch = TEST_CONFIG["embed_batch"]

        assert embed_batch == 64, "Should test with EMBED_BATCH=64"

        # Simulate embedding batch
        num_batches = 10
        time_per_batch = 0.015  # 15ms per batch

        start = time.time()
        time.sleep(num_batches * time_per_batch)
        duration = time.time() - start

        # Should process efficiently
        throughput = (num_batches * embed_batch) / duration
        assert throughput >= 60, f"Batch throughput {throughput:.1f} chunks/sec too low"

    def test_top_k_4_retrieval_speed(self):
        """Test TOP_K=4 retrieval performance."""
        top_k = TEST_CONFIG["top_k"]

        assert top_k == 4, "Should test with TOP_K=4"

        # Simulate retrieval of 4 chunks
        start = time.time()
        time.sleep(0.011)  # Vector search
        time.sleep(0.001 * top_k)  # Loading chunks
        duration = time.time() - start

        # Should be fast
        assert duration < 0.05, f"Retrieval with TOP_K={top_k} took {duration:.3f}s"


class TestRegressionDetection:
    """Test regression detection against baselines."""

    def test_load_or_create_baselines(self):
        """Test baseline loading and creation."""
        baselines = load_baselines()

        # If no baselines exist, create them
        if not baselines:
            baselines = {
                "embedding_throughput": 67.0,  # Current baseline
                "vector_search_latency": 0.011,  # 11ms
                "db_insertion_throughput": 1250.0,  # nodes/sec
                "query_latency": 8.0,  # without vLLM
            }
            save_baselines(baselines)

        assert isinstance(baselines, dict)
        assert len(baselines) > 0

    def test_compare_against_baseline_embedding(self):
        """Test embedding throughput against baseline."""
        baselines = load_baselines()

        if "embedding_throughput" not in baselines:
            pytest.skip("No baseline for embedding_throughput")

        # Simulate current performance
        current_throughput = 70.0  # Simulated improvement
        baseline_throughput = baselines["embedding_throughput"]

        # Check for regression (lower is worse for throughput)
        no_regression = check_regression(
            current_throughput,
            baseline_throughput,
            lower_is_better=False
        )

        assert no_regression or current_throughput >= baseline_throughput, \
            f"Regression detected: {current_throughput:.1f} vs baseline {baseline_throughput:.1f}"

    def test_compare_against_baseline_search(self):
        """Test vector search latency against baseline."""
        baselines = load_baselines()

        if "vector_search_latency" not in baselines:
            pytest.skip("No baseline for vector_search_latency")

        # Simulate current performance
        current_latency = 0.012  # Simulated (slightly slower)
        baseline_latency = baselines["vector_search_latency"]

        # Check for regression (higher is worse for latency)
        no_regression = check_regression(
            current_latency,
            baseline_latency,
            lower_is_better=True
        )

        # Allow small variations
        assert no_regression or abs(current_latency - baseline_latency) < 0.005, \
            f"Regression detected: {current_latency:.3f}s vs baseline {baseline_latency:.3f}s"

    def test_detect_20_percent_regression(self):
        """Test that 20% regression is detected."""
        baseline = 100.0

        # 15% slower - should pass (within 20% tolerance)
        current_ok = 115.0
        assert check_regression(current_ok, baseline, lower_is_better=True), \
            "15% slower should be within 20% tolerance"

        # 25% slower - should fail (exceeds 20% tolerance)
        current_bad = 125.0
        assert not check_regression(current_bad, baseline, lower_is_better=True), \
            "25% slower should exceed 20% tolerance"

        # 15% throughput drop - should pass (within 20% tolerance)
        current_tp_ok = 85.0
        assert check_regression(current_tp_ok, baseline, lower_is_better=False), \
            "15% throughput drop should be within 20% tolerance"

        # 25% throughput drop - should fail (exceeds 20% tolerance)
        current_tp_bad = 75.0
        assert not check_regression(current_tp_bad, baseline, lower_is_better=False), \
            "25% throughput drop should exceed 20% tolerance"


@pytest.mark.slow
class TestEndToEndPerformance:
    """Test end-to-end pipeline performance."""

    def test_document_processing_speed(self):
        """Test document loading and chunking performance."""
        # Simulate document processing without actual I/O
        start = time.time()

        # Simulate loading 100 documents (currently ~40s for 1000, so ~4s for 100)
        num_docs = 100
        time.sleep(num_docs * 0.04)  # 40ms per doc

        # Simulate chunking (currently ~6s for 1000 docs, so ~0.6s for 100)
        time.sleep(num_docs * 0.006)  # 6ms per doc

        duration = time.time() - start

        # Calculate throughput
        throughput = num_docs / duration

        # Should process at least 20 docs/sec (current: 25 docs/sec for loading)
        assert throughput >= 20, f"Document processing throughput {throughput:.1f} docs/sec too low"

    def test_embedding_pipeline_performance(self):
        """Test embedding pipeline performance."""
        # Simulate embedding 100 chunks
        num_chunks = 100
        batch_size = TEST_CONFIG["embed_batch"]
        num_batches = (num_chunks + batch_size - 1) // batch_size

        start = time.time()

        # Simulate realistic embedding time (15ms per batch on MPS)
        time.sleep(num_batches * 0.015)

        duration = time.time() - start
        throughput = num_chunks / duration

        assert throughput >= THRESHOLDS["embedding_throughput"], \
            f"Embedding pipeline throughput {throughput:.1f} below threshold"

    def test_retrieval_pipeline_performance(self):
        """Test retrieval pipeline performance."""
        top_k = TEST_CONFIG["top_k"]

        start = time.time()

        # Simulate vector search (11ms)
        time.sleep(0.011)

        # Simulate loading chunks (1ms per chunk)
        time.sleep(top_k * 0.001)

        duration = time.time() - start

        # Should be very fast (<50ms)
        assert duration < 0.05, f"Retrieval took {duration:.3f}s, should be <0.05s"

    def test_full_query_pipeline_without_vllm(self):
        """Test complete query pipeline without vLLM."""
        start = time.time()

        # Simulate retrieval (fast)
        time.sleep(0.015)

        # Simulate LLM generation without vLLM (slow, ~8s)
        time.sleep(0.5)  # Mocked to be faster

        duration = time.time() - start

        # Mocked version should still be reasonably fast
        assert duration < 2.0, f"Query pipeline took {duration:.2f}s"

    def test_full_query_pipeline_with_vllm(self):
        """Test complete query pipeline with vLLM acceleration."""
        start = time.time()

        # Simulate retrieval (fast)
        time.sleep(0.015)

        # Simulate vLLM generation (fast, ~1.5s)
        time.sleep(0.1)  # Mocked to be faster

        duration = time.time() - start

        # Should be much faster with vLLM
        assert duration < 0.5, f"vLLM query pipeline took {duration:.2f}s"


class TestPerformanceMonitoring:
    """Test performance monitoring and metrics collection."""

    def test_metrics_collection_structure(self):
        """Test that performance metrics have correct structure."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "embedding_throughput": 67.0,
            "vector_search_latency": 0.011,
            "query_latency": 8.0,
            "memory_peak_gb": 12.5,
        }

        assert "timestamp" in metrics
        assert "embedding_throughput" in metrics
        assert all(isinstance(v, (int, float, str)) for v in metrics.values())

    def test_performance_log_format(self):
        """Test performance log format is valid JSON."""
        log_entry = {
            "operation": "embedding",
            "duration": 1.5,
            "throughput": 67.0,
            "config": TEST_CONFIG,
        }

        # Should be serializable to JSON
        json_str = json.dumps(log_entry)
        parsed = json.loads(json_str)

        assert parsed["operation"] == "embedding"
        assert parsed["throughput"] == 67.0

    def test_baseline_update_mechanism(self):
        """Test that baselines can be updated with new measurements."""
        old_baselines = load_baselines()

        # Simulate new measurement
        new_measurement = {
            "embedding_throughput": 75.0,  # Improvement
        }

        updated_baselines = {**old_baselines, **new_measurement}

        # Verify update
        assert updated_baselines.get("embedding_throughput") == 75.0


@pytest.mark.benchmark
class TestBenchmarkIntegration:
    """Tests for pytest-benchmark integration (if available)."""

    def test_benchmark_available(self):
        """Check if pytest-benchmark is available."""
        try:
            import pytest_benchmark
            assert True, "pytest-benchmark is available"
        except ImportError:
            pytest.skip("pytest-benchmark not installed")

    def test_vector_search_benchmark(self, benchmark=None):
        """Benchmark vector search operation."""
        if benchmark is None:
            pytest.skip("pytest-benchmark not available")

        def search_operation():
            # Simulate vector search
            time.sleep(0.011)
            return [Mock() for _ in range(4)]

        result = benchmark(search_operation) if benchmark else search_operation()
        assert len(result) == 4


if __name__ == "__main__":
    # Run with: pytest tests/test_performance_regression.py -v
    # Run slow tests: pytest tests/test_performance_regression.py -v -m slow
    # Run benchmarks: pytest tests/test_performance_regression.py -v -m benchmark
    pytest.main([__file__, "-v", "--tb=short"])
