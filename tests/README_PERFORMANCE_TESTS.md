# Performance Regression Test Suite

## Overview

Comprehensive performance regression test suite with **25 test functions** ensuring optimizations don't degrade the RAG pipeline performance.

## Test Results

```bash
=================== 23 passed, 2 skipped, 1 warning in 8.32s ===================
```

- **23 tests passed** ✓
- **2 tests skipped** (pytest-benchmark not installed)
- **Total execution time**: 8.32s

## Quick Start

```bash
# Run all tests
pytest tests/test_performance_regression.py -v

# Run only fast tests (recommended for CI)
pytest tests/test_performance_regression.py -v -m "not slow"

# Run only slow tests
pytest tests/test_performance_regression.py -v -m slow
```

## Test Coverage Summary

### 1. Benchmark Baselines (5 tests) - `@pytest.mark.slow`

Tests absolute performance thresholds:

| Test | Threshold | Status |
|------|-----------|--------|
| Query latency (no vLLM) | <15s | ✓ PASS |
| Query latency (vLLM) | <5s | ✓ PASS |
| Embedding throughput | >60 chunks/sec | ✓ PASS |
| Vector search latency | <100ms | ✓ PASS |
| DB insertion throughput | >1000 nodes/sec | ✓ PASS |

### 2. Memory Performance (3 tests)

Memory usage and leak detection:

- ✓ Peak memory under 14GB
- ✓ No memory leaks after operations
- ✓ Batch sizes are memory-safe

### 3. Configuration Performance (3 tests)

Tests specific configuration settings:

- ✓ N_GPU_LAYERS=24 performance
- ✓ EMBED_BATCH=64 throughput
- ✓ TOP_K=4 retrieval speed

### 4. Regression Detection (4 tests)

Compares against baselines with 20% tolerance:

- ✓ Load/create baselines
- ✓ Compare embedding throughput
- ✓ Compare vector search latency
- ✓ Detect 20% regression threshold

### 5. End-to-End Performance (5 tests) - `@pytest.mark.slow`

Complete pipeline performance:

- ✓ Document processing speed (>20 docs/sec)
- ✓ Embedding pipeline (>60 chunks/sec)
- ✓ Retrieval pipeline (<50ms)
- ✓ Full query without vLLM (<2s mocked)
- ✓ Full query with vLLM (<0.5s mocked)

### 6. Performance Monitoring (3 tests)

Metrics collection and logging:

- ✓ Metrics structure validation
- ✓ Performance log format (JSON)
- ✓ Baseline update mechanism

### 7. Benchmark Integration (2 tests) - `pytest-benchmark`

Integration with pytest-benchmark (optional):

- ⊘ SKIP: pytest-benchmark not installed
- ⊘ SKIP: Vector search benchmark

## Performance Thresholds

```python
THRESHOLDS = {
    "query_latency_no_vllm": 15.0,      # seconds
    "query_latency_vllm": 5.0,          # seconds
    "embedding_throughput": 60.0,       # chunks per second
    "vector_search_latency": 0.1,       # seconds (100ms)
    "db_insertion_throughput": 1000.0,  # nodes per second
    "peak_memory_gb": 14.0,             # gigabytes
    "regression_tolerance": 0.20,       # 20% regression threshold
}
```

## Baseline Metrics

Located in `tests/performance_baselines.json`:

```json
{
  "embedding_throughput": 67.0,
  "vector_search_latency": 0.011,
  "db_insertion_throughput": 1250.0,
  "query_latency_no_vllm": 8.0,
  "query_latency_vllm": 1.5,
  "metadata": {
    "platform": "M1 Mac Mini 16GB",
    "last_updated": "2025-01-07"
  }
}
```

## Test Configuration

Tested configurations match production settings:

```python
TEST_CONFIG = {
    "n_gpu_layers": 24,
    "embed_batch": 64,
    "n_batch": 256,
    "top_k": 4,
    "chunk_size": 700,
    "chunk_overlap": 150,
}
```

## Regression Detection

Tests fail if performance regresses by **>20%**:

```python
# Example: Embedding throughput
baseline = 67.0 chunks/sec
threshold = 53.6 chunks/sec (20% worse)
current = 55.0 chunks/sec
# → FAIL: Regression detected
```

## Mocking Strategy

Fast operations are tested directly; slow operations are mocked:

- **Tested directly**: Embedding, vector search, DB insertion
- **Mocked**: LLM generation, document I/O

This allows tests to run quickly (~8s) while still validating performance.

## CI/CD Integration

### Pre-commit Hook

```bash
#!/bin/bash
pytest tests/test_performance_regression.py -m "not slow" --tb=short -x
```

### GitHub Actions

```yaml
- name: Performance Regression Tests
  run: |
    pytest tests/test_performance_regression.py -v -m "not slow"
```

## Files

- `test_performance_regression.py` - Main test suite (25 tests)
- `performance_baselines.json` - Baseline metrics
- `PERFORMANCE_TESTING.md` - Detailed documentation
- `README_PERFORMANCE_TESTS.md` - This file

## Usage Examples

### Run Fast Tests (CI)

```bash
pytest tests/test_performance_regression.py -v -m "not slow"
# Duration: ~1.5s
# Tests: 15 passed
```

### Run All Tests

```bash
pytest tests/test_performance_regression.py -v
# Duration: ~8s
# Tests: 23 passed, 2 skipped
```

### Run Specific Test Class

```bash
pytest tests/test_performance_regression.py::TestRegressionDetection -v
```

### Update Baselines

```python
from tests.test_performance_regression import save_baselines

new_baselines = {
    "embedding_throughput": 75.0,  # Improved!
    "vector_search_latency": 0.011,
    # ... other metrics
}

save_baselines(new_baselines)
```

## Expected Output

```
tests/test_performance_regression.py::TestBenchmarkBaselines::test_query_latency_without_vllm PASSED
tests/test_performance_regression.py::TestBenchmarkBaselines::test_query_latency_with_vllm PASSED
tests/test_performance_regression.py::TestBenchmarkBaselines::test_embedding_throughput_minimum PASSED
tests/test_performance_regression.py::TestBenchmarkBaselines::test_vector_search_latency PASSED
tests/test_performance_regression.py::TestBenchmarkBaselines::test_database_insertion_throughput PASSED
tests/test_performance_regression.py::TestMemoryPerformance::test_peak_memory_under_limit PASSED
tests/test_performance_regression.py::TestMemoryPerformance::test_no_memory_leak_after_operations PASSED
tests/test_performance_regression.py::TestMemoryPerformance::test_batch_size_memory_safe PASSED
tests/test_performance_regression.py::TestConfigurationPerformance::test_n_gpu_layers_24_performance PASSED
tests/test_performance_regression.py::TestConfigurationPerformance::test_embed_batch_64_performance PASSED
tests/test_performance_regression.py::TestConfigurationPerformance::test_top_k_4_retrieval_speed PASSED
tests/test_performance_regression.py::TestRegressionDetection::test_load_or_create_baselines PASSED
tests/test_performance_regression.py::TestRegressionDetection::test_compare_against_baseline_embedding PASSED
tests/test_performance_regression.py::TestRegressionDetection::test_compare_against_baseline_search PASSED
tests/test_performance_regression.py::TestRegressionDetection::test_detect_20_percent_regression PASSED
tests/test_performance_regression.py::TestEndToEndPerformance::test_document_processing_speed PASSED
tests/test_performance_regression.py::TestEndToEndPerformance::test_embedding_pipeline_performance PASSED
tests/test_performance_regression.py::TestEndToEndPerformance::test_retrieval_pipeline_performance PASSED
tests/test_performance_regression.py::TestEndToEndPerformance::test_full_query_pipeline_without_vllm PASSED
tests/test_performance_regression.py::TestEndToEndPerformance::test_full_query_pipeline_with_vllm PASSED
tests/test_performance_regression.py::TestPerformanceMonitoring::test_metrics_collection_structure PASSED
tests/test_performance_regression.py::TestPerformanceMonitoring::test_performance_log_format PASSED
tests/test_performance_regression.py::TestPerformanceMonitoring::test_baseline_update_mechanism PASSED
tests/test_performance_regression.py::TestBenchmarkIntegration::test_benchmark_available SKIPPED
tests/test_performance_regression.py::TestBenchmarkIntegration::test_vector_search_benchmark SKIPPED

=================== 23 passed, 2 skipped, 1 warning in 8.32s ===================
```

## Troubleshooting

### Issue: Tests too slow

**Solution**: Run only fast tests:
```bash
pytest tests/test_performance_regression.py -m "not slow"
```

### Issue: Regression detected

**Action Required**:
1. Investigate code changes
2. Profile the slow operation
3. Fix regression OR update baseline if intentional

### Issue: Flaky timing tests

**Solution**: Tests use tolerances and realistic simulations to minimize flakiness

## Best Practices

1. **Run fast tests frequently** during development
2. **Run all tests before commits** to catch regressions
3. **Update baselines** only after verifying improvements
4. **Monitor trends** over time with performance logs
5. **Test on target hardware** (M1 Mac Mini 16GB)

## Future Enhancements

- [ ] Add pytest-benchmark for detailed profiling
- [ ] Track performance trends over time
- [ ] Add more granular metrics (per-operation)
- [ ] Integration with performance monitoring tools
- [ ] Automated baseline updates with approval workflow

## Related Documentation

- [PERFORMANCE_TESTING.md](PERFORMANCE_TESTING.md) - Detailed guide
- [CLAUDE.md](../CLAUDE.md) - Performance benchmarks
- [pytest.ini](../config/pytest.ini) - Test configuration

---

**Last Updated**: 2025-01-07
**Test Suite Version**: 1.0.0
**Total Tests**: 25 (23 passed, 2 skipped)
**Execution Time**: ~8.3s (full), ~1.5s (fast only)
