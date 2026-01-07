# Performance Regression Testing Guide

## Overview

The performance regression test suite ensures that optimizations and changes to the RAG pipeline don't degrade performance. Tests focus on fast operations and mock slow operations like LLM generation.

## Quick Start

```bash
# Run all performance regression tests
pytest tests/test_performance_regression.py -v

# Run only fast tests (skip @pytest.mark.slow)
pytest tests/test_performance_regression.py -v -m "not slow"

# Run only slow tests
pytest tests/test_performance_regression.py -v -m slow

# Run with benchmark mode (requires pytest-benchmark)
pytest tests/test_performance_regression.py -v -m benchmark
```

## Test Categories

### 1. Benchmark Baselines (`TestBenchmarkBaselines`)

Tests that performance meets absolute thresholds:

- **Query Latency (no vLLM)**: <15s
- **Query Latency (with vLLM)**: <5s
- **Embedding Throughput**: >60 chunks/sec
- **Vector Search**: <100ms
- **DB Insertion**: >1000 nodes/sec

```bash
pytest tests/test_performance_regression.py::TestBenchmarkBaselines -v -m slow
```

### 2. Memory Performance (`TestMemoryPerformance`)

Tests memory usage and leak detection:

- Peak memory stays under 14GB
- No memory leaks after operations
- Batch sizes are memory-safe

```bash
pytest tests/test_performance_regression.py::TestMemoryPerformance -v
```

### 3. Configuration Performance (`TestConfigurationPerformance`)

Tests specific configuration settings:

- `N_GPU_LAYERS=24` performance
- `EMBED_BATCH=64` throughput
- `TOP_K=4` retrieval speed

```bash
pytest tests/test_performance_regression.py::TestConfigurationPerformance -v
```

### 4. Regression Detection (`TestRegressionDetection`)

Compares current performance against baselines:

- Loads baseline metrics from `performance_baselines.json`
- Fails if performance regresses by >20%
- Tracks: embedding throughput, search latency, insertion throughput

```bash
pytest tests/test_performance_regression.py::TestRegressionDetection -v
```

### 5. End-to-End Performance (`TestEndToEndPerformance`)

Tests complete pipeline performance with mocked components:

- Document processing speed
- Embedding pipeline
- Retrieval pipeline

```bash
pytest tests/test_performance_regression.py::TestEndToEndPerformance -v -m slow
```

## Performance Baselines

Baseline metrics are stored in `tests/performance_baselines.json`:

```json
{
  "embedding_throughput": 67.0,
  "vector_search_latency": 0.011,
  "db_insertion_throughput": 1250.0,
  "query_latency_no_vllm": 8.0,
  "query_latency_vllm": 1.5
}
```

### Updating Baselines

When you verify that a performance improvement is legitimate:

```python
# In Python
from tests.test_performance_regression import save_baselines

new_baselines = {
    "embedding_throughput": 75.0,  # Improved from 67.0
    "vector_search_latency": 0.011,
    "db_insertion_throughput": 1250.0,
    "query_latency_no_vllm": 8.0,
    "query_latency_vllm": 1.5
}

save_baselines(new_baselines)
```

Or manually edit `tests/performance_baselines.json`.

## Regression Thresholds

Tests fail if performance regresses by more than **20%**:

| Metric | Baseline | 20% Regression | Fail Threshold |
|--------|----------|----------------|----------------|
| Embedding throughput | 67 chunks/sec | 53.6 chunks/sec | <53.6 |
| Vector search | 11ms | 13.2ms | >13.2ms |
| DB insertion | 1250 nodes/sec | 1000 nodes/sec | <1000 |
| Query latency | 8.0s | 9.6s | >9.6s |

## CI/CD Integration

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run fast performance tests before commit
pytest tests/test_performance_regression.py -m "not slow" --tb=short
if [ $? -ne 0 ]; then
    echo "Performance regression detected!"
    exit 1
fi
```

### GitHub Actions

```yaml
name: Performance Tests

on: [push, pull_request]

jobs:
  performance:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov psutil
      - name: Run performance tests
        run: |
          pytest tests/test_performance_regression.py -v -m "not slow"
```

## Interpreting Results

### Successful Run

```
tests/test_performance_regression.py::TestBenchmarkBaselines::test_embedding_throughput_minimum PASSED
tests/test_performance_regression.py::TestRegressionDetection::test_compare_against_baseline_embedding PASSED
```

All tests pass - no performance regressions detected.

### Regression Detected

```
tests/test_performance_regression.py::TestRegressionDetection::test_compare_against_baseline_embedding FAILED
AssertionError: Regression detected: 55.0 vs baseline 67.0
```

**Action Required:**
1. Investigate what changed (code, config, dependencies)
2. Profile the slow operation
3. Fix the regression OR update baseline if intentional

### Near-Threshold Warning

If performance is close to threshold but passing:

```python
# Add warnings for near-threshold performance
if throughput < THRESHOLDS["embedding_throughput"] * 1.1:
    warnings.warn(f"Performance near threshold: {throughput:.1f}")
```

## Performance Profiling

If tests fail, use profiling to identify bottlenecks:

```bash
# Profile a specific test
python -m cProfile -o profile.stats -m pytest tests/test_performance_regression.py::test_embedding_throughput_minimum

# Analyze results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

## Mock Strategy

Tests mock slow operations to focus on fast operations:

```python
# Mock LLM generation (slow)
@patch('rag_low_level_m1_16gb_verbose.build_llm')
def test_query_without_llm_slowdown(mock_llm):
    # Test retrieval speed without waiting for generation
    pass

# Time actual embedding (fast enough to test)
def test_embedding_speed():
    # Actually run embedding with small batch
    pass
```

## Memory Testing

Memory tests use `psutil` to track usage:

```bash
# Ensure psutil is installed
pip install psutil

# Run memory tests
pytest tests/test_performance_regression.py::TestMemoryPerformance -v
```

## Troubleshooting

### Issue: Tests failing on different hardware

**Solution:** Adjust thresholds for your hardware in test file:

```python
THRESHOLDS = {
    "embedding_throughput": 40.0,  # Lower for slower CPU
    "peak_memory_gb": 8.0,  # Lower for 8GB system
}
```

### Issue: Flaky timing tests

**Solution:** Use tolerance and multiple runs:

```python
# Run test 3 times, take median
durations = [run_test() for _ in range(3)]
median_duration = sorted(durations)[1]
assert median_duration < threshold
```

### Issue: pytest-benchmark not available

**Solution:** Tests skip gracefully:

```python
if benchmark is None:
    pytest.skip("pytest-benchmark not installed")
```

## Best Practices

1. **Run Fast Tests Frequently**
   ```bash
   pytest tests/test_performance_regression.py -m "not slow"
   ```

2. **Run Full Suite Before Release**
   ```bash
   pytest tests/test_performance_regression.py -v
   ```

3. **Update Baselines After Verified Improvements**
   - Don't update for regressions
   - Document why baseline changed

4. **Monitor Trends Over Time**
   - Log metrics to file with timestamps
   - Track performance history

5. **Test on Target Hardware**
   - M1 Mac Mini 16GB for this project
   - Results vary significantly across systems

## Related Files

- `tests/test_performance_regression.py` - Main test suite
- `tests/performance_baselines.json` - Baseline metrics
- `tests/test_performance.py` - Configuration performance tests
- `config/pytest.ini` - Pytest configuration with markers
- `CLAUDE.md` - Performance benchmarks section

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [Python profiling](https://docs.python.org/3/library/profile.html)
- [psutil documentation](https://psutil.readthedocs.io/)
