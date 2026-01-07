# RAG Benchmark Suite Documentation

**Version:** 1.0.0
**Last Updated:** January 2026

## Overview

The RAG Benchmark Suite provides comprehensive performance and quality testing for RAG (Retrieval-Augmented Generation) pipelines. It measures three critical dimensions:

1. **Retrieval Quality** - MRR, nDCG, Recall, Precision
2. **Answer Quality** - Faithfulness, Relevancy, Context Precision/Recall
3. **Performance** - Latency, Throughput, Cache Hit Rate, Resource Usage

## Quick Start

### 1. Installation

The benchmark suite is included in the `utils/` directory and requires:

```bash
# Core dependencies (already in requirements.txt)
pip install numpy

# Optional: For HTML reports with visualizations
pip install plotly

# Optional: For performance metrics
pip install psutil
```

### 2. Basic Usage

```python
from utils.rag_benchmark import RAGBenchmark, TestQuery

# Initialize benchmark
benchmark = RAGBenchmark(output_dir="benchmarks")

# Create or load test queries
test_queries = [
    TestQuery(
        id="q1",
        query="What is retrieval-augmented generation?",
        relevant_doc_ids=["doc1", "doc2"],
        expected_keywords=["retrieval", "generation"],
        category="definition"
    ),
    # ... more queries
]

# Run end-to-end benchmark
results = benchmark.run_end_to_end_benchmark(
    query_engine=your_query_engine,
    test_queries=test_queries
)

# Generate report
benchmark.generate_report(results, "benchmark_report.html", format="html")
```

### 3. Run Example Benchmarks

```bash
# Run all examples
python examples/benchmark_example.py --mode all

# Run specific example
python examples/benchmark_example.py --mode comparison

# Generate synthetic test dataset
python examples/benchmark_example.py --mode generate
```

## Metrics Explained

### Retrieval Quality Metrics

#### Mean Reciprocal Rank (MRR)
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Position of the first relevant result
- **Interpretation:**
  - 1.0 = First result is relevant
  - 0.5 = Second result is relevant
  - 0.33 = Third result is relevant
- **Use case:** Evaluating if the most relevant document ranks first

#### Normalized Discounted Cumulative Gain (nDCG@k)
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Quality of ranking, considering both relevance and position
- **Interpretation:**
  - 1.0 = Perfect ranking
  - 0.8+ = Excellent ranking
  - 0.6-0.8 = Good ranking
- **Use case:** Evaluating overall ranking quality across top-k results

#### Recall@k
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Percentage of relevant documents retrieved
- **Formula:** `relevant_retrieved / total_relevant`
- **Interpretation:**
  - 1.0 = All relevant documents retrieved
  - 0.5 = Half of relevant documents retrieved
- **Use case:** Ensuring no important information is missed

#### Precision@k
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Percentage of retrieved documents that are relevant
- **Formula:** `relevant_retrieved / total_retrieved`
- **Interpretation:**
  - 1.0 = All retrieved documents are relevant
  - 0.5 = Half of retrieved documents are relevant
- **Use case:** Avoiding noise in retrieved context

### Answer Quality Metrics

#### Faithfulness
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Whether the answer is grounded in retrieved context
- **Method:** Checks if answer content appears in retrieved chunks
- **Interpretation:**
  - 0.9+ = Highly grounded
  - 0.7-0.9 = Mostly grounded
  - <0.7 = May contain hallucinations
- **Use case:** Detecting hallucinations and ensuring factual accuracy

#### Answer Relevancy
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Whether the answer addresses the query
- **Method:** Checks for query terms and expected keywords in answer
- **Interpretation:**
  - 0.9+ = Directly answers query
  - 0.7-0.9 = Partially addresses query
  - <0.7 = Off-topic or incomplete
- **Use case:** Ensuring answers are on-topic and complete

#### Context Precision
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Quality of retrieved context (same as Precision@k)
- **Use case:** Ensuring clean, relevant context for generation

#### Context Recall
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Completeness of retrieved context (same as Recall@k)
- **Use case:** Ensuring sufficient information for complete answers

### Performance Metrics

#### Latency Percentiles
- **p50 (median):** Half of queries finish faster
- **p95:** 95% of queries finish faster (typical SLA target)
- **p99:** 99% of queries finish faster (tail latency)
- **Interpretation:**
  - p50: 100ms = Very fast
  - p95: 500ms = Fast
  - p95: 2000ms = Acceptable
  - p95: 5000ms+ = Slow

#### Throughput (QPS)
- **Queries Per Second:** Number of queries the system can handle
- **Interpretation:**
  - 10+ QPS = High throughput
  - 1-10 QPS = Medium throughput
  - <1 QPS = Low throughput
- **Note:** Depends heavily on hardware and configuration

#### Cache Hit Rate
- **Range:** 0.0 - 1.0 (higher is better)
- **What it measures:** Percentage of queries served from cache
- **Interpretation:**
  - 0.5+ = Excellent caching
  - 0.2-0.5 = Good caching
  - <0.2 = Limited cache benefit
- **Impact:** Cache hits are 10-100x faster than full pipeline

#### Tokens Per Second
- **What it measures:** LLM generation speed
- **Typical ranges:**
  - 20+ TPS = Fast (GPU-accelerated)
  - 10-20 TPS = Medium (M1 Mac, llama.cpp)
  - <10 TPS = Slow (CPU-only)

## Benchmark Modes

### Quick Mode (Environment Variable)
```bash
BENCHMARK_MODE=quick python rag_low_level_m1_16gb_verbose.py
```
- 5-10 test queries
- Basic metrics only
- Fast execution (~1-2 minutes)
- Use for: Sanity checks, debugging

### Standard Mode (Default)
```bash
BENCHMARK_MODE=standard python rag_low_level_m1_16gb_verbose.py
```
- 20-50 test queries
- All retrieval and answer metrics
- Full reports
- Use for: Regular testing, development

### Comprehensive Mode
```bash
BENCHMARK_MODE=comprehensive python rag_low_level_m1_16gb_verbose.py
```
- 100+ test queries
- All metrics + detailed analysis
- Statistical significance testing
- Use for: Production validation, research

## Configuration Comparison

### Defining Configurations

```python
configs = [
    {
        "name": "baseline",
        "enable_reranking": False,
        "enable_cache": False,
        "top_k": 4,
        "chunk_size": 700
    },
    {
        "name": "with_reranking",
        "enable_reranking": True,
        "enable_cache": False,
        "top_k": 12,  # Retrieve more, rerank to 4
        "chunk_size": 700
    },
    {
        "name": "full_optimized",
        "enable_reranking": True,
        "enable_cache": True,
        "enable_hyde": True,
        "top_k": 12,
        "chunk_size": 600
    }
]
```

### Running Comparison

```python
def build_query_engine(config):
    # Build query engine based on config
    # ... implementation depends on your setup
    return query_engine

comparison_results = benchmark.compare_configurations(
    configs=configs,
    test_queries=test_queries,
    build_query_engine_fn=build_query_engine
)

benchmark.generate_comparison_report(
    comparison_results,
    "config_comparison.html"
)
```

## Test Dataset Creation

### Manual Test Queries

```python
from utils.rag_benchmark import TestQuery

test_queries = [
    TestQuery(
        id="q1",
        query="What is RAG?",
        ground_truth_answer="RAG is...",
        relevant_doc_ids=["doc1", "doc2"],
        expected_keywords=["retrieval", "generation"],
        category="definition"
    )
]
```

### Synthetic Dataset Generation

```python
benchmark = RAGBenchmark()

# Generate synthetic queries
synthetic_queries = benchmark.generate_synthetic_dataset(
    num_queries=50,
    categories=['factual', 'definition', 'howto', 'comparison']
)

# Save for reuse
benchmark.save_test_dataset(synthetic_queries, "test_queries.json")
```

### Loading from File

```python
# Load test queries from JSON
test_queries = benchmark.load_test_dataset("examples/sample_test_queries.json")
```

## Report Formats

### HTML Report (Recommended)
```python
benchmark.generate_report(results, "report.html", format="html")
```
- Interactive Plotly charts
- Detailed metrics tables
- Per-query results
- Requires: `pip install plotly`

### Markdown Report
```python
benchmark.generate_report(results, "report.md", format="markdown")
```
- Human-readable text format
- Easy to include in documentation
- Version control friendly

### CSV Report
```python
benchmark.generate_report(results, "report.csv", format="csv")
```
- Machine-readable format
- Import into spreadsheets
- Further analysis with pandas

## Regression Testing

### Setting Up Baseline

```python
# Run baseline benchmark
baseline_results = benchmark.run_end_to_end_benchmark(
    baseline_engine,
    test_queries
)

# Save baseline metrics
baseline_summary = benchmark._generate_summary("baseline", baseline_results)
with open("baseline_metrics.json", "w") as f:
    json.dump(asdict(baseline_summary), f)
```

### Testing New Version

```python
# Load baseline
with open("baseline_metrics.json") as f:
    baseline = json.load(f)

# Run new version
new_results = benchmark.run_end_to_end_benchmark(
    new_engine,
    test_queries
)
new_summary = benchmark._generate_summary("new", new_results)

# Compare
threshold = 0.95  # 95% of baseline

if new_summary.avg_faithfulness < baseline["avg_faithfulness"] * threshold:
    print("REGRESSION: Faithfulness degraded")
    sys.exit(1)

if new_summary.p95_latency_ms > baseline["p95_latency_ms"] / threshold:
    print("REGRESSION: Latency increased")
    sys.exit(1)

print("PASSED: No regressions detected")
```

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: RAG Benchmark

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run benchmark
        run: |
          python -m utils.rag_benchmark --mode standard

      - name: Check regression
        run: |
          python scripts/check_regression.py

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/
```

## Best Practices

### 1. Test Dataset Quality
- Use diverse query types (factual, reasoning, comparison)
- Include easy, medium, and hard queries
- Ensure ground truth accuracy
- Update dataset regularly

### 2. Baseline Maintenance
- Re-establish baseline after major changes
- Track baselines over time
- Document expected variations

### 3. Metric Selection
- Focus on metrics that matter for your use case
- Balance quality vs. performance
- Consider user experience

### 4. Iteration Frequency
- Quick benchmarks: Daily during development
- Standard benchmarks: Before each release
- Comprehensive benchmarks: Monthly or quarterly

### 5. Result Analysis
- Look for patterns across query types
- Investigate outliers
- Correlate metrics with user feedback

## Troubleshooting

### Issue: Low MRR/nDCG scores
**Causes:**
- Poor embedding model
- Insufficient training data
- Incorrect document metadata

**Solutions:**
- Try better embedding model (e.g., bge-large vs bge-small)
- Add query expansion
- Implement reranking

### Issue: Low faithfulness scores
**Causes:**
- LLM hallucination
- Insufficient context
- High temperature setting

**Solutions:**
- Lower temperature (0.1-0.3)
- Increase top-k
- Use larger context window
- Add faithfulness prompt

### Issue: High latency
**Causes:**
- Too many retrieved chunks
- Large chunk size
- No caching

**Solutions:**
- Reduce top-k
- Implement semantic caching
- Use smaller, faster models
- Add query batching

### Issue: Low cache hit rate
**Causes:**
- Threshold too high
- Insufficient traffic
- Diverse queries

**Solutions:**
- Lower similarity threshold (0.90-0.92)
- Increase cache size
- Pre-populate cache with common queries

## Examples

See `examples/benchmark_example.py` for complete working examples:

```bash
# Run all examples
python examples/benchmark_example.py --mode all

# Individual examples
python examples/benchmark_example.py --mode basic       # Basic metrics
python examples/benchmark_example.py --mode answer      # Answer quality
python examples/benchmark_example.py --mode generate    # Generate dataset
python examples/benchmark_example.py --mode mock        # Mock benchmark
python examples/benchmark_example.py --mode comparison  # Config comparison
python examples/benchmark_example.py --mode regression  # Regression test
```

## Sample Test Queries

Sample test queries are provided in `examples/sample_test_queries.json` covering:
- Definitions (What is X?)
- How-to (How do I X?)
- Comparisons (What's the difference between X and Y?)
- Factual (What are the benefits of X?)

## Contributing

To add new metrics or features:

1. Add metric calculation to `RAGBenchmark` class
2. Update `QueryBenchmarkResult` dataclass
3. Add visualization to report generation
4. Update documentation
5. Add tests

## References

- **MRR:** Mean Reciprocal Rank (information retrieval)
- **nDCG:** Normalized Discounted Cumulative Gain (ranking quality)
- **Faithfulness:** Grounding in retrieved context (RAG-specific)
- **Answer Relevancy:** Query-answer alignment (QA systems)

## Support

For questions or issues:
- Check examples in `examples/benchmark_example.py`
- Review this documentation
- Check main project README.md
- File an issue with benchmark results

## License

Same as main project.
