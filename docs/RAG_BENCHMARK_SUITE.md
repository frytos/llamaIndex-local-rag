# RAG Benchmark Suite - Complete Guide

**Version:** 1.0.0
**Last Updated:** January 2026
**Location:** `utils/rag_benchmark.py`

## Overview

The RAG Benchmark Suite is a comprehensive framework for measuring and tracking RAG pipeline performance improvements across three critical dimensions:

1. **Retrieval Quality** - MRR, nDCG, Recall@k, Precision@k
2. **Answer Quality** - Faithfulness, Relevancy, Context Precision/Recall
3. **Performance Metrics** - Latency (p50/p95/p99), Throughput, Cache Hit Rate

## Quick Start

### Basic Benchmark

```python
from utils.rag_benchmark import RAGBenchmark, TestQuery

# Initialize
benchmark = RAGBenchmark(output_dir="benchmarks")

# Create test queries
test_queries = [
    TestQuery(
        id="q1",
        query="What is RAG?",
        relevant_doc_ids=["doc1", "doc2"],
        expected_keywords=["retrieval", "generation"]
    )
]

# Run benchmark
results = benchmark.run_end_to_end_benchmark(
    query_engine=your_query_engine,
    test_queries=test_queries
)

# Generate report
benchmark.generate_report(results, "report.html")
```

### Configuration Comparison

```python
configs = [
    {"name": "baseline", "enable_reranking": False},
    {"name": "with_reranking", "enable_reranking": True},
]

comparison = benchmark.compare_configurations(
    configs=configs,
    test_queries=test_queries,
    build_query_engine_fn=build_engine
)

benchmark.generate_comparison_report(comparison, "comparison.html")
```

## Features

### 1. Retrieval Quality Metrics

#### Mean Reciprocal Rank (MRR)
- **Purpose:** Measures position of first relevant result
- **Range:** 0.0 - 1.0 (higher is better)
- **Formula:** `1 / rank_of_first_relevant_doc`
- **Use case:** When first result matters most (e.g., question answering)

**Example:**
```python
retrieved = ["doc3", "doc1", "doc5"]  # doc1 is relevant
relevant = ["doc1", "doc2"]
mrr = benchmark.calculate_mrr(retrieved, relevant)
# Result: 0.5 (relevant doc at position 2)
```

#### Normalized Discounted Cumulative Gain (nDCG@k)
- **Purpose:** Evaluates ranking quality with position penalty
- **Range:** 0.0 - 1.0 (higher is better)
- **Formula:** `DCG / IDCG` where DCG sums `rel_i / log2(i+1)`
- **Use case:** When ranking quality across all results matters

**Example:**
```python
ndcg = benchmark.calculate_ndcg_at_k(retrieved, relevant, k=5)
# Result: 0.63 (good ranking but not perfect)
```

#### Recall@k
- **Purpose:** Percentage of relevant docs retrieved
- **Range:** 0.0 - 1.0 (higher is better)
- **Formula:** `relevant_retrieved / total_relevant`
- **Use case:** Ensuring no important information is missed

**Example:**
```python
recall = benchmark.calculate_recall_at_k(retrieved, relevant, k=5)
# Result: 0.5 (retrieved 1 of 2 relevant docs)
```

#### Precision@k
- **Purpose:** Percentage of retrieved docs that are relevant
- **Range:** 0.0 - 1.0 (higher is better)
- **Formula:** `relevant_retrieved / total_retrieved`
- **Use case:** Avoiding noise in retrieved context

**Example:**
```python
precision = benchmark.calculate_precision_at_k(retrieved, relevant, k=5)
# Result: 0.33 (1 of 3 retrieved docs is relevant)
```

### 2. Answer Quality Metrics

#### Faithfulness
- **Purpose:** Measures if answer is grounded in retrieved context
- **Range:** 0.0 - 1.0 (higher is better)
- **Method:** Word overlap between answer and context
- **Use case:** Detecting hallucinations

**Example:**
```python
answer = "RAG combines retrieval and generation"
context = ["RAG uses retrieval before generation"]
faithfulness = benchmark.calculate_faithfulness(answer, [context])
# Result: 0.75 (75% of answer words appear in context)
```

#### Answer Relevancy
- **Purpose:** Measures if answer addresses the query
- **Range:** 0.0 - 1.0 (higher is better)
- **Method:** Query term coverage + keyword matching + length heuristics
- **Use case:** Ensuring on-topic, complete answers

**Example:**
```python
query = "What is RAG?"
answer = "RAG combines retrieval and generation..."
keywords = ["retrieval", "generation"]
relevancy = benchmark.calculate_answer_relevancy(query, answer, keywords)
# Result: 0.85 (good coverage of query terms and keywords)
```

#### Context Precision & Recall
- **Purpose:** Evaluate quality of retrieved context
- **Range:** 0.0 - 1.0 (higher is better)
- **Method:** Same as Precision@k and Recall@k
- **Use case:** Tuning retrieval before generation

### 3. Performance Metrics

#### Latency Percentiles
- **p50 (median):** Typical query latency
- **p95:** 95th percentile (common SLA target)
- **p99:** 99th percentile (tail latency)
- **Interpretation:**
  - p50 < 500ms: Fast
  - p95 < 2000ms: Acceptable
  - p95 > 5000ms: Slow

#### Throughput (QPS)
- **Queries Per Second:** System capacity
- **Typical ranges:**
  - GPU-accelerated: 5-20 QPS
  - M1 Mac (llama.cpp): 0.5-2 QPS
  - CPU-only: 0.1-0.5 QPS

#### Cache Hit Rate
- **Range:** 0.0 - 1.0 (higher is better)
- **Impact:** Cache hits are 10-100x faster
- **Target:** 20-50% for production systems

#### Tokens Per Second
- **LLM generation speed**
- **Typical ranges:**
  - vLLM (GPU): 50-200 TPS
  - llama.cpp (M1): 10-20 TPS
  - CPU-only: 2-8 TPS

## Benchmark Modes

### Quick Mode
```bash
BENCHMARK_MODE=quick python benchmark.py
```
- 5-10 queries
- Basic metrics
- ~1-2 minutes
- Use for: Quick validation

### Standard Mode (Default)
```bash
BENCHMARK_MODE=standard python benchmark.py
```
- 20-50 queries
- All metrics
- ~5-10 minutes
- Use for: Regular testing

### Comprehensive Mode
```bash
BENCHMARK_MODE=comprehensive python benchmark.py
```
- 100+ queries
- Statistical analysis
- ~30-60 minutes
- Use for: Production validation

## Test Dataset Creation

### Method 1: Manual Queries

```python
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

### Method 2: Load from File

```python
# JSON format
test_queries = benchmark.load_test_dataset("test_queries.json")
```

### Method 3: Generate Synthetic

```python
test_queries = benchmark.generate_synthetic_dataset(
    num_queries=50,
    categories=['factual', 'definition', 'howto', 'comparison']
)
```

## Report Formats

### HTML (Recommended)
- Interactive Plotly charts
- Detailed tables
- Requires: `pip install plotly`

```python
benchmark.generate_report(results, "report.html", format="html")
```

### Markdown
- Human-readable text
- Version control friendly
- No dependencies

```python
benchmark.generate_report(results, "report.md", format="markdown")
```

### CSV
- Machine-readable
- Import into spreadsheets
- Further analysis

```python
benchmark.generate_report(results, "report.csv", format="csv")
```

## Integration with RAG Pipeline

### Step 1: Import Components

```python
from rag_low_level_m1_16gb_verbose import (
    build_embed_model,
    build_llm,
    make_vector_store,
    VectorDBRetriever
)
from llama_index.core.query_engine import RetrieverQueryEngine
from utils.rag_benchmark import RAGBenchmark, TestQuery
```

### Step 2: Build Query Engine

```python
# Initialize components
vector_store = make_vector_store()
embed_model = build_embed_model()
retriever = VectorDBRetriever(vector_store, embed_model, top_k=4)
llm = build_llm()

# Create query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    llm=llm
)
```

### Step 3: Run Benchmark

```python
# Initialize benchmark
benchmark = RAGBenchmark(output_dir="benchmarks")

# Load test queries
test_queries = benchmark.load_test_dataset("test_queries.json")

# Run benchmark
results = benchmark.run_end_to_end_benchmark(
    query_engine=query_engine,
    test_queries=test_queries
)

# Generate report
benchmark.generate_report(results, "benchmark_report.html")
```

### Step 4: Compare Configurations

```python
def build_query_engine(config):
    """Build query engine from config dict."""
    # Set environment variables
    os.environ['TOP_K'] = str(config['top_k'])
    os.environ['CHUNK_SIZE'] = str(config['chunk_size'])

    # Build components
    vector_store = make_vector_store()
    embed_model = build_embed_model()
    retriever = VectorDBRetriever(vector_store, embed_model)

    # Add optimizations
    if config.get('enable_reranking'):
        from utils.reranker import Reranker
        reranker = Reranker()
        # Wrap retriever with reranker...

    llm = build_llm()
    return RetrieverQueryEngine(retriever=retriever, llm=llm)

# Compare configurations
configs = [
    {"name": "baseline", "enable_reranking": False, "top_k": 4},
    {"name": "optimized", "enable_reranking": True, "top_k": 12}
]

comparison = benchmark.compare_configurations(
    configs=configs,
    test_queries=test_queries,
    build_query_engine_fn=build_query_engine
)

benchmark.generate_comparison_report(comparison, "comparison.html")
```

## Regression Testing

### Setup Baseline

```python
# Run baseline
baseline_results = benchmark.run_end_to_end_benchmark(
    baseline_engine,
    test_queries
)

# Save baseline
baseline_summary = benchmark._generate_summary("baseline", baseline_results)
import json
with open("baseline_metrics.json", "w") as f:
    json.dump(asdict(baseline_summary), f)
```

### Test New Version

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

# Check regression (95% threshold)
threshold = 0.95

assert new_summary.avg_faithfulness >= baseline["avg_faithfulness"] * threshold
assert new_summary.avg_mrr >= baseline["avg_mrr"] * threshold
assert new_summary.p95_latency_ms <= baseline["p95_latency_ms"] / threshold

print("PASSED: No regressions detected")
```

## CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: RAG Benchmark

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run benchmark
        run: |
          BENCHMARK_MODE=standard python -m utils.rag_benchmark

      - name: Check regression
        run: |
          python scripts/check_regression.py

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/
```

## Examples

### Example 1: Quick Retrieval Benchmark

```bash
python examples/benchmark_example.py --mode basic
```

### Example 2: Generate Test Dataset

```bash
python examples/benchmark_example.py --mode generate
```

### Example 3: Mock End-to-End

```bash
python examples/benchmark_example.py --mode mock
```

### Example 4: Configuration Comparison

```bash
python examples/benchmark_example.py --mode comparison
```

### Example 5: All Examples

```bash
python examples/benchmark_example.py --mode all
```

## Expected Improvements

| Optimization | Retrieval | Answer | Latency | Cache |
|--------------|-----------|---------|---------|-------|
| Baseline | 0.70 MRR | 0.75 Faith | 2500ms | 0% |
| + Reranking | +15-30% | +10-20% | +100-200ms | 0% |
| + Semantic Cache | +0% | +0% | -70-90% (hits) | 20-50% |
| + Query Expansion | +10-15% | +5-10% | +50-100ms | 0% |
| + HyDE | +5-15% | +10-15% | +500-1000ms | 0% |
| Full Optimized | 0.85 MRR | 0.90 Faith | 500ms (avg) | 40% |

## Troubleshooting

### Low MRR/nDCG
- **Cause:** Poor embedding model, insufficient data
- **Solution:** Use better embeddings (bge-large), add reranking

### Low Faithfulness
- **Cause:** LLM hallucination, insufficient context
- **Solution:** Lower temperature, increase top-k, add grounding prompt

### High Latency
- **Cause:** Too many chunks, large chunk size
- **Solution:** Reduce top-k, enable caching, use faster model

### Low Cache Hit Rate
- **Cause:** Threshold too high, diverse queries
- **Solution:** Lower threshold (0.90-0.92), increase cache size

## Files

- **`utils/rag_benchmark.py`** - Main benchmark suite (1400+ lines)
- **`examples/benchmark_example.py`** - Working examples
- **`examples/integrate_with_rag.py`** - Integration guide
- **`examples/sample_test_queries.json`** - 20 sample queries
- **`examples/BENCHMARK_README.md`** - Detailed documentation
- **`docs/RAG_BENCHMARK_SUITE.md`** - This guide

## API Reference

### RAGBenchmark Class

```python
class RAGBenchmark:
    def __init__(output_dir, enable_detailed_logging)

    # Retrieval metrics
    def calculate_mrr(retrieved, relevant) -> float
    def calculate_ndcg_at_k(retrieved, relevant, scores, k) -> float
    def calculate_recall_at_k(retrieved, relevant, k) -> float
    def calculate_precision_at_k(retrieved, relevant, k) -> float

    # Answer metrics
    def calculate_faithfulness(answer, context) -> float
    def calculate_answer_relevancy(query, answer, keywords) -> float

    # Benchmarking
    def run_retrieval_benchmark(retriever, queries, top_k)
    def run_end_to_end_benchmark(query_engine, queries, config)
    def compare_configurations(configs, queries, build_fn)

    # Reports
    def generate_report(results, output_path, format)
    def generate_comparison_report(comparison, output_path)

    # Test data
    @staticmethod
    def generate_synthetic_dataset(num_queries, categories)
    @staticmethod
    def load_test_dataset(file_path)
    @staticmethod
    def save_test_dataset(queries, file_path)
```

### TestQuery Dataclass

```python
@dataclass
class TestQuery:
    id: str
    query: str
    ground_truth_answer: Optional[str]
    relevant_doc_ids: List[str]
    expected_keywords: List[str]
    category: str
```

### BenchmarkSummary Dataclass

```python
@dataclass
class BenchmarkSummary:
    config_name: str
    num_queries: int

    # Retrieval
    avg_mrr: float
    avg_ndcg: float
    avg_recall: float
    avg_precision: float

    # Answer
    avg_faithfulness: float
    avg_answer_relevancy: float

    # Performance
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_qps: float
    cache_hit_rate: float
```

## Contributing

To extend the benchmark suite:

1. Add new metrics to `RAGBenchmark` class
2. Update `QueryBenchmarkResult` dataclass
3. Add visualizations to report generation
4. Update documentation
5. Add examples

## License

Same as main project.

## Support

- Examples: `examples/benchmark_example.py`
- Documentation: `examples/BENCHMARK_README.md`
- Main README: `README.md`
