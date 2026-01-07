#!/usr/bin/env python3
"""
Example: Using the RAG Benchmark Suite

This script demonstrates how to use the RAG benchmark suite to measure
and compare different RAG configurations.

Usage:
    # Run basic benchmark
    python benchmark_example.py --mode basic

    # Compare configurations
    python benchmark_example.py --mode comparison

    # Generate synthetic test dataset
    python benchmark_example.py --mode generate
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rag_benchmark import RAGBenchmark, TestQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def create_sample_test_queries() -> list:
    """Create sample test queries for demonstration."""
    return [
        TestQuery(
            id="q1",
            query="What is retrieval-augmented generation?",
            ground_truth_answer=(
                "Retrieval-augmented generation (RAG) is a technique that combines "
                "retrieval of relevant documents with text generation to produce "
                "more accurate and contextual responses."
            ),
            relevant_doc_ids=["doc1", "doc2", "doc3"],
            expected_keywords=["retrieval", "generation", "documents", "context"],
            category="definition"
        ),
        TestQuery(
            id="q2",
            query="How do embeddings work in RAG systems?",
            ground_truth_answer=(
                "Embeddings convert text into dense vector representations that "
                "capture semantic meaning, enabling similarity-based retrieval."
            ),
            relevant_doc_ids=["doc4", "doc5"],
            expected_keywords=["embeddings", "vectors", "semantic", "similarity"],
            category="howto"
        ),
        TestQuery(
            id="q3",
            query="What are the benefits of using pgvector for RAG?",
            ground_truth_answer=(
                "pgvector provides efficient vector storage and similarity search "
                "in PostgreSQL, with good performance and SQL integration."
            ),
            relevant_doc_ids=["doc6", "doc7"],
            expected_keywords=["pgvector", "PostgreSQL", "vector", "storage"],
            category="factual"
        ),
        TestQuery(
            id="q4",
            query="Compare bi-encoder and cross-encoder for reranking",
            ground_truth_answer=(
                "Bi-encoders encode query and document separately for fast retrieval, "
                "while cross-encoders process them together for more accurate ranking."
            ),
            relevant_doc_ids=["doc8", "doc9"],
            expected_keywords=["bi-encoder", "cross-encoder", "reranking"],
            category="comparison"
        ),
        TestQuery(
            id="q5",
            query="What is semantic caching in RAG pipelines?",
            ground_truth_answer=(
                "Semantic caching stores query results and reuses them for similar "
                "queries based on embedding similarity, avoiding expensive recomputation."
            ),
            relevant_doc_ids=["doc10", "doc11"],
            expected_keywords=["semantic", "caching", "similarity", "performance"],
            category="definition"
        ),
    ]


def example_basic_benchmark():
    """Example: Run basic retrieval metrics benchmark."""
    log.info("\n" + "="*70)
    log.info("Example 1: Basic Retrieval Metrics")
    log.info("="*70)

    benchmark = RAGBenchmark(output_dir="benchmarks/examples")

    # Simulate retrieval results
    retrieved_docs = ["doc3", "doc1", "doc5", "doc2", "doc8"]
    relevant_docs = ["doc1", "doc2", "doc4", "doc7"]

    # Calculate metrics
    mrr = benchmark.calculate_mrr(retrieved_docs, relevant_docs)
    ndcg = benchmark.calculate_ndcg_at_k(retrieved_docs, relevant_docs, k=5)
    recall = benchmark.calculate_recall_at_k(retrieved_docs, relevant_docs, k=5)
    precision = benchmark.calculate_precision_at_k(retrieved_docs, relevant_docs, k=5)

    log.info(f"\nRetrieval Metrics:")
    log.info(f"  Retrieved: {retrieved_docs}")
    log.info(f"  Relevant:  {relevant_docs}")
    log.info(f"  MRR:         {mrr:.4f}")
    log.info(f"  nDCG@5:      {ndcg:.4f}")
    log.info(f"  Recall@5:    {recall:.4f}")
    log.info(f"  Precision@5: {precision:.4f}")


def example_answer_quality():
    """Example: Evaluate answer quality."""
    log.info("\n" + "="*70)
    log.info("Example 2: Answer Quality Metrics")
    log.info("="*70)

    benchmark = RAGBenchmark(output_dir="benchmarks/examples")

    query = "What is retrieval-augmented generation?"
    answer = (
        "Retrieval-augmented generation (RAG) combines document retrieval "
        "with text generation to create contextual responses based on "
        "retrieved information from a knowledge base."
    )
    context_chunks = [
        "RAG is a technique that retrieves relevant documents before generation.",
        "The system uses embeddings to find similar documents in the knowledge base.",
        "Retrieved context is provided to the language model for grounding."
    ]
    keywords = ["retrieval", "generation", "documents", "context"]

    # Calculate metrics
    faithfulness = benchmark.calculate_faithfulness(answer, context_chunks)
    relevancy = benchmark.calculate_answer_relevancy(query, answer, keywords)

    log.info(f"\nAnswer Quality:")
    log.info(f"  Query: {query}")
    log.info(f"  Answer: {answer[:100]}...")
    log.info(f"  Faithfulness: {faithfulness:.4f}")
    log.info(f"  Relevancy:    {relevancy:.4f}")


def example_generate_dataset():
    """Example: Generate synthetic test dataset."""
    log.info("\n" + "="*70)
    log.info("Example 3: Generate Synthetic Test Dataset")
    log.info("="*70)

    benchmark = RAGBenchmark(output_dir="benchmarks/examples")

    # Generate synthetic queries
    synthetic_queries = benchmark.generate_synthetic_dataset(
        num_queries=20,
        categories=['factual', 'definition', 'howto', 'comparison']
    )

    log.info(f"\nGenerated {len(synthetic_queries)} synthetic queries")
    log.info(f"Categories: {set(q.category for q in synthetic_queries)}")
    log.info(f"\nSample queries:")
    for i, q in enumerate(synthetic_queries[:5], 1):
        log.info(f"  {i}. [{q.category}] {q.query}")

    # Save to file
    output_file = "benchmarks/examples/synthetic_test_queries.json"
    benchmark.save_test_dataset(synthetic_queries, output_file)
    log.info(f"\nSaved to: {output_file}")


def example_mock_end_to_end():
    """Example: Mock end-to-end benchmark (without actual RAG pipeline)."""
    log.info("\n" + "="*70)
    log.info("Example 4: Mock End-to-End Benchmark")
    log.info("="*70)

    benchmark = RAGBenchmark(output_dir="benchmarks/examples")

    # Create test queries
    test_queries = create_sample_test_queries()

    log.info(f"Running mock benchmark with {len(test_queries)} queries...")

    # Mock results (in real use, you'd call benchmark.run_end_to_end_benchmark)
    from utils.rag_benchmark import (
        QueryBenchmarkResult,
        RetrievalMetrics,
        AnswerMetrics,
        PerformanceMetrics
    )
    import time
    import random

    results = []
    for query in test_queries:
        result = QueryBenchmarkResult(
            query_id=query.id,
            query_text=query.query,
            timestamp=time.time(),
            ground_truth_answer=query.ground_truth_answer
        )

        # Simulate retrieval
        result.retrieval = RetrievalMetrics(
            mrr=random.uniform(0.7, 1.0),
            ndcg_at_k=random.uniform(0.7, 0.95),
            recall_at_k=random.uniform(0.6, 0.9),
            precision_at_k=random.uniform(0.5, 0.8),
            avg_relevance_score=random.uniform(0.7, 0.9),
            num_retrieved=4
        )

        # Simulate answer quality
        result.answer = AnswerMetrics(
            faithfulness=random.uniform(0.7, 0.95),
            answer_relevancy=random.uniform(0.75, 0.95),
            context_precision=random.uniform(0.6, 0.85),
            context_recall=random.uniform(0.65, 0.9),
            answer_length=random.randint(50, 300),
            contains_keywords=True
        )

        # Simulate performance
        result.performance = PerformanceMetrics(
            total_latency_ms=random.uniform(100, 5000),
            embedding_latency_ms=random.uniform(10, 100),
            retrieval_latency_ms=random.uniform(20, 200),
            generation_latency_ms=random.uniform(50, 4000),
            cache_hit=random.random() < 0.2,
            tokens_generated=random.randint(20, 100),
            tokens_per_second=random.uniform(8, 15)
        )

        result.generated_answer = f"Mock answer for: {query.query}"
        results.append(result)

    # Generate reports
    log.info("\nGenerating reports...")

    html_report = benchmark.generate_report(results, "mock_benchmark.html", format="html")
    log.info(f"  HTML report: {html_report}")

    md_report = benchmark.generate_report(results, "mock_benchmark.md", format="markdown")
    log.info(f"  Markdown report: {md_report}")

    csv_report = benchmark.generate_report(results, "mock_benchmark.csv", format="csv")
    log.info(f"  CSV report: {csv_report}")


def example_comparison():
    """Example: Compare multiple configurations."""
    log.info("\n" + "="*70)
    log.info("Example 5: Configuration Comparison")
    log.info("="*70)

    benchmark = RAGBenchmark(output_dir="benchmarks/examples")

    # Mock comparison results
    from utils.rag_benchmark import BenchmarkSummary
    import random

    configs = ["baseline", "with_reranking", "with_reranking_and_cache"]
    comparison_results = {}

    for config in configs:
        # Simulate progressive improvements
        baseline_boost = configs.index(config) * 0.05

        summary = BenchmarkSummary(
            config_name=config,
            num_queries=10,
            num_successful=10,
            num_failed=0,
            avg_mrr=0.70 + baseline_boost + random.uniform(-0.03, 0.03),
            avg_ndcg=0.75 + baseline_boost + random.uniform(-0.03, 0.03),
            avg_recall=0.65 + baseline_boost + random.uniform(-0.03, 0.03),
            avg_precision=0.60 + baseline_boost + random.uniform(-0.03, 0.03),
            avg_faithfulness=0.80 + baseline_boost + random.uniform(-0.02, 0.02),
            avg_answer_relevancy=0.82 + baseline_boost + random.uniform(-0.02, 0.02),
            avg_context_precision=0.70 + baseline_boost + random.uniform(-0.03, 0.03),
            avg_context_recall=0.68 + baseline_boost + random.uniform(-0.03, 0.03),
            p50_latency_ms=2000 - (baseline_boost * 500),
            p95_latency_ms=4500 - (baseline_boost * 800),
            p99_latency_ms=6000 - (baseline_boost * 1000),
            avg_latency_ms=2500 - (baseline_boost * 600),
            throughput_qps=0.4 + (baseline_boost * 0.1),
            cache_hit_rate=0.0 if config == "baseline" else (0.15 + baseline_boost),
            avg_tokens_per_second=10.0 + baseline_boost
        )

        comparison_results[config] = summary

    log.info("\nComparison Summary:")
    for name, summary in comparison_results.items():
        log.info(f"\n  {name}:")
        log.info(f"    Avg MRR:         {summary.avg_mrr:.4f}")
        log.info(f"    Avg Faithfulness: {summary.avg_faithfulness:.4f}")
        log.info(f"    P95 Latency:     {summary.p95_latency_ms:.2f}ms")
        log.info(f"    Cache Hit Rate:  {summary.cache_hit_rate:.2%}")

    # Generate comparison report
    report_path = benchmark.generate_comparison_report(
        comparison_results,
        "configuration_comparison.html"
    )
    log.info(f"\nComparison report: {report_path}")


def example_regression_testing():
    """Example: Regression testing against baseline."""
    log.info("\n" + "="*70)
    log.info("Example 6: Regression Testing")
    log.info("="*70)

    # Simulate baseline and new version
    baseline_mrr = 0.75
    baseline_faithfulness = 0.82
    baseline_latency = 2500

    new_mrr = 0.78
    new_faithfulness = 0.85
    new_latency = 2200

    threshold = 0.95  # 95% of baseline

    log.info(f"\nBaseline Metrics:")
    log.info(f"  MRR:          {baseline_mrr:.4f}")
    log.info(f"  Faithfulness: {baseline_faithfulness:.4f}")
    log.info(f"  Latency:      {baseline_latency:.0f}ms")

    log.info(f"\nNew Version Metrics:")
    log.info(f"  MRR:          {new_mrr:.4f} ({new_mrr/baseline_mrr:.1%} of baseline)")
    log.info(f"  Faithfulness: {new_faithfulness:.4f} ({new_faithfulness/baseline_faithfulness:.1%} of baseline)")
    log.info(f"  Latency:      {new_latency:.0f}ms ({new_latency/baseline_latency:.1%} of baseline)")

    # Check regression
    passed = True
    if new_mrr < baseline_mrr * threshold:
        log.warning(f"  REGRESSION: MRR below {threshold:.0%} threshold")
        passed = False

    if new_faithfulness < baseline_faithfulness * threshold:
        log.warning(f"  REGRESSION: Faithfulness below {threshold:.0%} threshold")
        passed = False

    if new_latency > baseline_latency / threshold:
        log.warning(f"  REGRESSION: Latency above {1/threshold:.0%} threshold")
        passed = False

    if passed:
        log.info(f"\n  PASSED: No regressions detected")
    else:
        log.error(f"\n  FAILED: Regressions detected")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Benchmark Examples")
    parser.add_argument(
        "--mode",
        choices=["basic", "answer", "generate", "mock", "comparison", "regression", "all"],
        default="all",
        help="Example mode to run"
    )
    args = parser.parse_args()

    # Create output directory
    Path("benchmarks/examples").mkdir(parents=True, exist_ok=True)

    if args.mode in ["basic", "all"]:
        example_basic_benchmark()

    if args.mode in ["answer", "all"]:
        example_answer_quality()

    if args.mode in ["generate", "all"]:
        example_generate_dataset()

    if args.mode in ["mock", "all"]:
        example_mock_end_to_end()

    if args.mode in ["comparison", "all"]:
        example_comparison()

    if args.mode in ["regression", "all"]:
        example_regression_testing()

    log.info("\n" + "="*70)
    log.info("Examples Complete!")
    log.info("="*70)
    log.info("\nCheck benchmarks/examples/ for generated reports")


if __name__ == "__main__":
    main()
