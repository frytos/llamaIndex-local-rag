"""
RAG Benchmark Suite - Comprehensive Performance and Quality Testing

This module provides a complete benchmarking framework for measuring and
tracking RAG pipeline improvements across retrieval quality, answer quality,
and performance metrics.

Key Features:
  - Retrieval quality metrics (MRR, nDCG, Recall, Precision)
  - Answer quality metrics (Faithfulness, Relevancy, Context Precision/Recall)
  - Performance metrics (Latency, Throughput, Cache Hit Rate, Resource Usage)
  - Configuration comparison framework
  - HTML/Markdown/CSV report generation with visualizations
  - Synthetic test dataset generation
  - Regression testing support

Usage:
    # Basic benchmark
    from utils.rag_benchmark import RAGBenchmark

    benchmark = RAGBenchmark()
    results = benchmark.run_end_to_end_benchmark(test_queries)
    benchmark.generate_report(results, "benchmark_report.html")

    # Compare configurations
    configs = [
        {"name": "baseline", "enable_reranking": False},
        {"name": "with_reranking", "enable_reranking": True},
    ]
    comparison = benchmark.compare_configurations(configs, test_queries)
    benchmark.generate_comparison_report(comparison, "comparison.html")

    # Quick benchmark (environment variable)
    BENCHMARK_MODE=quick python rag_low_level_m1_16gb_verbose.py

Environment Variables:
    BENCHMARK_MODE=quick|standard|comprehensive  # Benchmark depth
    BENCHMARK_OUTPUT_DIR=./benchmarks            # Output directory
    BENCHMARK_DATASET=./test_queries.json        # Test dataset path
    ENABLE_REGRESSION_TEST=1                      # Enable regression testing
    REGRESSION_THRESHOLD=0.95                     # Min score vs baseline
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Optional dependencies
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    log.warning("Plotly not available. Install with: pip install plotly")

try:
    from utils.metrics import RAGMetrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    log.warning("RAGMetrics not available")

try:
    from utils.query_cache import semantic_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    log.warning("Semantic cache not available")


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics for a single query."""
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_k: float = 0.0  # Normalized Discounted Cumulative Gain
    recall_at_k: float = 0.0  # Recall@k
    precision_at_k: float = 0.0  # Precision@k
    avg_relevance_score: float = 0.0  # Average similarity score
    num_retrieved: int = 0


@dataclass
class AnswerMetrics:
    """Answer quality metrics for a single query."""
    faithfulness: float = 0.0  # Answer grounded in context (0-1)
    answer_relevancy: float = 0.0  # Answer addresses question (0-1)
    context_precision: float = 0.0  # Retrieved docs relevant (0-1)
    context_recall: float = 0.0  # All needed docs retrieved (0-1)
    answer_length: int = 0  # Characters in answer
    contains_keywords: bool = False  # Contains expected keywords


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single query."""
    total_latency_ms: float = 0.0  # Total end-to-end time
    embedding_latency_ms: float = 0.0  # Query embedding time
    retrieval_latency_ms: float = 0.0  # Vector search time
    rerank_latency_ms: float = 0.0  # Reranking time (if enabled)
    generation_latency_ms: float = 0.0  # LLM generation time
    cache_hit: bool = False  # Whether result was cached
    tokens_generated: int = 0  # Number of tokens generated
    tokens_per_second: float = 0.0  # Generation throughput


@dataclass
class QueryBenchmarkResult:
    """Complete benchmark result for a single query."""
    query_id: str
    query_text: str
    timestamp: float
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    answer: AnswerMetrics = field(default_factory=AnswerMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    ground_truth_answer: Optional[str] = None
    generated_answer: Optional[str] = None
    retrieved_chunks: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results across all queries."""
    config_name: str
    num_queries: int
    num_successful: int
    num_failed: int

    # Aggregated retrieval metrics
    avg_mrr: float = 0.0
    avg_ndcg: float = 0.0
    avg_recall: float = 0.0
    avg_precision: float = 0.0

    # Aggregated answer metrics
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0

    # Aggregated performance metrics
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_qps: float = 0.0  # Queries per second
    cache_hit_rate: float = 0.0
    avg_tokens_per_second: float = 0.0

    # Resource metrics
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TestQuery:
    """Test query with ground truth for evaluation."""
    id: str
    query: str
    ground_truth_answer: Optional[str] = None
    relevant_doc_ids: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    category: str = "general"


class RAGBenchmark:
    """
    Comprehensive RAG benchmarking suite.

    Measures retrieval quality, answer quality, and performance metrics
    across different configurations and test datasets.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        enable_detailed_logging: bool = True
    ):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory for benchmark results
            enable_detailed_logging: Enable detailed per-query logging
        """
        self.output_dir = Path(output_dir or os.getenv("BENCHMARK_OUTPUT_DIR", "benchmarks"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_detailed_logging = enable_detailed_logging
        self.results: List[QueryBenchmarkResult] = []

        log.info(f"RAG Benchmark initialized: {self.output_dir}")

    # ==================== Retrieval Quality Metrics ====================

    def calculate_mrr(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR measures the position of the first relevant result.
        Score of 1.0 means first result is relevant.

        Args:
            retrieved_doc_ids: Ordered list of retrieved document IDs
            relevant_doc_ids: List of relevant document IDs

        Returns:
            MRR score (0.0-1.0)
        """
        if not relevant_doc_ids or not retrieved_doc_ids:
            return 0.0

        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank

        return 0.0

    def calculate_ndcg_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        relevance_scores: Optional[List[float]] = None,
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.

        nDCG measures the quality of ranking, considering both relevance
        and position. Higher scores are better.

        Args:
            retrieved_doc_ids: Ordered list of retrieved document IDs
            relevant_doc_ids: List of relevant document IDs
            relevance_scores: Optional relevance scores (0-1) for each doc
            k: Cutoff position (None = use all)

        Returns:
            nDCG@k score (0.0-1.0)
        """
        if not relevant_doc_ids or not retrieved_doc_ids:
            return 0.0

        k = k or len(retrieved_doc_ids)
        retrieved = retrieved_doc_ids[:k]

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_doc_ids:
                # Binary relevance (1 if relevant, 0 otherwise)
                relevance = 1.0
                # Use relevance scores if provided
                if relevance_scores:
                    idx = retrieved_doc_ids.index(doc_id)
                    if idx < len(relevance_scores):
                        relevance = relevance_scores[idx]

                # DCG formula: rel_i / log2(i+1)
                dcg += relevance / np.log2(rank + 1)

        # Calculate IDCG (Ideal DCG) - best possible ranking
        ideal_scores = sorted(
            [1.0] * min(len(relevant_doc_ids), k),
            reverse=True
        )
        idcg = sum(
            score / np.log2(rank + 1)
            for rank, score in enumerate(ideal_scores, start=1)
        )

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_recall_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Recall@k.

        Recall measures what percentage of relevant documents were retrieved.

        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_doc_ids: List of relevant document IDs
            k: Cutoff position (None = use all)

        Returns:
            Recall@k (0.0-1.0)
        """
        if not relevant_doc_ids:
            return 0.0

        k = k or len(retrieved_doc_ids)
        retrieved = set(retrieved_doc_ids[:k])
        relevant = set(relevant_doc_ids)

        retrieved_relevant = retrieved.intersection(relevant)
        return len(retrieved_relevant) / len(relevant)

    def calculate_precision_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Precision@k.

        Precision measures what percentage of retrieved documents are relevant.

        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_doc_ids: List of relevant document IDs
            k: Cutoff position (None = use all)

        Returns:
            Precision@k (0.0-1.0)
        """
        if not retrieved_doc_ids:
            return 0.0

        k = k or len(retrieved_doc_ids)
        retrieved = set(retrieved_doc_ids[:k])
        relevant = set(relevant_doc_ids)

        retrieved_relevant = retrieved.intersection(relevant)
        return len(retrieved_relevant) / len(retrieved)

    # ==================== Answer Quality Metrics ====================

    def calculate_faithfulness(
        self,
        answer: str,
        context_chunks: List[str]
    ) -> float:
        """
        Calculate faithfulness score (answer grounded in context).

        Uses simple keyword overlap as a proxy for faithfulness.
        In production, use LLM-based evaluation or NLI models.

        Args:
            answer: Generated answer
            context_chunks: Retrieved context chunks

        Returns:
            Faithfulness score (0.0-1.0)
        """
        if not answer or not context_chunks:
            return 0.0

        # Simple implementation: check if answer words appear in context
        answer_words = set(answer.lower().split())
        context_text = " ".join(context_chunks).lower()
        context_words = set(context_text.split())

        # Remove common stop words for better signal
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was'
        }
        answer_words -= stop_words

        if not answer_words:
            return 0.0

        # Calculate overlap
        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(answer_words)

    def calculate_answer_relevancy(
        self,
        query: str,
        answer: str,
        expected_keywords: Optional[List[str]] = None
    ) -> float:
        """
        Calculate answer relevancy (answer addresses query).

        Uses keyword matching and length heuristics.
        In production, use semantic similarity or LLM-based evaluation.

        Args:
            query: User query
            answer: Generated answer
            expected_keywords: Optional list of expected keywords

        Returns:
            Relevancy score (0.0-1.0)
        """
        if not answer:
            return 0.0

        score = 0.0

        # Check for query terms in answer
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was',
            'what', 'how', 'why', 'when', 'where', 'who'
        }
        query_words -= stop_words

        if query_words:
            overlap = query_words.intersection(answer_words)
            score += 0.5 * (len(overlap) / len(query_words))

        # Check for expected keywords
        if expected_keywords:
            answer_lower = answer.lower()
            found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            score += 0.3 * (found / len(expected_keywords))

        # Penalty for very short answers (likely incomplete)
        if len(answer) < 20:
            score *= 0.5

        # Bonus for reasonable length (50-500 chars)
        if 50 <= len(answer) <= 500:
            score += 0.2

        return min(score, 1.0)

    def calculate_context_precision(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate context precision (retrieved docs are relevant).

        Same as precision@k but specific to RAG context.

        Args:
            retrieved_doc_ids: Retrieved document IDs
            relevant_doc_ids: Relevant document IDs

        Returns:
            Context precision (0.0-1.0)
        """
        return self.calculate_precision_at_k(retrieved_doc_ids, relevant_doc_ids)

    def calculate_context_recall(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate context recall (all needed docs retrieved).

        Same as recall@k but specific to RAG context.

        Args:
            retrieved_doc_ids: Retrieved document IDs
            relevant_doc_ids: Relevant document IDs

        Returns:
            Context recall (0.0-1.0)
        """
        return self.calculate_recall_at_k(retrieved_doc_ids, relevant_doc_ids)

    # ==================== Benchmarking Methods ====================

    def run_retrieval_benchmark(
        self,
        retriever,
        test_queries: List[TestQuery],
        top_k: int = 4
    ) -> List[QueryBenchmarkResult]:
        """
        Benchmark retrieval quality only (no generation).

        Args:
            retriever: Retriever instance (BaseRetriever)
            test_queries: List of test queries with ground truth
            top_k: Number of documents to retrieve

        Returns:
            List of benchmark results
        """
        log.info(f"\n{'='*70}")
        log.info(f"Running Retrieval Benchmark")
        log.info(f"{'='*70}")
        log.info(f"  Queries: {len(test_queries)}")
        log.info(f"  Top-K: {top_k}")

        results = []

        for test_query in test_queries:
            result = QueryBenchmarkResult(
                query_id=test_query.id,
                query_text=test_query.query,
                timestamp=time.time(),
                ground_truth_answer=test_query.ground_truth_answer
            )

            try:
                # Time retrieval
                start = time.time()
                retrieved_nodes = retriever.retrieve(test_query.query)
                retrieval_time = (time.time() - start) * 1000  # ms

                # Extract document IDs and scores
                retrieved_doc_ids = [
                    node.node.metadata.get("doc_id", f"doc_{i}")
                    for i, node in enumerate(retrieved_nodes)
                ]
                relevance_scores = [node.score for node in retrieved_nodes]

                # Store retrieved chunks
                result.retrieved_chunks = [
                    node.node.get_content()[:200]  # First 200 chars
                    for node in retrieved_nodes
                ]

                # Calculate retrieval metrics
                result.retrieval.mrr = self.calculate_mrr(
                    retrieved_doc_ids,
                    test_query.relevant_doc_ids
                )
                result.retrieval.ndcg_at_k = self.calculate_ndcg_at_k(
                    retrieved_doc_ids,
                    test_query.relevant_doc_ids,
                    relevance_scores,
                    k=top_k
                )
                result.retrieval.recall_at_k = self.calculate_recall_at_k(
                    retrieved_doc_ids,
                    test_query.relevant_doc_ids,
                    k=top_k
                )
                result.retrieval.precision_at_k = self.calculate_precision_at_k(
                    retrieved_doc_ids,
                    test_query.relevant_doc_ids,
                    k=top_k
                )
                result.retrieval.avg_relevance_score = (
                    np.mean(relevance_scores) if relevance_scores else 0.0
                )
                result.retrieval.num_retrieved = len(retrieved_nodes)

                # Performance metrics
                result.performance.retrieval_latency_ms = retrieval_time
                result.performance.total_latency_ms = retrieval_time

                if self.enable_detailed_logging:
                    log.info(f"\n  Query: {test_query.query}")
                    log.info(f"    MRR: {result.retrieval.mrr:.4f}")
                    log.info(f"    nDCG@{top_k}: {result.retrieval.ndcg_at_k:.4f}")
                    log.info(f"    Recall@{top_k}: {result.retrieval.recall_at_k:.4f}")
                    log.info(f"    Latency: {retrieval_time:.2f}ms")

            except Exception as e:
                log.error(f"Error benchmarking query '{test_query.query}': {e}")
                result.error = str(e)

            results.append(result)

        log.info(f"\n{'='*70}")
        log.info(f"Retrieval Benchmark Complete")
        log.info(f"  Successful: {sum(1 for r in results if not r.error)}/{len(results)}")
        log.info(f"{'='*70}\n")

        return results

    def run_end_to_end_benchmark(
        self,
        query_engine,
        test_queries: List[TestQuery],
        config: Optional[Dict[str, Any]] = None
    ) -> List[QueryBenchmarkResult]:
        """
        Benchmark full RAG pipeline (retrieval + generation).

        Args:
            query_engine: Query engine instance
            test_queries: List of test queries with ground truth
            config: Optional configuration dict for metadata

        Returns:
            List of benchmark results
        """
        log.info(f"\n{'='*70}")
        log.info(f"Running End-to-End RAG Benchmark")
        log.info(f"{'='*70}")
        log.info(f"  Queries: {len(test_queries)}")
        if config:
            log.info(f"  Config: {config.get('name', 'unnamed')}")

        results = []

        for test_query in test_queries:
            result = QueryBenchmarkResult(
                query_id=test_query.id,
                query_text=test_query.query,
                timestamp=time.time(),
                ground_truth_answer=test_query.ground_truth_answer
            )

            try:
                # Check cache first
                cache_hit = False
                if CACHE_AVAILABLE:
                    cache_stats_before = semantic_cache.stats()

                # Time end-to-end query
                start_total = time.time()
                response = query_engine.query(test_query.query)
                total_time = (time.time() - start_total) * 1000  # ms

                # Check if it was a cache hit
                if CACHE_AVAILABLE:
                    cache_stats_after = semantic_cache.stats()
                    cache_hit = cache_stats_after['hits'] > cache_stats_before['hits']

                result.generated_answer = str(response)

                # Extract retrieved chunks from response
                if hasattr(response, 'source_nodes'):
                    retrieved_nodes = response.source_nodes
                    result.retrieved_chunks = [
                        node.node.get_content()[:200]
                        for node in retrieved_nodes
                    ]

                    retrieved_doc_ids = [
                        node.node.metadata.get("doc_id", f"doc_{i}")
                        for i, node in enumerate(retrieved_nodes)
                    ]
                    relevance_scores = [node.score for node in retrieved_nodes]

                    # Calculate retrieval metrics
                    result.retrieval.mrr = self.calculate_mrr(
                        retrieved_doc_ids,
                        test_query.relevant_doc_ids
                    )
                    result.retrieval.ndcg_at_k = self.calculate_ndcg_at_k(
                        retrieved_doc_ids,
                        test_query.relevant_doc_ids,
                        relevance_scores
                    )
                    result.retrieval.recall_at_k = self.calculate_recall_at_k(
                        retrieved_doc_ids,
                        test_query.relevant_doc_ids
                    )
                    result.retrieval.precision_at_k = self.calculate_precision_at_k(
                        retrieved_doc_ids,
                        test_query.relevant_doc_ids
                    )
                    result.retrieval.avg_relevance_score = np.mean(relevance_scores)
                    result.retrieval.num_retrieved = len(retrieved_nodes)

                # Calculate answer metrics
                result.answer.faithfulness = self.calculate_faithfulness(
                    result.generated_answer,
                    result.retrieved_chunks
                )
                result.answer.answer_relevancy = self.calculate_answer_relevancy(
                    test_query.query,
                    result.generated_answer,
                    test_query.expected_keywords
                )
                result.answer.context_precision = self.calculate_context_precision(
                    retrieved_doc_ids if result.retrieved_chunks else [],
                    test_query.relevant_doc_ids
                )
                result.answer.context_recall = self.calculate_context_recall(
                    retrieved_doc_ids if result.retrieved_chunks else [],
                    test_query.relevant_doc_ids
                )
                result.answer.answer_length = len(result.generated_answer)
                result.answer.contains_keywords = any(
                    kw.lower() in result.generated_answer.lower()
                    for kw in test_query.expected_keywords
                ) if test_query.expected_keywords else False

                # Performance metrics
                result.performance.total_latency_ms = total_time
                result.performance.cache_hit = cache_hit

                # Estimate token count (rough: ~4 chars per token)
                result.performance.tokens_generated = len(result.generated_answer) // 4
                if total_time > 0 and not cache_hit:
                    result.performance.tokens_per_second = (
                        result.performance.tokens_generated / (total_time / 1000)
                    )

                if self.enable_detailed_logging:
                    log.info(f"\n  Query: {test_query.query}")
                    log.info(f"    Faithfulness: {result.answer.faithfulness:.4f}")
                    log.info(f"    Relevancy: {result.answer.answer_relevancy:.4f}")
                    log.info(f"    Latency: {total_time:.2f}ms")
                    log.info(f"    Cache Hit: {cache_hit}")

            except Exception as e:
                log.error(f"Error benchmarking query '{test_query.query}': {e}")
                result.error = str(e)

            results.append(result)

        log.info(f"\n{'='*70}")
        log.info(f"End-to-End Benchmark Complete")
        log.info(f"  Successful: {sum(1 for r in results if not r.error)}/{len(results)}")
        log.info(f"{'='*70}\n")

        self.results.extend(results)
        return results

    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        test_queries: List[TestQuery],
        build_query_engine_fn
    ) -> Dict[str, BenchmarkSummary]:
        """
        Compare multiple configurations on the same test dataset.

        Args:
            configs: List of configuration dicts, each with a 'name' key
            test_queries: Test queries to run
            build_query_engine_fn: Function that takes config dict and returns query engine

        Returns:
            Dict mapping config name to BenchmarkSummary
        """
        log.info(f"\n{'='*70}")
        log.info(f"Configuration Comparison Benchmark")
        log.info(f"{'='*70}")
        log.info(f"  Configurations: {len(configs)}")
        log.info(f"  Test Queries: {len(test_queries)}")

        comparison_results = {}

        for config in configs:
            config_name = config.get('name', 'unnamed')
            log.info(f"\n{'='*70}")
            log.info(f"Testing Configuration: {config_name}")
            log.info(f"{'='*70}")

            try:
                # Build query engine with this config
                query_engine = build_query_engine_fn(config)

                # Run benchmark
                results = self.run_end_to_end_benchmark(
                    query_engine,
                    test_queries,
                    config
                )

                # Generate summary
                summary = self._generate_summary(config_name, results)
                comparison_results[config_name] = summary

                log.info(f"\n  Summary for {config_name}:")
                log.info(f"    Success Rate: {summary.num_successful}/{summary.num_queries}")
                log.info(f"    Avg Latency: {summary.avg_latency_ms:.2f}ms")
                log.info(f"    Avg Faithfulness: {summary.avg_faithfulness:.4f}")

            except Exception as e:
                log.error(f"Error testing config '{config_name}': {e}")

        return comparison_results

    def _generate_summary(
        self,
        config_name: str,
        results: List[QueryBenchmarkResult]
    ) -> BenchmarkSummary:
        """Generate aggregated summary from individual results."""
        successful_results = [r for r in results if not r.error]

        summary = BenchmarkSummary(
            config_name=config_name,
            num_queries=len(results),
            num_successful=len(successful_results),
            num_failed=len(results) - len(successful_results)
        )

        if not successful_results:
            return summary

        # Aggregate retrieval metrics
        summary.avg_mrr = np.mean([r.retrieval.mrr for r in successful_results])
        summary.avg_ndcg = np.mean([r.retrieval.ndcg_at_k for r in successful_results])
        summary.avg_recall = np.mean([r.retrieval.recall_at_k for r in successful_results])
        summary.avg_precision = np.mean([r.retrieval.precision_at_k for r in successful_results])

        # Aggregate answer metrics
        summary.avg_faithfulness = np.mean([r.answer.faithfulness for r in successful_results])
        summary.avg_answer_relevancy = np.mean([r.answer.answer_relevancy for r in successful_results])
        summary.avg_context_precision = np.mean([r.answer.context_precision for r in successful_results])
        summary.avg_context_recall = np.mean([r.answer.context_recall for r in successful_results])

        # Aggregate performance metrics
        latencies = [r.performance.total_latency_ms for r in successful_results]
        summary.avg_latency_ms = np.mean(latencies)
        summary.p50_latency_ms = np.percentile(latencies, 50)
        summary.p95_latency_ms = np.percentile(latencies, 95)
        summary.p99_latency_ms = np.percentile(latencies, 99)

        total_time_s = sum(latencies) / 1000
        summary.throughput_qps = len(successful_results) / total_time_s if total_time_s > 0 else 0

        cache_hits = sum(1 for r in successful_results if r.performance.cache_hit)
        summary.cache_hit_rate = cache_hits / len(successful_results)

        tps_values = [
            r.performance.tokens_per_second
            for r in successful_results
            if r.performance.tokens_per_second > 0
        ]
        summary.avg_tokens_per_second = np.mean(tps_values) if tps_values else 0

        return summary

    # ==================== Report Generation ====================

    def generate_report(
        self,
        results: List[QueryBenchmarkResult],
        output_path: str,
        format: str = "html"
    ) -> Path:
        """
        Generate benchmark report.

        Args:
            results: Benchmark results
            output_path: Output file path
            format: Report format (html, markdown, csv)

        Returns:
            Path to generated report
        """
        output_file = self.output_dir / output_path

        if format == "html":
            return self._generate_html_report(results, output_file)
        elif format == "markdown":
            return self._generate_markdown_report(results, output_file)
        elif format == "csv":
            return self._generate_csv_report(results, output_file)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_html_report(
        self,
        results: List[QueryBenchmarkResult],
        output_path: Path
    ) -> Path:
        """Generate HTML report with Plotly visualizations."""
        if not PLOTLY_AVAILABLE:
            log.warning("Plotly not available, generating markdown instead")
            return self._generate_markdown_report(results, output_path.with_suffix('.md'))

        summary = self._generate_summary("benchmark", results)

        # Create visualizations
        figs = []

        # 1. Latency distribution
        latencies = [r.performance.total_latency_ms for r in results if not r.error]
        fig_latency = go.Figure(data=[go.Histogram(x=latencies, nbinsx=30)])
        fig_latency.update_layout(
            title="Query Latency Distribution",
            xaxis_title="Latency (ms)",
            yaxis_title="Count"
        )
        figs.append(fig_latency)

        # 2. Retrieval metrics radar chart
        successful = [r for r in results if not r.error]
        if successful:
            categories = ['MRR', 'nDCG', 'Recall', 'Precision']
            values = [
                summary.avg_mrr,
                summary.avg_ndcg,
                summary.avg_recall,
                summary.avg_precision
            ]

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],  # Close the loop
                theta=categories + [categories[0]],
                fill='toself'
            ))
            fig_radar.update_layout(
                title="Retrieval Quality Metrics",
                polar=dict(radialaxis=dict(range=[0, 1]))
            )
            figs.append(fig_radar)

        # 3. Answer quality metrics bar chart
        if successful:
            fig_answer = go.Figure(data=[
                go.Bar(
                    x=['Faithfulness', 'Relevancy', 'Context Precision', 'Context Recall'],
                    y=[
                        summary.avg_faithfulness,
                        summary.avg_answer_relevancy,
                        summary.avg_context_precision,
                        summary.avg_context_recall
                    ]
                )
            ])
            fig_answer.update_layout(
                title="Answer Quality Metrics",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1])
            )
            figs.append(fig_answer)

        # Generate HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>RAG Benchmark Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #333; }",
            "h2 { color: #666; margin-top: 30px; }",
            ".metric { display: inline-block; margin: 10px 20px; }",
            ".metric-label { font-weight: bold; }",
            ".metric-value { font-size: 1.2em; color: #0066cc; }",
            "table { border-collapse: collapse; width: 100%; margin-top: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style>",
            "</head><body>",
            f"<h1>RAG Benchmark Report</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",

            "<h2>Summary</h2>",
            f"<div class='metric'><span class='metric-label'>Queries:</span> ",
            f"<span class='metric-value'>{summary.num_successful}/{summary.num_queries}</span></div>",
            f"<div class='metric'><span class='metric-label'>Avg Latency:</span> ",
            f"<span class='metric-value'>{summary.avg_latency_ms:.2f}ms</span></div>",
            f"<div class='metric'><span class='metric-label'>P95 Latency:</span> ",
            f"<span class='metric-value'>{summary.p95_latency_ms:.2f}ms</span></div>",
            f"<div class='metric'><span class='metric-label'>Throughput:</span> ",
            f"<span class='metric-value'>{summary.throughput_qps:.2f} QPS</span></div>",
            f"<div class='metric'><span class='metric-label'>Cache Hit Rate:</span> ",
            f"<span class='metric-value'>{summary.cache_hit_rate:.2%}</span></div>",
        ]

        # Add visualizations
        for fig in figs:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # Add detailed table
        html_parts.extend([
            "<h2>Detailed Results</h2>",
            "<table>",
            "<tr>",
            "<th>Query</th>",
            "<th>MRR</th>",
            "<th>nDCG</th>",
            "<th>Faithfulness</th>",
            "<th>Relevancy</th>",
            "<th>Latency (ms)</th>",
            "<th>Status</th>",
            "</tr>"
        ])

        for r in results:
            status = "Error" if r.error else "Success"
            html_parts.append(
                f"<tr>"
                f"<td>{r.query_text[:50]}...</td>"
                f"<td>{r.retrieval.mrr:.3f}</td>"
                f"<td>{r.retrieval.ndcg_at_k:.3f}</td>"
                f"<td>{r.answer.faithfulness:.3f}</td>"
                f"<td>{r.answer.answer_relevancy:.3f}</td>"
                f"<td>{r.performance.total_latency_ms:.1f}</td>"
                f"<td>{status}</td>"
                f"</tr>"
            )

        html_parts.extend([
            "</table>",
            "</body></html>"
        ])

        # Write file
        output_path.write_text("\n".join(html_parts))
        log.info(f"HTML report generated: {output_path}")
        return output_path

    def _generate_markdown_report(
        self,
        results: List[QueryBenchmarkResult],
        output_path: Path
    ) -> Path:
        """Generate Markdown report."""
        summary = self._generate_summary("benchmark", results)

        lines = [
            "# RAG Benchmark Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary\n",
            f"- **Total Queries:** {summary.num_queries}",
            f"- **Successful:** {summary.num_successful}",
            f"- **Failed:** {summary.num_failed}",
            "\n### Retrieval Quality\n",
            f"- **Mean Reciprocal Rank:** {summary.avg_mrr:.4f}",
            f"- **nDCG@k:** {summary.avg_ndcg:.4f}",
            f"- **Recall@k:** {summary.avg_recall:.4f}",
            f"- **Precision@k:** {summary.avg_precision:.4f}",
            "\n### Answer Quality\n",
            f"- **Faithfulness:** {summary.avg_faithfulness:.4f}",
            f"- **Answer Relevancy:** {summary.avg_answer_relevancy:.4f}",
            f"- **Context Precision:** {summary.avg_context_precision:.4f}",
            f"- **Context Recall:** {summary.avg_context_recall:.4f}",
            "\n### Performance\n",
            f"- **Avg Latency:** {summary.avg_latency_ms:.2f}ms",
            f"- **P50 Latency:** {summary.p50_latency_ms:.2f}ms",
            f"- **P95 Latency:** {summary.p95_latency_ms:.2f}ms",
            f"- **P99 Latency:** {summary.p99_latency_ms:.2f}ms",
            f"- **Throughput:** {summary.throughput_qps:.2f} queries/second",
            f"- **Cache Hit Rate:** {summary.cache_hit_rate:.2%}",
            f"- **Avg Tokens/Second:** {summary.avg_tokens_per_second:.2f}",
            "\n## Detailed Results\n",
            "| Query | MRR | nDCG | Faith | Relev | Latency | Status |",
            "|-------|-----|------|-------|-------|---------|--------|"
        ]

        for r in results:
            status = "Error" if r.error else "OK"
            lines.append(
                f"| {r.query_text[:30]}... | "
                f"{r.retrieval.mrr:.3f} | "
                f"{r.retrieval.ndcg_at_k:.3f} | "
                f"{r.answer.faithfulness:.3f} | "
                f"{r.answer.answer_relevancy:.3f} | "
                f"{r.performance.total_latency_ms:.0f}ms | "
                f"{status} |"
            )

        output_path.write_text("\n".join(lines))
        log.info(f"Markdown report generated: {output_path}")
        return output_path

    def _generate_csv_report(
        self,
        results: List[QueryBenchmarkResult],
        output_path: Path
    ) -> Path:
        """Generate CSV report for analysis."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'query_id', 'query_text', 'timestamp',
                'mrr', 'ndcg', 'recall', 'precision', 'avg_score',
                'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall',
                'total_latency_ms', 'retrieval_latency_ms', 'generation_latency_ms',
                'cache_hit', 'tokens_generated', 'tokens_per_second',
                'error'
            ])

            # Data rows
            for r in results:
                writer.writerow([
                    r.query_id, r.query_text, r.timestamp,
                    r.retrieval.mrr, r.retrieval.ndcg_at_k,
                    r.retrieval.recall_at_k, r.retrieval.precision_at_k,
                    r.retrieval.avg_relevance_score,
                    r.answer.faithfulness, r.answer.answer_relevancy,
                    r.answer.context_precision, r.answer.context_recall,
                    r.performance.total_latency_ms,
                    r.performance.retrieval_latency_ms,
                    r.performance.generation_latency_ms,
                    r.performance.cache_hit,
                    r.performance.tokens_generated,
                    r.performance.tokens_per_second,
                    r.error or ''
                ])

        log.info(f"CSV report generated: {output_path}")
        return output_path

    def generate_comparison_report(
        self,
        comparison_results: Dict[str, BenchmarkSummary],
        output_path: str
    ) -> Path:
        """
        Generate comparison report for multiple configurations.

        Args:
            comparison_results: Dict mapping config name to summary
            output_path: Output file path

        Returns:
            Path to generated report
        """
        output_file = self.output_dir / output_path

        if not PLOTLY_AVAILABLE:
            log.warning("Plotly not available, generating markdown comparison")
            return self._generate_markdown_comparison(comparison_results, output_file)

        # Create comparison visualizations
        config_names = list(comparison_results.keys())
        summaries = list(comparison_results.values())

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Retrieval Quality',
                'Answer Quality',
                'Latency (P50, P95, P99)',
                'Performance Metrics'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ]
        )

        # Retrieval quality
        fig.add_trace(
            go.Bar(name='MRR', x=config_names, y=[s.avg_mrr for s in summaries]),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='nDCG', x=config_names, y=[s.avg_ndcg for s in summaries]),
            row=1, col=1
        )

        # Answer quality
        fig.add_trace(
            go.Bar(name='Faithfulness', x=config_names, y=[s.avg_faithfulness for s in summaries]),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Relevancy', x=config_names, y=[s.avg_answer_relevancy for s in summaries]),
            row=1, col=2
        )

        # Latency
        fig.add_trace(
            go.Bar(name='P50', x=config_names, y=[s.p50_latency_ms for s in summaries]),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='P95', x=config_names, y=[s.p95_latency_ms for s in summaries]),
            row=2, col=1
        )

        # Performance
        fig.add_trace(
            go.Bar(name='QPS', x=config_names, y=[s.throughput_qps for s in summaries]),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name='Cache Hit Rate', x=config_names, y=[s.cache_hit_rate * 100 for s in summaries]),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Configuration Comparison")

        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Configuration Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Configuration Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            {fig.to_html(full_html=False, include_plotlyjs='cdn')}

            <h2>Summary Table</h2>
            <table>
                <tr>
                    <th>Configuration</th>
                    <th>MRR</th>
                    <th>nDCG</th>
                    <th>Faithfulness</th>
                    <th>Relevancy</th>
                    <th>P95 Latency</th>
                    <th>QPS</th>
                </tr>
        """

        for name, summary in comparison_results.items():
            html += f"""
                <tr>
                    <td>{name}</td>
                    <td>{summary.avg_mrr:.4f}</td>
                    <td>{summary.avg_ndcg:.4f}</td>
                    <td>{summary.avg_faithfulness:.4f}</td>
                    <td>{summary.avg_answer_relevancy:.4f}</td>
                    <td>{summary.p95_latency_ms:.2f}ms</td>
                    <td>{summary.throughput_qps:.2f}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        output_file.write_text(html)
        log.info(f"Comparison report generated: {output_file}")
        return output_file

    def _generate_markdown_comparison(
        self,
        comparison_results: Dict[str, BenchmarkSummary],
        output_path: Path
    ) -> Path:
        """Generate markdown comparison report."""
        lines = [
            "# Configuration Comparison Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary\n",
            "| Config | MRR | nDCG | Faithfulness | Relevancy | P95 Latency | QPS |",
            "|--------|-----|------|--------------|-----------|-------------|-----|"
        ]

        for name, summary in comparison_results.items():
            lines.append(
                f"| {name} | "
                f"{summary.avg_mrr:.4f} | "
                f"{summary.avg_ndcg:.4f} | "
                f"{summary.avg_faithfulness:.4f} | "
                f"{summary.avg_answer_relevancy:.4f} | "
                f"{summary.p95_latency_ms:.2f}ms | "
                f"{summary.throughput_qps:.2f} |"
            )

        output_path.write_text("\n".join(lines))
        log.info(f"Markdown comparison generated: {output_path}")
        return output_path

    # ==================== Test Data Generation ====================

    @staticmethod
    def generate_synthetic_dataset(
        num_queries: int = 20,
        categories: Optional[List[str]] = None
    ) -> List[TestQuery]:
        """
        Generate synthetic test dataset for benchmarking.

        Args:
            num_queries: Number of queries to generate
            categories: Query categories (e.g., ['factual', 'reasoning'])

        Returns:
            List of TestQuery objects
        """
        if categories is None:
            categories = ['factual', 'definition', 'howto', 'comparison']

        templates = {
            'factual': [
                "What is {}?",
                "When did {} happen?",
                "Who created {}?",
                "Where is {} located?",
            ],
            'definition': [
                "Define {}.",
                "Explain what {} means.",
                "What does {} refer to?",
            ],
            'howto': [
                "How do I {}?",
                "What are the steps to {}?",
                "Guide me through {}.",
            ],
            'comparison': [
                "What is the difference between {} and {}?",
                "Compare {} with {}.",
                "Which is better: {} or {}?",
            ]
        }

        topics = [
            "machine learning", "neural networks", "transformers", "RAG",
            "embeddings", "vector databases", "semantic search", "LLMs",
            "fine-tuning", "prompt engineering", "retrieval", "ranking",
            "natural language processing", "deep learning", "attention mechanism"
        ]

        queries = []
        for i in range(num_queries):
            category = categories[i % len(categories)]
            template = np.random.choice(templates[category])

            if '{}' in template:
                # Count placeholders
                num_placeholders = template.count('{}')
                selected_topics = np.random.choice(topics, num_placeholders, replace=False)
                query_text = template.format(*selected_topics)
            else:
                query_text = template

            queries.append(TestQuery(
                id=f"test_{i+1}",
                query=query_text,
                category=category,
                # Note: Ground truth would need to be added manually or via LLM
                relevant_doc_ids=[f"doc_{j}" for j in range(1, 4)],
                expected_keywords=list(selected_topics) if '{}' in template else []
            ))

        log.info(f"Generated {num_queries} synthetic test queries")
        return queries

    @staticmethod
    def load_test_dataset(file_path: str) -> List[TestQuery]:
        """
        Load test dataset from JSON file.

        Expected format:
        [
            {
                "id": "q1",
                "query": "What is RAG?",
                "ground_truth_answer": "RAG stands for...",
                "relevant_doc_ids": ["doc1", "doc2"],
                "expected_keywords": ["retrieval", "generation"],
                "category": "definition"
            },
            ...
        ]

        Args:
            file_path: Path to JSON file

        Returns:
            List of TestQuery objects
        """
        with open(file_path) as f:
            data = json.load(f)

        queries = [TestQuery(**item) for item in data]
        log.info(f"Loaded {len(queries)} test queries from {file_path}")
        return queries

    @staticmethod
    def save_test_dataset(queries: List[TestQuery], file_path: str):
        """Save test dataset to JSON file."""
        data = [asdict(q) for q in queries]
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        log.info(f"Saved {len(queries)} test queries to {file_path}")


# ==================== CLI for Testing ====================

def main():
    """CLI for running benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Benchmark Suite")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "comprehensive"],
        default=os.getenv("BENCHMARK_MODE", "standard"),
        help="Benchmark mode"
    )
    parser.add_argument(
        "--generate-dataset",
        type=int,
        metavar="N",
        help="Generate N synthetic test queries"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks",
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Initialize benchmark
    benchmark = RAGBenchmark(output_dir=args.output_dir)

    # Generate synthetic dataset
    if args.generate_dataset:
        queries = benchmark.generate_synthetic_dataset(num_queries=args.generate_dataset)
        output_file = f"test_queries_{args.generate_dataset}.json"
        benchmark.save_test_dataset(queries, output_file)
        print(f"\nGenerated synthetic dataset: {output_file}")
        print(f"  Queries: {len(queries)}")
        print(f"  Categories: {set(q.category for q in queries)}")
        return

    # Example: Test metrics calculation
    print("\n" + "="*70)
    print("RAG Benchmark Suite - Example Metrics")
    print("="*70)

    # Test retrieval metrics
    print("\n1. Retrieval Metrics")
    retrieved = ["doc3", "doc1", "doc5", "doc2"]
    relevant = ["doc1", "doc2", "doc4"]

    mrr = benchmark.calculate_mrr(retrieved, relevant)
    ndcg = benchmark.calculate_ndcg_at_k(retrieved, relevant)
    recall = benchmark.calculate_recall_at_k(retrieved, relevant)
    precision = benchmark.calculate_precision_at_k(retrieved, relevant)

    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant:  {relevant}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  nDCG@4: {ndcg:.4f}")
    print(f"  Recall@4: {recall:.4f}")
    print(f"  Precision@4: {precision:.4f}")

    # Test answer metrics
    print("\n2. Answer Quality Metrics")
    query = "What is machine learning?"
    answer = "Machine learning is a subset of AI that enables systems to learn from data."
    context = [
        "Machine learning is a field of artificial intelligence.",
        "AI systems can learn patterns from data automatically."
    ]
    keywords = ["machine learning", "AI", "data"]

    faithfulness = benchmark.calculate_faithfulness(answer, context)
    relevancy = benchmark.calculate_answer_relevancy(query, answer, keywords)

    print(f"  Query: {query}")
    print(f"  Answer: {answer}")
    print(f"  Faithfulness: {faithfulness:.4f}")
    print(f"  Relevancy: {relevancy:.4f}")

    print("\n" + "="*70)
    print("Benchmark suite ready!")
    print("="*70)
    print("\nUsage examples:")
    print("  1. Generate test dataset:")
    print("     python rag_benchmark.py --generate-dataset 50")
    print("\n  2. Run benchmark in your RAG pipeline:")
    print("     from utils.rag_benchmark import RAGBenchmark")
    print("     benchmark = RAGBenchmark()")
    print("     results = benchmark.run_end_to_end_benchmark(engine, queries)")
    print("     benchmark.generate_report(results, 'report.html')")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
