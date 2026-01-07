#!/usr/bin/env python3
"""
Integration Example: Using RAG Benchmark Suite with Real Pipeline

This script shows how to integrate the benchmark suite with the actual
RAG pipeline (rag_low_level_m1_16gb_verbose.py).

Usage:
    # Run retrieval-only benchmark
    python integrate_with_rag.py --mode retrieval

    # Run full end-to-end benchmark
    python integrate_with_rag.py --mode end-to-end

    # Compare configurations
    python integrate_with_rag.py --mode compare
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


def load_test_queries():
    """Load test queries from sample file."""
    benchmark = RAGBenchmark()
    test_queries_path = Path(__file__).parent / "sample_test_queries.json"

    if test_queries_path.exists():
        return benchmark.load_test_dataset(str(test_queries_path))
    else:
        log.warning("Sample test queries not found, generating synthetic dataset")
        return benchmark.generate_synthetic_dataset(num_queries=20)


def benchmark_retrieval_only():
    """
    Example: Benchmark retrieval quality without LLM generation.

    This is faster and useful for tuning retrieval parameters.
    """
    log.info("\n" + "="*70)
    log.info("Retrieval-Only Benchmark Example")
    log.info("="*70)

    benchmark = RAGBenchmark(output_dir="benchmarks/integration")
    test_queries = load_test_queries()[:10]  # Use first 10 for quick test

    log.info(f"\nLoaded {len(test_queries)} test queries")
    log.info("\nTo run this benchmark with your actual RAG pipeline:")
    log.info("1. Import your retriever:")
    log.info("   from rag_low_level_m1_16gb_verbose import VectorDBRetriever")
    log.info("\n2. Initialize retriever:")
    log.info("   retriever = VectorDBRetriever(vector_store, embed_model, top_k=4)")
    log.info("\n3. Run benchmark:")
    log.info("   results = benchmark.run_retrieval_benchmark(")
    log.info("       retriever=retriever,")
    log.info("       test_queries=test_queries,")
    log.info("       top_k=4")
    log.info("   )")
    log.info("\n4. Generate report:")
    log.info("   benchmark.generate_report(results, 'retrieval_benchmark.html')")

    log.info("\n" + "="*70)
    log.info("Note: This example shows the workflow.")
    log.info("To run with actual pipeline, initialize your retriever first.")
    log.info("="*70)


def benchmark_end_to_end():
    """
    Example: Benchmark full RAG pipeline (retrieval + generation).

    This measures both retrieval and answer quality.
    """
    log.info("\n" + "="*70)
    log.info("End-to-End Benchmark Example")
    log.info("="*70)

    benchmark = RAGBenchmark(output_dir="benchmarks/integration")
    test_queries = load_test_queries()[:10]

    log.info(f"\nLoaded {len(test_queries)} test queries")
    log.info("\nTo run this benchmark with your actual RAG pipeline:")
    log.info("1. Build your query engine:")
    log.info("   from rag_low_level_m1_16gb_verbose import build_llm, VectorDBRetriever")
    log.info("   from llama_index.core.query_engine import RetrieverQueryEngine")
    log.info("")
    log.info("   retriever = VectorDBRetriever(vector_store, embed_model)")
    log.info("   llm = build_llm()")
    log.info("   query_engine = RetrieverQueryEngine(retriever=retriever, llm=llm)")
    log.info("\n2. Run benchmark:")
    log.info("   results = benchmark.run_end_to_end_benchmark(")
    log.info("       query_engine=query_engine,")
    log.info("       test_queries=test_queries")
    log.info("   )")
    log.info("\n3. Generate report:")
    log.info("   benchmark.generate_report(results, 'e2e_benchmark.html')")

    log.info("\n" + "="*70)
    log.info("Expected outputs:")
    log.info("  - Retrieval metrics (MRR, nDCG, Recall, Precision)")
    log.info("  - Answer quality (Faithfulness, Relevancy)")
    log.info("  - Performance (Latency, Throughput, Cache Hit Rate)")
    log.info("="*70)


def benchmark_configuration_comparison():
    """
    Example: Compare multiple RAG configurations.

    This helps identify which optimizations provide the best improvements.
    """
    log.info("\n" + "="*70)
    log.info("Configuration Comparison Example")
    log.info("="*70)

    benchmark = RAGBenchmark(output_dir="benchmarks/integration")
    test_queries = load_test_queries()[:10]

    # Define configurations to compare
    configs = [
        {
            "name": "baseline",
            "enable_reranking": False,
            "enable_cache": False,
            "enable_hyde": False,
            "top_k": 4,
            "chunk_size": 700,
            "chunk_overlap": 150
        },
        {
            "name": "with_reranking",
            "enable_reranking": True,
            "enable_cache": False,
            "enable_hyde": False,
            "top_k": 12,  # Retrieve more, rerank to 4
            "chunk_size": 700,
            "chunk_overlap": 150
        },
        {
            "name": "with_cache",
            "enable_reranking": True,
            "enable_cache": True,
            "enable_hyde": False,
            "top_k": 12,
            "chunk_size": 700,
            "chunk_overlap": 150
        },
        {
            "name": "full_optimized",
            "enable_reranking": True,
            "enable_cache": True,
            "enable_hyde": True,
            "top_k": 12,
            "chunk_size": 600,  # Slightly smaller chunks
            "chunk_overlap": 120
        }
    ]

    log.info(f"\nComparing {len(configs)} configurations:")
    for i, config in enumerate(configs, 1):
        log.info(f"  {i}. {config['name']}")
        log.info(f"     - Reranking: {config['enable_reranking']}")
        log.info(f"     - Caching: {config['enable_cache']}")
        log.info(f"     - HyDE: {config['enable_hyde']}")
        log.info(f"     - Top-K: {config['top_k']}")

    log.info("\nTo run this comparison:")
    log.info("1. Define a function to build query engine from config:")
    log.info("")
    log.info("   def build_query_engine(config):")
    log.info("       # Set environment variables")
    log.info("       os.environ['TOP_K'] = str(config['top_k'])")
    log.info("       os.environ['CHUNK_SIZE'] = str(config['chunk_size'])")
    log.info("       os.environ['ENABLE_RERANKING'] = str(int(config['enable_reranking']))")
    log.info("")
    log.info("       # Build components")
    log.info("       vector_store = make_vector_store()")
    log.info("       embed_model = build_embed_model()")
    log.info("       retriever = VectorDBRetriever(vector_store, embed_model)")
    log.info("")
    log.info("       # Add reranking if enabled")
    log.info("       if config['enable_reranking']:")
    log.info("           from utils.reranker import Reranker")
    log.info("           reranker = Reranker()")
    log.info("           # Wrap retriever with reranker...")
    log.info("")
    log.info("       llm = build_llm()")
    log.info("       return RetrieverQueryEngine(retriever=retriever, llm=llm)")
    log.info("\n2. Run comparison:")
    log.info("   comparison = benchmark.compare_configurations(")
    log.info("       configs=configs,")
    log.info("       test_queries=test_queries,")
    log.info("       build_query_engine_fn=build_query_engine")
    log.info("   )")
    log.info("\n3. Generate comparison report:")
    log.info("   benchmark.generate_comparison_report(")
    log.info("       comparison,")
    log.info("       'config_comparison.html'")
    log.info("   )")

    log.info("\n" + "="*70)
    log.info("Expected insights:")
    log.info("  - Reranking: +15-30% retrieval quality, +100-200ms latency")
    log.info("  - Caching: +50-90% cache hit rate, -70-90% latency for hits")
    log.info("  - HyDE: +5-15% relevancy, +500-1000ms latency")
    log.info("  - Smaller chunks: +10-20% precision, -5-10% recall")
    log.info("="*70)


def benchmark_with_custom_metrics():
    """
    Example: Using benchmark suite with custom evaluation logic.
    """
    log.info("\n" + "="*70)
    log.info("Custom Metrics Example")
    log.info("="*70)

    log.info("\nThe benchmark suite provides flexible metric calculation.")
    log.info("You can:")
    log.info("  1. Use built-in metrics (MRR, nDCG, Faithfulness, etc.)")
    log.info("  2. Add custom metrics by extending RAGBenchmark class")
    log.info("  3. Use external evaluation (e.g., LLM-as-judge)")

    log.info("\nExample: Add custom metric for answer conciseness")
    log.info("")
    log.info("class CustomBenchmark(RAGBenchmark):")
    log.info("    def calculate_conciseness(self, answer: str) -> float:")
    log.info('        """Score based on answer length (shorter is better)."""')
    log.info("        # Ideal length: 50-200 characters")
    log.info("        length = len(answer)")
    log.info("        if 50 <= length <= 200:")
    log.info("            return 1.0")
    log.info("        elif length < 50:")
    log.info("            return length / 50  # Penalty for too short")
    log.info("        else:")
    log.info("            return 1.0 - min((length - 200) / 300, 0.5)")
    log.info("")
    log.info("    def calculate_keyword_coverage(")
    log.info("        self,")
    log.info("        answer: str,")
    log.info("        required_keywords: list")
    log.info("    ) -> float:")
    log.info('        """Check if answer covers required keywords."""')
    log.info("        answer_lower = answer.lower()")
    log.info("        found = sum(1 for kw in required_keywords")
    log.info("                   if kw.lower() in answer_lower)")
    log.info("        return found / len(required_keywords)")

    log.info("\n" + "="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline Integration Examples")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "end-to-end", "compare", "custom", "all"],
        default="all",
        help="Example mode"
    )
    args = parser.parse_args()

    if args.mode in ["retrieval", "all"]:
        benchmark_retrieval_only()

    if args.mode in ["end-to-end", "all"]:
        benchmark_end_to_end()

    if args.mode in ["compare", "all"]:
        benchmark_configuration_comparison()

    if args.mode in ["custom", "all"]:
        benchmark_with_custom_metrics()

    log.info("\n" + "="*70)
    log.info("Integration Examples Complete!")
    log.info("="*70)
    log.info("\nNext steps:")
    log.info("  1. Initialize your RAG pipeline components")
    log.info("  2. Load or create test queries")
    log.info("  3. Run benchmarks using examples above")
    log.info("  4. Analyze reports in benchmarks/integration/")
    log.info("\nFor complete working code, see:")
    log.info("  - examples/benchmark_example.py (mock examples)")
    log.info("  - examples/BENCHMARK_README.md (documentation)")
    log.info("  - utils/rag_benchmark.py (benchmark suite)")


if __name__ == "__main__":
    main()
