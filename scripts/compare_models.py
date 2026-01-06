#!/usr/bin/env python3
"""
Compare embedding models (bge-large vs bge-small vs minilm).

Tests model performance, throughput, and storage requirements.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --backend mlx
    python scripts/compare_models.py --test-query "custom query"
"""

import argparse
import time
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model(
    model_name: str,
    embed_dim: int,
    query: str,
    backend: str = "mlx"
) -> Dict:
    """
    Test a single embedding model.

    Args:
        model_name: HuggingFace model name
        embed_dim: Expected embedding dimension
        query: Test query
        backend: Embedding backend (mlx or huggingface)

    Returns:
        Dict with performance metrics
    """
    os.environ["EMBED_MODEL"] = model_name
    os.environ["EMBED_DIM"] = str(embed_dim)
    os.environ["EMBED_BACKEND"] = backend

    # Force reimport to pick up new env vars
    import importlib
    import rag_low_level_m1_16gb_verbose
    importlib.reload(rag_low_level_m1_16gb_verbose)

    from rag_low_level_m1_16gb_verbose import build_embed_model

    print(f"\n{'='*70}")
    print(f"Testing: {model_name} ({embed_dim}d)")
    print(f"{'='*70}")

    # Load model
    start = time.perf_counter()
    model = build_embed_model()
    load_time = time.perf_counter() - start

    # Test single query embedding
    start = time.perf_counter()
    embedding = model._get_query_embedding(query)
    query_time = (time.perf_counter() - start) * 1000  # ms

    # Test batch embedding (100 texts)
    texts = [f"Sample text number {i} for embedding" for i in range(100)]
    start = time.perf_counter()
    embeddings = model._get_text_embeddings(texts)
    batch_time = time.perf_counter() - start
    throughput = len(texts) / batch_time

    # Storage calculation
    storage_per_chunk = len(embedding) * 4  # 4 bytes per float32
    storage_47k = (47651 * storage_per_chunk) / (1024 * 1024)  # MB

    print(f"\n📊 Results:")
    print(f"  Load time:        {load_time:.2f}s")
    print(f"  Query latency:    {query_time:.2f}ms")
    print(f"  Batch throughput: {throughput:.1f} emb/sec")
    print(f"  Vector dimension: {len(embedding)}")
    print(f"  Storage/chunk:    {storage_per_chunk} bytes")
    print(f"  Storage (47k):    {storage_47k:.1f} MB")

    return {
        "model": model_name,
        "dim": embed_dim,
        "load_time": load_time,
        "query_time": query_time,
        "throughput": throughput,
        "storage_mb": storage_47k,
    }


def estimate_indexing_time(throughput: float, num_chunks: int = 47651) -> float:
    """Estimate time to index full corpus"""
    return num_chunks / throughput / 60  # minutes


def main():
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument(
        "--test-query",
        default="What did Elena say about traveling to Morocco?"
    )
    parser.add_argument(
        "--backend",
        default="mlx",
        choices=["mlx", "huggingface"],
        help="Embedding backend to test"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=47651,
        help="Number of chunks for time estimation"
    )
    args = parser.parse_args()

    print("="*70)
    print("  Embedding Model Comparison")
    print("="*70)
    print(f"Backend: {args.backend.upper()}")
    print(f"Test query: {args.test_query}")
    print()

    # Models to test (name, dimension)
    models = [
        ("BAAI/bge-small-en", 384),
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
        ("BAAI/bge-large-en-v1.5", 1024),
    ]

    results: List[Dict] = []

    for model_name, dim in models:
        try:
            result = test_model(model_name, dim, args.test_query, args.backend)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*70}\n")

        baseline = results[-1]  # bge-large as baseline

        print(f"Baseline: {baseline['model']}\n")

        for r in results:
            speedup = baseline["query_time"] / r["query_time"]
            throughput_gain = r["throughput"] / baseline["throughput"]
            storage_ratio = r["dim"] / baseline["dim"]
            indexing_time = estimate_indexing_time(r["throughput"], args.num_chunks)

            print(f"{'─'*70}")
            print(f"{r['model']}")
            print(f"{'─'*70}")
            print(f"  Query speed:       {speedup:.1f}x faster")
            print(f"  Indexing speed:    {throughput_gain:.1f}x faster")
            print(f"  Storage size:      {storage_ratio:.1%} of baseline")
            print(f"  Storage (47k):     {r['storage_mb']:.1f} MB")
            print(f"  Est. index time:   {indexing_time:.1f} minutes")
            print()

        # Recommendations
        print(f"{'='*70}")
        print("  RECOMMENDATIONS")
        print(f"{'='*70}\n")

        print("For Chat Logs (prioritize speed):")
        print(f"  → BAAI/bge-small-en")
        print(f"    • 3x faster than bge-large")
        print(f"    • 1/3 storage size")
        print(f"    • Likely sufficient quality for conversations\n")

        print("For Documents (prioritize quality):")
        print(f"  → BAAI/bge-large-en-v1.5")
        print(f"    • Best semantic understanding")
        print(f"    • Worth the extra time for complex docs\n")

        print("For Budget/Speed (minimal setup):")
        print(f"  → sentence-transformers/all-MiniLM-L6-v2")
        print(f"    • Fast and lightweight")
        print(f"    • Good baseline performance\n")

    print(f"{'='*70}")
    print("  Test Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
