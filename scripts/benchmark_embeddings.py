#!/usr/bin/env python3
"""
Benchmark embedding backends (HuggingFace vs MLX) on Apple Silicon.

Usage:
    python scripts/benchmark_embeddings.py --backend mlx
    python scripts/benchmark_embeddings.py --compare
    python scripts/benchmark_embeddings.py --compare --model BAAI/bge-small-en
"""

import time
import argparse
import numpy as np
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_single_embedding(model, text: str, iterations: int = 100) -> float:
    """Benchmark single text embedding (query scenario)"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model._get_query_embedding(text)
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000  # Convert to ms


def benchmark_batch_embedding(model, texts: List[str]) -> Dict:
    """Benchmark batch embedding (indexing scenario)"""
    start = time.perf_counter()
    embeddings = model._get_text_embeddings(texts)
    elapsed = time.perf_counter() - start

    return {
        "elapsed": elapsed,
        "throughput": len(texts) / elapsed,
        "per_item_ms": elapsed / len(texts) * 1000
    }


def verify_consistency(emb1: List[float], emb2: List[float]) -> float:
    """Compare two embeddings (cosine similarity)"""
    v1 = np.array(emb1)
    v2 = np.array(emb2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding backends")
    parser.add_argument("--backend", default="mlx", choices=["mlx", "huggingface"],
                       help="Backend to test (default: mlx)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare both backends")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                       help="Model to test (default: BAAI/bge-large-en-v1.5)")
    parser.add_argument("--batch-sizes", default="10,50,100,500",
                       help="Comma-separated batch sizes to test (default: 10,50,100,500)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Embedding Benchmark - Apple Silicon")
    print("=" * 70)
    print(f"Model: {args.model}")
    print()

    # Test texts
    single_text = "What did Elena say about traveling to Morocco?"
    batch_texts = [f"Sample text number {i} for embedding benchmark" for i in range(500)]

    if args.compare:
        # Compare both backends
        print("Running comparison between HuggingFace and MLX...")
        print()
        results = {}

        for backend in ["huggingface", "mlx"]:
            print(f"{'─' * 70}")
            print(f"  Testing: {backend.upper()}")
            print(f"{'─' * 70}")

            os.environ["EMBED_BACKEND"] = backend
            os.environ["EMBED_MODEL"] = args.model

            try:
                # Import after setting env vars
                import importlib
                import rag_low_level_m1_16gb_verbose
                importlib.reload(rag_low_level_m1_16gb_verbose)

                model = rag_low_level_m1_16gb_verbose.build_embed_model()

                # Single embedding benchmark
                print("\n1. Single Query Embedding:")
                single_latency = benchmark_single_embedding(model, single_text, iterations=50)
                print(f"   Latency: {single_latency:.2f}ms (avg over 50 iterations)")

                # Batch embedding benchmarks
                print("\n2. Batch Embedding:")
                batch_results = []
                for batch_size in map(int, args.batch_sizes.split(",")):
                    texts = batch_texts[:batch_size]
                    result = benchmark_batch_embedding(model, texts)
                    batch_results.append((batch_size, result))
                    print(f"   Batch {batch_size:3d}: {result['throughput']:6.1f} emb/sec"
                          f" ({result['per_item_ms']:5.2f}ms per item)")

                results[backend] = {
                    "model": model,
                    "latency": single_latency,
                    "batch_results": batch_results
                }

            except Exception as e:
                print(f"\n❌ Error testing {backend}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Consistency check
        if len(results) == 2:
            print(f"\n{'=' * 70}")
            print("  Consistency Check")
            print(f"{'=' * 70}")
            try:
                emb_hf = results["huggingface"]["model"]._get_query_embedding(single_text)
                emb_mlx = results["mlx"]["model"]._get_query_embedding(single_text)
                similarity = verify_consistency(emb_hf, emb_mlx)
                print(f"Cosine similarity: {similarity:.6f}")
                if similarity > 0.999:
                    print("✅ Embeddings are consistent (similarity > 0.999)")
                elif similarity > 0.99:
                    print("⚠️  Embeddings slightly different (similarity > 0.99)")
                else:
                    print(f"❌ Warning: Embeddings differ significantly (similarity: {similarity:.6f})")
            except Exception as e:
                print(f"❌ Consistency check failed: {e}")

            # Performance summary
            print(f"\n{'=' * 70}")
            print("  Performance Summary")
            print(f"{'=' * 70}")

            hf_lat = results["huggingface"]["latency"]
            mlx_lat = results["mlx"]["latency"]
            speedup = hf_lat / mlx_lat

            print(f"\nSingle Query Embedding:")
            print(f"  HuggingFace: {hf_lat:.2f}ms")
            print(f"  MLX:         {mlx_lat:.2f}ms")
            print(f"  Speedup:     {speedup:.1f}x faster")

            print(f"\nBatch Embedding (largest batch):")
            hf_batch = results["huggingface"]["batch_results"][-1][1]
            mlx_batch = results["mlx"]["batch_results"][-1][1]
            batch_speedup = mlx_batch["throughput"] / hf_batch["throughput"]

            print(f"  HuggingFace: {hf_batch['throughput']:.1f} emb/sec")
            print(f"  MLX:         {mlx_batch['throughput']:.1f} emb/sec")
            print(f"  Speedup:     {batch_speedup:.1f}x faster")

            # Estimate full corpus indexing time
            chunks = 47651
            hf_time_min = chunks / hf_batch["throughput"] / 60
            mlx_time_min = chunks / mlx_batch["throughput"] / 60

            print(f"\nEstimated Time for 47,651 Chunks:")
            print(f"  HuggingFace: {hf_time_min:.1f} minutes")
            print(f"  MLX:         {mlx_time_min:.1f} minutes")
            print(f"  Time saved:  {hf_time_min - mlx_time_min:.1f} minutes")

    else:
        # Single backend test
        os.environ["EMBED_BACKEND"] = args.backend
        os.environ["EMBED_MODEL"] = args.model

        print(f"Testing backend: {args.backend.upper()}")
        print()

        from rag_low_level_m1_16gb_verbose import build_embed_model
        model = build_embed_model()

        print("\n1. Single Query Embedding:")
        single_latency = benchmark_single_embedding(model, single_text, iterations=50)
        print(f"   Latency: {single_latency:.2f}ms (avg over 50 iterations)")

        print("\n2. Batch Embedding:")
        for batch_size in map(int, args.batch_sizes.split(",")):
            texts = batch_texts[:batch_size]
            result = benchmark_batch_embedding(model, texts)
            print(f"   Batch {batch_size:3d}: {result['throughput']:6.1f} emb/sec"
                  f" ({result['per_item_ms']:5.2f}ms per item)")

    print(f"\n{'=' * 70}")
    print("  Benchmark Complete")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
