#!/usr/bin/env python3
"""
Debug script to test hybrid search with different parameters.
Helps diagnose why you're getting identical results.
"""

import os
import sys

# Disable semantic cache for testing
os.environ["ENABLE_SEMANTIC_CACHE"] = "0"

# Test different hybrid configurations
test_configs = [
    {"HYBRID_ALPHA": "1.0", "name": "Pure Vector"},
    {"HYBRID_ALPHA": "0.5", "name": "Balanced Hybrid"},
    {"HYBRID_ALPHA": "0.0", "name": "Pure BM25"},
]

test_query = "What is the main topic?"

print("=" * 70)
print("HYBRID SEARCH DEBUG TEST")
print("=" * 70)
print(f"Query: \"{test_query}\"")
print(f"Cache: DISABLED (ENABLE_SEMANTIC_CACHE=0)")
print("=" * 70)

for i, config in enumerate(test_configs, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}/3: {config['name']} (α={config['HYBRID_ALPHA']})")
    print(f"{'='*70}")

    # Set environment variable
    os.environ["HYBRID_ALPHA"] = config["HYBRID_ALPHA"]
    os.environ["TOP_K"] = "3"  # Get top 3 for comparison

    # Import fresh (to pick up new env vars)
    # Note: This may not work if modules are already loaded
    # Better to run this script 3 times with different HYBRID_ALPHA

    try:
        from rag_low_level_m1_16gb_verbose import (
            build_embed_model,
            make_vector_store,
            HybridRetriever
        )

        print(f"Loading models...")
        embed_model = build_embed_model()
        vector_store = make_vector_store()

        print(f"Creating retriever with α={config['HYBRID_ALPHA']}...")
        retriever = HybridRetriever(
            vector_store=vector_store,
            embed_model=embed_model,
            similarity_top_k=3,
            alpha=float(config["HYBRID_ALPHA"]),
            enable_metadata_filter=False,
            mmr_threshold=0.0
        )

        # Run query
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=test_query)

        print(f"\nRetrieving results...")
        results = retriever._retrieve(query_bundle)

        print(f"\n📊 Results for {config['name']}:")
        for j, node_with_score in enumerate(results, 1):
            node = node_with_score.node
            score = node_with_score.score
            text_preview = node.text[:100].replace('\n', ' ')

            print(f"\n  [{j}] Score: {score:.4f}")
            print(f"      ID: {node.node_id}")
            print(f"      Text: {text_preview}...")

        print(f"\n✓ Test {i} completed")

    except Exception as e:
        print(f"\n❌ Error in test {i}: {e}")
        import traceback
        traceback.print_exc()

    # Clean up to allow fresh import
    # (This won't fully work in Python - better to run script 3 times)
    print("\n" + "="*70)

print("\n" + "="*70)
print("DEBUG TEST COMPLETE")
print("="*70)
print("\nIMPORTANT: Due to Python module caching, run this script separately")
print("for each configuration:")
print()
print("  HYBRID_ALPHA=1.0 python debug_hybrid_search.py  # Pure vector")
print("  HYBRID_ALPHA=0.5 python debug_hybrid_search.py  # Balanced")
print("  HYBRID_ALPHA=0.0 python debug_hybrid_search.py  # Pure BM25")
print()
print("Then compare the results manually.")
print("="*70)
