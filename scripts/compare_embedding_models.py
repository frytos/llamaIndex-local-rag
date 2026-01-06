#!/usr/bin/env python3
"""
Compare embedding models for multilingual (French/English) RAG.

This script:
1. Queries the current index (bge-small-en) with French and English queries
2. Shows retrieval quality metrics
3. Recommends whether to switch to multilingual-e5-small

Usage:
    python scripts/compare_embedding_models.py

Requirements:
    - Current index must be completed (messenger_clean_cs700_ov150)
    - Both embedding models will be tested
"""

import os
import sys
import time
from typing import List, Dict, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_embedding_similarity(model_name: str, model_dim: int):
    """Test how well a model handles French/English semantic similarity."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}\n")

    # Initialize embedding model
    if "multilingual" in model_name.lower():
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        model = HuggingFaceEmbedding(model_name=model_name, device="mps")
        backend = "HuggingFace"
    else:
        # Try MLX first for bge-small-en
        try:
            from mlx_embedding import MLXEmbedding
            model = MLXEmbedding(model_name=model_name)
            backend = "MLX"
        except Exception as e:
            print(f"MLX not available ({e}), falling back to HuggingFace")
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            model = HuggingFaceEmbedding(model_name=model_name, device="mps")
            backend = "HuggingFace"

    print(f"✓ Model loaded ({backend})")

    # Test cases: pairs of semantically similar sentences in French and English
    test_pairs = [
        # French-French similarity
        ("J'adore les restaurants à Paris", "Les restaurants parisiens sont excellents"),

        # English-English similarity
        ("I love restaurants in Paris", "Parisian restaurants are excellent"),

        # French-English cross-lingual (same meaning)
        ("J'adore les restaurants à Paris", "I love restaurants in Paris"),

        # Unrelated sentences (should have low similarity)
        ("J'adore les restaurants à Paris", "Le football est un sport populaire"),
        ("I love restaurants in Paris", "Football is a popular sport"),
    ]

    print("\n📊 Semantic Similarity Tests:")
    print("-" * 70)

    results = []

    for i, (text1, text2) in enumerate(test_pairs, 1):
        # Get embeddings
        emb1 = model.get_text_embedding(text1)
        emb2 = model.get_text_embedding(text2)

        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Determine test type
        is_french_1 = any(c in text1.lower() for c in ['à', 'é', 'è', 'ê'])
        is_french_2 = any(c in text2.lower() for c in ['à', 'é', 'è', 'ê'])

        if is_french_1 and is_french_2:
            test_type = "FR-FR"
            color = "\033[94m"  # Blue
        elif not is_french_1 and not is_french_2:
            test_type = "EN-EN"
            color = "\033[92m"  # Green
        elif is_french_1 != is_french_2:
            test_type = "FR-EN"
            color = "\033[93m"  # Yellow
        else:
            test_type = "OTHER"
            color = "\033[0m"

        reset = "\033[0m"

        print(f"\n{color}Test {i} ({test_type}):{reset}")
        print(f"  Text 1: {text1[:60]}...")
        print(f"  Text 2: {text2[:60]}...")
        print(f"  Similarity: {similarity:.4f}")

        results.append({
            'test': i,
            'type': test_type,
            'similarity': similarity,
            'text1': text1,
            'text2': text2
        })

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    fr_fr = [r['similarity'] for r in results if r['type'] == 'FR-FR']
    en_en = [r['similarity'] for r in results if r['type'] == 'EN-EN']
    fr_en = [r['similarity'] for r in results if r['type'] == 'FR-EN']

    print(f"\nAverage Similarities:")
    print(f"  French-French:  {np.mean(fr_fr):.4f}" if fr_fr else "  French-French:  N/A")
    print(f"  English-English: {np.mean(en_en):.4f}" if en_en else "  English-English: N/A")
    print(f"  French-English:  {np.mean(fr_en):.4f}" if fr_en else "  French-English:  N/A")

    return {
        'model': model_name,
        'backend': backend,
        'results': results,
        'avg_fr_fr': np.mean(fr_fr) if fr_fr else 0,
        'avg_en_en': np.mean(en_en) if en_en else 0,
        'avg_fr_en': np.mean(fr_en) if fr_en else 0,
    }


def query_current_index(table_name: str = "messenger_clean_cs700_ov150"):
    """Query the current index with test queries."""
    print(f"\n{'='*70}")
    print(f"QUERYING CURRENT INDEX: {table_name}")
    print(f"{'='*70}\n")

    # Check if table exists and has data
    import psycopg2
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="fryt",
            password="frytos",
            database="vector_db"
        )
        conn.autocommit = True

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            print(f"✓ Table exists with {count:,} chunks")

            if count == 0:
                print("⚠️  Table is empty - indexing may still be in progress")
                return None

        conn.close()
    except Exception as e:
        print(f"❌ Cannot access table: {e}")
        print("   Make sure indexing is complete first")
        return None

    # Test queries
    test_queries = [
        ("Discussions sur les restaurants", "French query about restaurants"),
        ("Conversations about Paris", "English query about Paris"),
        ("Vacances et voyages", "French query about vacations"),
        ("Weekend plans", "English query about weekend"),
    ]

    print("\n📝 Test Queries:")
    for query, description in test_queries:
        print(f"  • {query} ({description})")

    print("\n⚠️  Full query testing requires running the RAG pipeline")
    print("    Run these queries manually:")
    print()

    for query, _ in test_queries:
        print(f"    python rag_low_level_m1_16gb_verbose.py \\")
        print(f"      --query-only --query \"{query}\"")
        print()

    return test_queries


def compare_models():
    """Compare bge-small-en vs multilingual-e5-small."""
    print("\n" + "="*70)
    print("MULTILINGUAL EMBEDDING MODEL COMPARISON")
    print("="*70)

    models = [
        ("BAAI/bge-small-en", 384, "Current model (English-focused)"),
        ("intfloat/multilingual-e5-small", 384, "Multilingual alternative"),
    ]

    all_results = []

    for model_name, dim, description in models:
        print(f"\n\n{'#'*70}")
        print(f"# {description}")
        print(f"# Model: {model_name}")
        print(f"{'#'*70}")

        try:
            result = test_embedding_similarity(model_name, dim)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final comparison
    if len(all_results) == 2:
        print("\n\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)

        bge = all_results[0]
        e5 = all_results[1]

        print(f"\n{'Metric':<25} {'bge-small-en':<15} {'multilingual-e5':<15} {'Winner'}")
        print("-" * 70)

        # French-French
        winner = "E5" if e5['avg_fr_fr'] > bge['avg_fr_fr'] else "BGE"
        diff = abs(e5['avg_fr_fr'] - bge['avg_fr_fr'])
        print(f"{'French-French':<25} {bge['avg_fr_fr']:<15.4f} {e5['avg_fr_fr']:<15.4f} {winner} (+{diff:.4f})")

        # English-English
        winner = "E5" if e5['avg_en_en'] > bge['avg_en_en'] else "BGE"
        diff = abs(e5['avg_en_en'] - bge['avg_en_en'])
        print(f"{'English-English':<25} {bge['avg_en_en']:<15.4f} {e5['avg_en_en']:<15.4f} {winner} (+{diff:.4f})")

        # French-English (cross-lingual)
        winner = "E5" if e5['avg_fr_en'] > bge['avg_fr_en'] else "BGE"
        diff = abs(e5['avg_fr_en'] - bge['avg_fr_en'])
        print(f"{'Cross-lingual (FR-EN)':<25} {bge['avg_fr_en']:<15.4f} {e5['avg_fr_en']:<15.4f} {winner} (+{diff:.4f})")

        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)

        # Calculate improvement
        fr_improvement = (e5['avg_fr_fr'] - bge['avg_fr_fr']) / bge['avg_fr_fr'] * 100
        cross_improvement = (e5['avg_fr_en'] - bge['avg_fr_en']) / bge['avg_fr_en'] * 100

        if fr_improvement > 5 or cross_improvement > 5:
            print("\n✅ RECOMMENDATION: Switch to multilingual-e5-small")
            print(f"\n   Improvements:")
            print(f"   • French similarity: +{fr_improvement:.1f}%")
            print(f"   • Cross-lingual:    +{cross_improvement:.1f}%")
            print(f"\n   Re-index with:")
            print(f"   export EMBED_MODEL=intfloat/multilingual-e5-small")
            print(f"   export EMBED_DIM=384")
            print(f"   export PGTABLE=messenger_clean_cs700_ov150_e5")
            print(f"   export RESET_TABLE=1")
            print(f"   python rag_low_level_m1_16gb_verbose.py --index-only")
        else:
            print("\n✅ RECOMMENDATION: Keep bge-small-en")
            print(f"\n   The difference is minimal (<5%):")
            print(f"   • French similarity: +{fr_improvement:.1f}%")
            print(f"   • Cross-lingual:    +{cross_improvement:.1f}%")
            print(f"\n   Not worth the re-indexing time.")

    return all_results


def main():
    """Main comparison script."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  Multilingual Embedding Model Comparison                            ║
║  Testing: bge-small-en vs multilingual-e5-small                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Test semantic similarity
    print("\nStep 1: Testing semantic similarity with sample sentences...")
    results = compare_models()

    # Step 2: Provide guidance for querying current index
    print("\n\nStep 2: Query your current index to verify in practice")
    query_current_index()

    print("\n" + "="*70)
    print("SCRIPT COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the semantic similarity scores above")
    print("2. Run the suggested test queries on your current index")
    print("3. Decide whether to re-index with multilingual-e5-small")
    print()


if __name__ == "__main__":
    main()
