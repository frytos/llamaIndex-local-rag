#!/usr/bin/env python3
"""
HyDE (Hypothetical Document Embeddings) Example

Demonstrates how to use HyDE retrieval to improve RAG quality by 10-20%
for technical and complex queries.

Usage:
    # Basic example
    python examples/hyde_example.py

    # With custom query
    python examples/hyde_example.py --query "What is attention mechanism?"

    # Compare HyDE vs regular retrieval
    python examples/hyde_example.py --compare

Requirements:
    - PostgreSQL database with indexed documents
    - Environment variables configured (see config/.env.example)
    - LLM available (llama.cpp or vLLM)
"""

import os
import sys
import argparse
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hyde_retrieval import HyDERetriever, create_hyde_retriever_from_config

# Import main pipeline functions (if available)
try:
    from rag_low_level_m1_16gb_verbose import (
        build_embed_model,
        build_llm,
        make_vector_store,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("Warning: Main pipeline not available, using example only")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
log = logging.getLogger(__name__)


def example_basic_usage():
    """
    Example 1: Basic HyDE usage with environment variables.
    """
    print("="*70)
    print("Example 1: Basic HyDE Usage")
    print("="*70)
    print()

    if not PIPELINE_AVAILABLE:
        print("❌ Pipeline not available. Please ensure:")
        print("  1. PostgreSQL is running")
        print("  2. Environment variables are set (see .env.example)")
        print("  3. Documents are indexed")
        return

    # Build components
    print("📦 Loading models and database...")
    vector_store = make_vector_store()
    embed_model = build_embed_model()
    llm = build_llm()
    print()

    # Create HyDE retriever (reads from environment variables)
    print("🔮 Creating HyDE retriever...")
    retriever = create_hyde_retriever_from_config(
        vector_store=vector_store,
        embed_model=embed_model,
        llm=llm,
        similarity_top_k=4
    )
    print()

    # Query
    query = os.getenv("QUERY", "What is attention mechanism in neural networks?")
    print(f"❓ Query: \"{query}\"")
    print()

    # Retrieve
    print("🔍 Retrieving documents...")
    results = retriever.retrieve(query)
    print()

    # Display results
    print("="*70)
    print(f"Retrieved {len(results)} documents")
    print("="*70)
    for i, nws in enumerate(results):
        print(f"\n{i+1}. Score: {nws.score:.4f}")
        content = nws.node.get_content()
        print(f"   Text: {content[:200]}...")
        if nws.node.metadata:
            print(f"   Metadata: {nws.node.metadata}")

    # Show generated hypotheses
    if retriever.last_hypotheses:
        print()
        print("="*70)
        print("Generated Hypotheses")
        print("="*70)
        for i, hypothesis in enumerate(retriever.last_hypotheses):
            print(f"\n{i+1}. {hypothesis}")

    print()
    print(f"⏱️  Total retrieval time: {retriever.last_retrieval_time:.2f}s")


def example_compare_retrievers():
    """
    Example 2: Compare HyDE vs regular retrieval.
    """
    print("="*70)
    print("Example 2: Compare HyDE vs Regular Retrieval")
    print("="*70)
    print()

    if not PIPELINE_AVAILABLE:
        print("❌ Pipeline not available")
        return

    # Build components
    print("📦 Loading models and database...")
    vector_store = make_vector_store()
    embed_model = build_embed_model()
    llm = build_llm()
    print()

    # Import regular retriever
    from rag_low_level_m1_16gb_verbose import VectorDBRetriever

    # Create both retrievers
    print("🔮 Creating retrievers...")
    regular_retriever = VectorDBRetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        similarity_top_k=4
    )

    hyde_retriever = HyDERetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        llm=llm,
        similarity_top_k=4,
        num_hypotheses=1,
        enable_hyde=True,
    )
    print()

    # Test queries
    queries = [
        "What is attention mechanism?",
        "Explain backpropagation in neural networks",
        "Compare LSTM and transformers",
    ]

    for query in queries:
        print("="*70)
        print(f"Query: \"{query}\"")
        print("="*70)
        print()

        # Regular retrieval
        print("📊 Regular Retrieval:")
        regular_results = regular_retriever.retrieve(query)
        print(f"  • Top score: {regular_results[0].score:.4f}")
        print(f"  • Time: {regular_retriever.last_retrieval_time:.2f}s")
        print(f"  • Top result: {regular_results[0].node.get_content()[:100]}...")
        print()

        # HyDE retrieval
        print("🔮 HyDE Retrieval:")
        hyde_results = hyde_retriever.retrieve(query)
        print(f"  • Top score: {hyde_results[0].score:.4f}")
        print(f"  • Time: {hyde_retriever.last_retrieval_time:.2f}s")
        print(f"  • Top result: {hyde_results[0].node.get_content()[:100]}...")
        if hyde_retriever.last_hypotheses:
            print(f"  • Hypothesis: {hyde_retriever.last_hypotheses[0][:80]}...")
        print()

        # Compare
        score_diff = hyde_results[0].score - regular_results[0].score
        time_diff = hyde_retriever.last_retrieval_time - regular_retriever.last_retrieval_time
        print("📈 Comparison:")
        print(f"  • Score difference: {score_diff:+.4f}")
        print(f"  • Time overhead: {time_diff:+.2f}s")
        print()


def example_multiple_hypotheses():
    """
    Example 3: Use multiple hypotheses with different fusion methods.
    """
    print("="*70)
    print("Example 3: Multiple Hypotheses with Different Fusion")
    print("="*70)
    print()

    if not PIPELINE_AVAILABLE:
        print("❌ Pipeline not available")
        return

    # Build components
    print("📦 Loading models and database...")
    vector_store = make_vector_store()
    embed_model = build_embed_model()
    llm = build_llm()
    print()

    query = "Compare attention mechanisms in transformers vs RNNs"
    print(f"❓ Query: \"{query}\"")
    print()

    # Test different configurations
    configs = [
        (1, "rrf", "Single hypothesis with RRF"),
        (2, "rrf", "Two hypotheses with RRF"),
        (2, "avg", "Two hypotheses with averaging"),
        (3, "rrf", "Three hypotheses with RRF"),
    ]

    for num_hyp, fusion, desc in configs:
        print("="*70)
        print(f"Config: {desc}")
        print("="*70)

        retriever = HyDERetriever(
            vector_store=vector_store,
            embed_model=embed_model,
            llm=llm,
            similarity_top_k=4,
            num_hypotheses=num_hyp,
            fusion_method=fusion,
            enable_hyde=True,
        )

        results = retriever.retrieve(query)

        print(f"  • Hypotheses generated: {len(retriever.last_hypotheses)}")
        print(f"  • Top score: {results[0].score:.4f}")
        print(f"  • Time: {retriever.last_retrieval_time:.2f}s")
        print()


def example_conditional_hyde():
    """
    Example 4: Conditionally enable HyDE based on query complexity.
    """
    print("="*70)
    print("Example 4: Conditional HyDE (Smart Toggle)")
    print("="*70)
    print()

    if not PIPELINE_AVAILABLE:
        print("❌ Pipeline not available")
        return

    # Build components
    print("📦 Loading models and database...")
    vector_store = make_vector_store()
    embed_model = build_embed_model()
    llm = build_llm()
    print()

    # Create retriever (HyDE disabled initially)
    retriever = HyDERetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        llm=llm,
        similarity_top_k=4,
        enable_hyde=False,  # Start disabled
    )

    # Query classifier
    def is_complex_query(query: str) -> bool:
        """Simple heuristic to classify query complexity"""
        complex_keywords = [
            "explain", "compare", "difference", "why", "how",
            "trade-off", "advantage", "disadvantage", "versus", "vs"
        ]
        is_long = len(query.split()) > 5
        has_complex_word = any(kw in query.lower() for kw in complex_keywords)
        return is_long or has_complex_word

    # Test queries
    queries = [
        ("What is Python?", "Simple factual"),
        ("What is attention mechanism in neural networks?", "Technical"),
        ("Compare LSTM vs transformers for sequence modeling", "Complex comparative"),
        ("List all Python functions", "Simple keyword"),
    ]

    for query, query_type in queries:
        print("="*70)
        print(f"Query: \"{query}\"")
        print(f"Type: {query_type}")
        print("="*70)

        # Classify and toggle HyDE
        if is_complex_query(query):
            print("✓ Complex query detected - enabling HyDE")
            retriever._enable_hyde = True
        else:
            print("✗ Simple query - using regular retrieval")
            retriever._enable_hyde = False

        # Retrieve
        results = retriever.retrieve(query)

        print(f"  • HyDE used: {retriever._enable_hyde}")
        print(f"  • Top score: {results[0].score:.4f}")
        print(f"  • Time: {retriever.last_retrieval_time:.2f}s")
        if retriever.last_hypotheses:
            print(f"  • Hypotheses: {len(retriever.last_hypotheses)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="HyDE Retrieval Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run basic example
    python examples/hyde_example.py

    # Compare HyDE vs regular retrieval
    python examples/hyde_example.py --compare

    # Test multiple hypotheses
    python examples/hyde_example.py --multi-hypothesis

    # Test conditional HyDE
    python examples/hyde_example.py --conditional

    # Run all examples
    python examples/hyde_example.py --all

Environment Variables:
    ENABLE_HYDE=1              # Enable HyDE
    HYDE_NUM_HYPOTHESES=2      # Number of hypotheses
    HYDE_HYPOTHESIS_LENGTH=100 # Hypothesis length
    HYDE_FUSION_METHOD=rrf     # Fusion method
    QUERY="your question"      # Query to use
        """
    )
    parser.add_argument("--compare", action="store_true",
                       help="Compare HyDE vs regular retrieval")
    parser.add_argument("--multi-hypothesis", action="store_true",
                       help="Test multiple hypotheses with different fusion methods")
    parser.add_argument("--conditional", action="store_true",
                       help="Test conditional HyDE (smart toggle)")
    parser.add_argument("--all", action="store_true",
                       help="Run all examples")
    parser.add_argument("--query", help="Custom query to use")

    args = parser.parse_args()

    # Override query if provided
    if args.query:
        os.environ["QUERY"] = args.query

    # Run examples
    if args.all:
        example_basic_usage()
        print("\n\n")
        example_compare_retrievers()
        print("\n\n")
        example_multiple_hypotheses()
        print("\n\n")
        example_conditional_hyde()
    elif args.compare:
        example_compare_retrievers()
    elif args.multi_hypothesis:
        example_multiple_hypotheses()
    elif args.conditional:
        example_conditional_hyde()
    else:
        # Default: basic usage
        example_basic_usage()

    print()
    print("="*70)
    print("✓ Examples complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("  • Review docs/HYDE_GUIDE.md for detailed documentation")
    print("  • Test with your own queries")
    print("  • Tune HYDE_NUM_HYPOTHESES and HYDE_FUSION_METHOD")
    print("  • Compare retrieval quality with/without HyDE")


if __name__ == "__main__":
    main()
