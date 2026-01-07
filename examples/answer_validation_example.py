"""
Example: Integrating Answer Validation into RAG Pipeline

This example demonstrates how to integrate the AnswerValidator
into your RAG query pipeline for quality assurance.

Features demonstrated:
1. Basic validation of RAG answers
2. Confidence scoring and thresholds
3. Hallucination detection
4. Citation extraction
5. Warning handling
6. Integration with existing pipeline

Run:
    python examples/answer_validation_example.py
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.answer_validator import AnswerValidator, ValidationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def example_1_basic_validation():
    """Example 1: Basic answer validation"""
    print("\n" + "="*70)
    print("Example 1: Basic Answer Validation")
    print("="*70)

    # Simulate embedding model (in real usage, use build_embed_model())
    # Import mock from test file for examples
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))
    from test_answer_validator import MockEmbeddingModel
    embed_model = MockEmbeddingModel()

    # Initialize validator
    validator = AnswerValidator(embed_model)

    # Simulate RAG query results
    query = "What is sliding window attention in Mistral 7B?"

    answer = (
        "Sliding window attention is an attention mechanism used in Mistral 7B "
        "where each token attends only to a fixed window of previous tokens "
        "instead of all previous tokens. This reduces computational complexity."
    )

    retrieved_chunks = [
        "Mistral 7B uses sliding window attention for efficient processing.",
        "In sliding window attention, each token attends to a fixed window of 4096 tokens.",
        "This mechanism reduces computational complexity from O(n²) to O(n*w).",
        "The window size is a hyperparameter that balances context and efficiency.",
    ]

    # Validate answer
    result = validator.validate_answer(answer, query, retrieved_chunks)

    # Display results
    print(f"\nQuery: {query}")
    print(f"\nAnswer: {answer[:100]}...")
    print(f"\nValidation Results:")
    print(f"  ✓ Confidence Score: {result['confidence_score']:.2f}")
    print(f"  ✓ Hallucinations: {result['hallucination_count']}")
    print(f"  ✓ Citations: {len(result['citations'])}")
    print(f"  ✓ Passed: {result['passed']}")

    if result['warnings']:
        print(f"\n  Warnings:")
        for warning in result['warnings']:
            print(f"    • {warning}")


def example_2_confidence_scoring():
    """Example 2: Detailed confidence scoring"""
    print("\n" + "="*70)
    print("Example 2: Confidence Scoring Details")
    print("="*70)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))
    from test_answer_validator import MockEmbeddingModel
    embed_model = MockEmbeddingModel()
    validator = AnswerValidator(embed_model)

    query = "How many parameters does Mistral 7B have?"
    answer = "Mistral 7B has 7 billion parameters, making it efficient while maintaining quality."

    chunks = [
        "Mistral 7B is a 7-billion parameter language model.",
        "The model achieves efficient performance with its parameter count.",
        "Despite having fewer parameters than larger models, it maintains high quality.",
    ]

    # Get detailed confidence breakdown
    confidence = validator.score_confidence(answer, chunks, query)

    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
    print(f"\nConfidence Breakdown:")
    print(f"  Overall Score: {confidence['overall_score']:.2f}")
    print(f"  ├─ Relevance (Query-Answer): {confidence['relevance']:.2f}")
    print(f"  ├─ Faithfulness (Answer-Context): {confidence['faithfulness']:.2f}")
    print(f"  ├─ Completeness: {confidence['completeness']:.2f}")
    print(f"  └─ Has Uncertainty: {confidence['has_uncertainty']}")

    if confidence['uncertainty_indicators']:
        print(f"\n  Uncertainty Indicators: {', '.join(confidence['uncertainty_indicators'])}")


def example_3_hallucination_detection():
    """Example 3: Detecting hallucinations"""
    print("\n" + "="*70)
    print("Example 3: Hallucination Detection")
    print("="*70)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))
    from test_answer_validator import MockEmbeddingModel
    embed_model = MockEmbeddingModel()
    validator = AnswerValidator(embed_model)

    query = "What is Mistral 7B?"

    # Answer with hallucinations (made-up facts)
    answer = (
        "Mistral 7B is a language model with 7 billion parameters. "
        "It was trained on 50 trillion tokens from the internet. "  # Hallucination
        "The model uses sliding window attention for efficiency. "
        "It achieves 99% accuracy on all benchmarks."  # Hallucination
    )

    chunks = [
        "Mistral 7B is a 7-billion parameter language model.",
        "It uses sliding window attention for efficient processing.",
        "The model shows strong performance on various benchmarks.",
    ]

    # Detect hallucinations
    hallucinations = validator.detect_hallucinations(answer, chunks)

    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
    print(f"\nHallucination Analysis:")

    for claim, support_score, is_hallucination in hallucinations:
        if is_hallucination:
            status = "❌ UNSUPPORTED"
            print(f"\n  {status} (support: {support_score:.2f})")
        else:
            status = "✓ Supported"
            print(f"\n  {status} (support: {support_score:.2f})")

        print(f"    Claim: {claim[:80]}...")


def example_4_citation_extraction():
    """Example 4: Extracting citations"""
    print("\n" + "="*70)
    print("Example 4: Citation Extraction")
    print("="*70)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))
    from test_answer_validator import MockEmbeddingModel
    embed_model = MockEmbeddingModel()
    validator = AnswerValidator(embed_model)

    query = "What makes Mistral 7B efficient?"

    answer = (
        "Mistral 7B achieves efficiency through sliding window attention. "
        "This mechanism limits each token's attention to a fixed window. "
        "The computational complexity is reduced from quadratic to linear."
    )

    chunks = [
        "Mistral 7B uses sliding window attention for efficient token processing.",
        "Each token attends only to a fixed window of 4096 previous tokens.",
        "This reduces computational complexity from O(n²) to O(n*w) where w is window size.",
        "The sliding window approach maintains quality while improving speed.",
    ]

    # Extract citations
    citations = validator.extract_citations(answer, chunks)

    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
    print(f"\nCitations (sentence → source mapping):")

    for i, citation in enumerate(citations, 1):
        print(f"\n  [{i}] {citation['sentence']}")

        if citation['sources']:
            print(f"      Sources:")
            for source in citation['sources']:
                print(f"        • Chunk {source['chunk_id']} (similarity: {source['score']:.2f})")
                print(f"          \"{source['text'][:70]}...\"")
        else:
            print(f"      ⚠ No supporting sources found")


def example_5_pipeline_integration():
    """Example 5: Full RAG pipeline integration"""
    print("\n" + "="*70)
    print("Example 5: RAG Pipeline Integration")
    print("="*70)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))
    from test_answer_validator import MockEmbeddingModel
    embed_model = MockEmbeddingModel()

    # Configure validation thresholds
    config = ValidationConfig(
        enabled=True,
        confidence_threshold=0.7,  # Warn if confidence < 70%
        hallucination_threshold=0.5,  # Flag claims with support < 50%
    )

    validator = AnswerValidator(embed_model, config)

    def rag_query_with_validation(query: str, answer: str, chunks: list) -> dict:
        """
        Complete RAG query with answer validation.

        This is how you would integrate validation into your
        existing RAG pipeline (e.g., in rag_low_level_m1_16gb_verbose.py).
        """
        # Validate answer
        validation = validator.validate_answer(answer, query, chunks)

        # Build response with validation metadata
        response = {
            'query': query,
            'answer': answer,
            'sources': chunks,
            'validation': {
                'confidence': validation['confidence_score'],
                'hallucination_count': validation['hallucination_count'],
                'citations': validation['citations'],
                'warnings': validation['warnings'],
                'passed': validation['passed'],
            }
        }

        # Log warnings
        if validation['warnings']:
            log.warning(f"Answer quality issues detected:")
            for warning in validation['warnings']:
                log.warning(f"  • {warning}")

        # Optional: Reject low-quality answers
        if not validation['passed']:
            log.error(f"Answer failed validation (confidence: {validation['confidence_score']:.2f})")
            # Could trigger retry with different parameters, fallback response, etc.

        return response

    # Test with good answer
    print("\n--- Test 1: High-Quality Answer ---")

    query1 = "What is Mistral 7B?"
    answer1 = "Mistral 7B is a 7-billion parameter language model that uses sliding window attention."
    chunks1 = [
        "Mistral 7B is a language model with 7 billion parameters.",
        "It uses sliding window attention for efficient processing.",
    ]

    result1 = rag_query_with_validation(query1, answer1, chunks1)
    print(f"\nQuery: {result1['query']}")
    print(f"Answer: {result1['answer']}")
    print(f"Confidence: {result1['validation']['confidence']:.2f}")
    print(f"Passed: {result1['validation']['passed']}")

    # Test with poor answer
    print("\n\n--- Test 2: Low-Quality Answer ---")

    query2 = "What is Mistral 7B?"
    answer2 = "I'm not sure, but it might be some kind of AI model possibly."
    chunks2 = [
        "Mistral 7B is a language model with 7 billion parameters.",
        "It uses sliding window attention for efficient processing.",
    ]

    result2 = rag_query_with_validation(query2, answer2, chunks2)
    print(f"\nQuery: {result2['query']}")
    print(f"Answer: {result2['answer']}")
    print(f"Confidence: {result2['validation']['confidence']:.2f}")
    print(f"Passed: {result2['validation']['passed']}")

    if result2['validation']['warnings']:
        print(f"\nWarnings:")
        for warning in result2['validation']['warnings']:
            print(f"  • {warning}")


def example_6_custom_thresholds():
    """Example 6: Custom validation thresholds"""
    print("\n" + "="*70)
    print("Example 6: Custom Validation Thresholds")
    print("="*70)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))
    from test_answer_validator import MockEmbeddingModel
    embed_model = MockEmbeddingModel()

    # Strict validation (high thresholds)
    strict_config = ValidationConfig(
        confidence_threshold=0.85,  # 85% confidence required
        hallucination_threshold=0.6,  # 60% support required
        relevance_threshold=0.7,
        faithfulness_threshold=0.8,
    )

    strict_validator = AnswerValidator(embed_model, strict_config)

    # Lenient validation (low thresholds)
    lenient_config = ValidationConfig(
        confidence_threshold=0.5,  # 50% confidence acceptable
        hallucination_threshold=0.3,  # 30% support acceptable
        relevance_threshold=0.4,
        faithfulness_threshold=0.5,
    )

    lenient_validator = AnswerValidator(embed_model, lenient_config)

    # Test answer
    query = "What attention mechanism does Mistral use?"
    answer = "Mistral uses sliding window attention for efficiency."
    chunks = [
        "Mistral 7B uses sliding window attention.",
        "This improves computational efficiency.",
    ]

    # Compare results
    strict_result = strict_validator.validate_answer(answer, query, chunks)
    lenient_result = lenient_validator.validate_answer(answer, query, chunks)

    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")

    print(f"\nStrict Validation (threshold=0.85):")
    print(f"  Confidence: {strict_result['confidence_score']:.2f}")
    print(f"  Passed: {strict_result['passed']}")
    print(f"  Warnings: {len(strict_result['warnings'])}")

    print(f"\nLenient Validation (threshold=0.5):")
    print(f"  Confidence: {lenient_result['confidence_score']:.2f}")
    print(f"  Passed: {lenient_result['passed']}")
    print(f"  Warnings: {len(lenient_result['warnings'])}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Answer Validation - Integration Examples")
    print("="*70)
    print("\nThese examples show how to integrate answer validation")
    print("into your RAG pipeline for quality assurance.\n")

    try:
        example_1_basic_validation()
        example_2_confidence_scoring()
        example_3_hallucination_detection()
        example_4_citation_extraction()
        example_5_pipeline_integration()
        example_6_custom_thresholds()

        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)

        print("\n📚 Next Steps:")
        print("  1. Integrate validator into your RAG pipeline")
        print("  2. Adjust thresholds based on your use case")
        print("  3. Monitor validation metrics over time")
        print("  4. Use warnings to improve retrieval/generation")
        print("\n💡 Integration Tips:")
        print("  • Add to rag_low_level_m1_16gb_verbose.py after LLM response")
        print("  • Log validation results for analysis")
        print("  • Use confidence scores to filter low-quality answers")
        print("  • Extract citations for source attribution")
        print("  • Set ENABLE_ANSWER_VALIDATION=1 in .env")

    except Exception as e:
        log.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
