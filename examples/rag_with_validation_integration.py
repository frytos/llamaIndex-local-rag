"""
Real Integration Example: RAG Pipeline with Answer Validation

This example shows how to integrate answer validation into the actual
RAG pipeline (rag_low_level_m1_16gb_verbose.py).

This is a TEMPLATE showing WHERE and HOW to add validation.
Copy these patterns into your actual pipeline code.

Key Integration Points:
1. Import validator after building embed_model
2. Initialize validator once
3. Validate each answer after LLM generation
4. Log warnings and include in response
5. Optionally retry or reject low-quality answers
"""

import logging
from typing import Any, Dict

# This is how you would integrate in rag_low_level_m1_16gb_verbose.py


def example_integration_pattern():
    """
    Shows WHERE to add validation in the RAG pipeline.

    Copy this pattern into your actual code.
    """

    # ===================================================================
    # STEP 1: Import (at top of file)
    # ===================================================================
    from utils.answer_validator import AnswerValidator, ValidationConfig

    # ===================================================================
    # STEP 2: Check if validation is available (with imports)
    # ===================================================================
    try:
        from utils.answer_validator import AnswerValidator
        ANSWER_VALIDATION_AVAILABLE = True
    except ImportError:
        ANSWER_VALIDATION_AVAILABLE = False
        logging.warning("Answer validation not available")

    # ===================================================================
    # STEP 3: Initialize validator (after building embed_model)
    # ===================================================================
    # This goes after: embed_model = build_embed_model()

    validator = None
    if ANSWER_VALIDATION_AVAILABLE:
        try:
            validator = AnswerValidator(embed_model)
            logging.info("Answer validation enabled")
        except Exception as e:
            logging.warning(f"Failed to initialize validator: {e}")
            validator = None

    # ===================================================================
    # STEP 4: Validate answers (in query loop)
    # ===================================================================
    # This goes after: response = query_engine.query(question)

    def query_with_validation(query_engine, question, validator=None):
        """
        Query function with integrated validation.

        Args:
            query_engine: LlamaIndex query engine
            question: User query
            validator: AnswerValidator instance (optional)

        Returns:
            Dict with answer, validation results, and metadata
        """
        # Run query
        response = query_engine.query(question)

        # Extract answer and sources
        answer_text = str(response)
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []

        # Initialize result
        result = {
            'query': question,
            'answer': answer_text,
            'sources': source_nodes,
            'validation': None,
            'warnings': [],
        }

        # Validate if validator available
        if validator is not None:
            try:
                validation = validator.validate_answer(
                    answer=answer_text,
                    query=question,
                    chunks=source_nodes
                )

                result['validation'] = {
                    'confidence': validation['confidence_score'],
                    'hallucination_count': validation['hallucination_count'],
                    'passed': validation['passed'],
                    'warnings': validation['warnings'],
                }

                # Log warnings
                if validation['warnings']:
                    logging.warning(f"Answer quality issues for query: {question[:50]}...")
                    for warning in validation['warnings']:
                        logging.warning(f"  • {warning}")

                # Add to result warnings
                result['warnings'].extend(validation['warnings'])

            except Exception as e:
                logging.error(f"Validation failed: {e}")
                result['validation'] = {'error': str(e)}

        return result

    return query_with_validation


def example_with_retry():
    """
    Advanced: Retry low-confidence answers with different parameters.
    """

    def query_with_retry(
        query_engine,
        question: str,
        validator: Any,
        max_retries: int = 2,
        min_confidence: float = 0.7
    ) -> Dict:
        """
        Query with automatic retry on low confidence.

        Args:
            query_engine: LlamaIndex query engine
            question: User query
            validator: AnswerValidator instance
            max_retries: Maximum retry attempts
            min_confidence: Minimum acceptable confidence

        Returns:
            Dict with best answer and validation results
        """
        best_result = None
        best_confidence = 0.0

        for attempt in range(max_retries):
            # Run query
            response = query_engine.query(question)
            answer = str(response)
            sources = response.source_nodes if hasattr(response, 'source_nodes') else []

            # Validate
            validation = validator.validate_answer(answer, question, sources)
            confidence = validation['confidence_score']

            # Track best result
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = {
                    'answer': answer,
                    'sources': sources,
                    'validation': validation,
                    'attempt': attempt + 1,
                }

            # Check if passed threshold
            if confidence >= min_confidence:
                logging.info(f"Query succeeded on attempt {attempt + 1} (confidence: {confidence:.2f})")
                return best_result

            # Log retry
            if attempt < max_retries - 1:
                logging.warning(
                    f"Attempt {attempt + 1} failed validation (confidence: {confidence:.2f}), "
                    f"retrying..."
                )

                # Could modify query parameters here:
                # - Increase TOP_K to retrieve more context
                # - Adjust temperature
                # - Use query expansion
                # - etc.

        # Return best attempt
        logging.warning(
            f"All {max_retries} attempts completed, "
            f"best confidence: {best_confidence:.2f}"
        )
        return best_result

    return query_with_retry


def example_with_filtering():
    """
    Advanced: Filter/reject low-quality answers.
    """

    def query_with_quality_filter(
        query_engine,
        question: str,
        validator: Any,
        min_confidence: float = 0.6
    ) -> Dict:
        """
        Query with quality filtering.

        Returns fallback response if answer quality too low.

        Args:
            query_engine: LlamaIndex query engine
            question: User query
            validator: AnswerValidator instance
            min_confidence: Minimum acceptable confidence

        Returns:
            Dict with answer and validation, or fallback response
        """
        # Run query
        response = query_engine.query(question)
        answer = str(response)
        sources = response.source_nodes if hasattr(response, 'source_nodes') else []

        # Validate
        validation = validator.validate_answer(answer, question, sources)

        # Check quality
        if validation['confidence_score'] >= min_confidence:
            # Quality OK - return answer
            return {
                'answer': answer,
                'sources': sources,
                'validation': validation,
                'quality': 'passed',
            }
        else:
            # Quality too low - return fallback
            logging.warning(
                f"Answer quality too low (confidence: {validation['confidence_score']:.2f}), "
                f"returning fallback response"
            )

            fallback_answer = (
                "I don't have enough information to answer that question confidently. "
                "Please try rephrasing your question or asking something more specific."
            )

            return {
                'answer': fallback_answer,
                'sources': sources,
                'validation': validation,
                'quality': 'rejected',
                'original_answer': answer,
            }

    return query_with_quality_filter


def example_with_citations():
    """
    Advanced: Include citations in answer formatting.
    """

    def query_with_citations(
        query_engine,
        question: str,
        validator: Any
    ) -> Dict:
        """
        Query with citation-enhanced answer.

        Args:
            query_engine: LlamaIndex query engine
            question: User query
            validator: AnswerValidator instance

        Returns:
            Dict with answer, citations, and formatted output
        """
        # Run query
        response = query_engine.query(question)
        answer = str(response)
        sources = response.source_nodes if hasattr(response, 'source_nodes') else []

        # Validate and extract citations
        validation = validator.validate_answer(answer, question, sources)
        citations = validation['citations']

        # Format answer with citations
        formatted_answer = _format_answer_with_citations(answer, citations)

        return {
            'answer': answer,
            'formatted_answer': formatted_answer,
            'sources': sources,
            'citations': citations,
            'validation': validation,
        }

    def _format_answer_with_citations(answer: str, citations: list) -> str:
        """Format answer with inline citations."""
        # Simple formatting - could be made more sophisticated
        formatted = answer + "\n\n**Sources:**\n"

        for i, citation in enumerate(citations, 1):
            formatted += f"\n[{i}] {citation['sentence']}\n"
            for source in citation['sources']:
                formatted += f"    • Chunk {source['chunk_id']} (confidence: {source['score']:.2f})\n"

        return formatted

    return query_with_citations


# ===================================================================
# Complete Integration Example
# ===================================================================

def complete_integration_example():
    """
    Complete example showing full integration pattern.

    This is what your main() function would look like with validation.
    """

    print("\n" + "="*70)
    print("Complete RAG Pipeline with Answer Validation")
    print("="*70)

    # Pseudo-code showing complete flow:
    print("""
    def main():
        # 1. Setup (existing code)
        setup_logging()
        load_config()

        # 2. Build components (existing code)
        embed_model = build_embed_model()
        vector_store = make_vector_store()
        llm = build_llm()

        # 3. Initialize validator (NEW)
        validator = None
        try:
            from utils.answer_validator import AnswerValidator
            validator = AnswerValidator(embed_model)
            log.info("Answer validation enabled")
        except Exception as e:
            log.warning(f"Validation not available: {e}")

        # 4. Create query engine (existing code)
        retriever = VectorDBRetriever(...)
        query_engine = RetrieverQueryEngine(...)

        # 5. Query loop (MODIFIED)
        while True:
            question = get_user_input()

            # Run query
            response = query_engine.query(question)
            answer = str(response)
            sources = response.source_nodes

            # Validate answer (NEW)
            if validator:
                validation = validator.validate_answer(
                    answer=answer,
                    query=question,
                    chunks=sources
                )

                # Log warnings
                if validation['warnings']:
                    for warning in validation['warnings']:
                        log.warning(warning)

                # Show confidence
                print(f"\\nConfidence: {validation['confidence_score']:.2f}")

                # Optionally reject low quality
                if not validation['passed']:
                    print("⚠️  Answer quality warnings detected")

            # Display answer
            print(f"\\nAnswer: {answer}")
    """)

    print("\n" + "="*70)
    print("Integration Points Summary")
    print("="*70)
    print("""
    1. Import: Add at top with other imports
       from utils.answer_validator import AnswerValidator

    2. Initialize: After building embed_model
       validator = AnswerValidator(embed_model)

    3. Validate: After each LLM response
       validation = validator.validate_answer(answer, query, chunks)

    4. Use Results: Log warnings, display confidence, etc.
       if validation['warnings']:
           for warning in validation['warnings']:
               log.warning(warning)

    5. Configure: Set thresholds in .env
       ENABLE_ANSWER_VALIDATION=1
       CONFIDENCE_THRESHOLD=0.7
    """)


if __name__ == "__main__":
    print("="*70)
    print("RAG Pipeline + Answer Validation Integration Guide")
    print("="*70)
    print("\nThis file shows HOW and WHERE to integrate answer validation")
    print("into your RAG pipeline (rag_low_level_m1_16gb_verbose.py).")
    print("\nSee function docstrings for specific integration patterns.")
    print("\n" + "="*70)

    # Show examples
    complete_integration_example()

    print("\n" + "="*70)
    print("Ready to Integrate!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Copy the import statement to your RAG script")
    print("2. Initialize validator after embed_model")
    print("3. Add validation after query_engine.query()")
    print("4. Log warnings and display confidence")
    print("5. Set ENABLE_ANSWER_VALIDATION=1 in .env")
    print("\nFor working examples, see:")
    print("  - examples/answer_validation_example.py")
    print("\nFor complete API reference, see:")
    print("  - docs/ANSWER_VALIDATION.md")
    print("="*70)
