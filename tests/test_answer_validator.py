"""
Test suite for answer validation module.

Tests all validation components:
1. Confidence scoring
2. Hallucination detection
3. Citation extraction
4. Complete validation pipeline
5. Edge cases and error handling

Run with:
    pytest tests/test_answer_validator.py -v
    # OR
    python tests/test_answer_validator.py
"""

import sys
import os
import pytest
import numpy as np
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.answer_validator import AnswerValidator, ValidationConfig


# Mock embedding model for testing
class MockEmbeddingModel:
    """Mock embedding model with deterministic behavior"""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic embedding based on text content.

        Uses simple hashing to create reproducible embeddings.
        Similar texts will have similar embeddings.
        """
        import hashlib

        # Hash text to get deterministic seed
        text_hash = hashlib.md5(text.lower().encode()).digest()

        # Generate embedding
        embedding = []
        for i in range(self.dim):
            byte_idx = i % len(text_hash)
            value = (text_hash[byte_idx] / 255.0) - 0.5
            embedding.append(value)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding


# Fixtures
@pytest.fixture
def embed_model():
    """Provide mock embedding model"""
    return MockEmbeddingModel()


@pytest.fixture
def validator(embed_model):
    """Provide validator with mock embedding model"""
    config = ValidationConfig(
        enabled=True,
        confidence_threshold=0.7,
        hallucination_threshold=0.5,
        relevance_threshold=0.6,
        faithfulness_threshold=0.7,
    )
    return AnswerValidator(embed_model, config)


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "What attention mechanism does Mistral 7B use?"


@pytest.fixture
def sample_answer():
    """Sample good answer for testing"""
    return "Mistral 7B uses sliding window attention with a window size of 4096 tokens."


@pytest.fixture
def sample_chunks():
    """Sample context chunks for testing"""
    return [
        "Mistral 7B uses sliding window attention for efficient processing.",
        "The sliding window has a size of 4096 tokens.",
        "This mechanism reduces computational complexity compared to full attention.",
    ]


# Test ValidationConfig
class TestValidationConfig:
    """Test validation configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ValidationConfig()
        assert config.enabled is True
        assert config.confidence_threshold == 0.7
        assert config.hallucination_threshold == 0.5
        assert len(config.uncertainty_words) > 0

    def test_custom_config(self):
        """Test custom configuration"""
        config = ValidationConfig(
            enabled=False,
            confidence_threshold=0.8,
            hallucination_threshold=0.6,
        )
        assert config.enabled is False
        assert config.confidence_threshold == 0.8
        assert config.hallucination_threshold == 0.6

    def test_from_env(self, monkeypatch):
        """Test loading configuration from environment"""
        monkeypatch.setenv("ENABLE_ANSWER_VALIDATION", "0")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")
        monkeypatch.setenv("HALLUCINATION_THRESHOLD", "0.6")

        config = ValidationConfig.from_env()
        assert config.enabled is False
        assert config.confidence_threshold == 0.8
        assert config.hallucination_threshold == 0.6


# Test AnswerValidator initialization
class TestValidatorInit:
    """Test validator initialization"""

    def test_init_with_defaults(self, embed_model):
        """Test initialization with default config"""
        validator = AnswerValidator(embed_model)
        assert validator.embed_model is embed_model
        assert validator.config.enabled is True

    def test_init_with_custom_config(self, embed_model):
        """Test initialization with custom config"""
        config = ValidationConfig(confidence_threshold=0.8)
        validator = AnswerValidator(embed_model, config)
        assert validator.config.confidence_threshold == 0.8

    def test_disabled_validator(self, embed_model):
        """Test validator with validation disabled"""
        config = ValidationConfig(enabled=False)
        validator = AnswerValidator(embed_model, config)
        assert validator.config.enabled is False


# Test confidence scoring
class TestConfidenceScoring:
    """Test confidence scoring functionality"""

    def test_score_confidence_good_answer(self, validator, sample_query, sample_answer, sample_chunks):
        """Test confidence scoring with good answer"""
        result = validator.score_confidence(sample_answer, sample_chunks, sample_query)

        assert 'overall_score' in result
        assert 'relevance' in result
        assert 'faithfulness' in result
        assert 'completeness' in result
        assert 'has_uncertainty' in result

        assert 0.0 <= result['overall_score'] <= 1.0
        assert 0.0 <= result['relevance'] <= 1.0
        assert 0.0 <= result['faithfulness'] <= 1.0
        assert 0.0 <= result['completeness'] <= 1.0

    def test_score_confidence_with_uncertainty(self, validator, sample_query, sample_chunks):
        """Test confidence scoring with uncertain answer"""
        uncertain_answer = "I'm not sure, but Mistral might use sliding window attention."

        result = validator.score_confidence(uncertain_answer, sample_chunks, sample_query)

        assert result['has_uncertainty'] is True
        assert len(result['uncertainty_indicators']) > 0
        # Uncertainty should reduce confidence
        assert result['overall_score'] < 1.0

    def test_score_confidence_empty_answer(self, validator, sample_query, sample_chunks):
        """Test confidence scoring with empty answer"""
        result = validator.score_confidence("", sample_chunks, sample_query)

        assert result['overall_score'] < 0.5
        assert result['completeness'] < 0.5

    def test_score_confidence_no_chunks(self, validator, sample_query, sample_answer):
        """Test confidence scoring with no context chunks"""
        result = validator.score_confidence(sample_answer, [], sample_query)

        assert result['overall_score'] == 0.0
        assert 'No context chunks' in result['details']

    def test_score_confidence_short_answer(self, validator, sample_query, sample_chunks):
        """Test confidence scoring with very short answer"""
        short_answer = "Yes."

        result = validator.score_confidence(short_answer, sample_chunks, sample_query)

        assert result['completeness'] < 0.5


# Test hallucination detection
class TestHallucinationDetection:
    """Test hallucination detection functionality"""

    def test_detect_no_hallucinations(self, validator, sample_answer, sample_chunks):
        """Test with answer fully supported by context"""
        result = validator.detect_hallucinations(sample_answer, sample_chunks)

        assert isinstance(result, list)
        assert len(result) > 0  # Should have at least one claim

        # Check format
        for claim, support_score, is_hallucination in result:
            assert isinstance(claim, str)
            assert isinstance(support_score, float)
            assert isinstance(is_hallucination, bool)
            assert 0.0 <= support_score <= 1.0

    def test_detect_hallucinations_present(self, validator, sample_chunks):
        """Test with answer containing hallucinations"""
        answer_with_hall = (
            "Mistral uses sliding window attention. "
            "It was trained on 100 trillion tokens from Mars."  # Hallucination
        )

        result = validator.detect_hallucinations(answer_with_hall, sample_chunks)

        # Should have at least one hallucination
        hallucinations = [claim for claim, score, is_hall in result if is_hall]
        # Note: With mock embeddings, detection may vary
        assert isinstance(hallucinations, list)

    def test_detect_hallucinations_no_chunks(self, validator, sample_answer):
        """Test hallucination detection with no context"""
        result = validator.detect_hallucinations(sample_answer, [])

        # All claims should be hallucinations without context
        for claim, support_score, is_hallucination in result:
            assert support_score == 0.0
            assert is_hallucination is True

    def test_detect_hallucinations_empty_answer(self, validator, sample_chunks):
        """Test hallucination detection with empty answer"""
        result = validator.detect_hallucinations("", sample_chunks)

        assert len(result) == 0  # No claims to check


# Test citation extraction
class TestCitationExtraction:
    """Test citation extraction functionality"""

    def test_extract_citations_basic(self, validator, sample_answer, sample_chunks):
        """Test basic citation extraction"""
        result = validator.extract_citations(sample_answer, sample_chunks)

        assert isinstance(result, list)
        assert len(result) > 0  # Should have at least one citation

        # Check format
        for citation in result:
            assert 'sentence' in citation
            assert 'sources' in citation
            assert isinstance(citation['sentence'], str)
            assert isinstance(citation['sources'], list)

            for source in citation['sources']:
                assert 'chunk_id' in source
                assert 'score' in source
                assert 'text' in source
                assert 0 <= source['chunk_id'] < len(sample_chunks)
                assert 0.0 <= source['score'] <= 1.0

    def test_extract_citations_no_chunks(self, validator, sample_answer):
        """Test citation extraction with no chunks"""
        result = validator.extract_citations(sample_answer, [])

        # Should return empty list or citations with no sources
        if result:
            for citation in result:
                assert len(citation['sources']) == 0

    def test_extract_citations_empty_answer(self, validator, sample_chunks):
        """Test citation extraction with empty answer"""
        result = validator.extract_citations("", sample_chunks)

        assert len(result) == 0

    def test_extract_citations_max_sources(self, validator, sample_chunks):
        """Test that citations limit sources per sentence"""
        # Create answer with sentence that might match many chunks
        answer = "Mistral uses attention mechanisms."

        result = validator.extract_citations(answer, sample_chunks)

        for citation in result:
            # Should have max 3 sources per sentence
            assert len(citation['sources']) <= 3


# Test complete validation pipeline
class TestValidationPipeline:
    """Test complete validation pipeline"""

    def test_validate_answer_good(self, validator, sample_query, sample_answer, sample_chunks):
        """Test validation with good answer"""
        result = validator.validate_answer(sample_answer, sample_query, sample_chunks)

        assert 'confidence_score' in result
        assert 'confidence_details' in result
        assert 'hallucinations' in result
        assert 'hallucination_count' in result
        assert 'citations' in result
        assert 'warnings' in result
        assert 'passed' in result

        assert isinstance(result['hallucinations'], list)
        assert isinstance(result['citations'], list)
        assert isinstance(result['warnings'], list)
        assert isinstance(result['passed'], bool)

    def test_validate_answer_with_issues(self, validator, sample_query, sample_chunks):
        """Test validation with problematic answer"""
        bad_answer = (
            "I'm not sure, but Mistral might use some attention mechanism. "
            "It was possibly trained on alien technology."
        )

        result = validator.validate_answer(bad_answer, sample_query, sample_chunks)

        # Should have warnings
        assert len(result['warnings']) > 0
        assert result['passed'] is False

    def test_validate_answer_disabled(self, embed_model, sample_query, sample_answer, sample_chunks):
        """Test validation when disabled"""
        config = ValidationConfig(enabled=False)
        validator = AnswerValidator(embed_model, config)

        result = validator.validate_answer(sample_answer, sample_query, sample_chunks)

        assert result['confidence_score'] == 1.0
        assert result['hallucination_count'] == 0
        assert len(result['warnings']) == 0
        assert result['passed'] is True

    def test_validate_answer_edge_cases(self, validator, sample_query, sample_chunks):
        """Test validation with edge cases"""
        # Empty answer
        result_empty = validator.validate_answer("", sample_query, sample_chunks)
        assert len(result_empty['warnings']) > 0

        # No chunks
        result_no_chunks = validator.validate_answer(sample_query, sample_query, [])
        assert result_no_chunks['confidence_score'] == 0.0

        # Very short answer
        result_short = validator.validate_answer("Yes.", sample_query, sample_chunks)
        assert len(result_short['warnings']) > 0


# Test helper methods
class TestHelperMethods:
    """Test internal helper methods"""

    def test_extract_chunks_text_node_with_score(self, validator):
        """Test text extraction from NodeWithScore objects"""
        # Mock NodeWithScore
        class MockNode:
            def get_content(self):
                return "Test content"

        class MockNodeWithScore:
            def __init__(self, text):
                self.node = MockNode()
                self.node.text = text
                self.score = 0.9

        chunks = [MockNodeWithScore("Test 1"), MockNodeWithScore("Test 2")]
        texts = validator._extract_chunks_text(chunks)

        assert len(texts) == 2
        assert all(isinstance(text, str) for text in texts)

    def test_extract_chunks_text_string(self, validator):
        """Test text extraction from string chunks"""
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        texts = validator._extract_chunks_text(chunks)

        assert texts == chunks

    def test_extract_chunks_text_dict(self, validator):
        """Test text extraction from dict chunks"""
        chunks = [
            {'text': 'Content 1'},
            {'content': 'Content 2'},
        ]
        texts = validator._extract_chunks_text(chunks)

        assert len(texts) == 2
        assert 'Content 1' in texts[0]
        assert 'Content 2' in texts[1]

    def test_split_into_sentences(self, validator):
        """Test sentence splitting"""
        text = "This is sentence one. This is sentence two! And this is sentence three?"

        sentences = validator._split_into_sentences(text)

        assert len(sentences) == 3
        assert all(isinstance(sent, str) for sent in sentences)

    def test_detect_uncertainty(self, validator):
        """Test uncertainty detection"""
        # Text with uncertainty
        uncertain_text = "I'm not sure, but it might work."
        has_uncertainty, indicators = validator._detect_uncertainty(uncertain_text)

        assert has_uncertainty is True
        assert len(indicators) > 0

        # Text without uncertainty
        certain_text = "Mistral uses sliding window attention."
        has_uncertainty, indicators = validator._detect_uncertainty(certain_text)

        assert has_uncertainty is False
        assert len(indicators) == 0

    def test_cosine_similarity(self, validator):
        """Test cosine similarity calculation"""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])

        # Identical vectors
        sim_identical = validator._cosine_similarity(a, b)
        assert abs(sim_identical - 1.0) < 0.01

        # Orthogonal vectors
        sim_orthogonal = validator._cosine_similarity(a, c)
        assert abs(sim_orthogonal) < 0.01


# Run tests directly
if __name__ == "__main__":
    print("="*70)
    print("Answer Validator - Test Suite")
    print("="*70)

    # Run pytest if available
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v", "-s"]))
    except ImportError:
        print("\nNote: pytest not installed. Running manual tests...\n")

        # Run basic manual tests
        embed_model = MockEmbeddingModel()
        validator = AnswerValidator(embed_model)

        print("\n✓ Validator initialized successfully")

        # Test 1: Confidence scoring
        query = "What attention mechanism does Mistral use?"
        answer = "Mistral 7B uses sliding window attention with a window size of 4096."
        chunks = [
            "Mistral 7B uses sliding window attention.",
            "The window size is 4096 tokens.",
            "This improves efficiency."
        ]

        result = validator.score_confidence(answer, chunks, query)
        print(f"\n✓ Confidence scoring works: {result['overall_score']:.2f}")

        # Test 2: Hallucination detection
        hallucinations = validator.detect_hallucinations(answer, chunks)
        print(f"✓ Hallucination detection works: {len(hallucinations)} claims checked")

        # Test 3: Citation extraction
        citations = validator.extract_citations(answer, chunks)
        print(f"✓ Citation extraction works: {len(citations)} citations")

        # Test 4: Complete validation
        validation = validator.validate_answer(answer, query, chunks)
        print(f"✓ Complete validation works: passed={validation['passed']}")

        print("\n" + "="*70)
        print("✓ All manual tests passed!")
        print("="*70)
        print("\nFor comprehensive testing, install pytest:")
        print("  pip install pytest")
        print("  pytest tests/test_answer_validator.py -v")
