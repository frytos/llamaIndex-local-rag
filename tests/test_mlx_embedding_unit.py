"""Tests for MLX embedding edge cases.

This module tests the MLX embedding implementation for Apple Silicon, including:
- Empty string handling
- Very long text truncation
- Invalid input types
- Batch embedding fallback
- Zero vector fallback on errors

Week 1 - Day 5: MLX edge case tests (5 tests)

Note: These are UNIT tests that mock the MLX model to avoid requiring
      Apple Silicon hardware. Integration tests exist in test_mlx_robustness.py.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# Mock llama_index modules before import
sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.embeddings'] = MagicMock()

# Create a mock BaseEmbedding class that can be inherited
class MockBaseEmbedding:
    def __init__(self, **kwargs):
        pass

sys.modules['llama_index.core.embeddings'].BaseEmbedding = MockBaseEmbedding


# Mock mlx modules before import
sys.modules['mlx'] = MagicMock()
sys.modules['mlx.core'] = MagicMock()
sys.modules['mlx_embedding_models'] = MagicMock()
sys.modules['mlx_embedding_models.embedding'] = MagicMock()


# ============================================================================
# Day 5: MLX Embedding Edge Case Tests (5 tests)
# ============================================================================


class TestMLXEmbeddingEdgeCases:
    """Test MLX embedding edge cases and error handling.

    These tests validate that MLXEmbedding properly:
    - Returns zero vector for empty strings (prevents crashes)
    - Truncates very long text (>32k chars)
    - Handles invalid input types gracefully
    - Falls back to individual embedding on batch failures
    - Returns valid embeddings with correct dimensions
    """

    @pytest.mark.unit
    @pytest.mark.mlx
    def test_embed_empty_string_returns_zero_vector(self, mock_mlx_model):
        """Test empty string returns zero vector (prevents crashes).

        Given: An empty string or whitespace-only string
        When: _get_text_embedding() is called
        Then: Returns zero vector with correct dimensions (no crash)
        """
        from utils.mlx_embedding import MLXEmbedding

        # Mock the EmbeddingModel.from_registry to return our mock
        with patch('mlx_embedding_models.embedding.EmbeddingModel') as mock_class:
            mock_class.from_registry.return_value = mock_mlx_model

            # Create MLX embedding instance
            mlx_emb = MLXEmbedding(model_name="BAAI/bge-m3")

            # Test empty string
            embedding_empty = mlx_emb._get_text_embedding("")
            assert embedding_empty == [0.0] * 1024  # bge-m3 is 1024-dim
            assert len(embedding_empty) == 1024

            # Test whitespace-only string
            embedding_spaces = mlx_emb._get_text_embedding("   \n\t  ")
            assert embedding_spaces == [0.0] * 1024
            assert len(embedding_spaces) == 1024

    @pytest.mark.unit
    @pytest.mark.mlx
    def test_embed_very_long_text_truncates(self, mock_mlx_model):
        """Test very long text is truncated to 32k chars.

        Given: Text with >32,000 characters
        When: _get_text_embedding() is called
        Then: Text is truncated and embedded without error
        """
        from utils.mlx_embedding import MLXEmbedding

        with patch('mlx_embedding_models.embedding.EmbeddingModel') as mock_class:
            mock_class.from_registry.return_value = mock_mlx_model

            mlx_emb = MLXEmbedding(model_name="BAAI/bge-m3")

            # Create text longer than 32k chars
            long_text = "A" * 50000

            embedding = mlx_emb._get_text_embedding(long_text)

            # Should return valid embedding (not zero vector)
            assert len(embedding) == 1024
            assert embedding != [0.0] * 1024  # Should be actual embedding

            # Verify model was called with truncated text (32000 chars)
            mock_mlx_model.encode.assert_called()
            call_args = mock_mlx_model.encode.call_args
            truncated_text = call_args[0][0][0]  # First arg, first text
            assert len(truncated_text) == 32000

    @pytest.mark.unit
    @pytest.mark.mlx
    def test_embed_invalid_input_type(self, mock_mlx_model):
        """Test handling of invalid input types (non-string).

        Given: Invalid input (None, int, list, etc.)
        When: _get_text_embedding() is called
        Then: Returns zero vector without crashing
        """
        from utils.mlx_embedding import MLXEmbedding

        with patch('mlx_embedding_models.embedding.EmbeddingModel') as mock_class:
            mock_class.from_registry.return_value = mock_mlx_model

            mlx_emb = MLXEmbedding(model_name="BAAI/bge-m3")

            # Test None
            embedding_none = mlx_emb._get_text_embedding(None)
            assert embedding_none == [0.0] * 1024

            # Test integer
            embedding_int = mlx_emb._get_text_embedding(123)
            assert embedding_int == [0.0] * 1024

            # Test list
            embedding_list = mlx_emb._get_text_embedding(["text"])
            assert embedding_list == [0.0] * 1024

    @pytest.mark.unit
    @pytest.mark.mlx
    def test_batch_embedding_fallback_on_error(self, mock_mlx_model):
        """Test batch embedding falls back to individual on error.

        Given: Batch embedding fails (model error)
        When: _get_text_embeddings() is called
        Then: Falls back to individual embedding for each text
        """
        from utils.mlx_embedding import MLXEmbedding

        # Configure mock to fail on batch, succeed on individual
        def mock_encode_with_failure(texts, show_progress=False):
            if len(texts) > 1:
                # Batch fails
                raise RuntimeError("Batch embedding failed")
            else:
                # Individual succeeds
                return [np.random.randn(1024)]

        mock_mlx_model.encode.side_effect = mock_encode_with_failure

        with patch('mlx_embedding_models.embedding.EmbeddingModel') as mock_class:
            mock_class.from_registry.return_value = mock_mlx_model

            mlx_emb = MLXEmbedding(model_name="BAAI/bge-m3")

            # Try batch embedding (should fall back to individual)
            texts = ["Text 1", "Text 2", "Text 3"]
            embeddings = mlx_emb._get_text_embeddings(texts)

            # Should return 3 embeddings (via fallback)
            assert len(embeddings) == 3
            assert all(len(emb) == 1024 for emb in embeddings)

    @pytest.mark.unit
    @pytest.mark.mlx
    def test_embed_determinism(self, mock_mlx_model):
        """Test embedding produces consistent results for same text.

        Given: The same text embedded twice
        When: _get_text_embedding() is called both times
        Then: Returns identical embeddings (deterministic)
        """
        from utils.mlx_embedding import MLXEmbedding

        with patch('mlx_embedding_models.embedding.EmbeddingModel') as mock_class:
            mock_class.from_registry.return_value = mock_mlx_model

            mlx_emb = MLXEmbedding(model_name="BAAI/bge-m3")

            text = "Test query for determinism"

            # Embed twice
            embedding1 = mlx_emb._get_text_embedding(text)
            embedding2 = mlx_emb._get_text_embedding(text)

            # Should be identical (deterministic)
            assert embedding1 == embedding2
            assert len(embedding1) == 1024


# ============================================================================
# Test Discovery and Execution Notes
# ============================================================================
"""
Run Day 5 tests:
    pytest tests/test_mlx_embedding_unit.py -v

Run with coverage:
    pytest tests/test_mlx_embedding_unit.py \
        --cov=utils.mlx_embedding \
        --cov-report=term-missing

Expected results:
    - 5 tests passing
    - Coverage increase: +30-40% for utils/mlx_embedding.py
    - Validates edge cases and prevents crashes on Apple Silicon

Note:
    These are unit tests with mocked MLX models.
    Integration tests (requiring real MLX hardware) exist in test_mlx_robustness.py
"""
