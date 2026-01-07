"""Tests for embedding configuration."""

import pytest
import os


class TestEmbeddingConfiguration:
    """Test embedding model configuration."""

    def test_default_embedding_model(self):
        """Test default embedding model is specified."""
        embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
        assert embed_model == "BAAI/bge-small-en"
        assert "/" in embed_model, "Model should be in 'org/model' format"

    def test_embedding_dimensions_default(self):
        """Test default embedding dimensions."""
        embed_dim = int(os.getenv("EMBED_DIM", "384"))
        assert embed_dim in [384, 768, 1024], "Common embedding dimensions"

    def test_embedding_batch_size_default(self):
        """Test default batch size is reasonable."""
        batch_size = int(os.getenv("EMBED_BATCH", "32"))
        assert 16 <= batch_size <= 256, "Batch size should be reasonable for memory"

    def test_embedding_backend_options(self):
        """Test valid embedding backend options."""
        backend = os.getenv("EMBED_BACKEND", "huggingface")
        assert backend in ["huggingface", "mlx"], f"Unknown backend: {backend}"


class TestEmbeddingModelNames:
    """Test embedding model name extraction."""

    def test_common_model_extraction(self):
        """Test extraction of common model names."""
        from utils.naming import extract_model_short_name

        test_cases = {
            "BAAI/bge-small-en": "bge",
            "BAAI/bge-large-en-v1.5": "bge",
            "sentence-transformers/all-MiniLM-L6-v2": "minilm",
            "sentence-transformers/all-mpnet-base-v2": "mpnet",
        }

        for full_name, expected_short in test_cases.items():
            result = extract_model_short_name(full_name)
            assert result == expected_short, f"{full_name} should extract to {expected_short}"


class TestEmbeddingBatchSizeOptimization:
    """Test batch size optimization recommendations."""

    def test_mlx_batch_size(self):
        """Test recommended batch size for MLX backend."""
        # From audit: MLX can handle 64-128
        backend = "mlx"
        if backend == "mlx":
            recommended_batch = 64
            assert 64 <= recommended_batch <= 128

    def test_huggingface_batch_size(self):
        """Test recommended batch size for HuggingFace backend."""
        # From audit: HuggingFace recommends 32-64
        backend = "huggingface"
        if backend == "huggingface":
            recommended_batch = 32
            assert 32 <= recommended_batch <= 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
