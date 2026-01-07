"""Tests for configuration constants."""

import pytest
from config.constants import (
    CHUNK,
    SIMILARITY,
    LLM,
    RETRIEVAL,
    EMBEDDING,
    DATABASE,
    PERFORMANCE,
    ChunkConfig,
    SimilarityThresholds,
    LLMConfig,
    RetrievalConfig,
    EmbeddingConfig,
    DatabaseConfig,
    PerformanceConfig,
)


class TestChunkConfig:
    """Test chunk configuration constants."""

    def test_default_values(self):
        """Test that chunk config has expected defaults."""
        assert CHUNK.DEFAULT_SIZE == 700
        assert CHUNK.DEFAULT_OVERLAP == 150
        assert CHUNK.MIN_SIZE == 100
        assert CHUNK.MAX_SIZE == 2000

    def test_frozen(self):
        """Test that config is immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            CHUNK.DEFAULT_SIZE = 999


class TestSimilarityThresholds:
    """Test similarity threshold constants."""

    def test_threshold_ordering(self):
        """Test that thresholds are properly ordered."""
        assert SIMILARITY.EXCELLENT > SIMILARITY.GOOD
        assert SIMILARITY.GOOD > SIMILARITY.FAIR
        assert SIMILARITY.FAIR > SIMILARITY.MINIMUM

    def test_threshold_values(self):
        """Test specific threshold values."""
        assert SIMILARITY.EXCELLENT == 0.8
        assert SIMILARITY.GOOD == 0.6
        assert SIMILARITY.FAIR == 0.4
        assert SIMILARITY.MINIMUM == 0.3


class TestLLMConfig:
    """Test LLM configuration constants."""

    def test_default_values(self):
        """Test LLM defaults."""
        assert LLM.DEFAULT_CONTEXT_WINDOW == 3072
        assert LLM.DEFAULT_MAX_TOKENS == 256
        assert LLM.DEFAULT_TEMPERATURE == 0.1
        assert LLM.DEFAULT_GPU_LAYERS == 24
        assert LLM.DEFAULT_BATCH_SIZE == 256

    def test_temperature_thresholds(self):
        """Test temperature classification thresholds."""
        assert LLM.TEMP_FACTUAL < LLM.TEMP_BALANCED
        assert LLM.TEMP_FACTUAL == 0.3
        assert LLM.TEMP_BALANCED == 0.7


class TestRetrievalConfig:
    """Test retrieval configuration constants."""

    def test_default_values(self):
        """Test retrieval defaults."""
        assert RETRIEVAL.DEFAULT_TOP_K == 4
        assert RETRIEVAL.DEFAULT_RERANK_TOP_K == 4
        assert RETRIEVAL.MIN_TOP_K == 1
        assert RETRIEVAL.MAX_TOP_K == 20


class TestEmbeddingConfig:
    """Test embedding configuration constants."""

    def test_default_values(self):
        """Test embedding defaults."""
        assert EMBEDDING.DEFAULT_MODEL == "BAAI/bge-small-en"
        assert EMBEDDING.DEFAULT_DIMENSION == 384
        assert EMBEDDING.DEFAULT_BATCH_SIZE == 64


class TestDatabaseConfig:
    """Test database configuration constants."""

    def test_default_values(self):
        """Test database defaults."""
        assert DATABASE.DEFAULT_HOST == "localhost"
        assert DATABASE.DEFAULT_PORT == "5432"
        assert DATABASE.DEFAULT_DB_NAME == "vector_db"
        assert DATABASE.DEFAULT_POOL_SIZE == 5
        assert DATABASE.DEFAULT_TIMEOUT == 30


class TestPerformanceConfig:
    """Test performance configuration constants."""

    def test_default_values(self):
        """Test performance constants."""
        assert PERFORMANCE.BYTES_PER_FLOAT32 == 4
        assert PERFORMANCE.DEFAULT_PREVIEW_LENGTH == 150
        assert PERFORMANCE.PROGRESS_BAR_MIN_ITEMS == 10


class TestConstantsIntegration:
    """Test that constants work together properly."""

    def test_all_constants_importable(self):
        """Test that all constants can be imported."""
        # If this test runs, imports succeeded
        assert CHUNK is not None
        assert SIMILARITY is not None
        assert LLM is not None
        assert RETRIEVAL is not None
        assert EMBEDDING is not None
        assert DATABASE is not None
        assert PERFORMANCE is not None

    def test_classes_match_instances(self):
        """Test that exported instances match their classes."""
        assert isinstance(CHUNK, ChunkConfig)
        assert isinstance(SIMILARITY, SimilarityThresholds)
        assert isinstance(LLM, LLMConfig)
        assert isinstance(RETRIEVAL, RetrievalConfig)
        assert isinstance(EMBEDDING, EmbeddingConfig)
        assert isinstance(DATABASE, DatabaseConfig)
        assert isinstance(PERFORMANCE, PerformanceConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
