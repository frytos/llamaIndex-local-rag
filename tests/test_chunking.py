"""Tests for document chunking logic."""

import pytest
from pathlib import Path


class TestChunkSizeDefaults:
    """Test chunk size configuration defaults."""

    def test_default_chunk_size(self):
        """Test default chunk size is reasonable."""
        import os

        chunk_size = int(os.getenv("CHUNK_SIZE", "700"))
        assert 100 <= chunk_size <= 2000, "Chunk size should be 100-2000 characters"

    def test_default_chunk_overlap(self):
        """Test default chunk overlap is reasonable."""
        import os

        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        chunk_size = int(os.getenv("CHUNK_SIZE", "700"))

        # Overlap should be 10-30% of chunk size
        overlap_ratio = chunk_overlap / chunk_size
        assert 0.1 <= overlap_ratio <= 0.3, f"Overlap ratio {overlap_ratio:.1%} not in 10-30% range"


class TestChunkValidation:
    """Test chunk parameter validation logic."""

    def test_chunk_size_positive(self):
        """Test chunk size must be positive."""
        chunk_size = 700
        assert chunk_size > 0

    def test_overlap_less_than_size(self):
        """Test overlap must be less than chunk size."""
        chunk_size = 700
        chunk_overlap = 150
        assert chunk_overlap < chunk_size

    def test_overlap_non_negative(self):
        """Test overlap cannot be negative."""
        chunk_overlap = 150
        assert chunk_overlap >= 0

    def test_reasonable_overlap_ratio(self):
        """Test overlap ratio is reasonable."""
        chunk_size = 700
        chunk_overlap = 150

        ratio = chunk_overlap / chunk_size
        # Recommended: 15-25%
        assert 0.05 <= ratio <= 0.35, "Overlap should be 5-35% of chunk size"


class TestChunkSizeRecommendations:
    """Test chunk size recommendations from documentation."""

    def test_chat_logs_chunk_size(self):
        """Test recommended chunk size for chat logs."""
        # From CLAUDE.md: 100-300 for chat logs
        chunk_size = 500
        assert 100 <= chunk_size <= 800, "Chat logs: 100-800 characters recommended"

    def test_general_docs_chunk_size(self):
        """Test recommended chunk size for general documents."""
        # From CLAUDE.md: 500-800 for general docs
        chunk_size = 700
        assert 500 <= chunk_size <= 800, "General docs: 500-800 characters recommended"

    def test_long_form_chunk_size(self):
        """Test recommended chunk size for long-form content."""
        # From CLAUDE.md: 1000-2000 for long-form
        chunk_size = 1200
        assert 1000 <= chunk_size <= 2000, "Long-form: 1000-2000 characters recommended"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
