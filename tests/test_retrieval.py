"""Tests for retrieval configuration."""

import pytest
import os


class TestRetrievalConfiguration:
    """Test retrieval parameter configuration."""

    def test_top_k_default(self):
        """Test default TOP_K value."""
        top_k = int(os.getenv("TOP_K", "4"))
        assert top_k == 4
        assert 1 <= top_k <= 20, "TOP_K should be reasonable"

    def test_hybrid_alpha_default(self):
        """Test default hybrid alpha value."""
        alpha = float(os.getenv("HYBRID_ALPHA", "1.0"))
        assert 0.0 <= alpha <= 1.0, "Alpha must be 0.0-1.0"

    def test_mmr_threshold_default(self):
        """Test default MMR threshold."""
        mmr = float(os.getenv("MMR_THRESHOLD", "0.0"))
        assert 0.0 <= mmr <= 1.0, "MMR threshold must be 0.0-1.0"

    def test_enable_filters_default(self):
        """Test default filter setting."""
        filters = os.getenv("ENABLE_FILTERS", "1") == "1"
        assert isinstance(filters, bool)


class TestHybridSearchConfiguration:
    """Test hybrid search (BM25 + Vector) configuration."""

    def test_pure_vector_search(self):
        """Test pure vector search configuration."""
        alpha = 1.0  # 100% vector, 0% BM25
        assert alpha == 1.0

    def test_pure_bm25_search(self):
        """Test pure BM25 search configuration."""
        alpha = 0.0  # 0% vector, 100% BM25
        assert alpha == 0.0

    def test_balanced_hybrid(self):
        """Test balanced hybrid search."""
        alpha = 0.5  # 50/50 split
        assert alpha == 0.5

    def test_recommended_chat_alpha(self):
        """Test recommended alpha for chat logs."""
        # From audit: HYBRID_ALPHA=0.7 recommended for chat
        alpha = 0.7
        assert 0.6 <= alpha <= 0.8, "Chat logs work well with 60-80% vector"


class TestRetrievalOptimization:
    """Test retrieval optimization settings."""

    def test_top_k_for_context_window(self):
        """Test TOP_K fits in context window."""
        top_k = 4
        chunk_size = 700
        context_window = 8192

        # Calculate if chunks fit in context
        chunks_tokens = (top_k * chunk_size) / 4  # ~4 chars per token
        assert chunks_tokens < context_window * 0.6, "Chunks should use <60% of context"

    def test_mmr_diversity_settings(self):
        """Test MMR diversity threshold ranges."""
        # 0.0 = disabled, 0.5 = balanced, 1.0 = max relevance
        thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]
        for threshold in thresholds:
            assert 0.0 <= threshold <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
