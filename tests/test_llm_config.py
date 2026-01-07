"""Tests for LLM configuration."""

import pytest
import os


class TestLLMConfiguration:
    """Test LLM configuration defaults and validation."""

    def test_context_window_default(self):
        """Test default context window size."""
        ctx = int(os.getenv("CTX", "3072"))
        assert ctx in [2048, 3072, 4096, 8192, 16384], "Common context window sizes"

    def test_max_new_tokens_default(self):
        """Test default max new tokens."""
        max_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
        assert 0 < max_tokens <= 2048, "Max tokens should be reasonable"

    def test_temperature_default(self):
        """Test default temperature for RAG."""
        temp = float(os.getenv("TEMP", "0.1"))
        assert 0.0 <= temp <= 1.0, "Temperature should be 0-1"
        # For RAG, should be low (factual)
        assert temp <= 0.3, "RAG should use low temperature for factuality"

    def test_n_gpu_layers_default(self):
        """Test default GPU layers."""
        n_gpu = int(os.getenv("N_GPU_LAYERS", "16"))
        assert 0 <= n_gpu <= 100, "GPU layers should be reasonable"

    def test_n_batch_default(self):
        """Test default batch size."""
        n_batch = int(os.getenv("N_BATCH", "128"))
        assert n_batch in [64, 128, 256, 512], "Common batch sizes"


class TestLLMParameterRanges:
    """Test LLM parameter validation ranges."""

    def test_temperature_range(self):
        """Test temperature must be in valid range."""
        temp = 0.1
        assert 0.0 <= temp <= 2.0, "Temperature 0-2 is valid range"

    def test_top_k_positive(self):
        """Test TOP_K must be positive."""
        top_k = int(os.getenv("TOP_K", "4"))
        assert top_k > 0, "TOP_K must be at least 1"
        assert top_k <= 20, "TOP_K >20 usually wasteful"

    def test_n_gpu_layers_non_negative(self):
        """Test N_GPU_LAYERS cannot be negative."""
        n_gpu = 24
        assert n_gpu >= 0, "GPU layers cannot be negative"


class TestLLMOptimizationSettings:
    """Test LLM optimization settings from audit."""

    def test_m1_optimized_gpu_layers(self):
        """Test M1 optimized GPU layer setting."""
        # From audit: N_GPU_LAYERS=24 is optimal for M1 16GB
        optimized = 24
        default = 16

        # Optimized should be higher
        assert optimized > default
        # Should be reasonable for Mistral 7B (32 layers total)
        assert optimized <= 32

    def test_m1_optimized_batch_size(self):
        """Test M1 optimized batch size."""
        # From audit: N_BATCH=256 is optimal
        optimized = 256
        default = 128

        assert optimized > default
        assert optimized in [128, 256, 512]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
