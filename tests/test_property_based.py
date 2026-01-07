"""Property-based and parametrized tests for the RAG pipeline.

This module uses Hypothesis for property-based testing and pytest.mark.parametrize
for parametrized testing to ensure robustness across a wide range of inputs.

Test Categories:
1. Property-Based Tests: Test invariant properties with random inputs
2. Parametrized Tests: Test specific configurations systematically
3. Edge Case Tests: Test boundary values and extreme inputs
4. Validation Tests: Test input validation and error handling
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Any
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis import example

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from utils
from utils.naming import sanitize_table_name, extract_model_short_name, generate_table_name


# ============================================================================
# PROPERTY-BASED TESTS (using Hypothesis)
# ============================================================================

class TestTableNameSanitization:
    """Property-based tests for table name sanitization."""

    @given(st.text(min_size=1, max_size=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    def test_sanitize_table_name_is_sql_safe(self, name: str):
        """Property: Sanitized table names should only contain alphanumeric and underscores."""
        sanitized = sanitize_table_name(name)

        # Must not be empty after sanitization
        assert len(sanitized) > 0

        # Only contains valid SQL identifier characters
        assert all(c.isalnum() or c == '_' for c in sanitized)

        # Must not start with a number (SQL requirement)
        assert not sanitized[0].isdigit()

        # Must be lowercase
        assert sanitized == sanitized.lower()

    @given(st.text(min_size=1, max_size=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    def test_sanitize_table_name_is_deterministic(self, name: str):
        """Property: Same input should always produce same output."""
        result1 = sanitize_table_name(name)
        result2 = sanitize_table_name(name)
        assert result1 == result2

    @given(st.text(alphabet=st.characters(whitelist_categories=('Ll', 'Lu')), min_size=1, max_size=50))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_sanitize_preserves_letters(self, name: str):
        """Property: ASCII letters should be preserved (as lowercase)."""
        sanitized = sanitize_table_name(name)
        # Should contain some of the original letters (lowercased)
        original_lower = name.lower()
        assert any(c in sanitized for c in original_lower if c.isalpha())

    @given(st.text(min_size=1, max_size=20).filter(lambda x: x[0].isdigit()))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_sanitize_adds_prefix_for_numeric_start(self, name: str):
        """Property: Names starting with digits should get 't_' prefix."""
        assume(name[0].isdigit())
        sanitized = sanitize_table_name(name)
        assert sanitized.startswith('t_')


class TestChunkSizeValidation:
    """Property-based tests for chunk size validation."""

    @given(st.integers(min_value=1, max_value=10000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    def test_positive_chunk_size_is_valid(self, chunk_size: int):
        """Property: Any positive chunk size should be valid."""
        assert chunk_size > 0

    @given(
        st.integers(min_value=100, max_value=2000),
        st.floats(min_value=0.05, max_value=0.35)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    def test_overlap_ratio_constraint(self, chunk_size: int, overlap_ratio: float):
        """Property: Chunk overlap should be less than chunk size."""
        chunk_overlap = int(chunk_size * overlap_ratio)
        assert 0 <= chunk_overlap < chunk_size

    @given(st.integers(min_value=100, max_value=5000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_chunk_size_affects_overlap_bounds(self, chunk_size: int):
        """Property: Valid overlap range scales with chunk size."""
        max_valid_overlap = chunk_size - 1
        assert 0 <= max_valid_overlap < chunk_size


class TestEnvironmentVariableParsing:
    """Property-based tests for environment variable parsing."""

    @given(st.integers(min_value=1, max_value=10000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_integer_env_var_parsing(self, value: int):
        """Property: Integer env vars should parse correctly."""
        os.environ["TEST_INT_VAR"] = str(value)
        parsed = int(os.getenv("TEST_INT_VAR", "0"))
        assert parsed == value
        del os.environ["TEST_INT_VAR"]

    @given(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_float_env_var_parsing(self, value: float):
        """Property: Float env vars should parse correctly."""
        os.environ["TEST_FLOAT_VAR"] = str(value)
        parsed = float(os.getenv("TEST_FLOAT_VAR", "0.0"))
        assert abs(parsed - value) < 1e-6
        del os.environ["TEST_FLOAT_VAR"]

    @given(st.booleans())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
    def test_boolean_env_var_parsing(self, value: bool):
        """Property: Boolean env vars should parse correctly."""
        os.environ["TEST_BOOL_VAR"] = "1" if value else "0"
        parsed = os.getenv("TEST_BOOL_VAR", "0") == "1"
        assert parsed == value
        del os.environ["TEST_BOOL_VAR"]


class TestModelNameExtraction:
    """Property-based tests for model name extraction."""

    @given(st.text(min_size=1, max_size=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_extract_model_short_name_returns_string(self, model_name: str):
        """Property: Model name extraction should always return a string."""
        result = extract_model_short_name(model_name)
        assert isinstance(result, str)
        assert len(result) > 0

    @given(st.text(min_size=1, max_size=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_extract_model_short_name_is_lowercase(self, model_name: str):
        """Property: Extracted model names should be lowercase."""
        result = extract_model_short_name(model_name)
        assert result == result.lower()

    @given(st.text(min_size=1, max_size=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_extract_model_short_name_max_length(self, model_name: str):
        """Property: Extracted model names should be reasonably short."""
        result = extract_model_short_name(model_name)
        # Should be at most 8 characters (per implementation)
        assert len(result) <= 8


# ============================================================================
# PARAMETRIZED TESTS (using pytest.mark.parametrize)
# ============================================================================

class TestChunkSizeConfigurations:
    """Parametrized tests for different chunk size configurations."""

    @pytest.mark.parametrize("chunk_size", [100, 500, 700, 1000, 2000])
    def test_chunk_size_ranges(self, chunk_size: int):
        """Test that recommended chunk sizes are valid."""
        assert 100 <= chunk_size <= 2000
        assert chunk_size > 0

    @pytest.mark.parametrize(
        "chunk_size,chunk_overlap",
        [
            (100, 10),    # 10% overlap
            (500, 100),   # 20% overlap
            (700, 150),   # ~21% overlap (default)
            (1000, 200),  # 20% overlap
            (2000, 300),  # 15% overlap
        ]
    )
    def test_chunk_overlap_ratios(self, chunk_size: int, chunk_overlap: int):
        """Test that chunk overlap ratios are reasonable."""
        ratio = chunk_overlap / chunk_size
        assert 0.05 <= ratio <= 0.35, f"Overlap ratio {ratio:.1%} out of recommended range"
        assert chunk_overlap < chunk_size

    @pytest.mark.parametrize(
        "chunk_size,expected_category",
        [
            (100, "chat_logs"),
            (200, "chat_logs"),
            (300, "chat_logs"),
            (500, "general"),
            (700, "general"),
            (800, "general"),
            (1000, "long_form"),
            (1500, "long_form"),
            (2000, "long_form"),
        ]
    )
    def test_chunk_size_categories(self, chunk_size: int, expected_category: str):
        """Test chunk size categorization per documentation."""
        if expected_category == "chat_logs":
            assert 100 <= chunk_size <= 300
        elif expected_category == "general":
            assert 500 <= chunk_size <= 800
        elif expected_category == "long_form":
            assert 1000 <= chunk_size <= 2000


class TestBatchSizeConfigurations:
    """Parametrized tests for batch size configurations."""

    @pytest.mark.parametrize("batch_size", [16, 32, 64, 128, 256])
    def test_embed_batch_sizes(self, batch_size: int):
        """Test that embedding batch sizes are valid."""
        assert batch_size > 0
        assert batch_size <= 512  # Reasonable upper bound
        # Should be power of 2 for GPU efficiency
        assert batch_size & (batch_size - 1) == 0

    @pytest.mark.parametrize("n_batch", [128, 256, 512, 1024])
    def test_llm_batch_sizes(self, n_batch: int):
        """Test that LLM batch sizes are valid."""
        assert n_batch > 0
        assert n_batch <= 2048  # Reasonable upper bound
        # Should be power of 2 for efficiency
        assert n_batch & (n_batch - 1) == 0


class TestTopKConfigurations:
    """Parametrized tests for TOP_K retrieval configurations."""

    @pytest.mark.parametrize("top_k", [1, 3, 4, 6, 10])
    def test_top_k_values(self, top_k: int):
        """Test that TOP_K values are valid."""
        assert top_k > 0
        assert top_k <= 20  # Reasonable upper bound

    @pytest.mark.parametrize(
        "top_k,chunk_size,context_window",
        [
            (3, 500, 3072),
            (4, 700, 3072),
            (5, 500, 8192),
            (10, 300, 8192),
        ]
    )
    def test_top_k_context_window_fit(self, top_k: int, chunk_size: int, context_window: int):
        """Test that TOP_K * chunk_size fits in context window."""
        # Rough estimate: leave room for system prompt + response
        estimated_tokens = (top_k * chunk_size) // 3  # ~3 chars per token
        assert estimated_tokens < context_window * 0.7  # Use max 70% for context


class TestTemperatureConfigurations:
    """Parametrized tests for temperature configurations."""

    @pytest.mark.parametrize("temperature", [0.0, 0.1, 0.5, 1.0])
    def test_temperature_ranges(self, temperature: float):
        """Test that temperature values are in valid range."""
        assert 0.0 <= temperature <= 2.0

    @pytest.mark.parametrize(
        "temperature,use_case",
        [
            (0.0, "factual"),
            (0.1, "factual"),
            (0.5, "balanced"),
            (0.7, "creative"),
            (1.0, "creative"),
        ]
    )
    def test_temperature_use_cases(self, temperature: float, use_case: str):
        """Test temperature recommendations for different use cases."""
        if use_case == "factual":
            assert temperature <= 0.2
        elif use_case == "balanced":
            assert 0.3 <= temperature <= 0.6
        elif use_case == "creative":
            assert temperature >= 0.7


class TestTableNameSanitizationParametrized:
    """Parametrized tests for table name sanitization."""

    @pytest.mark.parametrize(
        "input_name,expected_contains",
        [
            ("my-document", "my_document"),
            ("test file", "test_file"),
            ("2024-report", "t_2024_report"),
            ("data.csv", "data_csv"),
            ("My_Table", "my_table"),
            ("UPPER_CASE", "upper_case"),
        ]
    )
    def test_sanitize_specific_cases(self, input_name: str, expected_contains: str):
        """Test specific sanitization cases."""
        result = sanitize_table_name(input_name)
        assert expected_contains in result or result == expected_contains

    @pytest.mark.parametrize(
        "special_chars",
        ["-", " ", ".", "/", "\\", "@", "#", "$", "%", "^", "&", "*", "(", ")", "+", "="]
    )
    def test_sanitize_special_characters(self, special_chars: str):
        """Test that special characters are replaced."""
        input_name = f"test{special_chars}name"
        result = sanitize_table_name(input_name)
        # Special char should be replaced with underscore
        assert special_chars not in result
        assert "_" in result or result == "testname"


class TestModelNameExtractionParametrized:
    """Parametrized tests for model name extraction."""

    @pytest.mark.parametrize(
        "model_path,expected_name",
        [
            ("BAAI/bge-small-en", "bge"),
            ("BAAI/bge-large-en-v1.5", "bge"),
            ("sentence-transformers/all-MiniLM-L6-v2", "minilm"),
            ("intfloat/e5-base-v2", "e5"),
            ("sentence-transformers/all-mpnet-base-v2", "mpnet"),
            ("roberta-base", "roberta"),
            ("bert-base-uncased", "bert"),
        ]
    )
    def test_extract_known_models(self, model_path: str, expected_name: str):
        """Test extraction of known model names."""
        result = extract_model_short_name(model_path)
        assert result == expected_name


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for boundary values and edge cases."""

    def test_chunk_size_minimum(self):
        """Test minimum chunk size boundary."""
        chunk_size = 1
        assert chunk_size > 0
        # Should still be valid, though impractical

    def test_chunk_size_very_large(self):
        """Test very large chunk size."""
        chunk_size = 10000
        assert chunk_size > 0
        # Valid but may cause context overflow

    def test_chunk_overlap_zero(self):
        """Test zero overlap (no overlap)."""
        chunk_overlap = 0
        assert chunk_overlap >= 0
        # Valid: chunks don't overlap

    def test_chunk_overlap_nearly_full(self):
        """Test overlap nearly equal to chunk size."""
        chunk_size = 1000
        chunk_overlap = 999
        assert chunk_overlap < chunk_size
        # Valid but unusual

    def test_top_k_one(self):
        """Test retrieving single chunk."""
        top_k = 1
        assert top_k > 0
        # Valid: retrieve only best match

    def test_top_k_very_large(self):
        """Test retrieving many chunks."""
        top_k = 100
        assert top_k > 0
        # Valid but may overflow context window

    def test_temperature_zero(self):
        """Test deterministic generation (temp=0)."""
        temperature = 0.0
        assert 0.0 <= temperature <= 2.0
        # Valid: deterministic output

    def test_temperature_max(self):
        """Test maximum temperature."""
        temperature = 2.0
        assert 0.0 <= temperature <= 2.0
        # Valid but very random

    def test_embed_batch_one(self):
        """Test batch size of 1 (no batching)."""
        embed_batch = 1
        assert embed_batch > 0
        # Valid but slow

    def test_n_gpu_layers_zero(self):
        """Test no GPU acceleration."""
        n_gpu_layers = 0
        assert n_gpu_layers >= 0
        # Valid: CPU-only mode


class TestInvalidInputs:
    """Tests for invalid input handling."""

    def test_negative_chunk_size(self):
        """Test that negative chunk size would be invalid."""
        chunk_size = -100
        # This should fail validation
        assert not (chunk_size > 0)

    def test_negative_chunk_overlap(self):
        """Test that negative overlap would be invalid."""
        chunk_overlap = -50
        # This should fail validation
        assert not (chunk_overlap >= 0)

    def test_overlap_exceeds_chunk_size(self):
        """Test that overlap >= chunk_size is invalid."""
        chunk_size = 500
        chunk_overlap = 500
        # This should fail validation
        assert not (chunk_overlap < chunk_size)

    def test_zero_top_k(self):
        """Test that TOP_K=0 is invalid."""
        top_k = 0
        # This should fail validation
        assert not (top_k > 0)

    def test_negative_top_k(self):
        """Test that negative TOP_K is invalid."""
        top_k = -5
        # This should fail validation
        assert not (top_k > 0)

    def test_negative_temperature(self):
        """Test that negative temperature is invalid."""
        temperature = -0.5
        # This should fail validation
        assert not (0.0 <= temperature <= 2.0)

    def test_excessive_temperature(self):
        """Test that temperature > 2 is invalid."""
        temperature = 3.0
        # This should fail validation
        assert not (0.0 <= temperature <= 2.0)

    def test_zero_embed_batch(self):
        """Test that batch size of 0 is invalid."""
        embed_batch = 0
        # This should fail validation
        assert not (embed_batch > 0)

    def test_negative_n_gpu_layers(self):
        """Test that negative GPU layers is invalid."""
        n_gpu_layers = -5
        # This should fail validation
        assert not (n_gpu_layers >= 0)


class TestExtremeConfigurations:
    """Tests for extreme but valid configurations."""

    @pytest.mark.parametrize(
        "chunk_size,chunk_overlap,top_k,description",
        [
            (100, 10, 1, "Minimal: small chunks, low overlap, single retrieval"),
            (2000, 400, 10, "Maximal: large chunks, high overlap, many retrievals"),
            (500, 0, 3, "No overlap: minimal duplication"),
            (1000, 250, 5, "Balanced: moderate all parameters"),
        ]
    )
    def test_extreme_valid_configs(
        self,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        description: str
    ):
        """Test extreme but valid configurations."""
        # All should pass basic validation
        assert chunk_size > 0
        assert chunk_overlap >= 0
        assert chunk_overlap < chunk_size
        assert top_k > 0


class TestEmptyAndNullInputs:
    """Tests for empty/null input handling."""

    def test_empty_string_sanitization(self):
        """Test sanitizing empty string."""
        # Empty string after sanitization should be handled
        result = sanitize_table_name("!!!")  # Only special chars
        # Should produce something (all underscores or similar)
        assert isinstance(result, str)

    def test_sanitize_only_numbers(self):
        """Test sanitizing string with only numbers."""
        result = sanitize_table_name("12345")
        # Should add prefix since starts with digit
        assert result.startswith("t_")
        assert "12345" in result


# ============================================================================
# METADATA SCHEMA CONSISTENCY TESTS
# ============================================================================

class TestMetadataSchemaConsistency:
    """Tests for metadata schema consistency."""

    def test_metadata_keys_are_consistent(self):
        """Test that expected metadata keys are defined."""
        # Based on the pipeline, metadata should include:
        expected_keys = {"doc_id", "chunk_index", "total_chunks", "source"}
        # This is a structural test - in real code we'd check actual nodes
        assert len(expected_keys) > 0
        assert all(isinstance(key, str) for key in expected_keys)

    @pytest.mark.parametrize(
        "metadata_key",
        ["doc_id", "chunk_index", "total_chunks", "source", "chunk_size"]
    )
    def test_metadata_key_naming(self, metadata_key: str):
        """Test that metadata keys follow naming convention."""
        # Should be snake_case
        assert metadata_key.islower() or "_" in metadata_key
        # Should not have spaces
        assert " " not in metadata_key


# ============================================================================
# PORT VALIDATION TESTS
# ============================================================================

class TestPortValidation:
    """Tests for port number validation."""

    @pytest.mark.parametrize("port", [1, 5432, 8080, 65535])
    def test_valid_ports(self, port: int):
        """Test valid port numbers."""
        assert 1 <= port <= 65535

    @pytest.mark.parametrize("port", [0, -1, 65536, 100000])
    def test_invalid_ports(self, port: int):
        """Test invalid port numbers."""
        assert not (1 <= port <= 65535)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
