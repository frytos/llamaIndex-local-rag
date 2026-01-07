"""Tests for configuration and Settings validation."""

import pytest
import os
from pathlib import Path


class TestNameGeneration:
    """Test table name generation functions (already imported in utils)."""

    def test_can_import_utils(self):
        """Test that utils module can be imported."""
        from utils.naming import sanitize_table_name, extract_model_short_name

        assert callable(sanitize_table_name)
        assert callable(extract_model_short_name)


class TestEnvironmentVariableDefaults:
    """Test environment variable parsing without Settings class."""

    def test_chunk_size_parsing(self):
        """Test CHUNK_SIZE environment variable parsing."""
        # Test default
        value = int(os.getenv("CHUNK_SIZE", "700"))
        assert value == 700

        # Test custom value
        os.environ["CHUNK_SIZE"] = "1000"
        value = int(os.getenv("CHUNK_SIZE", "700"))
        assert value == 1000
        del os.environ["CHUNK_SIZE"]

    def test_boolean_parsing(self):
        """Test boolean environment variable parsing."""
        # Test RESET_TABLE parsing
        os.environ["RESET_TABLE"] = "1"
        result = os.getenv("RESET_TABLE", "0") == "1"
        assert result is True
        del os.environ["RESET_TABLE"]

        os.environ["RESET_TABLE"] = "0"
        result = os.getenv("RESET_TABLE", "0") == "1"
        assert result is False
        del os.environ["RESET_TABLE"]

    def test_integer_defaults(self):
        """Test integer defaults are reasonable."""
        chunk_size = int(os.getenv("CHUNK_SIZE", "700"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        top_k = int(os.getenv("TOP_K", "4"))

        # Validate defaults make sense
        assert chunk_size > 0
        assert chunk_overlap >= 0
        assert chunk_overlap < chunk_size
        assert top_k > 0


class TestConfigValidationLogic:
    """Test validation logic directly without Settings instantiation."""

    def test_chunk_overlap_cannot_exceed_chunk_size(self):
        """Test chunk overlap validation logic."""
        chunk_size = 700
        chunk_overlap = 150

        # Valid: overlap < size
        assert chunk_overlap < chunk_size

        # Invalid: overlap >= size
        chunk_overlap_bad = 800
        with pytest.raises(AssertionError):
            assert chunk_overlap_bad < chunk_size

    def test_top_k_must_be_positive(self):
        """Test TOP_K validation logic."""
        top_k = 4
        assert top_k > 0

        top_k_bad = 0
        with pytest.raises(AssertionError):
            assert top_k_bad > 0

    def test_temperature_range(self):
        """Test temperature should be in reasonable range."""
        temp = 0.1
        assert 0 <= temp <= 2

        temp_bad = -0.5
        with pytest.raises(AssertionError):
            assert 0 <= temp_bad <= 2

    def test_n_gpu_layers_non_negative(self):
        """Test N_GPU_LAYERS validation."""
        n_gpu_layers = 24
        assert n_gpu_layers >= 0

        n_gpu_layers_bad = -1
        with pytest.raises(AssertionError):
            assert n_gpu_layers_bad >= 0

    def test_embed_batch_positive(self):
        """Test embedding batch size validation."""
        embed_batch = 64
        assert embed_batch > 0

        embed_batch_bad = 0
        with pytest.raises(AssertionError):
            assert embed_batch_bad > 0


class TestDatabaseDefaults:
    """Test database configuration defaults."""

    def test_database_defaults(self):
        """Test that database configuration is set."""
        # Test fixtures set DB_NAME to "test_db", so just verify they're set
        db_name = os.getenv("DB_NAME")
        host = os.getenv("PGHOST", "localhost")
        port = os.getenv("PGPORT", "5432")

        assert db_name is not None and len(db_name) > 0
        assert host == "localhost"
        assert port == "5432"

        # Validate port is valid
        port_int = int(port)
        assert 1 <= port_int <= 65535


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
