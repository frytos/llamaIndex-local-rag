"""Tests for configuration and Settings validation."""

import pytest
import os
from unittest.mock import patch


class TestSettingsValidation:
    """Test Settings class validation logic."""

    def test_chunk_size_validation(self):
        """Test that invalid chunk sizes are caught."""
        # Import here to avoid import errors if Settings has issues
        from rag_low_level_m1_16gb_verbose import Settings

        # Test invalid chunk size
        with patch.dict(os.environ, {
            'CHUNK_SIZE': '-100',
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }):
            with pytest.raises(ValueError, match="CHUNK_SIZE must be positive"):
                s = Settings()
                s.validate()

    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        from rag_low_level_m1_16gb_verbose import Settings

        with patch.dict(os.environ, {
            'CHUNK_SIZE': '700',
            'CHUNK_OVERLAP': '800',  # Overlap > chunk_size
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }):
            with pytest.raises(ValueError, match="CHUNK_OVERLAP.*cannot exceed"):
                s = Settings()
                s.validate()

    def test_top_k_validation(self):
        """Test TOP_K parameter validation."""
        from rag_low_level_m1_16gb_verbose import Settings

        with patch.dict(os.environ, {
            'TOP_K': '0',
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }):
            with pytest.raises(ValueError, match="TOP_K must be.*positive"):
                s = Settings()
                s.validate()

    def test_missing_credentials_error(self):
        """Test that missing credentials raise helpful error."""
        from rag_low_level_m1_16gb_verbose import Settings

        # Clear credentials from environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Database credentials not set"):
                Settings()

    def test_valid_configuration(self):
        """Test that valid configuration passes."""
        from rag_low_level_m1_16gb_verbose import Settings

        with patch.dict(os.environ, {
            'PGUSER': 'testuser',
            'PGPASSWORD': 'testpass',
            'CHUNK_SIZE': '700',
            'CHUNK_OVERLAP': '150',
            'TOP_K': '4'
        }):
            s = Settings()
            s.validate()  # Should not raise

            assert s.chunk_size == 700
            assert s.chunk_overlap == 150
            assert s.top_k == 4
            assert s.user == 'testuser'
            assert s.password == 'testpass'

    def test_embed_batch_size_validation(self):
        """Test embedding batch size validation."""
        from rag_low_level_m1_16gb_verbose import Settings

        with patch.dict(os.environ, {
            'EMBED_BATCH': '0',
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }):
            with pytest.raises(ValueError):
                s = Settings()
                s.validate()

    def test_n_gpu_layers_validation(self):
        """Test N_GPU_LAYERS validation."""
        from rag_low_level_m1_16gb_verbose import Settings

        with patch.dict(os.environ, {
            'N_GPU_LAYERS': '-1',
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }):
            with pytest.raises(ValueError):
                s = Settings()
                s.validate()


class TestEnvironmentVariables:
    """Test environment variable parsing."""

    def test_boolean_parsing(self):
        """Test boolean environment variable parsing."""
        from rag_low_level_m1_16gb_verbose import Settings

        # Test RESET_TABLE parsing
        with patch.dict(os.environ, {
            'RESET_TABLE': '1',
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }):
            s = Settings()
            assert s.reset_table is True

        with patch.dict(os.environ, {
            'RESET_TABLE': '0',
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }):
            s = Settings()
            assert s.reset_table is False

    def test_integer_parsing(self):
        """Test integer environment variable parsing."""
        from rag_low_level_m1_16gb_verbose import Settings

        with patch.dict(os.environ, {
            'CHUNK_SIZE': '1000',
            'TOP_K': '6',
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }):
            s = Settings()
            assert isinstance(s.chunk_size, int)
            assert s.chunk_size == 1000
            assert isinstance(s.top_k, int)
            assert s.top_k == 6

    def test_default_values(self):
        """Test that default values are applied correctly."""
        from rag_low_level_m1_16gb_verbose import Settings

        with patch.dict(os.environ, {
            'PGUSER': 'test',
            'PGPASSWORD': 'test'
        }, clear=True):
            s = Settings()

            # Check defaults
            assert s.chunk_size == 700  # Default
            assert s.chunk_overlap == 150  # Default
            assert s.top_k == 4  # Default
            assert s.host == "localhost"  # Default
            assert s.port == "5432"  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
