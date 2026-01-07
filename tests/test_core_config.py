"""Tests for core.config module."""

import os
import pytest
from core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def set_test_credentials(monkeypatch):
    """Automatically set test database credentials for all tests."""
    monkeypatch.setenv("PGUSER", "test_user")
    monkeypatch.setenv("PGPASSWORD", "test_pass")


class TestSettings:
    """Test Settings dataclass."""

    def test_settings_uses_constants(self):
        """Test that Settings uses config constants."""
        settings = Settings()

        # Check that defaults come from constants
        assert settings.chunk_size == 700  # CHUNK.DEFAULT_SIZE
        assert settings.chunk_overlap == 150  # CHUNK.DEFAULT_OVERLAP
        assert settings.top_k == 4  # RETRIEVAL.DEFAULT_TOP_K
        assert settings.db_name == "vector_db"  # DATABASE.DEFAULT_DB_NAME

    def test_settings_env_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("CHUNK_SIZE", "500")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("TOP_K", "8")

        settings = Settings()

        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 100
        assert settings.top_k == 8

    def test_settings_requires_credentials(self, monkeypatch):
        """Test that Settings requires database credentials."""
        # Clear credentials by setting to None
        monkeypatch.delenv("PGUSER", raising=False)
        monkeypatch.delenv("PGPASSWORD", raising=False)

        with pytest.raises(ValueError, match="Database credentials not set"):
            Settings()


class TestSettingsValidation:
    """Test Settings validation methods."""

    def test_validate_chunk_size_bounds(self, monkeypatch):
        """Test chunk size must be within bounds."""
        monkeypatch.setenv("CHUNK_SIZE", "50")  # Below MIN_SIZE (100)

        settings = Settings()
        with pytest.raises(ValueError, match="chunk_size must be between"):
            settings.validate()

    def test_validate_chunk_overlap_vs_size(self, monkeypatch):
        """Test chunk overlap must be less than chunk size."""
        monkeypatch.setenv("CHUNK_SIZE", "500")
        monkeypatch.setenv("CHUNK_OVERLAP", "600")  # Greater than chunk_size

        settings = Settings()
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
            settings.validate()

    def test_validate_top_k_bounds(self, monkeypatch):
        """Test top_k must be within bounds."""
        monkeypatch.setenv("TOP_K", "0")  # Below MIN_TOP_K (1)

        settings = Settings()
        with pytest.raises(ValueError, match="top_k must be between"):
            settings.validate()

    def test_validate_temperature_range(self, monkeypatch):
        """Test temperature must be in valid range."""
        monkeypatch.setenv("TEMP", "3.0")  # Above 2.0

        settings = Settings()
        with pytest.raises(ValueError, match="temperature must be between"):
            settings.validate()

    def test_validate_success(self):
        """Test validation passes with valid settings."""
        settings = Settings()
        # Should not raise
        settings.validate()


class TestSettingsHelpers:
    """Test Settings helper methods."""

    def test_connection_string(self, monkeypatch):
        """Test connection string generation."""
        monkeypatch.setenv("PGUSER", "myuser")
        monkeypatch.setenv("PGPASSWORD", "mypass")
        monkeypatch.setenv("PGHOST", "localhost")
        monkeypatch.setenv("PGPORT", "5432")
        monkeypatch.setenv("DB_NAME", "mydb")

        settings = Settings()
        conn_str = settings.connection_string()

        assert conn_str == "postgresql://myuser:mypass@localhost:5432/mydb"


class TestGetSettings:
    """Test get_settings singleton function."""

    def test_get_settings_returns_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_singleton(self):
        """Test that get_settings returns the same instance."""
        # Clear any existing singleton
        import core.config
        core.config._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
