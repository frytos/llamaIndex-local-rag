"""Tests for database configuration and utilities."""

import pytest
import os
from pathlib import Path


class TestDatabaseConfiguration:
    """Test database configuration parsing."""

    def test_database_host_default(self):
        """Test default database host."""
        host = os.getenv("PGHOST", "localhost")
        assert host in ["localhost", "127.0.0.1", "::1"], "Should default to localhost"

    def test_database_port_default(self):
        """Test default database port."""
        port = int(os.getenv("PGPORT", "5432"))
        assert port == 5432, "PostgreSQL default port is 5432"

    def test_database_port_validation(self):
        """Test port number validation."""
        port = int(os.getenv("PGPORT", "5432"))
        assert 1 <= port <= 65535, "Port must be in valid range"

    def test_database_name_default(self):
        """Test database name is set."""
        db_name = os.getenv("DB_NAME")
        # Test fixtures set this to "test_db", so just verify it's set
        assert db_name is not None
        assert len(db_name) > 0, "Database name cannot be empty"


class TestTableNameValidation:
    """Test table name validation."""

    def test_table_name_prefix(self):
        """Test that table names starting with numbers get prefixed."""
        from utils.naming import sanitize_table_name

        # Numbers at start should be prefixed
        result = sanitize_table_name("2024_report")
        assert result.startswith("t_"), "Names starting with numbers should be prefixed"
        assert result == "t_2024_report"

    def test_table_name_length(self):
        """Test that table names don't exceed PostgreSQL limits."""
        from utils.naming import generate_table_name

        long_name = "a" * 100
        doc_path = Path(f"{long_name}.pdf")
        result = generate_table_name(doc_path, 700, 150)

        # PostgreSQL identifier limit is 63 characters
        # Our naming adds suffixes, so base name should be limited
        assert len(result) < 63, "Table name should not exceed PostgreSQL limit"

    def test_table_name_sql_safe(self):
        """Test that table names are SQL-safe."""
        from utils.naming import sanitize_table_name

        unsafe_names = [
            "my-table",
            "my table",
            "table@name",
            "table#name",
            "table$name",
        ]

        for unsafe in unsafe_names:
            result = sanitize_table_name(unsafe)
            # Should only contain alphanumeric and underscores
            assert all(
                c.isalnum() or c == "_" for c in result
            ), f"Result '{result}' contains unsafe characters"
            # Should not contain original unsafe characters
            assert "-" not in result
            assert " " not in result
            assert "@" not in result


class TestConnectionString:
    """Test database connection string construction."""

    def test_connection_parameters(self):
        """Test that all required connection parameters have defaults."""
        params = {
            "host": os.getenv("PGHOST", "localhost"),
            "port": int(os.getenv("PGPORT", "5432")),
            "dbname": os.getenv("DB_NAME", "vector_db"),
        }

        assert all(params.values()), "All connection parameters should have values"
        assert isinstance(params["port"], int), "Port should be integer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
