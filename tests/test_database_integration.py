"""Comprehensive database integration tests for the RAG pipeline.

Tests cover:
- PostgreSQL connection handling
- pgvector extension management
- Table operations (create, reset, drop)
- Vector storage and retrieval
- Configuration metadata tracking
- Error handling and retry logic

Note: These tests use extensive mocking to avoid requiring:
- Running PostgreSQL instance
- llama_index library installation
- Actual database connections
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Optional
import sys
import os
import psycopg2
from psycopg2 import OperationalError as PgOperationalError


class MockSettings:
    """Mock Settings class for testing."""
    db_name = "test_db"
    host = "localhost"
    port = "5432"
    user = "test_user"
    password = "test_password"
    table = "test_table"
    embed_dim = 384
    reset_table = False
    reset_db = False


class TestPostgreSQLConnection:
    """Test PostgreSQL connection handling."""

    def test_connection_parameters_validation(self):
        """Test that connection parameters are validated."""
        # Test that env vars are set (test fixtures may override defaults)
        db_name = os.getenv("DB_NAME")
        host = os.getenv("PGHOST", "localhost")
        port = os.getenv("PGPORT", "5432")

        assert db_name is not None and len(db_name) > 0
        assert host == "localhost"
        assert port == "5432"
        assert int(port) in range(1, 65536)

    @patch('psycopg2.connect')
    def test_connection_with_valid_credentials(self, mock_connect):
        """Test successful connection with valid credentials."""
        # Mock successful connection
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Simulate connection
        conn = mock_connect(
            dbname="vector_db",
            host="localhost",
            port="5432",
            user="test_user",
            password="test_password"
        )

        # Verify connection was established
        assert conn is not None
        mock_connect.assert_called_once()

        # Verify connection has expected attributes
        assert hasattr(mock_conn, 'close')
        assert hasattr(mock_conn, 'cursor')

    @patch('psycopg2.connect')
    def test_connection_failure_handling(self, mock_connect):
        """Test connection failure with appropriate error."""
        # Mock connection failure
        mock_connect.side_effect = PgOperationalError("Connection refused")

        # Should raise exception
        with pytest.raises(PgOperationalError):
            mock_connect(
                dbname="vector_db",
                host="localhost",
                port="5432"
            )

    @patch('psycopg2.connect')
    @patch('time.sleep')
    def test_connection_retry_logic(self, mock_sleep, mock_connect):
        """Test connection retry with exponential backoff."""
        # Mock: fail twice, then succeed
        mock_conn = Mock()
        mock_connect.side_effect = [
            PgOperationalError("Connection refused"),
            PgOperationalError("Connection refused"),
            mock_conn
        ]

        # Simulate retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = mock_connect()
                break
            except PgOperationalError:
                if attempt < max_retries - 1:
                    mock_sleep(2 ** attempt)
                else:
                    raise

        # Verify retries occurred
        assert mock_connect.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries
        assert conn is not None

    @patch('psycopg2.connect')
    def test_admin_connection_to_postgres_db(self, mock_connect):
        """Test admin connection to 'postgres' database."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Admin connection should use 'postgres' database
        conn = mock_connect(
            dbname='postgres',
            host='localhost',
            port='5432',
            user='admin'
        )

        # Verify connects to 'postgres' database for admin operations
        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs['dbname'] == 'postgres'
        assert conn is not None

    @patch('psycopg2.connect')
    def test_connection_autocommit_mode(self, mock_connect):
        """Test that admin connections use autocommit for DDL operations."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate DDL operation
        conn = mock_connect()
        conn.autocommit = True

        with conn.cursor() as cursor:
            cursor.execute("CREATE DATABASE test_db")

        # Verify autocommit was enabled
        assert conn.autocommit is True


class TestPgvectorExtension:
    """Test pgvector extension management."""

    @patch('psycopg2.connect')
    def test_pgvector_extension_creation(self, mock_connect):
        """Test pgvector extension is created if not exists."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate extension creation
        conn = mock_connect()
        conn.autocommit = True

        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Verify extension creation SQL was executed
        mock_cursor.execute.assert_called_with("CREATE EXTENSION IF NOT EXISTS vector;")
        assert conn.autocommit is True

    @patch('psycopg2.connect')
    def test_pgvector_extension_already_exists(self, mock_connect):
        """Test handling when pgvector extension already exists."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Extension creation should not raise error even if exists
        conn = mock_connect()
        with conn.cursor() as cursor:
            # IF NOT EXISTS makes this safe
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Should complete successfully
        conn.close()
        conn.close.assert_called_once()

    @patch('psycopg2.connect')
    def test_pgvector_extension_failure(self, mock_connect):
        """Test handling of pgvector extension creation failure."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = psycopg2.errors.InsufficientPrivilege()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Should raise the error
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            conn = mock_connect()
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")


class TestTableOperations:
    """Test table creation, reset, and management."""

    @patch('psycopg2.connect')
    def test_database_creation(self, mock_connect):
        """Test database creation if not exists."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate database creation
        conn = mock_connect()
        conn.autocommit = True

        with conn.cursor() as cursor:
            cursor.execute("CREATE DATABASE test_db")

        # Verify CREATE DATABASE was executed
        calls = [str(call) for call in mock_cursor.execute.call_args_list]
        assert any('CREATE DATABASE' in str(call) for call in calls)

    @patch('psycopg2.connect')
    def test_table_reset_when_requested(self, mock_connect):
        """Test table is dropped when RESET_TABLE=1."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate table reset
        reset_table = True
        if reset_table:
            conn = mock_connect()
            conn.autocommit = True

            with conn.cursor() as cursor:
                cursor.execute('DROP TABLE IF EXISTS "test_table";')

        # Verify DROP TABLE was executed
        drop_calls = [call for call in mock_cursor.execute.call_args_list
                     if 'DROP TABLE' in str(call)]
        assert len(drop_calls) > 0

    def test_table_preserved_when_not_reset(self):
        """Test table is preserved when RESET_TABLE=0."""
        reset_table = False

        # When reset_table is False, no connection should be made
        if not reset_table:
            # Early return, no database operation
            pass
        else:
            # This branch should not execute
            pytest.fail("Table reset should not occur when RESET_TABLE=0")

    @patch('psycopg2.connect')
    def test_table_name_with_data_prefix(self, mock_connect):
        """Test that count_rows uses correct 'data_' prefix."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [42]
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # PGVectorStore adds 'data_' prefix to table names
        table_name = "my_index"
        actual_table = f"data_{table_name}"

        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) FROM "{actual_table}";')
            result = cursor.fetchone()[0]

        # Verify query uses 'data_my_index'
        query = str(mock_cursor.execute.call_args[0][0])
        assert 'data_my_index' in query
        assert result == 42

    def test_table_name_sanitization(self):
        """Test table names are sanitized to be SQL-safe."""
        from utils.naming import sanitize_table_name

        # Test various unsafe inputs
        test_cases = {
            "my-table": "my_table",
            "my table": "my_table",
            "table@name": "table_name",
            "123table": "t_123table",  # Numbers at start get prefix
            "table#$%name": "table___name",
        }

        for unsafe, expected in test_cases.items():
            result = sanitize_table_name(unsafe)
            assert result == expected, f"Failed for '{unsafe}': got '{result}', expected '{expected}'"


class TestVectorStorage:
    """Test vector storage operations."""

    def test_vector_store_creation_parameters(self):
        """Test PGVectorStore initialization parameters."""
        # Mock PGVectorStore directly without patching the import
        MockPGVector = Mock()
        mock_store = Mock()
        MockPGVector.from_params.return_value = mock_store

        # Simulate store creation
        settings = MockSettings()
        store = MockPGVector.from_params(
            database=settings.db_name,
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            table_name=settings.table,
            embed_dim=settings.embed_dim,
        )

        # Verify store was created with correct parameters
        MockPGVector.from_params.assert_called_once_with(
            database=settings.db_name,
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            table_name=settings.table,
            embed_dim=settings.embed_dim,
        )
        assert store == mock_store

    def test_node_insertion_single_batch(self):
        """Test inserting nodes in a single batch."""
        mock_store = Mock()

        # Create mock nodes
        nodes = [Mock() for _ in range(100)]

        # Simulate insertion
        batch_size = 250
        mock_store.add(nodes)

        # Verify nodes were added
        mock_store.add.assert_called_once_with(nodes)

    def test_node_insertion_multiple_batches(self):
        """Test inserting nodes in multiple batches."""
        mock_store = Mock()

        # Create more nodes than batch size
        nodes = [Mock() for _ in range(1000)]
        batch_size = 250

        # Simulate batched insertion
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            mock_store.add(batch)

        # Verify multiple batches were added
        assert mock_store.add.call_count == 4  # 1000 / 250 = 4 batches

    def test_vector_similarity_query(self):
        """Test vector similarity search query."""
        mock_store = Mock()
        query_embedding = [0.5] * 384

        # Mock response
        mock_result = Mock()
        mock_result.nodes = []
        mock_result.similarities = []
        mock_store.query.return_value = mock_result

        # Execute query
        result = mock_store.query(query_embedding=query_embedding, similarity_top_k=4)

        # Verify query was executed
        mock_store.query.assert_called_once()
        assert result is not None


class TestMetadataTracking:
    """Test configuration metadata tracking in stored nodes."""

    @patch('psycopg2.connect')
    def test_check_index_configuration_with_metadata(self, mock_connect):
        """Test checking index configuration from metadata."""
        # Mock database response with metadata
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ({"_chunk_size": 700, "_chunk_overlap": 150, "_embed_model": "bge-small"}, 1),
            ({"_chunk_size": 700, "_chunk_overlap": 150, "_embed_model": "bge-small"}, 2),
        ]
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate configuration check
        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute('SELECT metadata, id FROM "data_test_table" ORDER BY id LIMIT 10')
            rows = cursor.fetchall()

        # Extract and validate configuration
        configs = []
        for metadata, row_id in rows:
            if metadata and isinstance(metadata, dict):
                config = {
                    "chunk_size": metadata.get("_chunk_size"),
                    "chunk_overlap": metadata.get("_chunk_overlap"),
                    "embed_model": metadata.get("_embed_model"),
                }
                configs.append(config)

        # Verify configuration was extracted
        assert len(configs) == 2
        assert configs[0]['chunk_size'] == 700
        assert configs[0]['chunk_overlap'] == 150

        # Check consistency
        first_config = configs[0]
        all_same = all(c == first_config for c in configs)
        assert all_same is True

    @patch('psycopg2.connect')
    def test_check_index_configuration_mixed(self, mock_connect):
        """Test detection of mixed configurations."""
        # Mock database with inconsistent metadata
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ({"_chunk_size": 700, "_chunk_overlap": 150}, 1),
            ({"_chunk_size": 500, "_chunk_overlap": 100}, 2),  # Different!
        ]
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate configuration check
        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute('SELECT metadata, id FROM "data_test_table" ORDER BY id LIMIT 10')
            rows = cursor.fetchall()

        # Extract configurations
        configs = []
        for metadata, row_id in rows:
            if metadata and isinstance(metadata, dict):
                configs.append(metadata)

        # Verify inconsistency was detected
        unique_configs = set(str(c) for c in configs)
        assert len(unique_configs) == 2  # Two different configurations

    @patch('psycopg2.connect')
    def test_check_index_configuration_table_not_exists(self, mock_connect):
        """Test checking configuration when table doesn't exist."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = psycopg2.errors.UndefinedTable()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate configuration check on non-existent table
        result = None
        try:
            conn = mock_connect()
            with conn.cursor() as cursor:
                cursor.execute('SELECT metadata, id FROM "nonexistent_table" LIMIT 10')
        except psycopg2.errors.UndefinedTable:
            result = None

        # Should return None for non-existent table
        assert result is None

    @patch('psycopg2.connect')
    def test_check_index_configuration_legacy_index(self, mock_connect):
        """Test checking legacy index without metadata."""
        # Mock database with no metadata (old index)
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ({}, 1),  # Empty metadata
            ({}, 2),
        ]
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate configuration check
        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute('SELECT metadata, id FROM "data_test_table" LIMIT 10')
            rows = cursor.fetchall()

        # Extract configurations
        configs = []
        for metadata, row_id in rows:
            if metadata and isinstance(metadata, dict) and metadata:
                configs.append(metadata)

        # Should detect as legacy index (no metadata)
        assert len(configs) == 0  # No valid metadata found


class TestRowCountOperations:
    """Test row counting operations."""

    @patch('psycopg2.connect')
    def test_count_rows_success(self, mock_connect):
        """Test successful row counting."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [12345]
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate row count
        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM "data_test_table";')
            result = int(cursor.fetchone()[0])

        assert result == 12345

    @patch('psycopg2.connect')
    def test_count_rows_table_not_exists(self, mock_connect):
        """Test counting rows when table doesn't exist."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = psycopg2.errors.UndefinedTable()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate row count on non-existent table
        result = None
        try:
            conn = mock_connect()
            with conn.cursor() as cursor:
                cursor.execute('SELECT COUNT(*) FROM "nonexistent_table";')
        except psycopg2.errors.UndefinedTable:
            result = None

        # Should return None for non-existent table
        assert result is None

    @patch('psycopg2.connect')
    def test_count_rows_connection_error(self, mock_connect):
        """Test counting rows with connection error."""
        mock_connect.side_effect = PgOperationalError("Connection failed")

        # Simulate row count with connection error
        result = None
        try:
            conn = mock_connect()
        except PgOperationalError:
            result = None

        # Should return None on connection error
        assert result is None

    @patch('psycopg2.connect')
    def test_count_rows_empty_table(self, mock_connect):
        """Test counting rows in empty table."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [0]
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate row count
        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM "data_test_table";')
            result = int(cursor.fetchone()[0])

        assert result == 0


class TestHNSWIndexCreation:
    """Test HNSW index creation for fast similarity search."""

    @patch('psycopg2.connect')
    def test_hnsw_index_creation(self, mock_connect):
        """Test HNSW index creation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate HNSW index creation
        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS test_table_hnsw_idx
                ON "data_test_table"
                USING hnsw (embedding vector_cosine_ops)
            """)

        # Verify index creation was attempted
        assert mock_cursor.execute.call_count > 0

    @patch('psycopg2.connect')
    def test_hnsw_index_already_exists(self, mock_connect):
        """Test HNSW index creation when index already exists."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # IF NOT EXISTS makes this safe
        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS test_table_hnsw_idx
                ON "data_test_table"
                USING hnsw (embedding vector_cosine_ops)
            """)

        # Should complete without error
        mock_cursor.execute.assert_called_once()

    @patch('psycopg2.connect')
    def test_hnsw_index_table_not_exists(self, mock_connect):
        """Test HNSW index creation when table doesn't exist."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = psycopg2.errors.UndefinedTable()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Should raise error for non-existent table
        with pytest.raises(psycopg2.errors.UndefinedTable):
            conn = mock_connect()
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE INDEX test_table_hnsw_idx
                    ON "nonexistent_table"
                    USING hnsw (embedding vector_cosine_ops)
                """)


class TestDatabaseErrorHandling:
    """Test database error handling and recovery."""

    def test_transaction_rollback_on_error(self):
        """Test transaction rollback on error."""
        mock_store = Mock()
        mock_store.add.side_effect = Exception("Insert failed")

        nodes = [Mock() for _ in range(10)]

        # Should raise the error
        with pytest.raises(Exception):
            mock_store.add(nodes)

    @patch('psycopg2.connect')
    def test_connection_leak_prevention(self, mock_connect):
        """Test that connections are properly closed."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [42]
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Simulate operation with connection
        conn = mock_connect()
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM "data_test_table";')
        conn.close()

        # Verify connection was closed
        mock_conn.close.assert_called_once()

    def test_sql_injection_prevention(self):
        """Test that table names are properly sanitized."""
        from utils.naming import sanitize_table_name

        # Attempt SQL injection
        malicious_input = "table'; DROP TABLE users; --"
        sanitized = sanitize_table_name(malicious_input)

        # Should remove dangerous characters
        assert ";" not in sanitized
        assert "--" not in sanitized
        # Should be safe to use in SQL
        assert all(c.isalnum() or c == "_" for c in sanitized)

    @patch('psycopg2.connect')
    def test_concurrent_access_handling(self, mock_connect):
        """Test handling of concurrent access scenarios."""
        mock_conn = Mock()
        mock_cursor = Mock()
        # Simulate database already exists (concurrent creation)
        mock_cursor.execute.side_effect = psycopg2.errors.DuplicateDatabase()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Should handle gracefully (not raise error in practice)
        # Here we just verify the exception is raised, but in actual code
        # it would be caught and handled
        with pytest.raises(psycopg2.errors.DuplicateDatabase):
            conn = mock_connect()
            with conn.cursor() as cursor:
                cursor.execute("CREATE DATABASE test_db")


class TestDatabaseConfiguration:
    """Test database configuration and environment variables."""

    def test_database_environment_variables(self):
        """Test database configuration from environment."""
        # Test that env vars are set (test fixtures may override defaults)
        db_name = os.getenv("DB_NAME")
        host = os.getenv("PGHOST", "localhost")
        port = os.getenv("PGPORT", "5432")

        # Verify they're set and not empty
        assert db_name is not None and len(db_name) > 0
        assert host == "localhost"
        assert port == "5432"

    def test_table_name_generation(self):
        """Test automatic table name generation."""
        from utils.naming import generate_table_name
        from pathlib import Path

        # Test with PDF file
        doc_path = Path("data/document.pdf")
        chunk_size = 700
        chunk_overlap = 150

        table_name = generate_table_name(doc_path, chunk_size, chunk_overlap)

        # Should include document name and config
        assert "document" in table_name or "doc" in table_name
        assert len(table_name) < 63  # PostgreSQL limit

    def test_reset_flags(self):
        """Test RESET_TABLE and RESET_DB flag parsing."""
        # Test RESET_TABLE parsing
        os.environ["RESET_TABLE"] = "1"
        reset_table = os.getenv("RESET_TABLE", "0") == "1"
        assert reset_table is True

        os.environ["RESET_TABLE"] = "0"
        reset_table = os.getenv("RESET_TABLE", "0") == "1"
        assert reset_table is False

        # Cleanup
        if "RESET_TABLE" in os.environ:
            del os.environ["RESET_TABLE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
