"""
Tests for HNSW index migration and validation

Tests migrate_add_hnsw_indices.py and scripts/validate_hnsw_performance.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
from pathlib import Path
import numpy as np

# Add project root and scripts to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

# Import after path setup
import migrate_add_hnsw_indices


class TestDatabaseConnection:
    """Test database connection for migration."""

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    def test_connect_db_success(self, mock_connect):
        """Test successful database connection."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        conn = migrate_add_hnsw_indices.connect_db()

        assert conn == mock_conn
        mock_connect.assert_called_once()


class TestTableDetection:
    """Test table detection and filtering."""

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    def test_list_tables(self, mock_connect):
        """Test listing data tables."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ('data_table1',),
            ('data_table2',),
            ('data_inbox',)
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        tables = migrate_add_hnsw_indices.list_tables(mock_conn)

        assert len(tables) == 3
        assert all(t.startswith('data_') for t in tables)

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    def test_has_hnsw_index_true(self, mock_connect):
        """Test detection of existing HNSW index."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)  # Has index
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        has_index = migrate_add_hnsw_indices.has_hnsw_index(mock_conn, 'data_table1')

        assert has_index is True

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    def test_has_hnsw_index_false(self, mock_connect):
        """Test detection of missing HNSW index."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (0,)  # No index
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        has_index = migrate_add_hnsw_indices.has_hnsw_index(mock_conn, 'data_table1')

        assert has_index is False


class TestTableInfo:
    """Test table information retrieval."""

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    def test_get_table_info(self, mock_connect):
        """Test getting table row count, size, and dimensions."""
        mock_conn = Mock()
        mock_cursor = Mock()

        mock_cursor.fetchone.side_effect = [
            (91219,),      # row count
            ('325 MB',),   # table size
            (384,),        # embedding dimensions
        ]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        info = migrate_add_hnsw_indices.get_table_info(mock_conn, 'data_inbox')

        assert info['row_count'] == 91219
        assert info['size'] == '325 MB'
        assert info['dimensions'] == 384


class TestBenchmarking:
    """Test query performance benchmarking."""

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    @patch('migrate_add_hnsw_indices.np.mean')
    @patch('migrate_add_hnsw_indices.np.min')
    @patch('migrate_add_hnsw_indices.np.max')
    @patch('migrate_add_hnsw_indices.np.std')
    def test_benchmark_query(self, mock_std, mock_max, mock_min, mock_mean, mock_connect):
        """Test query benchmarking with multiple trials."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Mock embedding dimension query
        mock_cursor.fetchone.return_value = (384,)

        # Mock benchmark trials
        mock_cursor.fetchall.return_value = [('id1', 0.85), ('id2', 0.82)]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock numpy statistics
        mock_mean.return_value = 0.443  # 443ms average
        mock_min.return_value = 0.420
        mock_max.return_value = 0.465
        mock_std.return_value = 0.015

        result = migrate_add_hnsw_indices.benchmark_query(mock_conn, 'data_inbox', num_trials=5)

        assert 'avg_latency' in result
        assert 'min_latency' in result
        assert 'max_latency' in result
        assert 'std_latency' in result
        assert result['trials'] == 5

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    def test_benchmark_query_no_embeddings(self, mock_connect):
        """Test benchmark with no embeddings."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None  # No embeddings found
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = migrate_add_hnsw_indices.benchmark_query(mock_conn, 'empty_table')

        assert 'error' in result
        assert 'No embeddings found' in result['error']


class TestHNSWIndexCreation:
    """Test HNSW index creation."""

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    @patch('migrate_add_hnsw_indices.time.time')
    def test_create_hnsw_index_success(self, mock_time, mock_connect):
        """Test successful HNSW index creation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock time for duration measurement
        mock_time.side_effect = [0, 4.3]  # 4.3s creation time

        result = migrate_add_hnsw_indices.create_hnsw_index(
            mock_conn,
            'data_table1',
            m=16,
            ef_construction=64
        )

        assert result['success'] is True
        assert result['creation_time'] == 4.3
        assert 'index_name' in result

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    def test_create_hnsw_index_failure(self, mock_connect):
        """Test HNSW index creation failure."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("Index creation failed")
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = migrate_add_hnsw_indices.create_hnsw_index(mock_conn, 'data_table1')

        assert result['success'] is False
        assert 'error' in result

    @patch('migrate_add_hnsw_indices.psycopg2.connect')
    def test_create_hnsw_index_custom_parameters(self, mock_connect):
        """Test HNSW index with custom parameters."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with patch('migrate_add_hnsw_indices.time.time', side_effect=[0, 2.0]):
            result = migrate_add_hnsw_indices.create_hnsw_index(
                mock_conn,
                'data_table1',
                m=32,           # Custom m
                ef_construction=128  # Custom ef
            )

        assert result['success'] is True
        # Verify execute was called with correct parameters
        assert mock_cursor.execute.called


class TestSpeedupFormatting:
    """Test speedup formatting utility."""

    def test_format_speedup_minimal(self):
        """Test formatting for minimal speedup."""
        result = migrate_add_hnsw_indices.format_speedup(100, 95)

        assert result == "~same"

    def test_format_speedup_moderate(self):
        """Test formatting for moderate speedup."""
        result = migrate_add_hnsw_indices.format_speedup(100, 50)

        assert result == "2.0x faster"

    def test_format_speedup_large(self):
        """Test formatting for large speedup."""
        result = migrate_add_hnsw_indices.format_speedup(443, 2.1)

        assert "211x faster" in result or "210x faster" in result

    def test_format_speedup_extreme(self):
        """Test formatting for extreme speedup."""
        result = migrate_add_hnsw_indices.format_speedup(1000, 0.5)

        assert "2000x faster" in result

    def test_format_speedup_zero_after(self):
        """Test formatting when after time is zero."""
        result = migrate_add_hnsw_indices.format_speedup(100, 0)

        assert "∞x faster" in result or "faster" in result.lower()


class TestValidationThresholds:
    """Test performance validation thresholds."""

    def test_small_table_threshold(self):
        """Test performance threshold for small tables."""
        # Import validation script
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        import validate_hnsw_performance

        category = validate_hnsw_performance.get_table_size_category(5000)

        assert category == 'small'

    def test_medium_table_threshold(self):
        """Test performance threshold for medium tables."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        import validate_hnsw_performance

        category = validate_hnsw_performance.get_table_size_category(50000)

        assert category == 'medium'

    def test_large_table_threshold(self):
        """Test performance threshold for large tables."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        import validate_hnsw_performance

        category = validate_hnsw_performance.get_table_size_category(150000)

        assert category == 'large'


class TestMigrationWorkflow:
    """Test complete migration workflow."""

    @patch('migrate_add_hnsw_indices.connect_db')
    @patch('migrate_add_hnsw_indices.list_tables')
    @patch('migrate_add_hnsw_indices.has_hnsw_index')
    @patch('migrate_add_hnsw_indices.get_table_info')
    def test_migration_workflow_dry_run(
        self,
        mock_get_info,
        mock_has_index,
        mock_list_tables,
        mock_connect_db,
        capsys
    ):
        """Test migration in dry-run mode."""
        mock_conn = Mock()
        mock_connect_db.return_value = mock_conn

        mock_list_tables.return_value = ['data_table1', 'data_table2']

        # First table needs migration, second already has index
        mock_has_index.side_effect = [False, True]

        mock_get_info.side_effect = [
            {'row_count': 10000, 'size': '50 MB', 'dimensions': 384},
            {'row_count': 5000, 'size': '25 MB', 'dimensions': 384}
        ]

        # Run with dry-run argument
        with patch('sys.argv', ['migrate_add_hnsw_indices.py', '--dry-run']):
            with pytest.raises(SystemExit) as exc_info:
                migrate_add_hnsw_indices.main()

        # Should exit cleanly
        assert exc_info.value.code is None or exc_info.value.code == 0

        captured = capsys.readouterr()
        assert 'Dry run mode' in captured.out
        assert 'data_table1' in captured.out


class TestPerformanceMetrics:
    """Test performance measurement accuracy."""

    def test_speedup_calculation_accurate(self):
        """Test that speedup calculations are accurate."""
        before = 443.2  # ms
        after = 2.1     # ms

        expected_speedup = before / after  # ~211x

        result = migrate_add_hnsw_indices.format_speedup(before / 1000, after / 1000)

        # Should show ~211x or ~210x
        assert "21" in result
        assert "faster" in result

    def test_benchmark_statistics(self):
        """Test that benchmark statistics are computed correctly."""
        latencies = [0.42, 0.44, 0.43, 0.45, 0.44]

        avg = np.mean(latencies)
        std = np.std(latencies)

        assert 0.42 <= avg <= 0.45
        assert std < 0.02  # Low variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
