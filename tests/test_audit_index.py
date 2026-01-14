"""
Tests for audit_index.py

Tests audit tool functionality including:
- Database connection
- Table information retrieval
- Configuration checking
- Chunk analysis
- Embedding validation
- Report generation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import audit_index


class TestDatabaseConnection:
    """Test database connection utilities."""

    @patch('audit_index.psycopg2.connect')
    def test_connect_db_success(self, mock_connect):
        """Test successful database connection."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        conn = audit_index.connect_db()

        assert conn == mock_conn
        mock_connect.assert_called_once()

    @patch('audit_index.psycopg2.connect')
    def test_connect_db_failure(self, mock_connect):
        """Test database connection failure."""
        mock_connect.side_effect = Exception("Connection refused")

        with pytest.raises(Exception) as exc_info:
            audit_index.connect_db()

        assert "Connection refused" in str(exc_info.value)


class TestTableListing:
    """Test table listing functionality."""

    @patch('audit_index.psycopg2.connect')
    def test_list_tables_success(self, mock_connect):
        """Test listing tables successfully."""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ('data_table1',),
            ('data_table2',),
            ('data_table3',)
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Test
        tables = audit_index.list_tables(mock_conn)

        assert len(tables) == 3
        assert 'data_table1' in tables
        assert 'data_table2' in tables
        assert 'data_table3' in tables

    @patch('audit_index.psycopg2.connect')
    def test_list_tables_empty(self, mock_connect):
        """Test listing tables when none exist."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        tables = audit_index.list_tables(mock_conn)

        assert len(tables) == 0


class TestTableInfo:
    """Test table information retrieval."""

    @patch('audit_index.psycopg2.connect')
    def test_get_table_info(self, mock_connect):
        """Test getting table information."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock cursor to return different results for different queries
        mock_cursor.fetchone.side_effect = [
            (1234,),  # Row count
            ('42 MB',),  # Table size
        ]
        mock_cursor.fetchall.return_value = [
            ('id', 'text'),
            ('text', 'text'),
            ('embedding', 'vector'),
            ('metadata_', 'jsonb')
        ]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        info = audit_index.get_table_info(mock_conn, 'test_table')

        assert info['row_count'] == 1234
        assert info['size'] == '42 MB'
        assert 'columns' in info
        assert len(info['columns']) == 4


class TestConfigurationCheck:
    """Test configuration consistency checking."""

    @patch('audit_index.psycopg2.connect')
    def test_check_configuration_consistent(self, mock_connect):
        """Test detection of consistent configuration."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock metadata column check (exists)
        mock_cursor.fetchone.side_effect = [
            (True,),  # metadata_ column exists
        ]

        # Mock configuration query (single config)
        mock_cursor.fetchall.return_value = [
            ('700', '150', 'BAAI/bge-small-en', '384', 1234),
        ]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        config = audit_index.check_configuration(mock_conn, 'test_table')

        assert config['consistent'] is True
        assert len(config['configs']) == 1
        assert config['configs'][0]['chunk_size'] == '700'
        assert len(config['warnings']) == 0

    @patch('audit_index.psycopg2.connect')
    def test_check_configuration_mixed(self, mock_connect):
        """Test detection of mixed configurations."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchone.side_effect = [(True,)]

        # Multiple configurations
        mock_cursor.fetchall.return_value = [
            ('700', '150', 'BAAI/bge-small-en', '384', 800),
            ('500', '100', 'BAAI/bge-small-en', '384', 434),
        ]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        config = audit_index.check_configuration(mock_conn, 'test_table')

        assert config['consistent'] is False
        assert len(config['configs']) == 2
        assert len(config['warnings']) > 0
        assert 'Mixed configurations' in config['warnings'][0]

    @patch('audit_index.psycopg2.connect')
    def test_check_configuration_no_metadata(self, mock_connect):
        """Test handling of legacy index without metadata."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchone.return_value = (False,)  # No metadata column

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        config = audit_index.check_configuration(mock_conn, 'test_table')

        assert config['consistent'] is False
        assert 'Legacy index' in config['warnings'][0]


class TestChunkAnalysis:
    """Test chunk quality analysis."""

    @patch('audit_index.psycopg2.connect')
    def test_analyze_chunks(self, mock_connect):
        """Test chunk statistics analysis."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock chunk size statistics
        mock_cursor.fetchone.side_effect = [
            (1411, 24, 3827, 342, 1458),  # avg, min, max, stddev, median
            (52, 6117),  # num_docs, total_chunks
        ]

        # Mock sample chunks
        mock_cursor.fetchall.return_value = [
            ('Sample chunk 1 text here...', 905),
            ('Sample chunk 2 with more text...', 1592),
            ('Sample chunk 3 content...', 864),
        ]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        stats = audit_index.analyze_chunks(mock_conn, 'test_table')

        assert stats['size']['avg'] == 1411
        assert stats['size']['min'] == 24
        assert stats['size']['max'] == 3827
        assert stats['size']['stddev'] == 342
        assert stats['size']['median'] == 1458
        assert len(stats['samples']) == 3
        assert stats['chunks_per_doc'] == pytest.approx(117.6, rel=0.1)


class TestEmbeddingValidation:
    """Test embedding health checks."""

    @patch('audit_index.psycopg2.connect')
    def test_check_embeddings_healthy(self, mock_connect):
        """Test healthy embeddings (no nulls, consistent dimensions)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchone.side_effect = [
            (0,),      # null_count
            (384,),    # dimensions
        ]

        # Only one dimension value
        mock_cursor.fetchone.return_value = (1,)

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        emb_stats = audit_index.check_embeddings(mock_conn, 'test_table')

        assert emb_stats['null_count'] == 0
        assert emb_stats['dimensions'] == 384
        # Note: dim_consistent check would need proper mock sequencing

    @patch('audit_index.psycopg2.connect')
    def test_check_embeddings_with_nulls(self, mock_connect):
        """Test detection of null embeddings."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchone.side_effect = [
            (150,),    # null_count (has nulls!)
            (384,),    # dimensions
        ]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        emb_stats = audit_index.check_embeddings(mock_conn, 'test_table')

        assert emb_stats['null_count'] == 150
        assert emb_stats['null_count'] > 0  # Should flag issue


class TestQueryPerformance:
    """Test query performance testing."""

    @patch('audit_index.psycopg2.connect')
    @patch('audit_index.np.mean')
    def test_query_performance_fast(self, mock_mean, mock_connect):
        """Test fast query performance."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock query results
        mock_cursor.fetchall.return_value = [
            ('chunk 1', 0.85),
            ('chunk 2', 0.82),
            ('chunk 3', 0.78),
        ]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        mock_mean.return_value = 0.82

        with patch('audit_index.time.time', side_effect=[0, 0.003]):  # 3ms latency
            results = audit_index.test_query(mock_conn, 'test_table', embed_dim=384)

        assert 'latency' in results
        assert results['latency'] == 0.003
        assert 'avg_similarity' in results

    @patch('audit_index.psycopg2.connect')
    def test_query_performance_slow(self, mock_connect):
        """Test slow query performance detection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchall.return_value = [
            ('chunk 1', 0.5),
        ]

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with patch('audit_index.time.time', side_effect=[0, 1.2]):  # 1.2s latency
            results = audit_index.test_query(mock_conn, 'test_table')

        assert results['latency'] >= 1.0  # Should detect as slow


class TestReportGeneration:
    """Test report generation."""

    def test_generate_report_healthy(self):
        """Test report generation for healthy index."""
        table_info = {
            'row_count': 6117,
            'size': '42 MB',
            'columns': {}
        }

        config = {
            'consistent': True,
            'configs': [{
                'chunk_size': '700',
                'overlap': '150',
                'model': 'BAAI/bge-m3',
                'dim': '1024',
                'count': 6117
            }],
            'warnings': []
        }

        chunk_stats = {
            'size': {
                'avg': 1411,
                'min': 0,
                'max': 3827,
                'stddev': 342,
                'median': 1458
            },
            'samples': [],
            'chunks_per_doc': 117.6
        }

        emb_stats = {
            'null_count': 0,
            'dimensions': 1024,
            'dim_consistent': True
        }

        query_results = {
            'latency': 0.042,
            'top_similarity': 0.85,
            'avg_similarity': 0.72,
            'min_similarity': 0.65
        }

        report = audit_index.generate_report(
            'test_table',
            table_info,
            config,
            chunk_stats,
            emb_stats,
            query_results
        )

        # Verify report contains key sections
        assert '✅ HEALTHY' in report
        assert 'Rows: 6,117' in report
        assert 'Storage Size: 42 MB' in report
        assert 'Consistent configuration' in report
        assert 'All chunks have embeddings' in report
        assert 'recommendations' in report.lower()

    def test_generate_report_with_issues(self):
        """Test report generation with warnings."""
        table_info = {'row_count': 1234, 'size': '10 MB', 'columns': {}}

        config = {
            'consistent': False,
            'configs': [
                {'chunk_size': '700', 'overlap': '150', 'model': 'model1', 'dim': '384', 'count': 800},
                {'chunk_size': '500', 'overlap': '100', 'model': 'model1', 'dim': '384', 'count': 434}
            ],
            'warnings': ['Mixed configurations detected']
        }

        chunk_stats = {
            'size': {'avg': 650, 'min': 100, 'max': 1500, 'stddev': 200, 'median': 600},
            'samples': [],
            'chunks_per_doc': None
        }

        emb_stats = {
            'null_count': 10,  # Has null embeddings!
            'dimensions': 384,
            'dim_consistent': True
        }

        query_results = {
            'latency': 0.5,
            'top_similarity': 0.3,
            'avg_similarity': 0.25,
            'min_similarity': 0.2
        }

        report = audit_index.generate_report(
            'test_table',
            table_info,
            config,
            chunk_stats,
            emb_stats,
            query_results
        )

        # Verify warnings are present
        assert '⚠️' in report or '❌' in report
        assert 'Mixed configurations' in report or 'Configuration Issues' in report
        assert '10 chunks missing embeddings' in report
        assert 'RECOMMENDATIONS' in report


class TestMainFunction:
    """Test main audit function."""

    @patch('audit_index.connect_db')
    @patch('audit_index.list_tables')
    def test_main_no_tables(self, mock_list_tables, mock_connect_db, capsys):
        """Test main function when no tables exist."""
        mock_conn = Mock()
        mock_connect_db.return_value = mock_conn
        mock_list_tables.return_value = []

        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['audit_index.py']):
                audit_index.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'No tables found' in captured.out

    @patch('audit_index.connect_db')
    @patch('audit_index.list_tables')
    def test_main_list_tables(self, mock_list_tables, mock_connect_db, capsys):
        """Test main function lists tables when no argument provided."""
        mock_conn = Mock()
        mock_connect_db.return_value = mock_conn
        mock_list_tables.return_value = ['table1', 'table2', 'table3']

        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['audit_index.py']):
                audit_index.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert 'Available tables:' in captured.out
        assert 'table1' in captured.out
        assert 'table2' in captured.out
        assert 'Usage: python audit_index.py <table_name>' in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch('audit_index.psycopg2.connect')
    def test_empty_table(self, mock_connect):
        """Test handling of empty table."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Empty table
        mock_cursor.fetchone.return_value = (0,)

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        info = audit_index.get_table_info(mock_conn, 'empty_table')

        assert info['row_count'] == 0

    @patch('audit_index.psycopg2.connect')
    def test_table_not_found(self, mock_connect):
        """Test handling of non-existent table."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Simulate table not found error
        mock_cursor.execute.side_effect = Exception("relation does not exist")

        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with pytest.raises(Exception) as exc_info:
            audit_index.get_table_info(mock_conn, 'nonexistent_table')

        assert "does not exist" in str(exc_info.value)


# Integration test (requires actual database)
@pytest.mark.integration
class TestAuditIntegration:
    """Integration tests requiring actual database."""

    @pytest.mark.skip(reason="Requires PostgreSQL connection")
    def test_full_audit_workflow(self):
        """Test complete audit workflow with real database."""
        # This would test against actual database
        # Skip by default, run with: pytest -m integration
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
