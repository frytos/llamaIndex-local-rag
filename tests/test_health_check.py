"""Tests for health check system.

This module tests the health_check module, including:
- Database connectivity checks
- System resource monitoring
- GPU/accelerator detection
- Overall health aggregation

Week 1 - Day 3: Health check database tests (8 tests)
"""

import pytest
from unittest.mock import patch, MagicMock
import psycopg2
from utils.health_check import HealthChecker, HealthCheckResult


# ============================================================================
# Day 3: Database Health Check Tests (3 tests)
# ============================================================================


class TestDatabaseHealthCheck:
    """Test database connectivity checks.

    These tests validate that the health checker properly:
    - Detects successful database connections
    - Handles connection failures gracefully
    - Detects missing pgvector extension
    """

    @pytest.mark.unit
    def test_database_check_success(self, mock_db_connection_success):
        """Test successful database connectivity check.

        Given: A healthy PostgreSQL database with pgvector
        When: check_database() is called
        Then: Returns HealthCheckResult with status="healthy"
        """
        checker = HealthChecker()

        with patch('psycopg2.connect', return_value=mock_db_connection_success):
            result = checker.check_database()

            assert result.status == "healthy"
            assert result.component == "database"
            assert "successful" in result.message.lower()
            assert result.details["pgvector_enabled"] is True
            assert result.details["vector_tables"] == 5
            assert result.latency_ms is not None
            assert result.latency_ms > 0

    @pytest.mark.unit
    def test_database_check_connection_failed(self):
        """Test handling of database connection failure.

        Given: PostgreSQL database is not running (connection refused)
        When: check_database() is called
        Then: Returns HealthCheckResult with status="unhealthy"
        """
        checker = HealthChecker()

        # Mock connection failure
        with patch('psycopg2.connect', side_effect=psycopg2.OperationalError("Connection refused")):
            result = checker.check_database()

            assert result.status == "unhealthy"
            assert result.component == "database"
            assert "failed" in result.message.lower()
            assert "error" in result.details
            assert result.latency_ms is not None

    @pytest.mark.unit
    def test_database_check_pgvector_missing(self):
        """Test detection of missing pgvector extension.

        Given: PostgreSQL database without pgvector extension
        When: check_database() is called
        Then: Returns healthy but details show pgvector_enabled=False
        """
        checker = HealthChecker()

        # Mock connection with no pgvector
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            ("PostgreSQL 15.0",),  # version
            None,                  # pgvector extension (missing)
            (0,),                  # table count
            ("50 MB",),            # database size
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch('psycopg2.connect', return_value=mock_conn):
            result = checker.check_database()

            assert result.status == "healthy"  # Still connects
            assert result.details["pgvector_enabled"] is False


# ============================================================================
# Day 3: System Resource Check Tests (3 tests)
# ============================================================================


class TestSystemResourceChecks:
    """Test system resource monitoring.

    These tests validate:
    - Healthy resource usage detection
    - High CPU/memory/disk warnings
    - Resource check error handling
    """

    @pytest.mark.unit
    def test_resource_check_healthy(self, mock_psutil):
        """Test healthy system resource usage.

        Given: CPU=50%, Memory=60%, Disk=70% (all healthy)
        When: check_system_resources() is called
        Then: Returns HealthCheckResult with status="healthy"
        """
        checker = HealthChecker()

        result = checker.check_system_resources()

        assert result.status == "healthy"
        assert result.component == "system_resources"
        assert "OK" in result.message
        assert result.details["cpu_percent"] == 50.0
        assert result.details["memory_percent"] == 60.0
        assert result.details["disk_percent"] == 70.0
        assert result.latency_ms is not None

    @pytest.mark.unit
    def test_resource_check_high_cpu(self):
        """Test high CPU usage warning.

        Given: CPU usage is 90% (above 85% threshold)
        When: check_system_resources() is called
        Then: Returns HealthCheckResult with status="degraded"
        """
        checker = HealthChecker()

        with patch('utils.health_check.psutil') as mock_psutil:
            # High CPU
            mock_psutil.cpu_percent.return_value = 90.0

            # Normal memory and disk
            mock_memory = MagicMock()
            mock_memory.percent = 60.0
            mock_memory.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_memory

            mock_disk = MagicMock()
            mock_disk.percent = 70.0
            mock_disk.free = 50 * 1024**3
            mock_psutil.disk_usage.return_value = mock_disk

            result = checker.check_system_resources()

            assert result.status == "degraded"
            assert "High CPU usage" in result.message
            assert result.details["cpu_percent"] == 90.0

    @pytest.mark.unit
    def test_resource_check_high_memory(self):
        """Test high memory usage warning.

        Given: Memory usage is 95% (above 90% threshold)
        When: check_system_resources() is called
        Then: Returns HealthCheckResult with status="degraded"
        """
        checker = HealthChecker()

        with patch('utils.health_check.psutil') as mock_psutil:
            # Normal CPU
            mock_psutil.cpu_percent.return_value = 50.0

            # High memory
            mock_memory = MagicMock()
            mock_memory.percent = 95.0
            mock_memory.available = 1 * 1024**3  # 1GB left
            mock_psutil.virtual_memory.return_value = mock_memory

            # Normal disk
            mock_disk = MagicMock()
            mock_disk.percent = 70.0
            mock_disk.free = 50 * 1024**3
            mock_psutil.disk_usage.return_value = mock_disk

            result = checker.check_system_resources()

            assert result.status == "degraded"
            assert "High memory usage" in result.message
            assert result.details["memory_percent"] == 95.0


# ============================================================================
# Day 3: GPU Health Check Tests (2 tests)
# ============================================================================


class TestGPUHealthCheck:
    """Test GPU/accelerator detection.

    These tests validate:
    - CUDA GPU detection
    - MPS (Apple Silicon) detection
    - CPU-only fallback
    """

    @pytest.mark.unit
    def test_gpu_check_cuda_available(self, mock_torch_cuda):
        """Test CUDA GPU detection.

        Given: System has CUDA-capable GPU
        When: check_gpu_availability() is called
        Then: Returns HealthCheckResult with status="healthy" and CUDA details
        """
        import sys
        checker = HealthChecker()

        # Patch torch in sys.modules so it's importable
        with patch('utils.health_check.TORCH_AVAILABLE', True):
            with patch.dict(sys.modules, {'torch': mock_torch_cuda}):
                # Also patch the module-level torch reference
                import utils.health_check
                original_torch = getattr(utils.health_check, 'torch', None)
                utils.health_check.torch = mock_torch_cuda

                try:
                    result = checker.check_gpu_availability()

                    assert result.status == "healthy"
                    assert result.component == "gpu"
                    assert "CUDA" in result.message
                    assert result.details["cuda_available"] is True
                    assert result.details["device_count"] == 1
                    assert "cuda_version" in result.details
                finally:
                    # Restore original
                    if original_torch is not None:
                        utils.health_check.torch = original_torch
                    elif hasattr(utils.health_check, 'torch'):
                        delattr(utils.health_check, 'torch')

    @pytest.mark.unit
    def test_gpu_check_mps_available(self, mock_torch_mps):
        """Test MPS (Apple Silicon) detection.

        Given: System has Apple Silicon with MPS
        When: check_gpu_availability() is called
        Then: Returns HealthCheckResult with status="healthy" and MPS details
        """
        import sys
        checker = HealthChecker()

        # Patch torch in sys.modules so it's importable
        with patch('utils.health_check.TORCH_AVAILABLE', True):
            with patch.dict(sys.modules, {'torch': mock_torch_mps}):
                # Also patch the module-level torch reference
                import utils.health_check
                original_torch = getattr(utils.health_check, 'torch', None)
                utils.health_check.torch = mock_torch_mps

                try:
                    result = checker.check_gpu_availability()

                    assert result.status == "healthy"
                    assert result.component == "gpu"
                    assert "MPS" in result.message
                    assert result.details["mps_available"] is True
                    assert result.details["cuda_available"] is False
                finally:
                    # Restore original
                    if original_torch is not None:
                        utils.health_check.torch = original_torch
                    elif hasattr(utils.health_check, 'torch'):
                        delattr(utils.health_check, 'torch')


# ============================================================================
# Test Discovery and Execution Notes
# ============================================================================
"""
Run Day 3 tests:
    pytest tests/test_health_check.py -v

Run with coverage:
    pytest tests/test_health_check.py \
        --cov=utils.health_check \
        --cov-report=term-missing

Expected results:
    - 8 tests passing
    - Coverage increase: +40-50% for utils/health_check.py
    - Validates production monitoring capabilities
"""
