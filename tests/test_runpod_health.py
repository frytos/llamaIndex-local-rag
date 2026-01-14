"""
Tests for RunPod Health Check Utilities (utils/runpod_health.py)

Tests health checking for SSH, vLLM, PostgreSQL, and GPU services.
Uses mocking to avoid actual service connections.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runpod_health import (
    check_ssh_connectivity,
    check_port_open,
    check_vllm_health,
    check_postgres_health,
    check_gpu_available,
    wait_for_service,
    wait_for_services,
    comprehensive_health_check
)


class TestSSHConnectivity:
    """Test SSH connectivity checking."""

    @patch('utils.runpod_health.subprocess.run')
    def test_check_ssh_success(self, mock_run):
        """Test successful SSH connection."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = check_ssh_connectivity("host123")

        assert result is True
        mock_run.assert_called_once()

    @patch('utils.runpod_health.subprocess.run')
    def test_check_ssh_failure(self, mock_run):
        """Test failed SSH connection."""
        mock_result = Mock()
        mock_result.returncode = 255
        mock_run.return_value = mock_result

        result = check_ssh_connectivity("host123")

        assert result is False

    @patch('utils.runpod_health.subprocess.run')
    def test_check_ssh_timeout(self, mock_run):
        """Test SSH connection timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('ssh', 10)

        result = check_ssh_connectivity("host123", timeout=10)

        assert result is False


class TestPortChecking:
    """Test port availability checking."""

    @patch('utils.runpod_health.socket.socket')
    def test_check_port_open_success(self, mock_socket_class):
        """Test successful port connection."""
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 0  # Success
        mock_socket_class.return_value = mock_sock

        result = check_port_open("localhost", 8000)

        assert result is True
        mock_sock.connect_ex.assert_called_once_with(("localhost", 8000))
        mock_sock.close.assert_called_once()

    @patch('utils.runpod_health.socket.socket')
    def test_check_port_open_closed(self, mock_socket_class):
        """Test port is closed."""
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1  # Connection refused
        mock_socket_class.return_value = mock_sock

        result = check_port_open("localhost", 8000)

        assert result is False

    @patch('utils.runpod_health.socket.socket')
    def test_check_port_exception(self, mock_socket_class):
        """Test handling of socket exception."""
        mock_sock = Mock()
        mock_sock.connect_ex.side_effect = Exception("Network error")
        mock_socket_class.return_value = mock_sock

        result = check_port_open("localhost", 8000)

        assert result is False


class TestVLLMHealth:
    """Test vLLM server health checking."""

    @patch('utils.runpod_health.requests.get')
    @patch('utils.runpod_health.time.time')
    def test_check_vllm_healthy(self, mock_time, mock_get):
        """Test vLLM server is healthy."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.elapsed.total_seconds.return_value = 0.025

        mock_get.return_value = mock_response
        mock_time.side_effect = [0, 0.025]

        result = check_vllm_health(host="localhost", port=8000)

        assert result['status'] == 'healthy'
        assert result['latency_ms'] == 25.0
        assert 'response' in result

    @patch('utils.runpod_health.requests.get')
    def test_check_vllm_unhealthy(self, mock_get):
        """Test vLLM server returns error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_get.return_value = mock_response

        result = check_vllm_health()

        assert result['status'] == 'unhealthy'
        assert 'error' in result

    @patch('utils.runpod_health.requests.get')
    def test_check_vllm_unreachable(self, mock_get):
        """Test vLLM server is unreachable."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = check_vllm_health()

        assert result['status'] == 'unreachable'
        assert 'Connection refused' in result['error']

    @patch('utils.runpod_health.requests.get')
    def test_check_vllm_timeout(self, mock_get):
        """Test vLLM server timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        result = check_vllm_health(timeout=5)

        assert result['status'] == 'timeout'
        assert 'timeout' in result['error'].lower()


class TestPostgreSQLHealth:
    """Test PostgreSQL health checking."""

    @patch('utils.runpod_health.psycopg2.connect')
    def test_check_postgres_healthy(self, mock_connect):
        """Test PostgreSQL is healthy."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [(1,), (True,)]  # SELECT 1, pgvector exists
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = check_postgres_health()

        assert result['status'] == 'healthy'
        assert result['pgvector'] is True

    @patch('utils.runpod_health.psycopg2.connect')
    def test_check_postgres_no_pgvector(self, mock_connect):
        """Test PostgreSQL without pgvector extension."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [(1,), (False,)]  # pgvector not installed
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = check_postgres_health()

        assert result['status'] == 'healthy'
        assert result['pgvector'] is False

    @patch('utils.runpod_health.psycopg2.connect')
    def test_check_postgres_connection_failed(self, mock_connect):
        """Test PostgreSQL connection failure."""
        mock_connect.side_effect = Exception("Connection refused")

        result = check_postgres_health()

        assert result['status'] == 'unhealthy'
        assert 'error' in result


class TestGPUAvailability:
    """Test GPU availability checking."""

    @patch('utils.runpod_health.subprocess.run')
    def test_check_gpu_local_available(self, mock_run):
        """Test GPU available locally."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA RTX 4090"
        mock_run.return_value = mock_result

        result = check_gpu_available()

        assert result['available'] is True
        assert 'RTX 4090' in result['name']
        assert result['via'] == 'local'

    @patch('utils.runpod_health.subprocess.run')
    def test_check_gpu_local_not_available(self, mock_run):
        """Test GPU not available locally."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = check_gpu_available()

        assert result['available'] is False

    @patch('utils.runpod_health.subprocess.run')
    def test_check_gpu_via_ssh(self, mock_run):
        """Test GPU check via SSH."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA RTX 4090, 24576 MiB, 8192 MiB"
        mock_run.return_value = mock_result

        result = check_gpu_available(ssh_host="host123")

        assert result['available'] is True
        assert 'RTX 4090' in result['name']
        assert result['via'] == 'ssh'


class TestServiceWaiting:
    """Test waiting for service readiness."""

    @patch('utils.runpod_health.time.time')
    @patch('utils.runpod_health.time.sleep')
    def test_wait_for_service_immediate(self, mock_sleep, mock_time):
        """Test service is immediately ready."""
        def mock_check():
            return {'status': 'healthy'}

        mock_time.side_effect = [0, 0]

        result = wait_for_service(
            mock_check,
            timeout=60,
            service_name="test_service"
        )

        assert result is True
        assert not mock_sleep.called  # No waiting needed

    @patch('utils.runpod_health.time.time')
    @patch('utils.runpod_health.time.sleep')
    def test_wait_for_service_eventually(self, mock_sleep, mock_time):
        """Test service becomes ready after waiting."""
        # First check fails, second succeeds
        check_count = [0]

        def mock_check():
            check_count[0] += 1
            if check_count[0] == 1:
                return {'status': 'starting'}
            else:
                return {'status': 'healthy'}

        mock_time.side_effect = [0, 10, 20]

        result = wait_for_service(
            mock_check,
            timeout=60,
            check_interval=10,
            service_name="test_service"
        )

        assert result is True
        assert mock_sleep.called

    @patch('utils.runpod_health.time.time')
    @patch('utils.runpod_health.time.sleep')
    def test_wait_for_service_timeout(self, mock_sleep, mock_time):
        """Test service never becomes ready (timeout)."""
        def mock_check():
            return {'status': 'starting'}  # Never becomes healthy

        # Simulate timeout
        mock_time.side_effect = [0, 10, 20, 30, 40, 50, 60, 70]

        result = wait_for_service(
            mock_check,
            timeout=60,
            check_interval=10,
            service_name="test_service"
        )

        assert result is False


class TestMultipleServices:
    """Test waiting for multiple services."""

    @patch('utils.runpod_health.time.time')
    @patch('utils.runpod_health.time.sleep')
    def test_wait_for_services_all_ready(self, mock_sleep, mock_time):
        """Test all services become ready."""
        def mock_vllm_check(**kwargs):
            return {'status': 'healthy'}

        def mock_pg_check(**kwargs):
            return {'status': 'healthy'}

        mock_time.side_effect = [0, 0, 0, 0]

        services = [
            {'name': 'vLLM', 'check_function': mock_vllm_check, 'host': 'localhost'},
            {'name': 'PostgreSQL', 'check_function': mock_pg_check, 'host': 'localhost'}
        ]

        results = wait_for_services(services, timeout=60)

        assert results['vLLM'] is True
        assert results['PostgreSQL'] is True

    @patch('utils.runpod_health.time.time')
    @patch('utils.runpod_health.time.sleep')
    def test_wait_for_services_partial(self, mock_sleep, mock_time):
        """Test some services ready, some timeout."""
        def mock_vllm_check(**kwargs):
            return {'status': 'healthy'}

        def mock_pg_check(**kwargs):
            return {'status': 'starting'}  # Never ready

        mock_time.side_effect = list(range(0, 200, 10))  # Simulate time passing

        services = [
            {'name': 'vLLM', 'check_function': mock_vllm_check},
            {'name': 'PostgreSQL', 'check_function': mock_pg_check}
        ]

        results = wait_for_services(services, timeout=30)

        assert results['vLLM'] is True
        assert results['PostgreSQL'] is False


class TestComprehensiveHealthCheck:
    """Test comprehensive health check."""

    @patch('utils.runpod_health.check_vllm_health')
    @patch('utils.runpod_health.check_postgres_health')
    @patch('utils.runpod_health.check_ssh_connectivity')
    @patch('utils.runpod_health.check_gpu_available')
    def test_comprehensive_check_all_healthy(
        self,
        mock_gpu,
        mock_ssh,
        mock_pg,
        mock_vllm
    ):
        """Test comprehensive check with all services healthy."""
        mock_vllm.return_value = {'status': 'healthy', 'latency_ms': 25}
        mock_pg.return_value = {'status': 'healthy', 'pgvector': True}
        mock_ssh.return_value = True
        mock_gpu.return_value = {'available': True, 'name': 'RTX 4090'}

        result = comprehensive_health_check(ssh_host="host123", local=True)

        assert result['overall_status'] == 'healthy'
        assert result['services']['vllm']['status'] == 'healthy'
        assert result['services']['postgres']['status'] == 'healthy'
        assert result['services']['ssh']['status'] == 'healthy'

    @patch('utils.runpod_health.check_vllm_health')
    @patch('utils.runpod_health.check_postgres_health')
    def test_comprehensive_check_partial(self, mock_pg, mock_vllm):
        """Test comprehensive check with some services down."""
        mock_vllm.return_value = {'status': 'healthy'}
        mock_pg.return_value = {'status': 'unhealthy', 'error': 'Connection refused'}

        result = comprehensive_health_check(local=True)

        assert result['overall_status'] == 'partial'
        assert result['services']['vllm']['status'] == 'healthy'
        assert result['services']['postgres']['status'] == 'unhealthy'

    @patch('utils.runpod_health.check_vllm_health')
    @patch('utils.runpod_health.check_postgres_health')
    def test_comprehensive_check_all_down(self, mock_pg, mock_vllm):
        """Test comprehensive check with all services down."""
        mock_vllm.return_value = {'status': 'unreachable'}
        mock_pg.return_value = {'status': 'unhealthy'}

        result = comprehensive_health_check(local=True)

        assert result['overall_status'] == 'unhealthy'


class TestHealthReporting:
    """Test health report printing."""

    @patch('builtins.print')
    def test_print_health_report(self, mock_print):
        """Test health report formatting."""
        from utils.runpod_health import print_health_report

        health = {
            'overall_status': 'healthy',
            'services': {
                'vllm': {'status': 'healthy', 'latency_ms': 25},
                'postgres': {'status': 'healthy', 'pgvector': True},
                'ssh': {'status': 'healthy'}
            }
        }

        print_health_report(health)

        # Verify print was called multiple times
        assert mock_print.called
        # Check that key information was printed
        printed_output = ' '.join(str(call[0][0]) for call in mock_print.call_args_list)
        assert 'HEALTH CHECK REPORT' in printed_output
        assert 'HEALTHY' in printed_output or 'healthy' in printed_output


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch('utils.runpod_health.requests.get')
    def test_vllm_health_network_error(self, mock_get):
        """Test handling of network errors."""
        mock_get.side_effect = Exception("Network unreachable")

        result = check_vllm_health()

        assert result['status'] == 'error'
        assert 'error' in result

    @patch('utils.runpod_health.psycopg2.connect')
    @patch.dict('os.environ', {'PGUSER': 'testuser', 'PGPASSWORD': 'testpass'})
    def test_postgres_health_uses_env(self, mock_connect):
        """Test PostgreSQL health uses environment variables."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [(1,), (True,)]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = check_postgres_health()

        # Verify environment variables were used
        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs['user'] == 'testuser'
        assert call_kwargs['password'] == 'testpass'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
