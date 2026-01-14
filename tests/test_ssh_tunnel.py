"""
Tests for SSH Tunnel Manager (utils/ssh_tunnel.py)

Tests SSH tunnel creation, management, and port forwarding.
Uses mocking to avoid actual SSH connections.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ssh_tunnel import SSHTunnelManager, create_tunnel, run_with_tunnel


class TestSSHTunnelInit:
    """Test SSH tunnel manager initialization."""

    def test_init_default_port(self):
        """Test initialization with default SSH port."""
        tunnel = SSHTunnelManager(ssh_host="host123")

        assert tunnel.ssh_host == "host123"
        assert tunnel.ssh_port == 22
        assert tunnel.process is None
        assert tunnel.ports == []

    def test_init_custom_port(self):
        """Test initialization with custom SSH port."""
        tunnel = SSHTunnelManager(ssh_host="host456", ssh_port=2222)

        assert tunnel.ssh_host == "host456"
        assert tunnel.ssh_port == 2222


class TestTunnelCreation:
    """Test tunnel creation."""

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    @patch('utils.ssh_tunnel.os.path.exists')
    def test_create_tunnel_success(self, mock_exists, mock_sleep, mock_popen):
        """Test successful tunnel creation."""
        mock_exists.return_value = True  # SSH key exists

        # Mock process
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process still running
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        tunnel = SSHTunnelManager(ssh_host="host123")
        result = tunnel.create_tunnel(ports=[8000, 5432])

        assert result is True
        assert tunnel.process == mock_process
        assert tunnel.ports == [8000, 5432]
        mock_popen.assert_called_once()

        # Verify SSH command construction
        call_args = mock_popen.call_args[0][0]
        assert 'ssh' in call_args
        assert '-L' in call_args
        assert '8000:localhost:8000' in call_args
        assert '5432:localhost:5432' in call_args

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    def test_create_tunnel_default_ports(self, mock_sleep, mock_popen):
        """Test tunnel creation with default ports."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        tunnel = SSHTunnelManager(ssh_host="host123")
        result = tunnel.create_tunnel()  # No ports specified

        assert result is True
        # Should use defaults: 8000, 5432, 3000
        assert tunnel.ports == [8000, 5432, 3000]

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    def test_create_tunnel_failure(self, mock_sleep, mock_popen):
        """Test tunnel creation failure."""
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process exited with error
        mock_process.stderr = Mock()
        mock_process.stderr.read.return_value = b"Connection refused"
        mock_popen.return_value = mock_process

        tunnel = SSHTunnelManager(ssh_host="host123")
        result = tunnel.create_tunnel(ports=[8000])

        assert result is False


class TestTunnelStatus:
    """Test tunnel status checking."""

    def test_is_active_no_process(self):
        """Test is_active when no tunnel exists."""
        tunnel = SSHTunnelManager(ssh_host="host123")

        assert tunnel.is_active() is False

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    def test_is_active_running(self, mock_sleep, mock_popen):
        """Test is_active when tunnel is running."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        tunnel = SSHTunnelManager(ssh_host="host123")
        tunnel.create_tunnel(ports=[8000])

        assert tunnel.is_active() is True

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    def test_is_active_stopped(self, mock_sleep, mock_popen):
        """Test is_active when tunnel has stopped."""
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, 0]  # Running, then stopped
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        tunnel = SSHTunnelManager(ssh_host="host123")
        tunnel.create_tunnel(ports=[8000])

        assert tunnel.is_active() is True
        # After process stops
        assert tunnel.is_active() is False


class TestTunnelTermination:
    """Test tunnel stopping."""

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    @patch('utils.ssh_tunnel.os.killpg')
    @patch('utils.ssh_tunnel.os.getpgid')
    def test_stop_tunnel_success(self, mock_getpgid, mock_killpg, mock_sleep, mock_popen):
        """Test successful tunnel stopping."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        mock_getpgid.return_value = 12345

        tunnel = SSHTunnelManager(ssh_host="host123")
        tunnel.create_tunnel(ports=[8000])

        result = tunnel.stop_tunnel()

        assert result is True
        assert tunnel.process is None
        mock_killpg.assert_called_once()

    def test_stop_tunnel_no_process(self):
        """Test stopping when no tunnel exists."""
        tunnel = SSHTunnelManager(ssh_host="host123")

        result = tunnel.stop_tunnel()

        assert result is False

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    @patch('utils.ssh_tunnel.os.killpg')
    @patch('utils.ssh_tunnel.os.getpgid')
    def test_stop_tunnel_force_kill(self, mock_getpgid, mock_killpg, mock_sleep, mock_popen):
        """Test force killing when graceful stop fails."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.wait.side_effect = subprocess.TimeoutExpired('ssh', 5)
        mock_popen.return_value = mock_process

        mock_getpgid.return_value = 12345

        tunnel = SSHTunnelManager(ssh_host="host123")
        tunnel.create_tunnel(ports=[8000])

        result = tunnel.stop_tunnel()

        assert result is True
        # Should call killpg twice (SIGTERM then SIGKILL)
        assert mock_killpg.call_count == 2


class TestTunnelCommandGeneration:
    """Test SSH tunnel command string generation."""

    def test_get_tunnel_command(self):
        """Test tunnel command string generation."""
        tunnel = SSHTunnelManager(ssh_host="host123")
        tunnel.ports = [8000, 5432, 3000]

        cmd = tunnel.get_tunnel_command()

        assert 'ssh' in cmd
        assert '-N' in cmd
        assert '-L 8000:localhost:8000' in cmd
        assert '-L 5432:localhost:5432' in cmd
        assert '-L 3000:localhost:3000' in cmd
        assert 'host123@ssh.runpod.io' in cmd


class TestContextManager:
    """Test context manager functionality."""

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    @patch('utils.ssh_tunnel.os.killpg')
    @patch('utils.ssh_tunnel.os.getpgid')
    def test_context_manager_auto_cleanup(self, mock_getpgid, mock_killpg, mock_sleep, mock_popen):
        """Test automatic cleanup with context manager."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        mock_getpgid.return_value = 12345

        with SSHTunnelManager("host123") as tunnel:
            tunnel.create_tunnel(ports=[8000])
            assert tunnel.is_active() is True

        # After exiting context, tunnel should be stopped
        # (process is set to None in stop_tunnel)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('utils.ssh_tunnel.SSHTunnelManager.create_tunnel')
    def test_create_tunnel_function(self, mock_create):
        """Test create_tunnel convenience function."""
        mock_create.return_value = True

        tunnel = create_tunnel(ssh_host="host123", ports=[8000], background=True)

        assert tunnel is not None
        mock_create.assert_called_once_with(ports=[8000], background=True)

    @patch('utils.ssh_tunnel.SSHTunnelManager.create_tunnel')
    @patch('utils.ssh_tunnel.SSHTunnelManager.stop_tunnel')
    @patch('utils.ssh_tunnel.time.sleep')
    def test_run_with_tunnel(self, mock_sleep, mock_stop, mock_create):
        """Test run_with_tunnel helper."""
        mock_create.return_value = True

        def dummy_func(arg1, arg2):
            return arg1 + arg2

        result = run_with_tunnel("host123", [8000], dummy_func, 5, 10)

        assert result == 15
        mock_create.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch('utils.ssh_tunnel.subprocess.Popen')
    def test_create_tunnel_exception(self, mock_popen):
        """Test handling of exception during tunnel creation."""
        mock_popen.side_effect = Exception("SSH not available")

        tunnel = SSHTunnelManager(ssh_host="host123")
        result = tunnel.create_tunnel(ports=[8000])

        assert result is False

    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.ssh_tunnel.time.sleep')
    def test_multiple_tunnels(self, mock_sleep, mock_popen):
        """Test creating multiple tunnels (should replace existing)."""
        mock_process1 = Mock()
        mock_process1.poll.return_value = None
        mock_process1.pid = 111

        mock_process2 = Mock()
        mock_process2.poll.return_value = None
        mock_process2.pid = 222

        mock_popen.side_effect = [mock_process1, mock_process2]

        tunnel = SSHTunnelManager(ssh_host="host123")

        # Create first tunnel
        result1 = tunnel.create_tunnel(ports=[8000])
        assert result1 is True
        assert tunnel.process.pid == 111

        # Create second tunnel (should replace first)
        result2 = tunnel.create_tunnel(ports=[5432])
        assert result2 is True
        assert tunnel.process.pid == 222


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
