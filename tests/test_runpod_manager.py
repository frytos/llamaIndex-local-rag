"""
Tests for RunPod Manager (utils/runpod_manager.py)

Tests pod lifecycle management, monitoring, and utilities.
Uses mocking to avoid actual RunPod API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunPodManagerInit:
    """Test RunPodManager initialization."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_init_with_api_key(self, mock_runpod):
        """Test initialization with API key parameter."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager(api_key="test_key_123")

        assert manager.api_key == "test_key_123"
        assert mock_runpod.api_key == "test_key_123"

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    @patch.dict(os.environ, {'RUNPOD_API_KEY': 'env_key_456'})
    def test_init_with_env_var(self, mock_runpod):
        """Test initialization with environment variable."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager()

        assert manager.api_key == "env_key_456"

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self, mock_runpod):
        """Test initialization fails without API key."""
        from utils.runpod_manager import RunPodManager

        with pytest.raises(ValueError) as exc_info:
            RunPodManager()

        assert "API key not found" in str(exc_info.value)

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', False)
    def test_init_without_runpod_package(self):
        """Test initialization fails when runpod package not installed."""
        from utils.runpod_manager import RunPodManager

        with pytest.raises(ImportError) as exc_info:
            RunPodManager(api_key="test_key")

        assert "runpod package not installed" in str(exc_info.value)


class TestPodListing:
    """Test pod listing functionality."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_list_pods_success(self, mock_runpod):
        """Test successful pod listing."""
        from utils.runpod_manager import RunPodManager

        # Mock pod data
        mock_pods = [
            {'id': 'pod1', 'name': 'rag-pipeline-1'},
            {'id': 'pod2', 'name': 'rag-pipeline-2'}
        ]
        mock_runpod.get_pods.return_value = mock_pods

        manager = RunPodManager(api_key="test_key")
        pods = manager.list_pods()

        assert len(pods) == 2
        assert pods[0]['id'] == 'pod1'
        assert pods[1]['name'] == 'rag-pipeline-2'

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_list_pods_empty(self, mock_runpod):
        """Test listing pods when none exist."""
        from utils.runpod_manager import RunPodManager

        mock_runpod.get_pods.return_value = []

        manager = RunPodManager(api_key="test_key")
        pods = manager.list_pods()

        assert len(pods) == 0

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_list_pods_api_error(self, mock_runpod):
        """Test handling of API error when listing pods."""
        from utils.runpod_manager import RunPodManager

        mock_runpod.get_pods.side_effect = Exception("API error")

        manager = RunPodManager(api_key="test_key")
        pods = manager.list_pods()

        assert len(pods) == 0  # Should return empty list on error


class TestPodCreation:
    """Test pod creation."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_create_pod_default_config(self, mock_runpod):
        """Test pod creation with default configuration."""
        from utils.runpod_manager import RunPodManager

        mock_pod = {
            'id': 'abc123',
            'name': 'rag-pipeline-vllm',
            'machine': {'podHostId': 'host123'}
        }
        mock_runpod.create_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")
        pod = manager.create_pod()

        assert pod['id'] == 'abc123'
        assert pod['name'] == 'rag-pipeline-vllm'
        mock_runpod.create_pod.assert_called_once()

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_create_pod_custom_config(self, mock_runpod):
        """Test pod creation with custom configuration."""
        from utils.runpod_manager import RunPodManager

        mock_pod = {
            'id': 'custom123',
            'name': 'custom-pod',
            'machine': {'podHostId': 'host456'}
        }
        mock_runpod.create_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")
        pod = manager.create_pod(
            name="custom-pod",
            gpu_type="NVIDIA RTX 3090",
            volume_gb=200
        )

        assert pod is not None
        call_kwargs = mock_runpod.create_pod.call_args[1]
        assert call_kwargs['name'] == 'custom-pod'
        assert call_kwargs['gpu_type_id'] == 'NVIDIA RTX 3090'
        assert call_kwargs['volume_in_gb'] == 200

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_create_pod_with_env_vars(self, mock_runpod):
        """Test pod creation with custom environment variables."""
        from utils.runpod_manager import RunPodManager

        mock_pod = {'id': 'env123', 'machine': {'podHostId': 'host'}}
        mock_runpod.create_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")
        pod = manager.create_pod(
            env={
                "CUSTOM_VAR": "custom_value",
                "CTX": "16384"
            }
        )

        call_kwargs = mock_runpod.create_pod.call_args[1]
        assert 'CUSTOM_VAR' in call_kwargs['env']
        assert call_kwargs['env']['CUSTOM_VAR'] == 'custom_value'
        assert call_kwargs['env']['CTX'] == '16384'

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_create_pod_failure(self, mock_runpod):
        """Test handling of pod creation failure."""
        from utils.runpod_manager import RunPodManager

        mock_runpod.create_pod.side_effect = Exception("GPU not available")

        manager = RunPodManager(api_key="test_key")
        pod = manager.create_pod()

        assert pod is None


class TestPodLifecycle:
    """Test pod lifecycle operations."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_stop_pod_success(self, mock_runpod):
        """Test stopping pod successfully."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager(api_key="test_key")
        result = manager.stop_pod('pod123')

        assert result is True
        mock_runpod.stop_pod.assert_called_once_with('pod123')

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_resume_pod_success(self, mock_runpod):
        """Test resuming pod successfully."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager(api_key="test_key")
        result = manager.resume_pod('pod123')

        assert result is True
        mock_runpod.resume_pod.assert_called_once_with('pod123', gpu_count=1)

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_terminate_pod_success(self, mock_runpod):
        """Test terminating pod successfully."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager(api_key="test_key")
        result = manager.terminate_pod('pod123')

        assert result is True
        mock_runpod.terminate_pod.assert_called_once_with('pod123')

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_pod_operation_failure(self, mock_runpod):
        """Test handling of failed pod operations."""
        from utils.runpod_manager import RunPodManager

        mock_runpod.stop_pod.side_effect = Exception("API error")

        manager = RunPodManager(api_key="test_key")
        result = manager.stop_pod('pod123')

        assert result is False


class TestPodStatus:
    """Test pod status retrieval."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_get_pod_status_running(self, mock_runpod):
        """Test getting status of running pod."""
        from utils.runpod_manager import RunPodManager

        mock_pod = {
            'id': 'pod123',
            'runtime': {
                'containerState': 'running',
                'uptimeInSeconds': 2700,
                'gpuUtilization': 45,
                'memoryUtilization': 60
            },
            'machine': {
                'podHostId': 'host123',
                'podPort': 22,
                'gpuTypeId': 'NVIDIA RTX 4090',
                'gpuCount': 1
            },
            'costPerHr': 0.50
        }
        mock_runpod.get_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")
        status = manager.get_pod_status('pod123')

        assert status['status'] == 'running'
        assert status['uptime_seconds'] == 2700
        assert status['gpu_utilization'] == 45
        assert status['memory_utilization'] == 60
        assert status['ssh_host'] == 'host123'
        assert status['cost_per_hour'] == 0.50

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_get_pod_status_not_found(self, mock_runpod):
        """Test handling of non-existent pod."""
        from utils.runpod_manager import RunPodManager

        mock_runpod.get_pod.return_value = None

        manager = RunPodManager(api_key="test_key")
        status = manager.get_pod_status('nonexistent')

        assert status['status'] == 'not_found'
        assert 'error' in status


class TestWaitForReady:
    """Test waiting for pod readiness."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    @patch('utils.runpod_manager.time.sleep')
    @patch('utils.runpod_manager.time.time')
    def test_wait_for_ready_immediate(self, mock_time, mock_sleep, mock_runpod):
        """Test pod already ready."""
        from utils.runpod_manager import RunPodManager

        # Mock pod as immediately running
        mock_pod = {
            'runtime': {'containerState': 'running', 'uptimeInSeconds': 10},
            'machine': {'gpuTypeId': 'RTX 4090', 'podHostId': 'host'},
            'costPerHr': 0.50
        }
        mock_runpod.get_pod.return_value = mock_pod

        mock_time.side_effect = [0, 0]  # No time elapsed

        manager = RunPodManager(api_key="test_key")
        result = manager.wait_for_ready('pod123', timeout=60)

        assert result is True

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    @patch('utils.runpod_manager.time.sleep')
    @patch('utils.runpod_manager.time.time')
    def test_wait_for_ready_eventually(self, mock_time, mock_sleep, mock_runpod):
        """Test pod becomes ready after waiting."""
        from utils.runpod_manager import RunPodManager

        # First check: not ready, second check: ready
        mock_pods = [
            {'runtime': {'containerState': 'starting', 'uptimeInSeconds': 0}, 'machine': {}, 'costPerHr': 0},
            {'runtime': {'containerState': 'running', 'uptimeInSeconds': 30}, 'machine': {'gpuTypeId': 'RTX 4090', 'podHostId': 'host'}, 'costPerHr': 0.50}
        ]

        mock_runpod.get_pod.side_effect = mock_pods
        mock_time.side_effect = [0, 10, 20]  # Time progression

        manager = RunPodManager(api_key="test_key")
        result = manager.wait_for_ready('pod123', timeout=60, check_interval=10)

        assert result is True
        assert mock_sleep.called

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    @patch('utils.runpod_manager.time.sleep')
    @patch('utils.runpod_manager.time.time')
    def test_wait_for_ready_timeout(self, mock_time, mock_sleep, mock_runpod):
        """Test timeout when pod doesn't become ready."""
        from utils.runpod_manager import RunPodManager

        # Pod never becomes ready
        mock_pod = {
            'runtime': {'containerState': 'starting', 'uptimeInSeconds': 0},
            'machine': {},
            'costPerHr': 0
        }
        mock_runpod.get_pod.return_value = mock_pod

        # Simulate timeout
        mock_time.side_effect = [0, 10, 20, 30, 40, 50, 60, 70]

        manager = RunPodManager(api_key="test_key")
        result = manager.wait_for_ready('pod123', timeout=60, check_interval=10)

        assert result is False


class TestSSHCommands:
    """Test SSH command generation."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_get_ssh_command_default_ports(self, mock_runpod):
        """Test SSH command with default ports."""
        from utils.runpod_manager import RunPodManager

        mock_pod = {
            'runtime': {'containerState': 'running'},
            'machine': {'podHostId': 'host123', 'podPort': 22},
            'costPerHr': 0.50
        }
        mock_runpod.get_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")
        ssh_cmd = manager.get_ssh_command('pod123')

        assert 'ssh' in ssh_cmd
        assert 'host123@ssh.runpod.io' in ssh_cmd
        assert '-L 8000:localhost:8000' in ssh_cmd
        assert '-L 5432:localhost:5432' in ssh_cmd
        assert '-L 3000:localhost:3000' in ssh_cmd

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_get_ssh_command_custom_ports(self, mock_runpod):
        """Test SSH command with custom ports."""
        from utils.runpod_manager import RunPodManager

        mock_pod = {
            'runtime': {'containerState': 'running'},
            'machine': {'podHostId': 'host456', 'podPort': 22},
            'costPerHr': 0.50
        }
        mock_runpod.get_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")
        ssh_cmd = manager.get_ssh_command('pod123', ports=[8000, 9000])

        assert '-L 8000:localhost:8000' in ssh_cmd
        assert '-L 9000:localhost:9000' in ssh_cmd
        assert '-L 5432' not in ssh_cmd  # Default ports not included


class TestCostEstimation:
    """Test cost estimation utilities."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_estimate_cost_development(self, mock_runpod):
        """Test cost estimation for development usage."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager(api_key="test_key")
        costs = manager.estimate_cost(
            hours_per_day=2,
            days=30,
            cost_per_hour=0.50
        )

        assert costs['hours_per_day'] == 2
        assert costs['days'] == 30
        assert costs['total_hours'] == 60
        assert costs['cost_per_hour'] == 0.50
        assert costs['daily_cost'] == 1.00
        assert costs['total_cost'] == 30.00

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_estimate_cost_production(self, mock_runpod):
        """Test cost estimation for production usage."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager(api_key="test_key")
        costs = manager.estimate_cost(
            hours_per_day=8,
            cost_per_hour=0.50
        )

        assert costs['daily_cost'] == 4.00
        assert costs['total_cost'] == 120.00

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_estimate_cost_24_7(self, mock_runpod):
        """Test cost estimation for 24/7 usage."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager(api_key="test_key")
        costs = manager.estimate_cost(
            hours_per_day=24,
            cost_per_hour=0.50
        )

        assert costs['total_cost'] == 360.00


class TestGPUListing:
    """Test GPU availability listing."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_list_available_gpus(self, mock_runpod):
        """Test listing available GPUs."""
        from utils.runpod_manager import RunPodManager

        mock_gpus = [
            {
                'displayName': 'NVIDIA RTX 4090',
                'memoryInGb': 24,
                'lowestPrice': {'uninterruptablePrice': 0.50}
            },
            {
                'displayName': 'NVIDIA RTX 3090',
                'memoryInGb': 24,
                'lowestPrice': {'uninterruptablePrice': 0.24}
            }
        ]
        mock_runpod.get_gpus.return_value = mock_gpus

        manager = RunPodManager(api_key="test_key")
        gpus = manager.list_available_gpus()

        assert len(gpus) == 2
        assert gpus[0]['displayName'] == 'NVIDIA RTX 4090'
        assert gpus[1]['memoryInGb'] == 24


class TestConvenienceFunctions:
    """Test convenience helper functions."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    @patch('utils.runpod_manager.RunPodManager.create_pod')
    @patch('utils.runpod_manager.RunPodManager.wait_for_ready')
    def test_create_rag_pod(self, mock_wait, mock_create, mock_runpod):
        """Test create_rag_pod convenience function."""
        from utils.runpod_manager import create_rag_pod

        mock_pod = {'id': 'pod123', 'machine': {'podHostId': 'host'}}
        mock_create.return_value = mock_pod
        mock_wait.return_value = True

        pod = create_rag_pod(api_key="test_key", name="my-rag", wait=True)

        assert pod is not None
        assert pod['id'] == 'pod123'
        mock_create.assert_called_once()
        mock_wait.assert_called_once_with('pod123')

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    @patch('utils.runpod_manager.RunPodManager.get_pod_status')
    @patch('utils.runpod_manager.RunPodManager.get_ssh_command')
    def test_get_pod_info(self, mock_ssh_cmd, mock_status, mock_runpod):
        """Test get_pod_info convenience function."""
        from utils.runpod_manager import get_pod_info

        mock_status.return_value = {
            'status': 'running',
            'gpu_utilization': 45,
            'cost_per_hour': 0.50
        }
        mock_ssh_cmd.return_value = "ssh host@ssh.runpod.io"

        info = get_pod_info(api_key="test_key", pod_id="pod123")

        assert info['status'] == 'running'
        assert info['ssh_command'] == "ssh host@ssh.runpod.io"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
