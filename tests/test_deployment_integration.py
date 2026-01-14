"""
Integration Tests for RunPod Deployment Workflow

Tests end-to-end deployment scenarios combining multiple components.
These tests validate the complete workflow from pod creation to health validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDeploymentWorkflow:
    """Test complete deployment workflow."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    @patch('utils.ssh_tunnel.subprocess.Popen')
    @patch('utils.runpod_health.requests.get')
    @patch('utils.runpod_health.psycopg2.connect')
    @patch('utils.ssh_tunnel.time.sleep')
    def test_full_deployment_workflow(
        self,
        mock_sleep,
        mock_pg_connect,
        mock_requests,
        mock_popen,
        mock_runpod
    ):
        """Test complete deployment: create → tunnel → validate."""
        from utils.runpod_manager import RunPodManager
        from utils.ssh_tunnel import SSHTunnelManager
        from utils.runpod_health import check_vllm_health, check_postgres_health

        # 1. Create pod
        mock_pod = {
            'id': 'integration_pod',
            'name': 'test-pod',
            'machine': {'podHostId': 'host123', 'podPort': 22},
            'runtime': {'containerState': 'running', 'uptimeInSeconds': 30},
            'costPerHr': 0.50
        }
        mock_runpod.create_pod.return_value = mock_pod
        mock_runpod.get_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")
        pod = manager.create_pod(name="test-pod")

        assert pod is not None
        assert pod['id'] == 'integration_pod'

        # 2. Wait for ready
        with patch('utils.runpod_manager.time.time', side_effect=[0, 0]):
            ready = manager.wait_for_ready(pod['id'], timeout=60)

        assert ready is True

        # 3. Create SSH tunnel
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 99999
        mock_popen.return_value = mock_process

        tunnel = SSHTunnelManager(ssh_host=pod['machine']['podHostId'])
        tunnel_created = tunnel.create_tunnel(ports=[8000, 5432])

        assert tunnel_created is True
        assert tunnel.is_active() is True

        # 4. Check service health
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.025
        mock_requests.return_value = mock_response

        vllm_health = check_vllm_health()
        assert vllm_health['status'] == 'healthy'

        # PostgreSQL health
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [(1,), (True,)]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pg_connect.return_value = mock_conn

        pg_health = check_postgres_health()
        assert pg_health['status'] == 'healthy'


class TestDeploymentErrorRecovery:
    """Test error handling in deployment workflow."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_pod_creation_failure_recovery(self, mock_runpod):
        """Test handling of pod creation failure."""
        from utils.runpod_manager import RunPodManager

        # First attempt fails
        mock_runpod.create_pod.side_effect = [
            Exception("GPU not available"),
            {'id': 'retry_pod', 'machine': {'podHostId': 'host'}}
        ]

        manager = RunPodManager(api_key="test_key")

        # First attempt fails
        pod1 = manager.create_pod()
        assert pod1 is None

        # Retry succeeds
        mock_runpod.create_pod.side_effect = None
        mock_runpod.create_pod.return_value = {'id': 'retry_pod', 'machine': {'podHostId': 'host'}}

        pod2 = manager.create_pod()
        assert pod2 is not None
        assert pod2['id'] == 'retry_pod'


class TestCostTracking:
    """Test cost tracking across deployment lifecycle."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_cost_tracking_workflow(self, mock_runpod):
        """Test cost estimation and tracking."""
        from utils.runpod_manager import RunPodManager

        manager = RunPodManager(api_key="test_key")

        # Estimate costs before deployment
        estimated_costs = manager.estimate_cost(hours_per_day=8, cost_per_hour=0.50)

        assert estimated_costs['daily_cost'] == 4.00
        assert estimated_costs['total_cost'] == 120.00

        # Create pod
        mock_pod = {
            'id': 'cost_pod',
            'machine': {'podHostId': 'host'},
            'runtime': {'uptimeInSeconds': 3600},  # 1 hour uptime
            'costPerHr': 0.50
        }
        mock_runpod.create_pod.return_value = mock_pod
        mock_runpod.get_pod.return_value = mock_pod

        pod = manager.create_pod()

        # Get actual cost
        status = manager.get_pod_status(pod['id'])

        assert status['cost_per_hour'] == 0.50
        assert status['uptime_seconds'] == 3600

        # Calculate actual cost so far
        hours_running = status['uptime_seconds'] / 3600
        cost_so_far = hours_running * status['cost_per_hour']

        assert cost_so_far == 0.50  # 1 hour at $0.50/hr


class TestMultiPodManagement:
    """Test managing multiple pods simultaneously."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_manage_multiple_pods(self, mock_runpod):
        """Test creating and managing multiple pods."""
        from utils.runpod_manager import RunPodManager

        # Mock multiple pods
        mock_pods = [
            {'id': 'pod1', 'name': 'rag-dev', 'runtime': {'containerState': 'running'}, 'costPerHr': 0.50, 'machine': {}},
            {'id': 'pod2', 'name': 'rag-test', 'runtime': {'containerState': 'stopped'}, 'costPerHr': 0, 'machine': {}},
            {'id': 'pod3', 'name': 'rag-prod', 'runtime': {'containerState': 'running'}, 'costPerHr': 0.50, 'machine': {}}
        ]
        mock_runpod.get_pods.return_value = mock_pods

        manager = RunPodManager(api_key="test_key")
        pods = manager.list_pods()

        # Count running pods
        running_pods = [p for p in pods if p['runtime']['containerState'] == 'running']
        assert len(running_pods) == 2

        # Calculate total cost
        total_cost = sum(p['costPerHr'] for p in running_pods)
        assert total_cost == 1.00  # 2 pods * $0.50/hr


class TestDeploymentScenarios:
    """Test various deployment scenarios."""

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_development_scenario(self, mock_runpod):
        """Test typical development deployment scenario."""
        from utils.runpod_manager import RunPodManager

        # Small pod for development
        mock_pod = {
            'id': 'dev_pod',
            'name': 'rag-dev',
            'machine': {'podHostId': 'dev_host', 'gpuTypeId': 'NVIDIA RTX 3090'},
            'runtime': {'containerState': 'running', 'uptimeInSeconds': 0},
            'costPerHr': 0.24
        }
        mock_runpod.create_pod.return_value = mock_pod
        mock_runpod.get_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")

        # Create development pod (smaller, cheaper)
        pod = manager.create_pod(
            name="rag-dev",
            gpu_type="NVIDIA RTX 3090",  # Cheaper GPU
            volume_gb=50  # Smaller storage
        )

        assert pod is not None
        assert pod['costPerHr'] < 0.50  # Cheaper than RTX 4090

    @patch('utils.runpod_manager.RUNPOD_AVAILABLE', True)
    @patch('utils.runpod_manager.runpod')
    def test_production_scenario(self, mock_runpod):
        """Test production deployment scenario."""
        from utils.runpod_manager import RunPodManager

        # Large pod for production
        mock_pod = {
            'id': 'prod_pod',
            'name': 'rag-prod',
            'machine': {'podHostId': 'prod_host', 'gpuTypeId': 'NVIDIA RTX 4090'},
            'runtime': {'containerState': 'running', 'uptimeInSeconds': 0},
            'costPerHr': 0.50
        }
        mock_runpod.create_pod.return_value = mock_pod

        manager = RunPodManager(api_key="test_key")

        # Create production pod (best performance)
        pod = manager.create_pod(
            name="rag-prod",
            gpu_type="NVIDIA RTX 4090",
            volume_gb=200,  # Larger storage
            env={
                "CTX": "16384",  # Larger context
                "TOP_K": "10"     # More retrieval
            }
        )

        assert pod is not None


# Mark as integration test (requires external resources in real scenario)
@pytest.mark.integration
class TestRealDeployment:
    """Integration tests requiring actual RunPod API."""

    @pytest.mark.skip(reason="Requires RunPod API key and creates real pods")
    def test_real_pod_creation(self):
        """Test actual pod creation (disabled by default)."""
        # This would create a real pod
        # Only run with: pytest -m integration
        pass

    @pytest.mark.skip(reason="Requires running pod and SSH access")
    def test_real_ssh_tunnel(self):
        """Test actual SSH tunnel creation (disabled by default)."""
        # This would create a real SSH tunnel
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
