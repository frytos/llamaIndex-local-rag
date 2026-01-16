"""
RunPod Pod Management Utilities

Provides a high-level interface to manage RunPod GPU pods for RAG deployment.

Features:
- Pod lifecycle management (create, stop, resume, terminate)
- Status monitoring with GPU metrics
- Health checks and readiness waiting
- SSH connection helpers
- Cost tracking

Usage:
    from utils.runpod_manager import RunPodManager

    # Initialize manager
    manager = RunPodManager(api_key="your_api_key")

    # Create pod
    pod = manager.create_pod(
        name="rag-pipeline-vllm",
        gpu_type="NVIDIA RTX 4090"
    )

    # Wait for ready
    if manager.wait_for_ready(pod['id']):
        print("Pod is ready!")

    # Get status
    status = manager.get_pod_status(pod['id'])
    print(f"GPU: {status['gpu_utilization']}%")

    # Stop pod
    manager.stop_pod(pod['id'])
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    logging.warning("runpod package not installed. Run: pip install runpod")

log = logging.getLogger(__name__)


class RunPodManager:
    """
    Manage RunPod GPU pods for RAG pipeline deployment.

    Provides high-level interface to RunPod Python SDK with:
    - Pod creation with RTX 4090 defaults
    - Lifecycle management
    - Status monitoring
    - Health checks
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize RunPod manager.

        Args:
            api_key: RunPod API key. If not provided, reads from RUNPOD_API_KEY env var.

        Raises:
            ValueError: If API key not found
            ImportError: If runpod package not installed
        """
        if not RUNPOD_AVAILABLE:
            raise ImportError(
                "runpod package not installed. Install with: pip install runpod"
            )

        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")

        if not self.api_key:
            raise ValueError(
                "RunPod API key not found. Either:\n"
                "  1. Pass api_key parameter\n"
                "  2. Set RUNPOD_API_KEY environment variable\n"
                "  Get your API key from https://runpod.io/settings"
            )

        # Set API key globally for runpod SDK
        runpod.api_key = self.api_key
        log.info("✅ RunPod API initialized")

    def list_pods(self) -> List[Dict]:
        """
        Get all pods for this account.

        Returns:
            List of pod dictionaries with details

        Example:
            >>> manager = RunPodManager()
            >>> pods = manager.list_pods()
            >>> for pod in pods:
            ...     print(f"{pod['name']}: {pod['runtime']['containerState']}")
        """
        try:
            pods = runpod.get_pods()
            log.debug(f"Found {len(pods)} pods")
            return pods if pods else []
        except Exception as e:
            log.error(f"Failed to list pods: {e}")
            return []

    def get_pod(self, pod_id: str) -> Optional[Dict]:
        """
        Get specific pod details.

        Args:
            pod_id: Pod ID

        Returns:
            Pod details dict or None if not found
        """
        try:
            pod = runpod.get_pod(pod_id)
            log.debug(f"Retrieved pod {pod_id}")
            return pod
        except Exception as e:
            log.error(f"Failed to get pod {pod_id}: {e}")
            return None

    def create_pod(
        self,
        name: str = "rag-pipeline-vllm",
        gpu_type: str = "NVIDIA RTX 4090",
        gpu_count: int = 1,
        image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        volume_gb: int = 100,
        container_disk_gb: int = 50,
        ports: str = "5432/tcp,8000/http,8001/tcp,22/tcp,3000/http",
        env: Optional[Dict[str, str]] = None,
        docker_args: Optional[str] = None,
        template_id: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create new RunPod pod optimized for RAG + vLLM.

        Args:
            name: Pod name (must be unique)
            gpu_type: GPU type (default: RTX 4090)
            gpu_count: Number of GPUs (default: 1)
            image: Docker image (default: PyTorch 2.4.0 + CUDA 12.4)
            volume_gb: Persistent storage size in GB (default: 100)
            container_disk_gb: Container disk size in GB (default: 50)
            ports: Exposed ports (default: PostgreSQL, vLLM, SSH, Grafana)
            env: Environment variables dict
            docker_args: Docker startup command
            template_id: RunPod template ID (optional, overrides image)

        Returns:
            Pod details dict or None on error

        Example:
            >>> manager = RunPodManager()
            >>> pod = manager.create_pod(
            ...     name="my-rag-prod",
            ...     gpu_type="NVIDIA RTX 4090",
            ...     volume_gb=100
            ... )
            >>> print(f"Pod ID: {pod['id']}")
            >>> print(f"SSH: ssh {pod['machine']['podHostId']}@ssh.runpod.io")
        """
        try:
            log.info(f"Creating pod: {name} on {gpu_type}")

            # Default environment variables for RAG pipeline
            default_env = {
                # vLLM Configuration
                "USE_VLLM": "1",
                "VLLM_MODEL": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                "VLLM_PORT": "8000",

                # Embedding Configuration
                "EMBED_BACKEND": "torch",
                "EMBED_MODEL": "BAAI/bge-small-en",
                "EMBED_DIM": "384",
                "EMBED_BATCH": "128",

                # PostgreSQL Configuration
                "PGHOST": "localhost",
                "PGPORT": "5432",
                "PGUSER": "fryt",
                "PGPASSWORD": "frytos",
                "DB_NAME": "vector_db",

                # RAG Configuration
                "CHUNK_SIZE": "700",
                "CHUNK_OVERLAP": "150",
                "TOP_K": "5",
                "CTX": "8192",
                "MAX_NEW_TOKENS": "512",
                "TEMP": "0.1",

                # Logging
                "LOG_LEVEL": "INFO",
                "LOG_QUERIES": "1",
            }

            # Merge with user-provided env vars
            if env:
                default_env.update(env)

            # Create pod using RunPod SDK
            pod = runpod.create_pod(
                name=name,
                image_name=image,
                gpu_type_id=gpu_type,
                gpu_count=gpu_count,
                volume_in_gb=volume_gb,
                container_disk_in_gb=container_disk_gb,
                ports=ports,
                env=default_env,
                docker_args=docker_args,  # Use None to let container use default CMD
                template_id=template_id
            )

            if not pod:
                log.error("Pod creation returned None")
                return None

            pod_id = pod.get('id', 'unknown')
            machine = pod.get('machine', {})
            ssh_host = machine.get('podHostId', 'unknown')

            log.info(f"✅ Pod created successfully!")
            log.info(f"   Pod ID: {pod_id}")
            log.info(f"   Name: {name}")
            log.info(f"   GPU: {gpu_type}")
            log.info(f"   SSH: ssh {ssh_host}@ssh.runpod.io")
            log.info(f"   Ports: {ports}")

            return pod

        except Exception as e:
            log.error(f"Failed to create pod: {e}")
            log.error("Check your API key and GPU availability")
            return None

    def stop_pod(self, pod_id: str) -> bool:
        """
        Stop pod to save costs.

        Note: Stopped pods don't incur GPU costs but still charge for storage.

        Args:
            pod_id: Pod ID

        Returns:
            True if successful, False otherwise

        Example:
            >>> manager.stop_pod("abc123")
            True
        """
        try:
            runpod.stop_pod(pod_id)
            log.info(f"✅ Pod {pod_id} stopped successfully")
            log.info("   Storage costs still apply. Terminate to remove completely.")
            return True
        except Exception as e:
            log.error(f"Failed to stop pod {pod_id}: {e}")
            return False

    def resume_pod(self, pod_id: str, gpu_count: int = 1) -> bool:
        """
        Resume stopped pod.

        Args:
            pod_id: Pod ID
            gpu_count: Number of GPUs to resume with

        Returns:
            True if successful, False otherwise

        Example:
            >>> manager.resume_pod("abc123")
            True
        """
        try:
            runpod.resume_pod(pod_id, gpu_count=gpu_count)
            log.info(f"✅ Pod {pod_id} resumed successfully")
            return True
        except Exception as e:
            log.error(f"Failed to resume pod {pod_id}: {e}")
            return False

    def terminate_pod(self, pod_id: str) -> bool:
        """
        Permanently terminate pod.

        WARNING: This action cannot be undone. All data on the pod will be lost
        unless you have a persistent volume attached.

        Args:
            pod_id: Pod ID

        Returns:
            True if successful, False otherwise

        Example:
            >>> manager.terminate_pod("abc123")
            True
        """
        try:
            runpod.terminate_pod(pod_id)
            log.info(f"✅ Pod {pod_id} terminated successfully")
            log.info("   All pod data has been deleted")
            return True
        except Exception as e:
            log.error(f"Failed to terminate pod {pod_id}: {e}")
            return False

    def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        """
        Get detailed pod status with metrics.

        Args:
            pod_id: Pod ID

        Returns:
            Dict with status, metrics, and connection info

        Example:
            >>> status = manager.get_pod_status("abc123")
            >>> print(f"Status: {status['status']}")
            >>> print(f"GPU: {status['gpu_utilization']}%")
            >>> print(f"Cost: ${status['cost_per_hour']}/hour")
        """
        pod = self.get_pod(pod_id)

        if not pod:
            return {
                "status": "not_found",
                "error": f"Pod {pod_id} not found"
            }

        runtime = pod.get("runtime") or {}
        machine = pod.get("machine") or {}
        gpu = pod.get("gpu") or {}

        # Get basic status from API
        container_state = runtime.get("containerState", "unknown")
        uptime = runtime.get("uptimeInSeconds", 0)
        ssh_host = machine.get("podHostId", "")

        # Improve status detection: if API says "unknown" but ports are available, pod is likely running
        ports = runtime.get("ports", [])
        if container_state == "unknown" and ports:
            # If we have port mappings, pod is probably running (API just hasn't updated)
            container_state = "running"
            log.debug(f"Pod {pod_id}: API status is 'unknown' but ports are available - treating as 'running'")

        # Try to construct SSH host if not available from API
        if not ssh_host and machine:
            # Sometimes podHostId is in the machine dict but with a different key
            # Or we can construct it from the pod creation response
            ssh_host = machine.get("podHostId") or machine.get("id", "")

        # If still no SSH host but we have pod ID, construct expected format
        if not ssh_host and pod_id:
            # RunPod SSH hosts follow pattern: {pod_id}-{random}@ssh.runpod.io
            # We can at least provide the pod ID for manual construction
            log.debug(f"SSH host not available from API for pod {pod_id}")

        status = {
            # State
            "status": container_state,
            "uptime_seconds": uptime,

            # GPU Metrics
            "gpu_utilization": runtime.get("gpuUtilization", 0),
            "gpu_type": machine.get("gpuTypeId", "unknown"),
            "gpu_count": machine.get("gpuCount", 0),

            # Memory
            "memory_utilization": runtime.get("memoryUtilization", 0),

            # Connection
            "ssh_host": ssh_host,
            "ssh_port": machine.get("podPort", 22),

            # Cost
            "cost_per_hour": pod.get("costPerHr", 0),

            # Timestamps
            "last_status_change": runtime.get("lastStatusChange", ""),

            # Extra: port mappings (useful for debugging)
            "ports": ports,
        }

        return status

    def wait_for_ready(
        self,
        pod_id: str,
        timeout: int = 300,
        check_interval: int = 10
    ) -> bool:
        """
        Wait for pod to be ready (running state).

        Args:
            pod_id: Pod ID
            timeout: Maximum wait time in seconds (default: 300 = 5 minutes)
            check_interval: Seconds between status checks (default: 10)

        Returns:
            True if pod is ready, False on timeout

        Example:
            >>> pod = manager.create_pod("my-pod")
            >>> if manager.wait_for_ready(pod['id']):
            ...     print("Pod is ready for use!")
        """
        start = time.time()
        elapsed = 0

        log.info(f"⏳ Waiting for pod {pod_id} to be ready...")
        log.info(f"   Timeout: {timeout}s | Check interval: {check_interval}s")

        while elapsed < timeout:
            status = self.get_pod_status(pod_id)

            if status.get("status") == "not_found":
                log.error(f"❌ Pod {pod_id} not found")
                return False

            container_state = status["status"]
            uptime = status["uptime_seconds"]

            # Pod is ready if running with uptime, OR if it has SSH host (even if API hasn't updated)
            ssh_host = status.get('ssh_host', '')

            if container_state == "running" and uptime > 0:
                log.info(f"✅ Pod {pod_id} is ready!")
                log.info(f"   Status: {container_state}")
                log.info(f"   Uptime: {uptime}s")
                log.info(f"   GPU: {status['gpu_type']} ({status['gpu_utilization']}%)")
                return True

            # If status is unknown but SSH host exists and we've waited >30s, assume ready
            if container_state == "unknown" and ssh_host and elapsed > 30:
                log.info(f"✅ Pod {pod_id} appears ready (SSH available)!")
                log.info(f"   Status: {container_state} (API may not be updated yet)")
                log.info(f"   SSH: {ssh_host}")
                log.info(f"   Try connecting via SSH to verify")
                return True

            elapsed = time.time() - start
            remaining = timeout - elapsed

            log.info(
                f"   Status: {container_state} | "
                f"Uptime: {uptime}s | "
                f"SSH: {ssh_host[:20] if ssh_host else 'N/A'} | "
                f"Remaining: {remaining:.0f}s"
            )

            time.sleep(check_interval)
            elapsed = time.time() - start

        log.error(f"❌ Pod {pod_id} not ready after {timeout}s")
        return False

    def get_ssh_command(self, pod_id: str, ports: Optional[List[int]] = None) -> str:
        """
        Generate SSH command for connecting to pod.

        Args:
            pod_id: Pod ID
            ports: List of local ports to forward (default: [8000, 5432, 3000])

        Returns:
            SSH command string

        Example:
            >>> cmd = manager.get_ssh_command("abc123")
            >>> print(cmd)
            ssh -L 8000:localhost:8000 -L 5432:localhost:5432 pod@ssh.runpod.io
        """
        # Try to get pod info directly first (more reliable)
        pod = self.get_pod(pod_id)

        if not pod:
            log.error(f"Pod {pod_id} not found when generating SSH command")
            return f"# Pod {pod_id} not found"

        # Extract SSH host from machine info
        machine = pod.get("machine") or {}
        ssh_host = machine.get("podHostId", "")

        log.debug(f"SSH host from machine info: {ssh_host}")

        # If empty, retry multiple times with delays (RunPod API is slow)
        if not ssh_host:
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    time.sleep(2)  # Wait 2 seconds before retry
                    fresh_pod = runpod.get_pod(pod_id)
                    if fresh_pod:
                        fresh_machine = fresh_pod.get("machine") or {}
                        ssh_host = fresh_machine.get("podHostId", "")
                        if ssh_host:
                            log.info(f"✅ Found SSH host on retry {attempt + 1}: {ssh_host}")
                            break
                        # Try alternate fields
                        ssh_host = fresh_machine.get("id", "")
                        if ssh_host:
                            log.info(f"✅ Found SSH host in alternate field: {ssh_host}")
                            break
                except Exception as e:
                    log.debug(f"Retry {attempt + 1} failed: {e}")

        # If STILL empty after retries, provide helpful placeholder with manual instructions
        if not ssh_host:
            log.warning(f"SSH host not available after retries for pod {pod_id}")
            return f"""# SSH host not available yet for pod: {pod_id}
#
# This usually means the pod was just created. Try:
#   1. Wait 2-3 minutes for RunPod API to update
#   2. Click 'Refresh Pod List' button above
#   3. Manual check: https://runpod.io/console/pods
#
# To get SSH host manually:
#   - Go to RunPod dashboard
#   - Click on your pod
#   - Look for 'SSH over exposed TCP' section
#   - Copy the ssh command (it will have the host)"""

        # Default ports: vLLM (8000), PostgreSQL (5432), Grafana (3000)
        if ports is None:
            ports = [8000, 5432, 3000]

        # Build SSH command with port forwarding
        port_forwards = " ".join([f"-L {p}:localhost:{p}" for p in ports])

        # Include SSH key hint if user has custom key
        ssh_cmd = f"ssh -i ~/.ssh/runpod_key -N {port_forwards} {ssh_host}@ssh.runpod.io"

        log.info(f"Generated SSH command for {pod_id}: {ssh_cmd}")

        return ssh_cmd

    def estimate_cost(
        self,
        hours_per_day: float,
        days: int = 30,
        cost_per_hour: float = 0.50
    ) -> Dict[str, float]:
        """
        Estimate monthly RunPod costs.

        Args:
            hours_per_day: Average hours of usage per day
            days: Number of days (default: 30)
            cost_per_hour: Cost per hour (default: $0.50 for RTX 4090)

        Returns:
            Dict with cost breakdown

        Example:
            >>> costs = manager.estimate_cost(hours_per_day=8, cost_per_hour=0.50)
            >>> print(f"Monthly cost: ${costs['total']:.2f}")
        """
        total_hours = hours_per_day * days
        total_cost = total_hours * cost_per_hour
        daily_cost = hours_per_day * cost_per_hour

        return {
            "hours_per_day": hours_per_day,
            "days": days,
            "total_hours": total_hours,
            "cost_per_hour": cost_per_hour,
            "daily_cost": daily_cost,
            "total_cost": total_cost,
        }

    def list_available_gpus(self) -> List[Dict]:
        """
        List available GPU types and their specs.

        Returns:
            List of GPU dictionaries with specs and pricing

        Example:
            >>> gpus = manager.list_available_gpus()
            >>> for gpu in gpus:
            ...     print(f"{gpu['displayName']}: ${gpu['lowestPrice']['uninterruptablePrice']}/hr")
        """
        try:
            gpus = runpod.get_gpus()
            log.debug(f"Found {len(gpus)} GPU types")
            return gpus if gpus else []
        except Exception as e:
            log.error(f"Failed to list GPUs: {e}")
            return []


# Convenience functions for common operations

def create_auto_pod(
    api_key: Optional[str] = None,
    name: str = "rag-pipeline-auto",
    github_repo: str = "",
    github_branch: str = "main",
    gpu_type: str = "NVIDIA GeForce RTX 4090",
    volume_gb: int = 100,
    wait: bool = True
) -> Optional[Dict]:
    """
    Create RunPod pod with automatic GitHub cloning and initialization.

    This creates a pod that:
    1. Clones your repository from GitHub automatically
    2. Runs the init script to set up PostgreSQL, vLLM, etc.
    3. Starts all services in the background
    4. Is fully ready to use when you SSH in!

    Args:
        api_key: RunPod API key (or uses RUNPOD_API_KEY env var)
        name: Pod name
        github_repo: GitHub repository URL (e.g., https://github.com/user/repo.git)
        github_branch: Git branch to clone (default: main)
        gpu_type: GPU type
        volume_gb: Storage size in GB
        wait: Wait for pod to be ready

    Returns:
        Pod details dict or None on error

    Example:
        >>> pod = create_auto_pod(
        ...     github_repo="https://github.com/username/llamaIndex-local-rag.git",
        ...     wait=True
        ... )
        >>> print(f"Pod ready! SSH: ssh {pod['machine']['podHostId']}@ssh.runpod.io")
        >>> # Services are already initialized! Just SSH in and use it.
    """
    import os

    api_key = api_key or os.getenv('RUNPOD_API_KEY')

    if not api_key:
        log.error("API key required (set RUNPOD_API_KEY or pass api_key parameter)")
        return None

    if not github_repo:
        log.error("github_repo is required for auto-setup")
        log.info("Example: create_auto_pod(github_repo='https://github.com/user/repo.git')")
        return None

    manager = RunPodManager(api_key=api_key)

    # Startup command that clones repo and runs init
    startup_cmd = f'''bash -c "
        cd /workspace && \\
        git clone --branch {github_branch} {github_repo} rag-pipeline && \\
        cd rag-pipeline && \\
        bash scripts/init_runpod_services.sh && \\
        exec sleep infinity
    "'''

    # Add GitHub info to environment
    env = {
        "GITHUB_REPO": github_repo,
        "GITHUB_BRANCH": github_branch,
    }

    log.info(f"Creating auto-setup pod from GitHub: {github_repo}")

    pod = manager.create_pod(
        name=name,
        gpu_type=gpu_type,
        volume_gb=volume_gb,
        env=env,
        docker_args=startup_cmd
    )

    if not pod:
        return None

    pod_id = pod['id']

    if wait:
        log.info("Waiting for pod to be ready and services to initialize...")
        log.info("This may take 10-15 minutes (repo clone + service setup)")

        # Wait longer for auto-setup (includes init time)
        if manager.wait_for_ready(pod_id, timeout=600):  # 10 minutes
            log.info("✅ Pod is ready!")
            log.info("")
            log.info("Services are initializing in background...")
            log.info("Wait 5 more minutes, then verify:")
            log.info("  curl http://localhost:8000/health")
            log.info("  psql -h localhost -U fryt -d vector_db -c 'SELECT 1'")

    return pod


def create_rag_pod(
    api_key: str,
    name: str = "rag-pipeline-vllm",
    wait: bool = True
) -> Optional[Dict]:
    """
    Convenience function to create and wait for RAG pod.

    Args:
        api_key: RunPod API key
        name: Pod name
        wait: Wait for pod to be ready

    Returns:
        Pod dict or None on error

    Example:
        >>> pod = create_rag_pod(api_key="your_key", name="my-rag-prod")
        >>> print(f"Pod ready: {pod['id']}")
    """
    manager = RunPodManager(api_key=api_key)

    log.info(f"Creating RAG pod: {name}")
    pod = manager.create_pod(name=name)

    if not pod:
        return None

    if wait:
        log.info("Waiting for pod to be ready...")
        if not manager.wait_for_ready(pod['id']):
            log.error("Pod failed to start")
            return None

    return pod


def get_pod_info(api_key: str, pod_id: str) -> Dict:
    """
    Get comprehensive pod information.

    Args:
        api_key: RunPod API key
        pod_id: Pod ID

    Returns:
        Dict with pod details and status

    Example:
        >>> info = get_pod_info(api_key="your_key", pod_id="abc123")
        >>> print(f"Status: {info['status']}")
        >>> print(f"SSH: {info['ssh_command']}")
    """
    manager = RunPodManager(api_key=api_key)

    status = manager.get_pod_status(pod_id)
    ssh_cmd = manager.get_ssh_command(pod_id)

    return {
        **status,
        "ssh_command": ssh_cmd,
    }
