# 🚀 Production Deployment Workflow: Streamlit → RunPod

**Target**: Seamless deployment from Streamlit UI to RunPod RTX 4090
**Stack**: PyTorch 2.4.0 (CUDA 12.4) + vLLM + PostgreSQL + HNSW
**Goal**: One-click deployment with automated management

**Status**: Implementation Ready
**Estimated Effort**: 8-12 hours full implementation

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: RunPod API Integration](#phase-1-runpod-api-integration)
3. [Phase 2: Deployment Automation](#phase-2-deployment-automation)
4. [Phase 3: Streamlit UI Integration](#phase-3-streamlit-ui-integration)
5. [Phase 4: Monitoring & Health Checks](#phase-4-monitoring--health-checks)
6. [Phase 5: Cost Optimization](#phase-5-cost-optimization)
7. [Testing & Validation](#testing--validation)
8. [Production Checklist](#production-checklist)

---

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL MACHINE                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           STREAMLIT WEB UI (rag_web.py)              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐ │  │
│  │  │ Indexing   │  │ Querying   │  │  Deployment    │ │  │
│  │  │ Controls   │  │ Interface  │  │  Manager       │ │  │
│  │  └────────────┘  └────────────┘  └────────────────┘ │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  RunPod Controls:                                 │ │  │
│  │  │  [Create Pod] [Stop Pod] [Resume Pod]            │ │  │
│  │  │  Status: ● Running | GPU: 45% | Queries: 127     │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ RunPod Python SDK
                             │ (GraphQL API)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              RUNPOD CLOUD (RTX 4090)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Pod: rag-pipeline-prod-vllm                         │  │
│  │  Template: RunPod PyTorch 2.4.0 (CUDA 12.4)         │  │
│  │  GPU: RTX 4090 (24GB) | Cost: $0.50/hour            │  │
│  │                                                       │  │
│  │  Services:                                           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────────┐ │  │
│  │  │PostgreSQL  │  │vLLM Server │  │ RAG Pipeline  │ │  │
│  │  │:5432       │  │:8000       │  │               │ │  │
│  │  │+ pgvector  │  │Mistral 7B  │  │BGE embeddings │ │  │
│  │  │+ HNSW      │  │AWQ 4-bit   │  │HNSW indices   │ │  │
│  │  └────────────┘  └────────────┘  └───────────────┘ │  │
│  │                                                       │  │
│  │  SSH Tunnel: ssh pod@ssh.runpod.io -L 8000:8000     │  │
│  │  Metrics: Prometheus + Grafana (optional)           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query → Streamlit UI → RunPod Pod → vLLM Server → Response
     ↓              ↓             ↓            ↓           ↓
  Input        Validation    PostgreSQL   Mistral 7B   Display
               + SSH         + HNSW       120 tok/s    Results
               Tunnel        215x faster
```

---

## Phase 1: RunPod API Integration

### 1.1 Setup RunPod Python SDK

**File**: `utils/runpod_manager.py`

```python
"""
RunPod Pod Management Utilities

Handles pod creation, lifecycle management, and monitoring.
"""
import os
import time
import runpod
from typing import Dict, List, Optional
import logging

log = logging.getLogger(__name__)


class RunPodManager:
    """Manage RunPod pods via Python SDK."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize RunPod manager.

        Args:
            api_key: RunPod API key (from settings or env)
        """
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RunPod API key not found. Set RUNPOD_API_KEY env var.")

        runpod.api_key = self.api_key
        log.info("✅ RunPod API initialized")

    def list_pods(self) -> List[Dict]:
        """Get all pods."""
        try:
            pods = runpod.get_pods()
            return pods
        except Exception as e:
            log.error(f"Failed to list pods: {e}")
            return []

    def get_pod(self, pod_id: str) -> Optional[Dict]:
        """Get specific pod details."""
        try:
            return runpod.get_pod(pod_id)
        except Exception as e:
            log.error(f"Failed to get pod {pod_id}: {e}")
            return None

    def create_pod(
        self,
        name: str = "rag-pipeline-vllm",
        gpu_type: str = "NVIDIA RTX 4090",
        gpu_count: int = 1,
        image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel",
        volume_gb: int = 100,
        ports: str = "5432/tcp,8000/http,22/tcp,3000/http",
        env: Dict[str, str] = None,
        docker_args: str = None
    ) -> Optional[Dict]:
        """
        Create new RunPod pod with vLLM configuration.

        Args:
            name: Pod name
            gpu_type: GPU type (RTX 4090 recommended)
            gpu_count: Number of GPUs
            image: Docker image (PyTorch 2.4.0 + CUDA 12.4)
            volume_gb: Persistent storage size
            ports: Exposed ports
            env: Environment variables
            docker_args: Docker startup command

        Returns:
            Pod details dict or None on error
        """
        try:
            log.info(f"Creating pod: {name} on {gpu_type}")

            # Default environment variables for RAG pipeline
            default_env = {
                "USE_VLLM": "1",
                "VLLM_MODEL": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                "EMBED_BACKEND": "torch",
                "EMBED_MODEL": "BAAI/bge-small-en",
                "CTX": "8192",
                "MAX_NEW_TOKENS": "512",
                "TOP_K": "5",
                "PGHOST": "localhost",
                "PGPORT": "5432",
                "PGUSER": "fryt",
                "PGPASSWORD": "frytos",
                "DB_NAME": "vector_db"
            }

            if env:
                default_env.update(env)

            # Create pod
            pod = runpod.create_pod(
                name=name,
                image_name=image,
                gpu_type_id=gpu_type,
                gpu_count=gpu_count,
                volume_in_gb=volume_gb,
                ports=ports,
                env=default_env,
                docker_args=docker_args or "bash"
            )

            log.info(f"✅ Pod created: {pod['id']}")
            log.info(f"   SSH: ssh {pod['machine']['podHostId']}@ssh.runpod.io")

            return pod

        except Exception as e:
            log.error(f"Failed to create pod: {e}")
            return None

    def stop_pod(self, pod_id: str) -> bool:
        """Stop pod to save costs."""
        try:
            runpod.stop_pod(pod_id)
            log.info(f"✅ Pod {pod_id} stopped")
            return True
        except Exception as e:
            log.error(f"Failed to stop pod: {e}")
            return False

    def resume_pod(self, pod_id: str, gpu_count: int = 1) -> bool:
        """Resume stopped pod."""
        try:
            runpod.resume_pod(pod_id, gpu_count=gpu_count)
            log.info(f"✅ Pod {pod_id} resumed")
            return True
        except Exception as e:
            log.error(f"Failed to resume pod: {e}")
            return False

    def terminate_pod(self, pod_id: str) -> bool:
        """Permanently terminate pod."""
        try:
            runpod.terminate_pod(pod_id)
            log.info(f"✅ Pod {pod_id} terminated")
            return True
        except Exception as e:
            log.error(f"Failed to terminate pod: {e}")
            return False

    def get_pod_status(self, pod_id: str) -> Dict:
        """Get pod status with metrics."""
        pod = self.get_pod(pod_id)

        if not pod:
            return {"status": "not_found"}

        runtime = pod.get("runtime", {})
        machine = pod.get("machine", {})

        return {
            "status": runtime.get("containerState", "unknown"),
            "uptime": runtime.get("uptimeInSeconds", 0),
            "gpu_utilization": runtime.get("gpuUtilization", 0),
            "memory_utilization": runtime.get("memoryUtilization", 0),
            "ssh_host": machine.get("podHostId", ""),
            "ssh_port": machine.get("podPort", 22),
            "cost_per_hour": pod.get("costPerHr", 0)
        }

    def wait_for_ready(self, pod_id: str, timeout: int = 300) -> bool:
        """
        Wait for pod to be ready.

        Args:
            pod_id: Pod ID
            timeout: Max wait time in seconds

        Returns:
            True if ready, False on timeout
        """
        start = time.time()

        while time.time() - start < timeout:
            status = self.get_pod_status(pod_id)

            if status["status"] == "running":
                log.info(f"✅ Pod {pod_id} is ready (uptime: {status['uptime']}s)")
                return True

            log.info(f"⏳ Waiting for pod... Status: {status['status']}")
            time.sleep(10)

        log.error(f"❌ Pod {pod_id} not ready after {timeout}s")
        return False
```

### 1.2 Configuration

**File**: `config/runpod_deployment.env`

```bash
# RunPod API Configuration
RUNPOD_API_KEY=your_api_key_here  # Get from https://runpod.io/settings

# Pod Configuration
RUNPOD_POD_NAME=rag-pipeline-vllm
RUNPOD_GPU_TYPE=NVIDIA RTX 4090
RUNPOD_GPU_COUNT=1
RUNPOD_VOLUME_GB=100

# Docker Image
RUNPOD_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel

# Exposed Ports
RUNPOD_PORTS=5432/tcp,8000/http,22/tcp,3000/http

# Auto-start script (on pod creation)
RUNPOD_STARTUP_SCRIPT=/workspace/rag-pipeline/scripts/runpod_startup_verbose.sh
```

---

## Phase 2: Deployment Automation

### 2.1 Automated Deployment Script

**File**: `scripts/deploy_to_runpod.py`

```python
#!/usr/bin/env python3
"""
Automated RunPod Deployment Script

Handles full deployment: create pod → setup environment → start services
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runpod_manager import RunPodManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def deploy_rag_pipeline(
    api_key: str,
    pod_name: str = "rag-pipeline-vllm",
    wait_for_ready: bool = True
):
    """
    Deploy complete RAG pipeline to RunPod.

    Steps:
        1. Create pod with RTX 4090
        2. Wait for pod to be ready
        3. Setup environment (PostgreSQL, vLLM)
        4. Start services
        5. Validate deployment
    """
    log.info("=" * 70)
    log.info("RUNPOD DEPLOYMENT - RAG Pipeline + vLLM")
    log.info("=" * 70)

    # Initialize manager
    manager = RunPodManager(api_key=api_key)

    # Step 1: Create pod
    log.info("\n🚀 Step 1: Creating RunPod pod...")

    pod = manager.create_pod(
        name=pod_name,
        gpu_type="NVIDIA RTX 4090",
        gpu_count=1,
        volume_gb=100,
        docker_args="bash /workspace/rag-pipeline/scripts/runpod_startup_verbose.sh"
    )

    if not pod:
        log.error("❌ Failed to create pod")
        return None

    pod_id = pod['id']
    ssh_host = pod['machine']['podHostId']

    log.info(f"✅ Pod created: {pod_id}")
    log.info(f"   SSH: ssh {ssh_host}@ssh.runpod.io")

    # Step 2: Wait for pod to be ready
    if wait_for_ready:
        log.info("\n⏳ Step 2: Waiting for pod to be ready...")

        if not manager.wait_for_ready(pod_id, timeout=300):
            log.error("❌ Pod failed to start")
            return None

    # Step 3: Get final status
    status = manager.get_pod_status(pod_id)

    log.info("\n✅ Deployment Complete!")
    log.info(f"   Pod ID: {pod_id}")
    log.info(f"   Status: {status['status']}")
    log.info(f"   SSH: ssh {ssh_host}@ssh.runpod.io")
    log.info(f"   Cost: ${status['cost_per_hour']:.2f}/hour")

    log.info("\n📝 Next Steps:")
    log.info("   1. SSH into pod and verify services")
    log.info("   2. Create SSH tunnel for vLLM: ssh -L 8000:localhost:8000 ...")
    log.info("   3. Test query: python rag_low_level_m1_16gb_verbose.py --query-only")

    return {
        "pod_id": pod_id,
        "ssh_host": ssh_host,
        "status": status
    }


def main():
    parser = argparse.ArgumentParser(description='Deploy RAG pipeline to RunPod')
    parser.add_argument('--api-key', required=True, help='RunPod API key')
    parser.add_argument('--pod-name', default='rag-pipeline-vllm', help='Pod name')
    parser.add_argument('--no-wait', action='store_true', help='Don\'t wait for pod to be ready')

    args = parser.parse_args()

    result = deploy_rag_pipeline(
        api_key=args.api_key,
        pod_name=args.pod_name,
        wait_for_ready=not args.no_wait
    )

    if result:
        print(f"\n✅ Deployment successful! Pod ID: {result['pod_id']}")
        sys.exit(0)
    else:
        print("\n❌ Deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 2.2 Usage

```bash
# One-command deployment
python scripts/deploy_to_runpod.py --api-key your_api_key_here

# Custom pod name
python scripts/deploy_to_runpod.py --api-key KEY --pod-name my-rag-prod
```

---

## Phase 3: Streamlit UI Integration

### 3.1 Deployment Tab in Streamlit

**File**: `rag_web.py` (add new tab)

```python
# Add to imports
from utils.runpod_manager import RunPodManager
import subprocess

# Add after existing tabs
def render_deployment_tab():
    """Render RunPod deployment management tab."""
    st.header("☁️ RunPod Deployment")

    # API Key input
    st.subheader("1. Configuration")

    api_key = st.text_input(
        "RunPod API Key",
        type="password",
        help="Get your API key from https://runpod.io/settings"
    )

    if not api_key:
        st.warning("⚠️ Enter your RunPod API key to continue")
        return

    # Initialize manager
    try:
        manager = RunPodManager(api_key=api_key)
        st.success("✅ API key validated")
    except Exception as e:
        st.error(f"❌ Invalid API key: {e}")
        return

    # List existing pods
    st.subheader("2. Existing Pods")

    pods = manager.list_pods()

    if pods:
        pod_data = []
        for pod in pods:
            runtime = pod.get('runtime', {})
            machine = pod.get('machine', {})

            pod_data.append({
                "Name": pod.get('name', 'N/A'),
                "Status": runtime.get('containerState', 'unknown'),
                "GPU": machine.get('gpuTypeId', 'N/A'),
                "Uptime": f"{runtime.get('uptimeInSeconds', 0) // 60}min",
                "Cost/hr": f"${pod.get('costPerHr', 0):.2f}",
                "ID": pod.get('id', '')[:12]
            })

        df = pd.DataFrame(pod_data)
        st.dataframe(df, use_container_width=True)

        # Pod management
        selected_pod = st.selectbox(
            "Select pod to manage",
            options=[p['id'] for p in pods],
            format_func=lambda x: next(p['name'] for p in pods if p['id'] == x)
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("▶️ Resume", key="resume"):
                with st.spinner("Resuming pod..."):
                    if manager.resume_pod(selected_pod):
                        st.success("Pod resumed!")
                        st.rerun()

        with col2:
            if st.button("⏸️ Stop", key="stop"):
                with st.spinner("Stopping pod..."):
                    if manager.stop_pod(selected_pod):
                        st.success("Pod stopped!")
                        st.rerun()

        with col3:
            if st.button("🗑️ Terminate", key="terminate"):
                if st.checkbox("Confirm termination"):
                    with st.spinner("Terminating pod..."):
                        if manager.terminate_pod(selected_pod):
                            st.success("Pod terminated!")
                            st.rerun()

    else:
        st.info("No existing pods found")

    # Create new pod
    st.subheader("3. Deploy New Pod")

    col1, col2 = st.columns(2)

    with col1:
        pod_name = st.text_input(
            "Pod Name",
            value="rag-pipeline-vllm",
            help="Unique name for your pod"
        )

        gpu_type = st.selectbox(
            "GPU Type",
            options=["NVIDIA RTX 4090", "NVIDIA RTX 4070 Ti", "NVIDIA RTX 3090"],
            index=0,
            help="RTX 4090 recommended for best performance"
        )

    with col2:
        volume_gb = st.number_input(
            "Storage (GB)",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Persistent storage for models and data"
        )

        auto_start = st.checkbox(
            "Auto-start services",
            value=True,
            help="Automatically start PostgreSQL and vLLM on pod creation"
        )

    if st.button("🚀 Deploy Pod", type="primary"):
        with st.spinner("Creating pod... This may take 2-3 minutes"):
            progress = st.progress(0)
            status_text = st.empty()

            # Create pod
            status_text.text("Creating pod on RunPod...")
            progress.progress(20)

            pod = manager.create_pod(
                name=pod_name,
                gpu_type=gpu_type,
                volume_gb=volume_gb,
                docker_args="bash /workspace/rag-pipeline/scripts/runpod_startup_verbose.sh" if auto_start else "bash"
            )

            if not pod:
                st.error("❌ Failed to create pod")
                return

            pod_id = pod['id']
            ssh_host = pod['machine']['podHostId']

            progress.progress(50)
            status_text.text("Waiting for pod to be ready...")

            # Wait for ready
            if manager.wait_for_ready(pod_id, timeout=180):
                progress.progress(100)
                status_text.text("✅ Pod is ready!")

                st.success("🎉 Deployment successful!")

                st.info(f"""
                **Pod Details:**
                - ID: `{pod_id}`
                - SSH: `ssh {ssh_host}@ssh.runpod.io`
                - vLLM Port: 8000
                - PostgreSQL Port: 5432

                **Next Steps:**
                1. Create SSH tunnel: `ssh -L 8000:localhost:8000 {ssh_host}@ssh.runpod.io`
                2. Test vLLM: `curl http://localhost:8000/health`
                3. Run query in "Querying" tab
                """)

                st.balloons()

            else:
                progress.progress(100)
                st.warning("⚠️ Pod created but not responding. Check SSH manually.")
                st.info(f"SSH: ssh {ssh_host}@ssh.runpod.io")


# Add tab to main UI
def main():
    st.title("🔍 RAG Pipeline Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Indexing",
        "💬 Querying",
        "📊 Analytics",
        "☁️ Deployment"  # New tab
    ])

    with tab1:
        render_indexing_tab()

    with tab2:
        render_querying_tab()

    with tab3:
        render_analytics_tab()

    with tab4:
        render_deployment_tab()  # New tab
```

### 3.2 UI Screenshots (Design)

```
┌────────────────────────────────────────────────────────────┐
│  ☁️ RunPod Deployment                                      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Configuration                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  RunPod API Key: ●●●●●●●●●●●●●●●●●●●●●●●●  🔑       │ │
│  └──────────────────────────────────────────────────────┘ │
│  ✅ API key validated                                       │
│                                                             │
│  2. Existing Pods                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Name              Status    GPU        Uptime   Cost  │ │
│  │ rag-pipeline-vllm Running   RTX 4090   45min  $0.50  │ │
│  │ test-pod          Stopped   RTX 3090   0min    -     │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  Select: [rag-pipeline-vllm ▼]                             │
│  [▶️ Resume]  [⏸️ Stop]  [🗑️ Terminate]                    │
│                                                             │
│  3. Deploy New Pod                                          │
│  Pod Name: [rag-pipeline-vllm___________]                  │
│  GPU Type: [NVIDIA RTX 4090 ▼]                             │
│  Storage:  [100___] GB                                      │
│  ☑ Auto-start services                                     │
│                                                             │
│  [🚀 Deploy Pod]                                            │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Phase 4: Monitoring & Health Checks

### 4.1 Health Check Module

**File**: `utils/runpod_health.py`

```python
"""
RunPod Health Monitoring

Checks pod health, service status, and GPU metrics.
"""
import requests
import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)


def check_vllm_health(host: str = "localhost", port: int = 8000) -> Dict:
    """Check vLLM server health."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)

        if response.status_code == 200:
            return {
                "status": "healthy",
                "latency_ms": response.elapsed.total_seconds() * 1000
            }
        else:
            return {
                "status": "unhealthy",
                "error": f"HTTP {response.status_code}"
            }

    except Exception as e:
        return {
            "status": "unreachable",
            "error": str(e)
        }


def check_postgres_health(host: str = "localhost", port: int = 5432) -> Dict:
    """Check PostgreSQL health."""
    import psycopg2

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=os.getenv("PGUSER", "fryt"),
            password=os.getenv("PGPASSWORD", "frytos"),
            database=os.getenv("DB_NAME", "vector_db"),
            connect_timeout=5
        )

        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

        conn.close()

        return {"status": "healthy"}

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def get_gpu_metrics(ssh_host: str) -> Optional[Dict]:
    """
    Get GPU metrics via SSH.

    Requires SSH tunnel or direct pod access.
    """
    # This would use paramiko or subprocess to SSH and run nvidia-smi
    # Simplified for brevity
    return {
        "gpu_utilization": 45,
        "memory_used_mb": 8192,
        "memory_total_mb": 24576,
        "temperature_c": 65,
        "power_watts": 250
    }
```

---

## Phase 5: Cost Optimization

### 5.1 Auto-Stop on Idle

**File**: `utils/cost_optimizer.py`

```python
"""
Cost Optimization for RunPod

Auto-stop pods when idle to save costs.
"""
import time
from datetime import datetime, timedelta
from utils.runpod_manager import RunPodManager


class CostOptimizer:
    """Optimize RunPod costs by auto-stopping idle pods."""

    def __init__(self, manager: RunPodManager):
        self.manager = manager
        self.last_query_time = {}

    def record_query(self, pod_id: str):
        """Record query timestamp for pod."""
        self.last_query_time[pod_id] = datetime.now()

    def check_idle_pods(self, idle_threshold_minutes: int = 30):
        """
        Check for idle pods and stop them.

        Args:
            idle_threshold_minutes: Minutes of inactivity before stop
        """
        pods = self.manager.list_pods()
        threshold = timedelta(minutes=idle_threshold_minutes)

        for pod in pods:
            pod_id = pod['id']
            runtime = pod.get('runtime', {})

            # Skip if already stopped
            if runtime.get('containerState') != 'running':
                continue

            # Check last query time
            last_query = self.last_query_time.get(pod_id)

            if last_query and datetime.now() - last_query > threshold:
                print(f"⚠️ Pod {pod['name']} idle for >{idle_threshold_minutes}min. Stopping...")
                self.manager.stop_pod(pod_id)
                print(f"✅ Pod stopped. Saved ${pod.get('costPerHr', 0):.2f}/hour")
```

### 5.2 Cost Tracking

```python
def calculate_monthly_cost(hours_per_day: float, cost_per_hour: float = 0.50) -> float:
    """Calculate monthly RunPod cost."""
    return hours_per_day * 30 * cost_per_hour


# Examples
print(f"8 hours/day: ${calculate_monthly_cost(8):.2f}/month")   # $120/month
print(f"24/7 usage: ${calculate_monthly_cost(24):.2f}/month")   # $360/month
print(f"2 hours/day: ${calculate_monthly_cost(2):.2f}/month")   # $30/month
```

---

## Testing & Validation

### Test Checklist

```bash
# 1. API Connection
python -c "from utils.runpod_manager import RunPodManager; m = RunPodManager('YOUR_API_KEY'); print(m.list_pods())"

# 2. Pod Creation (dry run would be manual verification)
python scripts/deploy_to_runpod.py --api-key YOUR_KEY --no-wait

# 3. Service Health Checks
python -c "from utils.runpod_health import check_vllm_health; print(check_vllm_health())"

# 4. SSH Tunnel
ssh -L 8000:localhost:8000 POD_HOST@ssh.runpod.io
curl http://localhost:8000/health

# 5. End-to-end Query
VLLM_API_BASE=http://localhost:8000/v1 python rag_low_level_m1_16gb_verbose.py --query-only --query "test"

# 6. GPU Monitoring
ssh POD_HOST@ssh.runpod.io "nvidia-smi"
```

---

## Production Checklist

### Pre-Deployment

- [ ] RunPod API key configured
- [ ] SSH keys generated and uploaded
- [ ] Environment variables validated
- [ ] Docker image tested locally
- [ ] Cost budget approved

### Deployment

- [ ] Pod created successfully
- [ ] Services started (PostgreSQL, vLLM)
- [ ] SSH tunnel established
- [ ] Health checks passing
- [ ] Test query successful

### Monitoring

- [ ] Grafana dashboard configured
- [ ] Cost tracking enabled
- [ ] Auto-stop idle pods configured
- [ ] Alert thresholds set

### Documentation

- [ ] Deployment guide updated
- [ ] Runbook created for incidents
- [ ] Team trained on UI controls
- [ ] Backup/recovery procedures documented

---

## Estimated Timeline

| Phase | Tasks | Effort | Dependencies |
|-------|-------|--------|--------------|
| **Phase 1** | RunPod API integration | 2-3 hours | None |
| **Phase 2** | Deployment automation | 2-3 hours | Phase 1 |
| **Phase 3** | Streamlit UI | 3-4 hours | Phase 2 |
| **Phase 4** | Monitoring | 1-2 hours | Phase 3 |
| **Phase 5** | Cost optimization | 1 hour | Phase 4 |
| **Testing** | Validation & docs | 2 hours | All phases |
| **Total** | **11-15 hours** | **Full implementation** | - |

### Minimum Viable Product (MVP)

**4-6 hours**:
- Phase 1: API integration (2-3h)
- Phase 2: Basic deployment script (2-3h)
- Skip Streamlit UI initially (use CLI)

### Production-Ready

**11-15 hours**: All phases + comprehensive testing

---

## Cost Analysis

### RunPod Costs

| Usage Pattern | Hours/Month | Cost/Month | Use Case |
|---------------|-------------|------------|----------|
| **Development** | 40h | **$20** | 2h/day during work |
| **Testing** | 80h | **$40** | 4h/day |
| **Light Production** | 160h | **$80** | 8h/day business hours |
| **Medium Production** | 360h | **$180** | 12h/day |
| **24/7 Production** | 720h | **$360** | Always-on service |

### Cost Optimization Strategies

1. **Auto-stop idle pods** → Save 40-60%
2. **Use Spot instances** → Save 50-70% (with interruptions)
3. **Schedule on/off** → Save 30-50% for business-hours usage
4. **Right-size GPU** → RTX 3090 vs 4090 saves 30%

### Recommended Strategy

**For Development/Testing**:
- Auto-stop after 30 min idle
- Use during work hours only
- **Cost**: $20-40/month

**For Production**:
- Schedule business hours (8h/day)
- Auto-scale based on query volume
- **Cost**: $80-120/month

---

## Next Steps

### Immediate (Phase 1-2)
1. Implement `RunPodManager` class
2. Create deployment automation script
3. Test pod creation and management

### Short-term (Phase 3)
1. Add deployment tab to Streamlit UI
2. Integrate SSH tunnel management
3. Add real-time pod monitoring

### Long-term (Phase 4-5)
1. Implement cost optimization
2. Add Grafana monitoring integration
3. Create comprehensive documentation

---

## Resources

### Documentation
- RunPod Python SDK: https://github.com/runpod/runpod-python
- RunPod GraphQL API: https://docs.runpod.io/sdks/graphql
- vLLM Documentation: `docs/VLLM_SERVER_GUIDE.md`
- RunPod Guide: `docs/RUNPOD_COMPLETE_GUIDE.md`

### Tools
- `utils/runpod_manager.py` - Pod management utilities
- `scripts/deploy_to_runpod.py` - Deployment automation
- `rag_web.py` - Streamlit UI with deployment tab

### Support
- RunPod Discord: https://discord.gg/runpod
- GitHub Issues: https://github.com/runpod/runpod-python/issues

---

**Status**: Ready for implementation
**Priority**: High (unlocks 15x LLM speedup + 215x query speedup)
**ROI**: Excellent (save dev time, enable production scale)

**Questions?** See troubleshooting section or consult deployment guides.
