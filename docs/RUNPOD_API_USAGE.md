# RunPod API Integration - Usage Guide

Complete guide to using the RunPod API integration for deploying RAG pipelines.

**Status**: Phase 1 Complete ✅
**Version**: 1.0.0

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Basic Usage](#basic-usage)
5. [Advanced Usage](#advanced-usage)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install RunPod SDK

```bash
pip install "runpod>=1.7.5"
```

### 2. Get API Key

Visit https://runpod.io/settings and copy your API key.

### 3. Test Connection

```bash
export RUNPOD_API_KEY=your_api_key_here
python scripts/test_runpod_connection.py
```

### 4. Create Your First Pod

```python
from utils.runpod_manager import RunPodManager

manager = RunPodManager(api_key="your_key")
pod = manager.create_pod(name="my-rag-pod")

# Wait for ready
if manager.wait_for_ready(pod['id']):
    print("Pod is ready!")

# Get SSH command
print(manager.get_ssh_command(pod['id']))
```

---

## Installation

### Requirements

- Python 3.11+
- RunPod account (https://runpod.io/signup)
- RunPod API key

### Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install "runpod>=1.7.5"
```

### Verify Installation

```python
import runpod
print(f"RunPod SDK version: {runpod.__version__}")
```

---

## Configuration

### Option 1: Environment Variable (Recommended)

```bash
# Add to .env file
echo "RUNPOD_API_KEY=your_api_key_here" >> .env

# Or export directly
export RUNPOD_API_KEY=your_api_key_here
```

### Option 2: Configuration File

```bash
# Edit config file
cp config/runpod_deployment.env.example config/runpod_deployment.env
nano config/runpod_deployment.env

# Set your API key
RUNPOD_API_KEY=your_api_key_here

# Load config
source config/runpod_deployment.env
```

### Option 3: Pass Directly

```python
from utils.runpod_manager import RunPodManager

manager = RunPodManager(api_key="your_api_key_here")
```

---

## Basic Usage

### Initialize Manager

```python
from utils.runpod_manager import RunPodManager

# Using env var
manager = RunPodManager()

# Or pass API key
manager = RunPodManager(api_key="your_key")
```

### List Existing Pods

```python
pods = manager.list_pods()

for pod in pods:
    print(f"Name: {pod['name']}")
    print(f"Status: {pod['runtime']['containerState']}")
    print(f"GPU: {pod['machine']['gpuTypeId']}")
    print()
```

### Create Pod

```python
pod = manager.create_pod(
    name="my-rag-pipeline",
    gpu_type="NVIDIA RTX 4090",
    volume_gb=100
)

print(f"Pod ID: {pod['id']}")
print(f"SSH: ssh {pod['machine']['podHostId']}@ssh.runpod.io")
```

### Wait for Pod to be Ready

```python
if manager.wait_for_ready(pod['id'], timeout=300):
    print("✅ Pod is ready!")
else:
    print("❌ Pod failed to start")
```

### Get Pod Status

```python
status = manager.get_pod_status(pod['id'])

print(f"Status: {status['status']}")
print(f"Uptime: {status['uptime_seconds']}s")
print(f"GPU Usage: {status['gpu_utilization']}%")
print(f"Cost: ${status['cost_per_hour']}/hour")
```

### Stop Pod (Save Costs)

```python
manager.stop_pod(pod['id'])
# Pod is stopped, no GPU costs
# Storage costs still apply
```

### Resume Pod

```python
manager.resume_pod(pod['id'])
# Pod is running again
```

### Terminate Pod (Permanent)

```python
manager.terminate_pod(pod['id'])
# Pod is deleted, all data lost
# No more costs
```

---

## Advanced Usage

### Custom Environment Variables

```python
pod = manager.create_pod(
    name="my-custom-pod",
    env={
        "USE_VLLM": "1",
        "VLLM_MODEL": "mistralai/Mistral-7B-Instruct-v0.2",
        "CTX": "16384",
        "TOP_K": "10",
        "CUSTOM_VAR": "my_value"
    }
)
```

### Custom Docker Command

```python
pod = manager.create_pod(
    name="my-pod",
    docker_args="bash /workspace/startup.sh"
)
```

### SSH Connection

```python
# Get SSH command
ssh_cmd = manager.get_ssh_command(pod['id'])
print(ssh_cmd)
# Output: ssh -L 8000:localhost:8000 -L 5432:localhost:5432 pod@ssh.runpod.io

# Custom port forwarding
ssh_cmd = manager.get_ssh_command(pod['id'], ports=[8000, 5432, 3000])
```

### Cost Estimation

```python
# Estimate monthly cost
costs = manager.estimate_cost(
    hours_per_day=8,
    days=30,
    cost_per_hour=0.50
)

print(f"Daily cost: ${costs['daily_cost']:.2f}")
print(f"Monthly cost: ${costs['total_cost']:.2f}")
```

### List Available GPUs

```python
gpus = manager.list_available_gpus()

for gpu in gpus:
    name = gpu['displayName']
    memory = gpu['memoryInGb']
    price = gpu['lowestPrice']['uninterruptablePrice']

    print(f"{name}: {memory}GB VRAM, ${price:.2f}/hour")
```

---

## Examples

### Example 1: Create and Monitor Pod

```python
from utils.runpod_manager import RunPodManager
import time

manager = RunPodManager()

# Create pod
print("Creating pod...")
pod = manager.create_pod(name="rag-prod")

if not pod:
    print("Failed to create pod")
    exit(1)

pod_id = pod['id']

# Wait for ready
print("Waiting for pod...")
if manager.wait_for_ready(pod_id):
    print("✅ Pod is ready!")

    # Get status
    status = manager.get_pod_status(pod_id)
    print(f"\nPod Status:")
    print(f"  GPU: {status['gpu_type']}")
    print(f"  Usage: {status['gpu_utilization']}%")
    print(f"  Cost: ${status['cost_per_hour']}/hour")

    # Get SSH command
    ssh_cmd = manager.get_ssh_command(pod_id)
    print(f"\nSSH Command:")
    print(f"  {ssh_cmd}")
else:
    print("❌ Pod failed to start")
```

### Example 2: Manage Multiple Pods

```python
from utils.runpod_manager import RunPodManager

manager = RunPodManager()

# List all pods
pods = manager.list_pods()

print(f"Found {len(pods)} pods:")

for pod in pods:
    name = pod['name']
    pod_id = pod['id']
    runtime = pod.get('runtime', {})
    state = runtime.get('containerState', 'unknown')

    print(f"\n{name} ({pod_id[:8]})")
    print(f"  Status: {state}")

    if state == "running":
        # Get detailed status
        status = manager.get_pod_status(pod_id)
        print(f"  GPU: {status['gpu_utilization']}%")
        print(f"  Cost: ${status['cost_per_hour']}/hour")
    elif state == "stopped":
        print(f"  ⏸️  Pod is stopped (not incurring GPU costs)")
```

### Example 3: Auto-Stop Idle Pods

```python
from utils.runpod_manager import RunPodManager
import time

manager = RunPodManager()

# Monitor pods and auto-stop idle ones
idle_threshold = 30 * 60  # 30 minutes

while True:
    pods = manager.list_pods()

    for pod in pods:
        runtime = pod.get('runtime', {})
        state = runtime.get('containerState')

        if state == 'running':
            # Check if idle (no queries in last 30 min)
            # (You'd track this separately in practice)
            pod_id = pod['id']
            status = manager.get_pod_status(pod_id)

            # If GPU idle and been running > 30 min
            if status['gpu_utilization'] < 5 and status['uptime_seconds'] > idle_threshold:
                print(f"⚠️  Pod {pod['name']} is idle. Stopping...")
                manager.stop_pod(pod_id)
                print("✅ Pod stopped (saved ${:.2f}/hour)".format(status['cost_per_hour']))

    time.sleep(300)  # Check every 5 minutes
```

### Example 4: Create Pod with Full Configuration

```python
from utils.runpod_manager import RunPodManager

manager = RunPodManager()

# Create fully configured pod
pod = manager.create_pod(
    name="rag-pipeline-prod",
    gpu_type="NVIDIA RTX 4090",
    gpu_count=1,
    image="runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel",
    volume_gb=100,
    container_disk_gb=50,
    ports="5432/tcp,8000/http,22/tcp,3000/http",
    env={
        # vLLM
        "USE_VLLM": "1",
        "VLLM_MODEL": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "VLLM_PORT": "8000",

        # Embeddings
        "EMBED_BACKEND": "torch",
        "EMBED_MODEL": "BAAI/bge-small-en",
        "EMBED_BATCH": "128",

        # PostgreSQL
        "PGHOST": "localhost",
        "PGPORT": "5432",
        "PGUSER": "fryt",
        "PGPASSWORD": "secure_password",
        "DB_NAME": "vector_db",

        # RAG
        "CHUNK_SIZE": "700",
        "TOP_K": "5",
        "CTX": "8192"
    },
    docker_args="bash /workspace/rag-pipeline/scripts/runpod_startup_verbose.sh"
)

if pod:
    print("✅ Pod created successfully!")
    print(f"   ID: {pod['id']}")
    print(f"   SSH: ssh {pod['machine']['podHostId']}@ssh.runpod.io")

    # Wait and get status
    if manager.wait_for_ready(pod['id']):
        status = manager.get_pod_status(pod['id'])
        print(f"\n📊 Pod Status:")
        print(f"   Running: {status['uptime_seconds']}s")
        print(f"   GPU: {status['gpu_type']}")
        print(f"   Cost: ${status['cost_per_hour']}/hour")
```

---

## Troubleshooting

### Error: API Key Not Found

```
ValueError: RunPod API key not found
```

**Solution**:
```bash
export RUNPOD_API_KEY=your_api_key_here
# Or pass directly to RunPodManager
```

### Error: Pod Creation Failed

```
Failed to create pod: GPU type not available
```

**Solution**:
```python
# Check available GPUs
manager = RunPodManager()
gpus = manager.list_available_gpus()

for gpu in gpus:
    print(f"{gpu['displayName']}: Available")

# Try different GPU type
pod = manager.create_pod(gpu_type="NVIDIA RTX 3090")
```

### Error: Pod Not Ready After Timeout

```
Pod abc123 not ready after 300s
```

**Solution**:
```python
# Increase timeout
manager.wait_for_ready(pod_id, timeout=600)  # 10 minutes

# Or check pod manually
status = manager.get_pod_status(pod_id)
print(f"Status: {status['status']}")

# SSH and check logs
ssh_cmd = manager.get_ssh_command(pod_id)
print(f"SSH: {ssh_cmd}")
```

### Error: Import Error

```
ImportError: runpod package not installed
```

**Solution**:
```bash
pip install "runpod>=1.7.5"
```

### Error: Connection Timeout

```
Failed to list pods: Connection timeout
```

**Solution**:
```python
# Check internet connection
# Verify API key is valid
# Try again (RunPod API may be temporarily down)

import time
time.sleep(10)  # Wait and retry
pods = manager.list_pods()
```

---

## Best Practices

### 1. Always Stop Pods When Not in Use

```python
# Stop pod when done
manager.stop_pod(pod_id)

# Or use context manager pattern
class PodContext:
    def __init__(self, manager, pod_id):
        self.manager = manager
        self.pod_id = pod_id

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.manager.stop_pod(self.pod_id)
```

### 2. Monitor Costs

```python
# Track running pods and costs
total_cost = 0

for pod in manager.list_pods():
    runtime = pod.get('runtime', {})
    if runtime.get('containerState') == 'running':
        cost = pod.get('costPerHr', 0)
        total_cost += cost

print(f"Current hourly cost: ${total_cost:.2f}/hour")
print(f"Estimated monthly (24/7): ${total_cost * 24 * 30:.2f}")
```

### 3. Use Persistent Volumes

```python
# Always specify volume_gb for data persistence
pod = manager.create_pod(
    name="my-pod",
    volume_gb=100  # Persistent storage
)
```

### 4. Validate Pod Readiness

```python
# Always wait for ready before using
pod = manager.create_pod("my-pod")

if manager.wait_for_ready(pod['id']):
    # Pod is ready, safe to use
    run_queries(pod)
else:
    # Handle failure
    log.error("Pod failed to start")
    manager.terminate_pod(pod['id'])
```

---

## API Reference

See `utils/runpod_manager.py` for complete API documentation.

### Key Methods

- `list_pods()` - List all pods
- `get_pod(pod_id)` - Get pod details
- `create_pod(**kwargs)` - Create new pod
- `stop_pod(pod_id)` - Stop pod (save costs)
- `resume_pod(pod_id)` - Resume stopped pod
- `terminate_pod(pod_id)` - Permanently delete pod
- `get_pod_status(pod_id)` - Get status with metrics
- `wait_for_ready(pod_id)` - Wait for pod to be running
- `get_ssh_command(pod_id)` - Generate SSH command
- `estimate_cost()` - Calculate costs
- `list_available_gpus()` - List GPU types

---

## Next Steps

1. **Test Connection**: Run `python scripts/test_runpod_connection.py`
2. **Create Pod**: Try example code above
3. **Deploy RAG**: Follow `docs/RUNPOD_DEPLOYMENT_WORKFLOW.md`
4. **Add to Streamlit**: Phase 3 integration

---

## Resources

- **RunPod Docs**: https://docs.runpod.io/
- **Python SDK**: https://github.com/runpod/runpod-python
- **API Reference**: https://graphql-spec.runpod.io/
- **Support**: https://discord.gg/runpod

---

**Status**: Phase 1 Complete ✅
**Next**: Phase 2 - Deployment Automation
