# Complete RunPod GPU Deployment Guide

**Last Updated:** 2026-01-07
**Status:** Production-ready with all bugs fixed
**Version:** 2.0 - Comprehensive Edition

---

## Table of Contents

1. [Strategic Overview](#1-strategic-overview)
2. [Quick Start](#2-quick-start)
3. [Deployment Methods](#3-deployment-methods)
4. [Verification](#4-verification)
5. [Troubleshooting](#5-troubleshooting)
6. [Cost Analysis](#6-cost-analysis)
7. [Production Deployment](#7-production-deployment)
8. [Performance Benchmarks](#8-performance-benchmarks)
9. [Additional Resources](#9-additional-resources)

---

## 1. Strategic Overview

<!-- Source: RUNPOD_DEPLOYMENT_GUIDE.md -->

### 1.1 Why Use Cloud GPUs?

#### Current Setup: M1 Mac Mini 16GB

**Pros:**
- Local, private, no egress costs
- Apple Metal GPU decent for inference
- Zero marginal cost after hardware purchase

**Cons:**
- Limited VRAM (shared with system RAM)
- Slower than dedicated NVIDIA GPUs (~3-5x)
- Cannot scale (1 machine limit)
- MLX ecosystem less mature than CUDA

#### Cloud GPU: RTX 4090 24GB

**Pros:**
- 24GB dedicated VRAM
- 10-20x faster LLM inference
- Mature CUDA ecosystem
- Scale to multiple instances
- Pay only when running

**Cons:**
- Costs ~$0.50/hour
- Data transfer costs (if large datasets)
- Internet dependency

### 1.2 Service Comparison Matrix

#### RunPod (Recommended)

```
Pricing: $0.19-0.69/h (RTX 3060-4090)
Payment: Credit card, pay-per-second
Setup: Docker-based, templates available
Storage: Persistent volumes (paid separately)
Network: Fast, reliable uptime
Support: Good community, Discord active

Best for: Docker deployment, flexible GPU choice
```

#### Vast.ai (Budget Option)

```
Pricing: $0.10-0.40/h (marketplace, varies)
Payment: BTC/ETH/Credit card
Setup: SSH access, more manual
Storage: Provider-dependent
Network: Variable (peer-to-peer marketplace)
Support: Community forums

Best for: Cost-sensitive, comfortable with SSH/manual setup
```

#### Lambda Labs

```
Pricing: $0.50-1.50/h (fixed pricing)
Payment: Credit card, monthly billing
Setup: Very simple UI
Storage: Persistent, included
Network: Excellent, professional
Support: Excellent, enterprise-grade

Best for: Teams, production workloads, simplicity
```

#### Modal (Serverless)

```
Pricing: ~$0.60/h + per-request
Payment: Credit card, pay-as-you-go
Setup: Python decorators, zero DevOps
Storage: Automatic, ephemeral
Network: Auto-scaling, global
Support: Good docs, Slack community

Best for: API endpoints, auto-scaling, modern stack
```

### 1.3 GPU Selection Guide

**Recommended for RAG Pipeline:**

| GPU | VRAM | Price/h | When to Use |
|-----|------|---------|-------------|
| RTX 3060 | 12GB | ~$0.19 | Testing, small models |
| RTX 4070 Ti | 12GB | ~$0.29 | Budget production |
| **RTX 4090** | 24GB | **~$0.50** | **Production (recommended)** |
| RTX 6000 Ada | 48GB | ~$0.89 | Large models (overkill for Mistral 7B) |
| A100 (40GB) | 40GB | ~$1.50 | Enterprise (overkill) |

**For Mistral 7B + BGE embeddings: RTX 4090 is the sweet spot**

---

## 2. Quick Start

<!-- Source: RUNPOD_FINAL_SETUP.md -->

### 2.1 One-Command Deployment (Recommended)

This is the fastest way to get your RAG pipeline running on RunPod with all bugs fixed.

#### Step 1: Create RunPod Pod

**In RunPod UI:**

1. **Template:** RunPod PyTorch 2.4.0 (CUDA 12.4)
2. **GPU:** RTX 4090 (24GB VRAM)
3. **Container Disk:** 50 GB
4. **Volume Disk:** 100 GB (optional, for persistence)
5. **Expose Ports:** `5432,8000,22`

#### Step 2: Environment Variables

**Option A - Private Repository (Recommended):**

Get your GitHub token:
```bash
gh auth token
```

Add environment variable in RunPod UI:
```
Key: GH_TOKEN
Value: ghp_your_token_here
```

**Option B - Public Repository (Temporary):**
```bash
gh repo edit llamaIndex-local-rag --visibility public
```

#### Step 3: Docker Command

**Copy this EXACT command for private repo:**

```bash
bash -c "apt-get update -qq && apt-get install -y git && rm -rf /workspace/rag-pipeline && git clone https://\${GH_TOKEN}@github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && chmod +x scripts/runpod_startup.sh && SETUP_POSTGRES=1 DOWNLOAD_MODELS=1 bash scripts/runpod_startup.sh"
```

**For public repo:**

```bash
bash -c "apt-get update -qq && apt-get install -y git && rm -rf /workspace/rag-pipeline && git clone https://github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && chmod +x scripts/runpod_startup.sh && SETUP_POSTGRES=1 DOWNLOAD_MODELS=1 bash scripts/runpod_startup.sh"
```

#### Step 4: Deploy

Click **"Deploy On-Demand"**

### 2.2 Startup Timeline (~5-7 minutes)

Here's what you'll see in the logs:

```
[0:00] ==========
       == CUDA ==
       ==========
       CUDA Version 12.4.1

[0:10] Cloning into '/workspace/rag-pipeline'...
       🚀 RAG Pipeline Auto-Startup
       📊 System Information:
       NVIDIA GeForce RTX 4090, 24564 MiB

[0:15] 📦 Installing dependencies...
       [1/3] Upgrading pip...
       Successfully installed pip-25.3

[0:20] [2/3] Installing requirements.txt...
       ────────────────────────────────────
       Collecting aiohappyeyeballs==2.6.1
       Collecting aiohttp==3.13.2
       ... (100+ packages)

[3:00] Downloading torch-2.9.1... (900 MB)
       Downloading nvidia-cudnn-cu12... (700 MB)
       ... (downloading ~3.5 GB CUDA libraries)

[5:00] ✅ Requirements installed

[5:10] [3/3] PyTorch with CUDA already installed ✅
       ⚙️  Loading configuration...

[5:15] 🐘 Setting up PostgreSQL...
       Installing PostgreSQL...
       🔨 Compiling pgvector from source...
       Starting PostgreSQL...
       CREATE DATABASE
       CREATE ROLE
       GRANT
       CREATE EXTENSION
       ✅ PostgreSQL ready

[5:30] 🧪 Testing GPU + PyTorch...
       ✅ PyTorch: 2.9.1+cu128
       ✅ CUDA: 12.8
       ✅ GPU: NVIDIA GeForce RTX 4090
       ✅ VRAM: 25.4 GB

[6:00] 📥 Pre-downloading models...
       Downloading BAAI/bge-small-en...
       Fetching 13 files: 100%|██████████| 13/13
       ✅ Model cached

[6:30] ================================================
       ✅ RAG Pipeline Ready!
       ================================================

       🔄 Keeping container alive...
       SSH into the pod to use it

       [Container stays running - NO MORE RESTARTS!]
```

---

## 3. Deployment Methods

<!-- Sources: All 3 files combined and translated -->

### 3.1 Method 1: One-Liner Command (Recommended)

**Best for:** Quick deployment, automated setup

This method uses the Docker command field in RunPod UI to automatically:
- Install git
- Clone your repository
- Run the startup script
- Setup PostgreSQL and download models

**See Section 2.1 for complete instructions.**

### 3.2 Method 2: Custom Docker Image

<!-- Source: RUNPOD_DEPLOYMENT_GUIDE.md -->

**Best for:** Production, reproducible deployments, version control

#### Create Dockerfile

Create `Dockerfile.runpod`:

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    postgresql-client \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY . .

# Download models to cache (optional, speeds up first run)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('BAAI/bge-small-en', cache_dir='/root/.cache/huggingface')"

# Environment variables for GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV EMBED_BACKEND=torch
ENV N_GPU_LAYERS=99
ENV N_BATCH=512
ENV CTX=16384

# PostgreSQL connection (override at runtime)
ENV PGHOST=localhost
ENV PGPORT=5432
ENV PGUSER=fryt
ENV PGPASSWORD=frytos
ENV DB_NAME=vector_db

# Expose ports
EXPOSE 5432 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["python3", "rag_low_level_m1_16gb_verbose.py", "--query-only"]
```

#### Build and Push

```bash
# Build image
docker build -f Dockerfile.runpod -t your-dockerhub-username/rag-pipeline:latest .

# Test locally (if you have NVIDIA GPU)
docker run --gpus all -it your-dockerhub-username/rag-pipeline:latest

# Push to Docker Hub
docker login
docker push your-dockerhub-username/rag-pipeline:latest
```

#### Deploy on RunPod

1. In RunPod UI: **Deploy** → **GPU Pods**
2. Select **"Custom"** template
3. Docker Image: `your-dockerhub-username/rag-pipeline:latest`
4. GPU: RTX 4090
5. Environment variables:
   ```
   PGHOST=your-postgres-host
   PGPASSWORD=your-password
   PGTABLE=your_table
   ```
6. Deploy!

### 3.3 Method 3: Manual Setup via SSH

<!-- Source: RUNPOD_STARTUP_INSTRUCTIONS.md - Translated from French -->

**Best for:** Learning, customization, debugging

#### Step 1: Create Basic Pod

1. Deploy a pod with PyTorch 2.4.0 template
2. Select RTX 4090 GPU
3. No Docker command needed
4. Click Deploy

#### Step 2: SSH and Manual Setup

```bash
# SSH into the pod
ssh root@your-pod-ip -p your-port

# Clone the repository
rm -rf /workspace/rag-pipeline
git clone https://github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline
cd /workspace/rag-pipeline

# Make startup script executable
chmod +x scripts/runpod_startup.sh

# Run the setup script
bash scripts/runpod_startup.sh
```

#### Step 3: Setup with Options

```bash
# Complete setup with PostgreSQL and model downloads
SETUP_POSTGRES=1 \
DOWNLOAD_MODELS=1 \
bash scripts/runpod_startup.sh
```

### 3.4 Method 4: Persistent Volume Configuration

<!-- Source: RUNPOD_STARTUP_INSTRUCTIONS.md - Translated from French -->

**Best for:** Cost optimization, faster restarts, data persistence

#### Why Use Persistent Volumes?

Persistent volumes retain data between pod restarts:
- `/workspace/rag-pipeline` (your code)
- `/workspace/huggingface_cache` (pre-downloaded models)
- `/var/lib/postgresql` (database data)

**Benefits:**
- Avoid re-downloading 3.5GB packages every time
- Startup time: 30 seconds instead of 7 minutes
- Preserve your data and indexes

#### Setup Instructions

**In RunPod UI:**

1. Go to **Storage** → **Network Volumes**
2. Create a new volume: 100 GB
3. When deploying pod, mount at: `/workspace`
4. First deployment: Full setup (7 minutes)
5. Subsequent deployments: Fast startup (30 seconds)

**Environment Variables:**

```bash
SETUP_POSTGRES=1
DOWNLOAD_MODELS=1
HF_HOME=/workspace/huggingface_cache
```

**Docker Command (same as Method 1):**

```bash
bash -c "apt-get update -qq && apt-get install -y git && rm -rf /workspace/rag-pipeline && git clone https://github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && chmod +x scripts/runpod_startup.sh && SETUP_POSTGRES=1 DOWNLOAD_MODELS=1 bash scripts/runpod_startup.sh"
```

### 3.5 Advanced Configuration Options

<!-- Source: RUNPOD_STARTUP_INSTRUCTIONS.md - Translated from French -->

**Available Environment Variables:**

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `REPO_URL` | Git repository URL | https://github.com/frytos/llamaIndex-local-rag.git | Your fork URL |
| `SETUP_POSTGRES` | Install PostgreSQL locally | 0 | `1` (to enable) |
| `DOWNLOAD_MODELS` | Pre-download embedding models | 0 | `1` (to enable) |
| `RUN_COMMAND` | Command to run after setup | (none) | `python3 rag_low_level_m1_16gb_verbose.py --help` |
| `KEEP_ALIVE` | Keep container active | 0 | `1` (for debugging) |
| `EMBED_BACKEND` | Embedding backend | torch | `torch` or `mlx` |
| `N_GPU_LAYERS` | GPU layers to offload | 24 | `99` (for full offload) |
| `N_BATCH` | LLM batch size | 256 | `512` (for more VRAM) |
| `CTX` | Context window size | 3072 | `16384` (for larger context) |
| `HF_HOME` | HuggingFace cache directory | /root/.cache | `/workspace/huggingface_cache` |

**Full Configuration Example:**

```bash
SETUP_POSTGRES=1
DOWNLOAD_MODELS=1
EMBED_BACKEND=torch
N_GPU_LAYERS=99
N_BATCH=512
CTX=16384
PGHOST=localhost
PGUSER=fryt
PGPASSWORD=frytos
DB_NAME=vector_db
PGTABLE=messenger_runpod
HF_HOME=/workspace/huggingface_cache
```

---

## 4. Verification

<!-- Source: RUNPOD_FINAL_SETUP.md -->

### 4.1 SSH into Pod

After setup completes:

```bash
ssh <your-pod-id>@ssh.runpod.io -i ~/.ssh/runpod_key
```

### 4.2 Run Verification Script

```bash
cd /workspace/rag-pipeline
source .venv/bin/activate

# Run comprehensive verification
bash scripts/verify_runpod_setup.sh
```

**Expected output:**

```
🔍 RunPod Setup Verification Script
====================================

🐘 [1/6] Restarting PostgreSQL... ✅
🔌 [2/6] Creating pgvector extension... ✅
✅ [3/6] Verifying pgvector extension... ✅
👤 [4/6] Testing user connection... ✅
🎮 [5/6] Testing GPU + PyTorch... ✅
🧪 [6/6] Testing RAG script... ✅

================================================
✅ ALL CHECKS PASSED!
================================================
```

### 4.3 Manual Verification Steps

```bash
cd /workspace/rag-pipeline
source .venv/bin/activate

# Test GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: GPU: NVIDIA GeForce RTX 4090

# Test PyTorch CUDA
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Expected: CUDA Available: True

# Test PostgreSQL connection
psql -h localhost -U fryt -d vector_db -c "SELECT version();"
# Expected: PostgreSQL version info

# Test pgvector extension
psql -h localhost -U fryt -d vector_db -c "SELECT * FROM pg_extension WHERE extname='vector';"
# Expected: Row showing vector extension

# Test RAG script
python3 rag_low_level_m1_16gb_verbose.py --help
# Expected: Help text with all options
```

### 4.4 Test Your First Query

```bash
# Quick test query (query-only mode, no indexing)
python3 rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "test query"

# Full test with small dataset
PDF_PATH=data/sample.pdf \
PGTABLE=test_index \
RESET_TABLE=1 \
python3 rag_low_level_m1_16gb_verbose.py
```

---

## 5. Troubleshooting

<!-- Source: RUNPOD_FINAL_SETUP.md -->

### 5.1 All Fixed Issues (Production-Ready)

#### Issue 1: git clone "directory exists"

**Symptom:** `fatal: destination path '/workspace/rag-pipeline' already exists`

**Fix Applied:** Added `rm -rf /workspace/rag-pipeline` before clone

**Why it happened:** Pod restarts or previous deployments left the directory

**Status:** ✅ Fixed in one-liner command

---

#### Issue 2: sudo command not found

**Symptom:** `sudo: command not found` when trying to run PostgreSQL commands

**Fix Applied:** Use `su - postgres -c "command"` instead of `sudo -u postgres command`

**Why it happened:** RunPod's base images don't include sudo by default

**Status:** ✅ Fixed in startup script

---

#### Issue 3: pgvector extension missing

**Symptom:** `ERROR: could not open extension control file "/usr/share/postgresql/14/extension/vector.control"`

**Fix Applied:** Compile pgvector from source (v0.7.4)

```bash
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make && make install
```

**Why it happened:** pgvector not included in PostgreSQL by default

**Status:** ✅ Fixed in startup script (automated compilation)

---

#### Issue 4: Silent installation (can't tell if stuck)

**Symptom:** Script appears frozen during package installation

**Fix Applied:** Show package installation progress with grep filters

```bash
pip install -r requirements.txt 2>&1 | grep -E "Collecting|Downloading|Installing|Successfully"
```

**Why it happened:** pip's progress bars don't work well in non-interactive terminals

**Status:** ✅ Fixed in startup script (verbose logging)

---

#### Issue 5: Infinite restart loop (CRITICAL)

**Symptom:** Container exits immediately after setup completes, triggering restart loop

**Fix Applied:** Add `tail -f /dev/null` at end to keep container running

```bash
echo "🔄 Keeping container alive (press Ctrl+C to exit)..."
tail -f /dev/null
```

**Why it happened:** Docker containers exit when the main process completes

**Status:** ✅ Fixed in startup script

**This was the critical bug preventing production use!**

---

#### Issue 6: verify script venv activation

**Symptom:** Verification script couldn't find Python packages

**Fix Applied:** Use direct path `/workspace/rag-pipeline/.venv/bin/python3`

```bash
# Instead of:
source .venv/bin/activate && python3 script.py

# Use:
/workspace/rag-pipeline/.venv/bin/python3 script.py
```

**Why it happened:** Sourcing venv in a script doesn't persist to subprocess calls

**Status:** ✅ Fixed in verification script

---

### 5.2 Common Issues and Solutions

#### Container Still Restarts

**Check:** The script should end with:
```
🔄 Keeping container alive (press Ctrl+C to exit)...
```

**If you don't see this:**
1. Make sure you pulled the latest code
2. Check that `tail -f /dev/null` is at the end of `scripts/runpod_startup.sh`
3. SSH into pod and run verification script

---

#### pgvector Installation Failed

**Check if installed:**
```bash
ls -la /usr/share/postgresql/14/extension/vector.control
```

**Manual installation:**
```bash
cd /tmp
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make && make install
service postgresql restart
```

---

#### CUDA Out of Memory

<!-- Source: RUNPOD_DEPLOYMENT_GUIDE.md -->

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Process killed during embedding/inference

**Solutions:**

```bash
# Reduce batch size
export N_BATCH=256  # Default was 512

# Reduce context window
export CTX=8192  # Default was 16384

# Use smaller quantization
# Q4_K_M → Q3_K_M (smaller, faster, slight quality loss)

# Reduce embedding batch size
export EMBED_BATCH=32  # Default was 64
```

---

#### Connection to PostgreSQL Refused

**Check if running:**
```bash
service postgresql status
```

**Start if stopped:**
```bash
service postgresql start
```

**Test connection:**
```bash
psql -h $PGHOST -p $PGPORT -U $PGUSER -d $DB_NAME
```

**If using external managed Postgres:**
- Whitelist RunPod's IP in your firewall
- Verify credentials in environment variables

---

#### Model Download Timeout

**Pre-download in Dockerfile:**
```bash
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('BAAI/bge-small-en')"
```

**Or use persistent volume:**
```bash
export HF_HOME=/workspace/huggingface_cache
```

---

#### PyTorch CUDA Not Available

**Verify CUDA setup:**
```bash
nvidia-smi  # Should show RTX 4090
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
```

**If False:**
- Check template: Must be PyTorch 2.4.0 with CUDA
- Reinstall PyTorch:
  ```bash
  pip3 install torch --index-url https://download.pytorch.org/whl/cu121
  ```

---

#### git clone Still Fails (Private Repo)

**Check token:**
```bash
echo $GH_TOKEN  # Should show your token
```

**Test clone manually:**
```bash
git clone https://${GH_TOKEN}@github.com/frytos/llamaIndex-local-rag.git /tmp/test
```

**If fails:**
- Generate new token: `gh auth token`
- Check token has `repo` scope
- Or temporarily make repo public

---

#### Slow Network/Downloads

**Symptoms:** Setup takes > 15 minutes

**Solutions:**
- Use persistent volume to cache downloads
- Pre-build Docker image with models included
- Choose RunPod datacenter closer to you

---

### 5.3 Debugging Checklist

If something goes wrong:

```bash
# 1. Check system info
nvidia-smi
df -h  # Check disk space

# 2. Check logs
cat /workspace/rag-pipeline/setup.log
tail -f /var/log/postgresql/postgresql-14-main.log

# 3. Verify dependencies
cd /workspace/rag-pipeline
source .venv/bin/activate
pip list | grep -E "torch|llama|postgres"

# 4. Test each component
python3 -c "import torch; print(torch.cuda.is_available())"
psql -h localhost -U fryt -d vector_db -c "SELECT 1;"
python3 -c "from llama_index.embeddings.huggingface import HuggingFaceEmbedding"

# 5. Run verification
bash scripts/verify_runpod_setup.sh
```

---

## 6. Cost Analysis

<!-- Sources: RUNPOD_FINAL_SETUP.md + RUNPOD_DEPLOYMENT_GUIDE.md -->

### 6.1 Real Testing Costs

**Your testing session:**
- Time spent: ~30 minutes debugging
- GPU: RTX 4090 @ $0.50/hour
- **Total: ~$0.25**

**Per successful deployment:**
- Setup time: 5-7 minutes
- Cost: ~$0.05
- After setup: $0.50/hour while running

### 6.2 Usage Scenarios

#### Scenario 1: Development/Testing

**Usage pattern:** 2 hours/day, 20 days/month

```
GPU: RTX 4090 @ $0.50/h
Time: 40 hours/month
GPU Cost: $20/month

Storage: 100 GB @ $0.10/GB/month
Storage Cost: $10/month

Total: ~$30/month
```

**vs M1 Mac Mini:** $0/month (already owned)
**Trade-off:** 10x faster development for $30/month

---

#### Scenario 2: Production API (24/7)

**Usage pattern:** Always-on service

```
GPU: RTX 4090 @ $0.50/h
Time: 720 hours/month
GPU Cost: $360/month

Storage: 500 GB @ $0.10/GB/month
Storage Cost: $50/month

Total: ~$410/month
```

**Optimization - Spot Instances:**
```
Price: ~$0.25/h (50% cheaper)
Total: ~$230/month
Trade-off: Can be interrupted (need restart logic)
```

---

#### Scenario 3: Batch Processing

**Usage pattern:** Run queries in batches, 4 hours/week

```
GPU: RTX 4090 @ $0.50/h
Time: 16 hours/month
GPU Cost: $8/month

Storage: 50 GB @ $0.10/GB/month
Storage Cost: $5/month

Total: ~$13/month
```

**Best approach:** Start pod when needed, stop when done

---

#### Scenario 4: Benchmarking (One-Time)

**Usage pattern:** Run performance comparisons

```
Setup: 7 minutes @ $0.50/h = $0.06
Indexing: 2 hours @ $0.50/h = $1.00
Queries: 30 minutes @ $0.50/h = $0.25

Total: ~$1.31 for complete benchmark
```

---

### 6.3 Cost Optimization Tips

1. **Use persistent volumes** to avoid re-downloading models ($0.10/GB/month vs re-download every time)
2. **Stop pods when not in use** - Pay per second billing
3. **Use Spot instances** for non-critical workloads (50% cheaper)
4. **Set budget alerts** in RunPod dashboard
5. **Pre-build Docker images** to reduce startup time/cost
6. **Use external managed PostgreSQL** (don't waste GPU cost on database)

### 6.4 Break-Even Analysis

**M1 Mac Mini vs Cloud GPU:**

| Scenario | M1 Cost | Cloud Cost/mo | Break-even |
|----------|---------|---------------|------------|
| Already own M1 | $0 | $30 (dev) | Never (but faster) |
| Need to buy M1 | $799 | $30 (dev) | 27 months |
| Already own M1 | $0 | $410 (prod) | Never (scale limits) |
| Need to buy M1 | $799 | $410 (prod) | 2 months |

**Recommendation:** Use M1 for local development, Cloud GPU for production and benchmarking

---

## 7. Production Deployment

<!-- Source: RUNPOD_DEPLOYMENT_GUIDE.md -->

### 7.1 Code Adaptations for NVIDIA GPU

#### Environment Variables

**On M1 Mac (MLX):**
```bash
EMBED_BACKEND=mlx
N_GPU_LAYERS=24  # Partial offload (M1 shared memory)
```

**On RunPod (CUDA):**
```bash
EMBED_BACKEND=torch  # Use PyTorch with CUDA
N_GPU_LAYERS=99      # Full offload (24GB VRAM available)
N_BATCH=512          # Increase batch size (more VRAM)
CTX=16384            # Increase context (if needed)
```

#### Enhanced GPU Detection

**Add to your code:**

```python
import torch

def detect_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        log.info(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
        log.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        log.info("✅ Apple Metal (MPS) detected")
    else:
        device = "cpu"
        log.warning("⚠️  No GPU detected, using CPU")
    return device

# Use in build_embed_model()
device = detect_device()
embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    device=device,  # "cuda", "mps", or "cpu"
)
```

#### Optimal Settings for RTX 4090

```python
# For Mistral 7B Q4 on RTX 4090 (24GB VRAM)
N_GPU_LAYERS=99      # All layers on GPU
N_BATCH=512          # Larger batches (more VRAM)
CTX=16384            # 4x larger context window
MAX_NEW_TOKENS=1024  # 2x longer responses
```

### 7.2 Production Checklist

- [ ] Use persistent volume for models and data
- [ ] Setup monitoring (GPU usage, memory, query latency)
- [ ] Configure automated backups (PostgreSQL data)
- [ ] Setup health checks and auto-restart
- [ ] Configure logging to persistent storage
- [ ] Setup alerts (cost, errors, performance)
- [ ] Test failure scenarios and recovery
- [ ] Document runbooks for common issues
- [ ] Setup staging environment for testing
- [ ] Configure CI/CD for automated deployments

### 7.3 Monitoring and Observability

**GPU Monitoring:**
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Log GPU stats
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv -l 5 > gpu_stats.log
```

**Application Monitoring:**
```bash
# Track query performance
python3 rag_low_level_m1_16gb_verbose.py --log-queries --log-file query_metrics.json

# Monitor PostgreSQL
psql -h localhost -U fryt -d vector_db -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema') ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"
```

### 7.4 Managed PostgreSQL (Recommended)

**Why external database?**
- Don't waste GPU cost on database hosting
- Better reliability and backups
- Easier scaling
- Persistent across pod restarts

**Options:**
1. **Supabase** (PostgreSQL + pgvector support)
2. **AWS RDS** (PostgreSQL + pgvector extension)
3. **Google Cloud SQL** (PostgreSQL + pgvector)
4. **Railway** (PostgreSQL + automatic backups)

**Configuration:**
```bash
# In RunPod environment variables
PGHOST=your-managed-postgres.com
PGPORT=5432
PGUSER=your_user
PGPASSWORD=your_secure_password
DB_NAME=vector_db
PGTABLE=production_index
```

### 7.5 Automation and CI/CD

**RunPod API Integration:**

```python
import runpodctl

# Start pod when needed
runpodctl.start_pod(pod_id="your-pod-id")

# Run job
result = runpodctl.exec_command(
    pod_id="your-pod-id",
    command="cd /workspace/rag-pipeline && python3 rag_low_level_m1_16gb_verbose.py"
)

# Stop pod when done
runpodctl.stop_pod(pod_id="your-pod-id")
```

**GitHub Actions Example:**

```yaml
name: Deploy to RunPod

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to RunPod
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
        run: |
          # Start pod, deploy code, run tests, stop pod
          ./scripts/runpod_deploy.sh
```

---

## 8. Performance Benchmarks

<!-- Source: RUNPOD_DEPLOYMENT_GUIDE.md -->

### 8.1 Current Performance (M1 Mac Mini 16GB)

**From your benchmarks (Run 1):**
```
Embedding: ~67 chunks/sec
Query time: ~65s (retrieval + generation)
LLM generation: ~10 tokens/sec
Memory: 0 swap (optimal)
```

### 8.2 Expected Performance (RTX 4090 24GB)

**Conservative estimates:**
```
Embedding: ~500-800 chunks/sec (7-12x faster)
Query time: ~10-15s (4-6x faster)
LLM generation: ~50-80 tokens/sec (5-8x faster)
Memory: 8-12 GB VRAM used (lots of headroom)
```

**Aggressive estimates (with optimization):**
```
Embedding: ~1000+ chunks/sec (15x faster)
Query time: ~5-8s (8-13x faster)
LLM generation: ~100+ tokens/sec (10x faster)
  - Use FP16 instead of Q4 quantization
  - Batch multiple queries
  - Use Flash Attention 2
```

### 8.3 Benchmark Predictions

**Your current 58,703 chunks (Messenger data):**

| Operation | M1 (Current) | RTX 4090 (Expected) | Speedup |
|-----------|--------------|---------------------|---------|
| Embed all chunks | ~15 minutes | **~1-2 minutes** | 7-15x |
| Single query | ~65s | **~10-15s** | 4-6x |
| 12 queries (benchmark) | ~13 minutes | **~2-3 minutes** | 4-6x |

### 8.4 Real-World Performance

**Actual results will vary based on:**
- Model quantization level (Q4 vs FP16 vs FP32)
- Batch size settings
- Context window size
- Concurrent requests
- Network latency (if using external PostgreSQL)

**Recommendation:** Run your own benchmarks and compare!

---

## 9. Additional Resources

### 9.1 Documentation

**RunPod:**
- Official Docs: https://docs.runpod.io/
- Templates: https://runpod.io/console/explore
- API Reference: https://docs.runpod.io/reference/overview
- Discord Community: https://discord.gg/runpod

**Alternative Services:**
- Vast.ai: https://vast.ai/
- Lambda Labs: https://lambdalabs.com/
- Modal: https://modal.com/
- Together.ai: https://www.together.ai/

**Optimization Guides:**
- llama.cpp GPU acceleration: https://github.com/ggerganov/llama.cpp#gpu-acceleration
- PyTorch CUDA best practices: https://pytorch.org/docs/stable/notes/cuda.html
- HuggingFace GPU inference: https://huggingface.co/docs/transformers/perf_infer_gpu_one

### 9.2 Related Files

- Startup script: `scripts/runpod_startup.sh`
- Verification script: `scripts/verify_runpod_setup.sh`
- Config example: `runpod_config.env`
- Original guides:
  - `docs/RUNPOD_FINAL_SETUP.md` (bug fixes)
  - `docs/RUNPOD_DEPLOYMENT_GUIDE.md` (strategic overview)
  - `docs/RUNPOD_STARTUP_INSTRUCTIONS.md` (deployment methods)

### 9.3 Success Checklist

- [ ] RunPod account created
- [ ] Pod deployed with fixed command
- [ ] Setup completed (saw "✅ RAG Pipeline Ready!")
- [ ] Container stays alive (no restart loop)
- [ ] SSH connection works
- [ ] `bash scripts/verify_runpod_setup.sh` passes all 6 checks
- [ ] GPU test successful (`nvidia-smi` shows RTX 4090)
- [ ] PyTorch CUDA test successful
- [ ] PostgreSQL + pgvector working
- [ ] First test query completed
- [ ] Ready to index production data

### 9.4 What This Guide Combined

This comprehensive guide merged content from:

1. **RUNPOD_FINAL_SETUP.md** (2026-01-07)
   - Production-ready deployment command
   - All 6 critical bug fixes documented
   - Real startup timeline with logs
   - Verification procedures
   - Real cost data from testing

2. **RUNPOD_DEPLOYMENT_GUIDE.md** (2025-12-20)
   - Strategic overview and service comparison
   - Dockerfile for custom images
   - Code adaptations for CUDA
   - Performance predictions
   - Cost analysis for different scenarios
   - Production best practices

3. **RUNPOD_STARTUP_INSTRUCTIONS.md** (French → English)
   - Deployment method variations
   - Environment variable documentation
   - Persistent volume setup
   - Manual setup via SSH
   - Troubleshooting tips

---

## Final Notes

**You're now ready to deploy with confidence!**

All issues discovered and fixed:
- ✅ Directory cleanup before clone
- ✅ PostgreSQL without sudo
- ✅ pgvector automatic compilation
- ✅ Visible installation progress
- ✅ Container lifecycle management (no more restarts!)
- ✅ Verification script venv handling

**Deployment costs:**
- Testing session: ~$0.25-0.50
- Per successful deploy: ~$0.05 (5-7 minutes @ $0.50/h)
- Development usage: ~$30/month (2h/day)
- Production 24/7: ~$410/month (or $230 with spot instances)

**Expected speedup over M1:**
- Embedding: 7-15x faster
- Query: 4-8x faster
- Total workflow: 5-10x faster

---

**Happy benchmarking on RTX 4090!** 🚀

Questions? Check the troubleshooting section or refer to the original documentation files.
