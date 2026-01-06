# 🚀 Runpod GPU Cloud Deployment Guide

**Last Updated:** 2025-12-20
**Target:** Deploy RAG pipeline to NVIDIA GPU cloud

---

## 📋 Table of Contents

1. [Why Use Cloud GPUs?](#why-use-cloud-gpus)
2. [Service Comparison](#service-comparison)
3. [Runpod Setup Guide](#runpod-setup-guide)
4. [Code Adaptations](#code-adaptations)
5. [Docker Deployment](#docker-deployment)
6. [Cost Analysis](#cost-analysis)
7. [Performance Expectations](#performance-expectations)

---

## 🤔 Why Use Cloud GPUs?

### Current Setup (M1 Mac Mini 16GB)
```
✅ Pros:
  • Local, private, no egress costs
  • Apple Metal GPU decent for inference
  • 0 marginal cost after hardware purchase

❌ Cons:
  • Limited VRAM (shared with system RAM)
  • Slower than dedicated NVIDIA GPUs (~3-5x)
  • Can't scale (1 machine limit)
  • MLX ecosystem less mature than CUDA
```

### Cloud GPU (e.g., RTX 4090)
```
✅ Pros:
  • 24GB dedicated VRAM
  • 10-20x faster LLM inference
  • Mature CUDA ecosystem
  • Scale to multiple instances
  • Pay only when running

❌ Cons:
  • Costs ~$0.50/hour
  • Data transfer costs (if large datasets)
  • Internet dependency
```

---

## 🏆 Service Comparison

### Detailed Breakdown

#### 1. Runpod (⭐ RECOMMENDED)
```
Pricing: $0.19-0.69/h (RTX 3060-4090)
Payment: Credit card, pay-per-second
Setup: Docker-based, templates available
Storage: Persistent volumes (paid separately)
Network: Fast, reliable uptime
Support: Good community, Discord active

Best for: Your use case (Docker deployment, flexible GPU choice)
```

#### 2. Vast.ai (💰 Budget Option)
```
Pricing: $0.10-0.40/h (marketplace, varies)
Payment: BTC/ETH/Credit card
Setup: SSH access, more manual
Storage: Provider-dependent
Network: Variable (peer-to-peer marketplace)
Support: Community forums

Best for: Cost-sensitive, comfortable with SSH/manual setup
```

#### 3. Lambda Labs
```
Pricing: $0.50-1.50/h (fixed pricing)
Payment: Credit card, monthly billing
Setup: Very simple UI
Storage: Persistent, included
Network: Excellent, professional
Support: Excellent, enterprise-grade

Best for: Teams, production workloads, simplicity
```

#### 4. Modal (🔮 Serverless)
```
Pricing: ~$0.60/h + per-request
Payment: Credit card, pay-as-you-go
Setup: Python decorators, zero DevOps
Storage: Automatic, ephemeral
Network: Auto-scaling, global
Support: Good docs, Slack community

Best for: API endpoints, auto-scaling, modern stack
```

---

## 🛠️ Runpod Setup Guide

### Step 1: Create Account

1. Go to https://www.runpod.io/
2. Sign up (email + password)
3. Add billing method (credit card)
4. **Add credits:** Start with $10-20 for testing

### Step 2: Choose GPU

**Recommended for your pipeline:**

| GPU | VRAM | Price/h | When to Use |
|-----|------|---------|-------------|
| RTX 3060 | 12GB | ~$0.19 | Testing, small models |
| RTX 4070 Ti | 12GB | ~$0.29 | Budget production |
| **RTX 4090** | 24GB | **~$0.50** | **Production (recommended)** |
| RTX 6000 Ada | 48GB | ~$0.89 | Large models (overkill for you) |
| A100 (40GB) | 40GB | ~$1.50 | Enterprise (overkill) |

**For Mistral 7B + BGE embeddings: RTX 4090 is perfect**

### Step 3: Launch Pod

#### Option A: Quick Deploy (Community Cloud)

1. Click **"Deploy"** → **"GPU Pods"**
2. Select GPU: **RTX 4090** (24GB)
3. Template: **PyTorch 2.1** (has CUDA pre-installed)
4. Container Disk: **50 GB** (for models + data)
5. Volume Disk: **100 GB** (persistent storage for PostgreSQL)
6. Expose ports: `5432` (PostgreSQL), `8000` (API)
7. Click **"Deploy On-Demand"**

#### Option B: Custom Docker (Advanced)

1. Build your custom Docker image (see below)
2. Push to Docker Hub or Runpod's registry
3. Deploy using custom image URL

### Step 4: Connect to Pod

```bash
# SSH into pod (Runpod provides SSH command in UI)
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/id_runpod

# Or use Runpod's web terminal (click "Connect" in UI)
```

---

## 🔧 Code Adaptations

### Changes Needed for NVIDIA GPU

#### 1. Environment Variables

**On M1 Mac (MLX):**
```bash
EMBED_BACKEND=mlx
N_GPU_LAYERS=24  # Partial offload (M1 shared memory)
```

**On Runpod (CUDA):**
```bash
EMBED_BACKEND=torch  # Use PyTorch with CUDA
N_GPU_LAYERS=99      # Full offload (24GB VRAM available)
N_BATCH=512          # Increase batch size (more VRAM)
CTX=16384            # Increase context (if needed)
```

#### 2. Embedding Model Detection

Your code already has some GPU detection. Enhance it:

**Current:**
```python
# In build_embed_model()
embed_backend = os.getenv("EMBED_BACKEND", "auto")
if embed_backend == "mlx":
    # Use MLX
elif embed_backend == "torch":
    # Use PyTorch
```

**Enhanced for CUDA:**
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

#### 3. LLM Configuration

**For RTX 4090 (24GB VRAM):**
```python
# Optimal settings for Mistral 7B Q4 on RTX 4090
N_GPU_LAYERS=99      # All layers on GPU
N_BATCH=512          # Larger batches (more VRAM)
CTX=16384            # 4x larger context window
MAX_NEW_TOKENS=1024  # 2x longer responses
```

**Expected speed:**
- Embedding: ~500-1000 chunks/sec (vs ~67 on M1)
- LLM generation: ~50-100 tokens/sec (vs ~10 on M1)

---

## 🐳 Docker Deployment

### Dockerfile

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

### Build and Push

```bash
# Build image
docker build -f Dockerfile.runpod -t your-dockerhub-username/rag-pipeline:latest .

# Test locally (if you have NVIDIA GPU)
docker run --gpus all -it your-dockerhub-username/rag-pipeline:latest

# Push to Docker Hub
docker login
docker push your-dockerhub-username/rag-pipeline:latest
```

### Deploy on Runpod

1. In Runpod UI: **Deploy** → **GPU Pods**
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

---

## 💰 Cost Analysis

### Scenario 1: Development/Testing

**Usage:** 2 hours/day, 20 days/month

```
GPU: RTX 4090 @ $0.50/h
Time: 40 hours/month
Cost: $20/month

Storage: 100 GB @ $0.10/GB/month
Cost: $10/month

Total: ~$30/month
```

**vs M1 Mac Mini:** $0/month (already owned)
**Break-even:** Never (but 10x faster development)

### Scenario 2: Production API

**Usage:** 24/7 always-on

```
GPU: RTX 4090 @ $0.50/h
Time: 720 hours/month
Cost: $360/month

Storage: 500 GB @ $0.10/GB/month
Cost: $50/month

Total: ~$410/month
```

**Alternative:** Spot instances (interruptible)
- Price: ~$0.25/h (50% cheaper)
- Total: ~$230/month
- Trade-off: Can be interrupted (need restart logic)

### Scenario 3: Batch Processing

**Usage:** Run queries in batches, 4 hours/week

```
GPU: RTX 4090 @ $0.50/h
Time: 16 hours/month
Cost: $8/month

Storage: 50 GB @ $0.10/GB/month
Cost: $5/month

Total: ~$13/month
```

**Best approach:** Start pod when needed, stop when done

---

## 📊 Performance Expectations

### Current (M1 Mac Mini 16GB)

**From your benchmarks (Run 1):**
```
Embedding: ~67 chunks/sec
Query time: ~65s (retrieval + generation)
LLM generation: ~10 tokens/sec
Memory: 0 swap (optimal)
```

### Expected (RTX 4090 24GB)

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

### Benchmark Predictions

**Your current 58,703 chunks (Messenger data):**

| Operation | M1 (Current) | RTX 4090 (Expected) | Speedup |
|-----------|--------------|---------------------|---------|
| Embed all chunks | ~15 minutes | **~1-2 minutes** | 7-15x |
| Single query | ~65s | **~10-15s** | 4-6x |
| 12 queries (benchmark) | ~13 minutes | **~2-3 minutes** | 4-6x |

---

## 🚀 Quick Start Commands

### 1. Minimal Runpod Deployment

```bash
# After SSHing into Runpod pod:

# Clone your repo
git clone https://github.com/your-username/llamaIndex-local-rag.git
cd llamaIndex-local-rag

# Install dependencies
pip3 install -r requirements.txt

# Install PyTorch with CUDA
pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Set environment for NVIDIA GPU
export EMBED_BACKEND=torch
export N_GPU_LAYERS=99
export N_BATCH=512

# Setup PostgreSQL (use Runpod's or external)
# Option 1: Install local PostgreSQL
apt-get install -y postgresql postgresql-contrib
service postgresql start

# Option 2: Use external managed PostgreSQL (better)
export PGHOST=your-managed-postgres.com
export PGPORT=5432

# Run indexing
python3 rag_low_level_m1_16gb_verbose.py

# Run query
python3 rag_low_level_m1_16gb_verbose.py --query-only --query "when did I go to New York"
```

### 2. Production Deployment with Docker

```bash
# Build
docker build -f Dockerfile.runpod -t rag-pipeline .

# Run locally with GPU
docker run --gpus all \
  -e PGHOST=your-postgres \
  -e PGPASSWORD=your-password \
  -e PGTABLE=messenger_prod \
  -p 8000:8000 \
  rag-pipeline

# Deploy to Runpod
# (Use UI to deploy custom Docker image)
```

---

## 🎯 Recommendations

### For Your Use Case (Messenger RAG)

1. **Start with Runpod RTX 4090**
   - Best performance/price ratio
   - 24GB VRAM = comfortable headroom
   - ~$0.50/h reasonable for testing

2. **Use Docker deployment**
   - Reproducible environment
   - Easy to scale to multiple pods
   - Portable to other services

3. **Managed PostgreSQL**
   - Use Runpod's Network Volume OR
   - External managed Postgres (AWS RDS, Supabase, etc.)
   - Don't run PostgreSQL inside GPU pod (waste of GPU cost)

4. **Start-stop workflow**
   - Don't run 24/7 initially
   - Start pod when benchmarking/testing
   - Stop pod when done (pay per second!)
   - Automate with Runpod API

5. **Monitor costs**
   - Set budget alerts in Runpod
   - Start with $10-20 credit
   - Monitor usage in first week

---

## 📚 Additional Resources

### Runpod Docs
- Official: https://docs.runpod.io/
- Templates: https://runpod.io/console/explore
- API: https://docs.runpod.io/reference/overview

### Alternatives
- Vast.ai: https://vast.ai/
- Lambda Labs: https://lambdalabs.com/
- Modal: https://modal.com/
- Together.ai: https://www.together.ai/

### Optimization Guides
- llama.cpp GPU acceleration: https://github.com/ggerganov/llama.cpp#gpu-acceleration
- PyTorch CUDA best practices: https://pytorch.org/docs/stable/notes/cuda.html
- HuggingFace GPU inference: https://huggingface.co/docs/transformers/perf_infer_gpu_one

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
export N_BATCH=256  # Default was 512

# Reduce context window
export CTX=8192  # Default was 16384

# Use smaller quantization
# Q4_K_M → Q3_K_M (smaller, faster, slight quality loss)
```

### "Connection to PostgreSQL refused"
```bash
# Check if PostgreSQL is running
service postgresql status

# Check connection from pod
psql -h $PGHOST -p $PGPORT -U $PGUSER -d $DB_NAME

# If using external managed Postgres, whitelist Runpod's IP
```

### "Model download timeout"
```bash
# Pre-download models in Dockerfile
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('BAAI/bge-small-en')"

# Or use HuggingFace cache in persistent volume
export HF_HOME=/workspace/huggingface_cache
```

---

## ✅ Next Steps

1. [ ] Create Runpod account
2. [ ] Test with $10 credit on RTX 4090
3. [ ] Run your benchmark (12 queries)
4. [ ] Compare performance vs M1
5. [ ] Decide: Local (M1) for dev, Cloud (GPU) for production?
6. [ ] Setup automated deployment (optional)

**Good luck!** Feel free to ask if you need help with any step. 🚀
