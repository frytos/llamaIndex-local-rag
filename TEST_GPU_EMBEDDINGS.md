# GPU Embeddings Test Procedure

Complete testing guide for the RunPod GPU embedding feature.

## Test Suite Overview

1. **Local Test (Mac)** - Verify service works before RunPod deployment
2. **RunPod Test** - Test on actual GPU hardware
3. **Railway Integration Test** - End-to-end with Streamlit UI
4. **Performance Verification** - Confirm 100x speedup

---

## Test 1: Local Service Test (Mac)

**Purpose:** Verify the embedding service works on your Mac before deploying to RunPod.

### 1A. Start Service Locally

```bash
cd /Users/frytos/code/llamaIndex-local-rag
source .venv/bin/activate
export RUNPOD_EMBEDDING_API_KEY="test-key-12345"
./scripts/start_embedding_service.sh
```

**Expected Output:**
```
✅ Dependencies OK
CUDA available: False  # Normal on Mac (no CUDA)
Device: cpu  # or mps
...
✅ Embedding service ready!
Service will be available at: http://0.0.0.0:8001
```

**Keep this terminal open!**

---

### 1B. Test Health Endpoint

Open a **new terminal** and run:

```bash
curl http://localhost:8001/health | python3 -m json.tool
```

**Expected:**
```json
{
  "status": "healthy",
  "gpu_available": false,
  "model_loaded": true,
  "model_name": "BAAI/bge-small-en"
}
```

**✅ Pass if:** `status: "healthy"` and `model_loaded: true`

---

### 1C. Test Embedding Endpoint

```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key-12345" \
  -d '{
    "texts": ["The quick brown fox", "jumps over the lazy dog"],
    "model": "BAAI/bge-small-en",
    "batch_size": 128
  }' | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✅ Count: {d[\"count\"]}, Dimension: {d[\"dimension\"]}, Time: {d[\"processing_time_ms\"]:.1f}ms, GPU: {d[\"gpu_used\"]}')"
```

**Expected:**
```
✅ Count: 2, Dimension: 384, Time: 50.0ms, GPU: False
```

**✅ Pass if:** Returns 2 embeddings with 384 dimensions each

---

### 1D. Test Large Batch

```bash
python3 << 'EOF'
import requests
import json

API_KEY = "test-key-12345"
BASE_URL = "http://localhost:8001"

# Generate 100 test texts
texts = [f"Document {i} with various content about topic {i%10}" for i in range(100)]

payload = {
    "texts": texts,
    "model": "BAAI/bge-small-en",
    "batch_size": 128
}

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

print(f"Sending {len(texts)} texts...")
response = requests.post(f"{BASE_URL}/embed", json=payload, headers=headers, timeout=60)
data = response.json()

print(f"✅ Processed {data['count']} texts")
print(f"   Dimension: {data['dimension']}")
print(f"   Time: {data['processing_time_ms']/1000:.2f}s")
print(f"   Throughput: {data['count'] / (data['processing_time_ms']/1000):.1f} texts/sec")
EOF
```

**Expected:**
```
✅ Processed 100 texts
   Dimension: 384
   Time: 2.5s
   Throughput: 40.0 texts/sec
```

**✅ Pass if:** All 100 texts embedded successfully

---

### 1E. Stop Local Service

```bash
# Press Ctrl+C in the terminal running the service
# Or in another terminal:
pkill -f "uvicorn services.embedding_service"
```

---

## Test 2: RunPod Service Test

**Purpose:** Test on actual NVIDIA GPU hardware.

### 2A. Start Service on RunPod

SSH into your RunPod pod:

```bash
ssh -i ~/.ssh/runpod_key <ssh-host>@ssh.runpod.io
```

Then run:

```bash
cd /workspace/rag-pipeline
source .venv/bin/activate
export RUNPOD_EMBEDDING_API_KEY="<your-actual-api-key>"
./scripts/start_embedding_service.sh
```

**Expected Output:**
```
Checking GPU...
CUDA available: True
Device: NVIDIA GeForce RTX 4090  ← IMPORTANT!
...
✅ Embedding service ready!
```

**✅ Pass if:** Shows RTX 4090 and CUDA available

**Keep this terminal/SSH session open!**

---

### 2B. Test RunPod Service (From RunPod Pod)

In **another SSH session** to the same pod:

```bash
# Test health
curl http://localhost:8001/health | python3 -m json.tool
```

**Expected:**
```json
{
  "status": "healthy",
  "gpu_available": true,  ← IMPORTANT!
  "model_loaded": true,
  "model_name": "BAAI/bge-small-en"
}
```

**✅ Pass if:** `gpu_available: true`

---

### 2C. Test GPU Performance

```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "texts": ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5"],
    "model": "BAAI/bge-small-en",
    "batch_size": 128
  }' | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✅ GPU: {d[\"gpu_used\"]}, Time: {d[\"processing_time_ms\"]:.1f}ms, Throughput: {d[\"count\"]/(d[\"processing_time_ms\"]/1000):.0f} texts/sec')"
```

**Expected:**
```
✅ GPU: True, Time: 15.0ms, Throughput: 333 texts/sec
```

**✅ Pass if:**
- `GPU: True`
- Time < 50ms (GPU is MUCH faster than CPU)
- Throughput > 100 texts/sec

---

### 2D. Test From Your Mac (Public IP)

**On your Mac**, test the public endpoint:

```bash
# Use the public IP and port from RunPod
curl http://38.65.239.5:<port-8001-mapping>/health

# Example if port 8001 maps to 18833:
curl http://38.65.239.5:18833/health | python3 -m json.tool
```

**✅ Pass if:** You can reach the service from outside RunPod

**To find the port mapping:**
```bash
# On RunPod pod:
curl http://localhost:8001/health  # This works
# Then check RunPod dashboard for port 8001 → external port mapping
```

---

## Test 3: Railway Streamlit Integration

**Purpose:** Test end-to-end workflow from web UI.

### Prerequisites

Railway must have these variables set:
```
RUNPOD_EMBEDDING_API_KEY=<your-generated-embedding-key>
RUNPOD_API_KEY=<your-runpod-account-api-key>
```

(No need for RUNPOD_EMBEDDING_ENDPOINT - auto-detected!)

---

### 3A. Check GPU Indicator

1. Go to https://rag.groussard.xyz
2. Login
3. Navigate to **"Index Documents"**
4. **Look at the top** of the page

**Expected:**
```
🚀 GPU Acceleration Enabled - Using RunPod RTX 4090 (~100x faster)
Endpoint: http://38.65.239.5:XXXXX
```

**✅ Pass if:** Green GPU message appears (not yellow CPU warning)

**❌ Fail if:** Yellow "💻 CPU Mode" warning shows
- Check RunPod service is running
- Check port 8001 is exposed on RunPod
- Check Railway has RUNPOD_EMBEDDING_API_KEY set

---

### 3B. Upload and Index Small File

1. Select **"📤 Upload files from your computer"**
2. Upload a small text file (~1-2 KB)
3. Configure:
   - Preset: "General documents (700/150)"
   - Model: "bge-small-en (384d, fast)"
   - Index name: `gpu_test_small`
4. Click **"🚀 Start Indexing"**

**Watch the progress:**
- Step 1: Load documents (<1 sec)
- Step 2: Chunking (<1 sec)
- Step 3: **Embedding** ← THIS IS THE KEY TEST
  - **With GPU:** Should complete in 1-3 seconds
  - **Without GPU:** Would take 30-60 seconds even for small file

**✅ Pass if:** Embedding step completes in under 5 seconds

---

### 3C. Upload and Index Medium File

Create a test file with ~500 words (or use existing document):

```bash
# On your Mac, create test file:
cat > /tmp/test_doc.txt << 'EOF'
[Paste several paragraphs of text here - ~500 words total]
EOF
```

Then in Streamlit:
1. Upload `/tmp/test_doc.txt`
2. Use same settings as 3B
3. Index name: `gpu_test_medium`
4. Start indexing

**Timing Test:**
- Note the time when "Step 3: Embedding" starts
- Note when it completes
- Calculate duration

**Expected results:**

| Chunks | GPU Time | CPU Time |
|--------|----------|----------|
| 50 | 1-2 sec | ~1 minute |
| 100 | 2-3 sec | ~2 minutes |
| 200 | 4-5 sec | ~4 minutes |

**✅ Pass if:** Completes in seconds (not minutes)

---

### 3D. Check Browser Console Logs

1. Open browser DevTools (F12)
2. Go to Console tab
3. Look for embedding-related logs

**Expected to see:**
```
🚀 Using RunPod GPU for embeddings
✅ Auto-detected embedding endpoint: http://38.65.239.5:XXXXX
```

**✅ Pass if:** Shows "Using RunPod GPU" in logs

---

### 3E. Verify in Database

1. Go to **"View Indexes"** page
2. Find your test index
3. Click to view details

**✅ Pass if:**
- Row count matches expected chunks
- Shows embedding dimension (384)
- Vectors contain actual numbers

---

## Test 4: Performance Benchmark

**Purpose:** Quantify the speedup.

### Create Benchmark Document

```bash
# Generate 1000-word document
cat > /tmp/benchmark.txt << 'EOF'
[Paste ~1000 words of text - can be Lorem Ipsum or any content]
EOF
```

### Run Benchmark

1. Upload `/tmp/benchmark.txt` in Streamlit
2. Settings: Default (700/150, bge-small-en)
3. **Start timer** when you click "Start Indexing"
4. **Note Step 3 (Embedding) time**
5. Record total time

**Expected for ~300 chunks:**
- ✅ **GPU:** 3-5 seconds
- ❌ **CPU:** 5-8 minutes

**Speedup ratio: ~100x**

---

## Test 5: Fallback Test

**Purpose:** Verify system works even if GPU service is down.

### 5A. Stop RunPod Service

In your RunPod SSH session:
```bash
# Press Ctrl+C to stop the embedding service
```

### 5B. Try Indexing Again

1. Go to Streamlit UI
2. Upload another small file
3. Start indexing

**Expected:**
- Yellow warning: "💻 CPU Mode"
- Indexing still works (but slower)
- Logs show: "RunPod embedding service unhealthy, falling back to local embedding"

**✅ Pass if:** Indexing completes successfully (even if slower)

### 5C. Restart RunPod Service

```bash
# In RunPod SSH:
./scripts/start_embedding_service.sh
```

Verify green "🚀 GPU Enabled" message returns.

---

## Test Results Checklist

After completing tests, verify:

- [ ] Local service starts and responds to health checks
- [ ] RunPod service shows `gpu_available: true`
- [ ] Railway UI shows green "🚀 GPU Acceleration Enabled"
- [ ] Small file (50 chunks) indexes in <2 seconds
- [ ] Medium file (200 chunks) indexes in <5 seconds
- [ ] Large file (500+ chunks) shows dramatic speedup vs CPU
- [ ] Fallback works when service is stopped
- [ ] Database shows correct vectors after indexing
- [ ] Query functionality still works with GPU-embedded documents

---

## Quick Reference Commands

### On RunPod:
```bash
# Start service
cd /workspace/rag-pipeline && source .venv/bin/activate
export RUNPOD_EMBEDDING_API_KEY="<your-key>"
./scripts/start_embedding_service.sh

# Check health
curl http://localhost:8001/health | python3 -m json.tool

# View logs
tail -f /workspace/embedding_service.log

# Stop service
pkill -f "uvicorn services.embedding_service"
```

### On Your Mac:
```bash
# Check Railway is configured
# Go to: https://railway.app → Your Project → Variables
# Verify: RUNPOD_EMBEDDING_API_KEY is set (your generated key)
# Verify: RUNPOD_API_KEY is set (your RunPod account API key)
```

---

## Troubleshooting

### Problem: "GPU: false" on RunPod

**Solution:**
```bash
# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Should show: CUDA: True
# If False, check PyTorch installation
```

### Problem: "Port already in use"

**Solution:**
```bash
# Kill existing service
lsof -ti:8001 | xargs kill -9

# Restart
./scripts/start_embedding_service.sh
```

### Problem: "Module not found: llama_index"

**Solution:**
```bash
# Install in virtual environment
source .venv/bin/activate
pip install -r requirements.txt
```

### Problem: Railway shows "CPU Mode" (yellow warning)

**Check:**
1. Is RunPod service running? `curl http://localhost:8001/health` from RunPod
2. Is port 8001 exposed? Check RunPod dashboard → Pod → Ports
3. Is API key set on Railway? Check Railway Variables
4. Check Railway logs for auto-detection messages

---

## Success Criteria

**ALL of these must be true:**

✅ RunPod service shows `gpu_available: true`
✅ Railway UI shows green "GPU Enabled" message
✅ 100 chunks embed in <3 seconds (not 2+ minutes)
✅ Database receives correct 384-dim vectors
✅ Queries return relevant results
✅ Fallback to CPU works if GPU unavailable

---

## Performance Targets

| Chunks | GPU Target | CPU Baseline |
|--------|-----------|--------------|
| 50 | <2 sec | ~1 min |
| 100 | <3 sec | ~2 min |
| 500 | <10 sec | ~10 min |
| 1000 | <15 sec | ~20 min |

**If you're not hitting these targets, the GPU isn't being used!**
