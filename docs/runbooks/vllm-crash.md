# Runbook: vLLM Server Crash

**Severity**: P1 (High - Performance degradation, fallback available)
**Estimated Resolution Time**: 5-10 minutes
**Last Updated**: 2026-01-07

## Overview

This runbook covers vLLM server crashes, hangs, and GPU-related issues. The system can fall back to llama.cpp, so this is not a complete outage.

## Symptoms

- `ConnectionError: Cannot connect to vLLM server`
- vLLM process died unexpectedly
- GPU out of memory errors
- Inference requests timing out (>60s)
- Server responding but returning errors
- Slow generation speed (<5 tokens/s)

## Quick Diagnosis

```bash
# 1. Check if vLLM process is running
ps aux | grep vllm

# 2. Check vLLM server health
curl http://localhost:8000/health

# 3. Check GPU usage
nvidia-smi
# OR on Mac:
sudo powermetrics --samplers gpu_power -i1 -n1

# 4. Check server logs
tail -f logs/vllm_server.log
# OR if running in screen/tmux:
screen -r vllm
```

## Common Issues & Solutions

### Issue 1: vLLM Server Not Running

**Symptoms**: `ConnectionError: Cannot connect to http://localhost:8000`

**Quick Fix**:
```bash
# Start vLLM server
./scripts/start_vllm_server.sh

# OR manually:
vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    > logs/vllm_server.log 2>&1 &

# Wait for warmup (30-60s)
sleep 60

# Verify
curl http://localhost:8000/health
```

**Expected Output**:
```json
{"status": "ok"}
```

### Issue 2: GPU Out of Memory

**Symptoms**:
- `CUDA out of memory`
- `RuntimeError: CUDA error: out of memory`
- vLLM crashes after starting

**Diagnosis**:
```bash
# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check which model is loaded
curl http://localhost:8000/v1/models
```

**Solution 1**: Reduce GPU memory utilization
```bash
# Stop vLLM
pkill -f vllm

# Restart with lower memory
vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --port 8000 \
    --gpu-memory-utilization 0.6 \  # Reduced from 0.8
    --max-model-len 4096 \           # Reduced from 8192
    > logs/vllm_server.log 2>&1 &
```

**Solution 2**: Use smaller model
```bash
# Stop vLLM
pkill -f vllm

# Use smaller model (TinyLlama)
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 2048 \
    > logs/vllm_server.log 2>&1 &
```

**Solution 3**: Fall back to llama.cpp
```bash
# Disable vLLM in environment
export USE_VLLM=0

# OR edit .env:
# USE_VLLM=0

# Run pipeline with CPU/Metal inference
python rag_low_level_m1_16gb_verbose.py --query-only
```

### Issue 3: vLLM Hung / Not Responding

**Symptoms**:
- Health check timeout
- Requests hang indefinitely
- CPU/GPU usage at 0%

**Solution**:
```bash
# 1. Try graceful shutdown
pkill -TERM -f vllm
sleep 5

# 2. If still running, force kill
pkill -9 -f vllm

# 3. Clear GPU memory
nvidia-smi --gpu-reset

# 4. Restart
./scripts/start_vllm_server.sh

# 5. Monitor startup
tail -f logs/vllm_server.log
```

**Watch for**: Model loading progress, "Uvicorn running" message

### Issue 4: Model Download Failed

**Symptoms**:
- `OSError: Model not found`
- `HTTPError: 404 Client Error`
- vLLM crashes during startup

**Solution**:
```bash
# Pre-download model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('TheBloke/Mistral-7B-Instruct-v0.2-AWQ')
"

# Verify download
ls -lh ~/.cache/huggingface/hub/

# Restart vLLM
./scripts/start_vllm_server.sh
```

**Alternative**: Use local model path
```bash
vllm serve /path/to/local/model \
    --port 8000 \
    --gpu-memory-utilization 0.8
```

### Issue 5: Slow Generation (<5 tokens/s)

**Symptoms**: Queries taking >30s, low GPU utilization

**Diagnosis**:
```bash
# Check GPU utilization
nvidia-smi dmon -s pucvmet

# Check vLLM stats
curl http://localhost:8000/v1/models
```

**Solutions**:

1. **Increase batch size**:
```bash
pkill -f vllm
vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --port 8000 \
    --max-num-seqs 256 \          # Increase batch size
    --gpu-memory-utilization 0.8
```

2. **Use AWQ quantization** (if not already):
```bash
# AWQ models are 2-3x faster than GPTQ
vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --port 8000 \
    --quantization awq
```

3. **Reduce context length**:
```bash
# Shorter context = faster generation
export CTX=2048  # Instead of 8192
export MAX_NEW_TOKENS=128  # Instead of 256
```

### Issue 6: Port Already in Use

**Symptoms**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000
# OR
netstat -anp | grep 8000

# Kill the process
kill -9 <PID>

# Restart vLLM
./scripts/start_vllm_server.sh
```

### Issue 7: CUDA Driver Version Mismatch

**Symptoms**:
- `CUDA driver version is insufficient`
- `RuntimeError: CUDA initialization: CUDA driver version is insufficient for CUDA runtime version`

**Solution**:
```bash
# Check versions
nvidia-smi
nvcc --version

# Update NVIDIA drivers
# Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-driver-535  # Or latest

# Restart system
sudo reboot

# Verify
nvidia-smi
```

## Health Monitoring

### Automated Health Check

```bash
#!/bin/bash
# vllm-health-check.sh

echo "=== vLLM Server Health Check ==="

# 1. Process check
echo "1. Process Status:"
if pgrep -f vllm > /dev/null; then
    echo "✓ vLLM process running (PID: $(pgrep -f vllm))"
else
    echo "✗ vLLM process not running"
    exit 1
fi

# 2. Port check
echo -e "\n2. Port Status:"
if lsof -i :8000 > /dev/null 2>&1; then
    echo "✓ Port 8000 listening"
else
    echo "✗ Port 8000 not listening"
fi

# 3. Health endpoint
echo -e "\n3. Health Endpoint:"
if curl -s http://localhost:8000/health | grep -q "ok"; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
fi

# 4. GPU check (if NVIDIA)
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n4. GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
fi

# 5. Test inference
echo -e "\n5. Test Inference:"
response=$(curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "prompt": "Say hello",
        "max_tokens": 5
    }')

if echo "$response" | grep -q "choices"; then
    echo "✓ Inference working"
else
    echo "✗ Inference failed"
    echo "Response: $response"
fi

echo -e "\n✓ Health check complete"
```

Save and run:
```bash
chmod +x vllm-health-check.sh
./vllm-health-check.sh
```

### Performance Metrics

Monitor these metrics:

1. **Throughput**: tokens/second
2. **Latency**: time to first token
3. **GPU utilization**: should be >70%
4. **Memory usage**: should be stable
5. **Error rate**: should be <1%

```bash
# Get vLLM metrics
curl http://localhost:8000/metrics
```

## Graceful Degradation

### Fallback to llama.cpp

If vLLM is unavailable, system automatically falls back to llama.cpp:

```python
# In code, this happens automatically:
if USE_VLLM and not vllm_available():
    logger.warning("vLLM unavailable, falling back to llama.cpp")
    USE_VLLM = False
    llm = build_llm_llamacpp()
```

**Performance Impact**:
- 3-4x slower inference (8-15s vs 2-3s)
- Higher CPU/RAM usage
- Lower throughput

**To force llama.cpp**:
```bash
USE_VLLM=0 python rag_low_level_m1_16gb_verbose.py --query-only
```

## Complete vLLM Reset

**Nuclear option** - use if server is completely broken:

```bash
# 1. Kill all vLLM processes
pkill -9 -f vllm

# 2. Clear GPU memory
nvidia-smi --gpu-reset

# 3. Clear model cache (optional)
rm -rf ~/.cache/huggingface/hub/models--*vllm*

# 4. Clear Python cache
python3 -c "import vllm; import os; os.system('rm -rf ' + os.path.dirname(vllm.__file__) + '/__pycache__')"

# 5. Restart fresh
./scripts/start_vllm_server.sh

# 6. Monitor startup
tail -f logs/vllm_server.log
```

## Prevention

### Best Practices

1. **Start vLLM in screen/tmux**: Prevents accidental termination
   ```bash
   screen -S vllm
   ./scripts/start_vllm_server.sh
   # Ctrl+A, D to detach
   ```

2. **Monitor GPU temperature**: Keep <85°C
   ```bash
   nvidia-smi dmon -s pucvmet
   ```

3. **Set up systemd service**: Auto-restart on crash
   ```bash
   # /etc/systemd/system/vllm.service
   [Unit]
   Description=vLLM Server
   After=network.target

   [Service]
   Type=simple
   User=your_user
   WorkingDirectory=/path/to/llamaIndex-local-rag
   ExecStart=/path/to/scripts/start_vllm_server.sh
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

4. **Log rotation**: Prevent disk fill
   ```bash
   # Add to /etc/logrotate.d/vllm
   /path/to/logs/vllm_server.log {
       daily
       rotate 7
       compress
       missingok
       notifempty
   }
   ```

### Resource Limits

**Recommended GPU Memory by Model**:
- Mistral 7B AWQ: 4-6GB
- Llama 2 7B AWQ: 4-6GB
- Llama 2 13B AWQ: 8-10GB
- Mixtral 8x7B AWQ: 20-24GB

**System Requirements**:
- NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- CUDA 11.8 or higher
- 16GB+ system RAM
- Fast SSD for model cache

## Escalation Path

If issue persists:

1. **Check vLLM logs**:
   ```bash
   cat logs/vllm_server.log
   ```

2. **Check system logs**:
   ```bash
   dmesg | tail -50
   journalctl -xe
   ```

3. **Report to vLLM GitHub**:
   - Include error messages
   - GPU model and driver version
   - vLLM version: `pip show vllm`
   - CUDA version: `nvcc --version`

4. **Temporary workaround**:
   ```bash
   USE_VLLM=0 python rag_low_level_m1_16gb_verbose.py
   ```

## Related Runbooks

- [Database Failure](database-failure.md)
- [Out of Memory](out-of-memory.md)

## Change Log

- 2026-01-07: Initial version
