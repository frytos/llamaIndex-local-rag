# Runbook: Out of Memory Errors

**Severity**: P0-P1 (Critical to High, depending on component)
**Estimated Resolution Time**: 2-10 minutes
**Last Updated**: 2026-01-07

## Overview

This runbook covers memory exhaustion issues across different components: system RAM, GPU VRAM, and disk space. Includes both quick fixes and long-term solutions.

## Symptoms

**System RAM:**
- `MemoryError: Unable to allocate array`
- System becomes unresponsive
- Swap usage at 100%
- Kernel OOM killer triggered

**GPU VRAM:**
- `CUDA out of memory`
- `RuntimeError: CUDA error: out of memory`
- `torch.cuda.OutOfMemoryError`
- GPU process crashes

**Disk Space:**
- `OSError: [Errno 28] No space left on device`
- Database writes fail
- Model downloads fail
- Log files stop growing

## Quick Diagnosis

```bash
# 1. Check system RAM
free -h
# OR on Mac:
vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f Mi\n", "$1:", $2 * $size / 1048576);'

# 2. Check GPU VRAM (NVIDIA)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
# OR on Mac:
sudo powermetrics --samplers gpu_power -i1 -n1 | grep "GPU"

# 3. Check disk space
df -h

# 4. Check process memory
ps aux --sort=-%mem | head -10
```

## Component-Specific Solutions

### 1. Embedding Model OOM

**Symptoms**: Crash during document indexing, high RAM usage during embedding

**Quick Diagnosis**:
```bash
# Check embedding batch size
grep EMBED_BATCH .env

# Check model size
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en')
print(f'Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters')
"
```

**Solution 1**: Reduce batch size
```bash
# In .env:
EMBED_BATCH=8  # Reduce from 32/64

# OR temporarily:
EMBED_BATCH=8 python rag_low_level_m1_16gb_verbose.py
```

**Memory Savings**: 4-8x reduction (e.g., 4GB → 512MB)

**Solution 2**: Use smaller embedding model
```bash
# In .env:
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2  # 80MB vs 130MB
EMBED_DIM=384
```

**Solution 3**: Use CPU for embeddings (slower but less memory)
```bash
# In .env:
EMBED_BACKEND=huggingface  # Instead of mlx/torch
```

### 2. LLM Inference OOM

**Symptoms**: Crash during query generation, high RAM/VRAM during inference

**Quick Diagnosis**:
```bash
# Check LLM settings
grep -E "(N_GPU_LAYERS|N_BATCH|CTX|MAX_NEW_TOKENS)" .env

# Check model size
ls -lh ~/.cache/huggingface/hub/ | grep -i mistral
```

**Solution 1**: Reduce GPU layers (for llama.cpp)
```bash
# In .env:
N_GPU_LAYERS=8  # Reduce from 16/24/32
N_BATCH=64      # Reduce from 128/256
```

**Memory Savings**: Proportional to layers (e.g., 24 → 8 layers = 3x reduction)

**Solution 2**: Use CPU-only inference
```bash
# In .env:
N_GPU_LAYERS=0  # Full CPU inference
N_BATCH=32      # Smaller batch for CPU
```

**Solution 3**: Use smaller/more quantized model
```bash
# Q2_K (2-bit, ~2.5GB):
MODEL_URL=https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf

# Q3_K_S (3-bit, ~3GB):
MODEL_URL=https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q3_K_S.gguf

# TinyLlama (1.1B params, ~700MB):
MODEL_URL=https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Solution 4**: Reduce context window
```bash
# In .env:
CTX=2048          # Reduce from 3072/8192
MAX_NEW_TOKENS=128  # Reduce from 256
TOP_K=2           # Retrieve fewer chunks
```

### 3. vLLM GPU OOM

**Symptoms**: vLLM crashes on startup or first query

**Quick Diagnosis**:
```bash
# Check vLLM settings
ps aux | grep vllm

# Check GPU memory
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

**Solution 1**: Reduce GPU memory utilization
```bash
# Stop vLLM
pkill -f vllm

# Restart with lower memory
vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --port 8000 \
    --gpu-memory-utilization 0.5 \  # Reduce from 0.8
    --max-model-len 2048              # Reduce from 8192
```

**Memory Savings**: 40-50% reduction

**Solution 2**: Use smaller model
```bash
pkill -f vllm
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 8000 \
    --gpu-memory-utilization 0.6
```

**Solution 3**: Fall back to llama.cpp
```bash
# Disable vLLM
export USE_VLLM=0

# Run with llama.cpp
python rag_low_level_m1_16gb_verbose.py --query-only
```

### 4. Database OOM

**Symptoms**: PostgreSQL crashes, high memory usage, slow queries

**Quick Diagnosis**:
```bash
# Check PostgreSQL memory
docker stats postgres

# Check database size
psql -h localhost -U $PGUSER -d vector_db -c \
    "SELECT pg_size_pretty(pg_database_size('vector_db'));"

# Check shared_buffers
psql -h localhost -U $PGUSER -d postgres -c \
    "SHOW shared_buffers;"
```

**Solution 1**: Reduce batch insert size
```bash
# In .env:
DB_INSERT_BATCH=100  # Reduce from 250/500
```

**Solution 2**: Add memory limit to PostgreSQL
```bash
# Edit config/docker-compose.yml
services:
  postgres:
    # Add memory limit
    mem_limit: 4g
    memswap_limit: 4g

# Restart
docker compose -f config/docker-compose.yml restart
```

**Solution 3**: Optimize PostgreSQL config
```bash
# Reduce shared_buffers
docker exec -it postgres bash
echo "shared_buffers = 256MB" >> /var/lib/postgresql/data/postgresql.conf
exit

docker compose -f config/docker-compose.yml restart
```

### 5. Chunking/Document Loading OOM

**Symptoms**: Crash during PDF loading, high memory spike

**Quick Diagnosis**:
```bash
# Check document size
ls -lh data/

# Check chunk settings
grep -E "(CHUNK_SIZE|CHUNK_OVERLAP)" .env
```

**Solution 1**: Process documents one at a time
```bash
# Instead of indexing entire folder:
PDF_PATH=data/  # Don't do this with large folders

# Index files individually:
for file in data/*.pdf; do
    PDF_PATH="$file" PGTABLE=docs RESET_TABLE=0 \
        python rag_low_level_m1_16gb_verbose.py
done
```

**Solution 2**: Increase chunk size (fewer chunks)
```bash
# In .env:
CHUNK_SIZE=1200   # Increase from 700 (30% fewer chunks)
CHUNK_OVERLAP=150
```

**Solution 3**: Disable metadata extraction
```bash
# In .env:
EXTRACT_ENHANCED_METADATA=0
EXTRACT_TOPICS=0
EXTRACT_ENTITIES=0
```

**Memory Savings**: 20-30% reduction

### 6. Disk Space Exhaustion

**Symptoms**: `No space left on device`, writes fail

**Quick Diagnosis**:
```bash
# Check disk usage
df -h

# Find large files
du -sh * | sort -hr | head -10

# Check specific directories
du -sh ~/.cache/huggingface/hub/
du -sh logs/
du -sh query_logs/
```

**Solution 1**: Clean model cache
```bash
# Remove unused models
rm -rf ~/.cache/huggingface/hub/models--*

# Keep only current model
ls -d ~/.cache/huggingface/hub/models--* | \
    grep -v "Mistral-7B-Instruct" | xargs rm -rf
```

**Space Freed**: 5-50GB per model

**Solution 2**: Clean logs
```bash
# Compress old logs
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# Delete very old logs
find logs/ -name "*.log.gz" -mtime +30 -delete

# Limit log size
truncate -s 100M logs/vllm_server.log
```

**Solution 3**: Clean query logs
```bash
# Disable query logging
export LOG_QUERIES=0

# Clean old query logs
find query_logs/ -name "*.json" -mtime +14 -delete
```

**Solution 4**: Clean Docker volumes
```bash
# Remove unused Docker images
docker image prune -a

# Remove unused volumes
docker volume prune
```

**Space Freed**: 1-20GB

## Memory Budget Guidelines

### For 16GB System RAM

**Optimal Configuration**:
```bash
# System: 2GB
# OS/Background: 4GB
# Available for RAG: 10GB

# Allocate:
EMBED_BATCH=32        # 1-2GB
N_GPU_LAYERS=16       # 3-4GB (Metal/GPU)
N_BATCH=128           # 2-3GB
DB_INSERT_BATCH=250   # 1GB
# Headroom: 2-3GB
```

**Conservative Configuration** (8GB system):
```bash
EMBED_BATCH=8
N_GPU_LAYERS=8
N_BATCH=64
DB_INSERT_BATCH=100
```

**Aggressive Configuration** (32GB+ system):
```bash
EMBED_BATCH=128
N_GPU_LAYERS=32
N_BATCH=512
DB_INSERT_BATCH=500
```

### For GPU VRAM

**4GB VRAM** (e.g., RTX 3050):
```bash
# Use Q2_K model or TinyLlama
# OR CPU-only with N_GPU_LAYERS=0
```

**8GB VRAM** (e.g., RTX 3060):
```bash
# Mistral 7B AWQ or Q4_K_M
USE_VLLM=1
VLLM_GPU_MEMORY=0.6
# OR
N_GPU_LAYERS=24
```

**16GB+ VRAM** (e.g., RTX 4080):
```bash
# Mistral 7B full precision or larger models
USE_VLLM=1
VLLM_GPU_MEMORY=0.8
# OR
N_GPU_LAYERS=99
```

## Emergency Memory Clearing

If system is unresponsive:

```bash
# 1. Kill heavy processes
pkill -f python
pkill -f vllm
pkill -f streamlit

# 2. Clear system cache
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 3. Clear GPU memory (NVIDIA)
nvidia-smi --gpu-reset

# 4. Restart Docker
docker compose -f config/docker-compose.yml restart

# 5. Check memory
free -h
nvidia-smi
```

## Monitoring & Prevention

### Continuous Monitoring

```bash
# Monitor system memory
watch -n 5 'free -h'

# Monitor GPU memory
watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'

# Monitor disk space
watch -n 60 'df -h'
```

### Automated Monitoring Script

```bash
#!/bin/bash
# memory-monitor.sh

while true; do
    # Check system RAM
    mem_used=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    if (( $(echo "$mem_used > 90" | bc -l) )); then
        echo "WARNING: RAM usage at ${mem_used}%"
    fi

    # Check disk space
    disk_used=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_used" -gt 90 ]; then
        echo "WARNING: Disk usage at ${disk_used}%"
    fi

    # Check GPU (if available)
    if command -v nvidia-smi &> /dev/null; then
        gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        gpu_percent=$(( gpu_mem * 100 / gpu_total ))
        if [ "$gpu_percent" -gt 90 ]; then
            echo "WARNING: GPU memory at ${gpu_percent}%"
        fi
    fi

    sleep 60
done
```

### Log Rotation

Set up automatic log rotation:

```bash
# /etc/logrotate.d/rag-pipeline
/path/to/llamaIndex-local-rag/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    maxsize 100M
}
```

### Disk Space Alerts

```bash
# Add to crontab
# Run daily at 2 AM
0 2 * * * /path/to/cleanup-script.sh
```

cleanup-script.sh:
```bash
#!/bin/bash
# Clean up old files automatically

# Clean logs older than 14 days
find /path/to/logs/ -name "*.log" -mtime +14 -delete

# Clean query logs older than 30 days
find /path/to/query_logs/ -name "*.json" -mtime +30 -delete

# Clean old model cache (keep last 2 accessed)
find ~/.cache/huggingface/hub/ -type d -name "models--*" \
    -atime +30 ! -atime -60 -exec rm -rf {} +
```

## Tuning Recommendations

### Indexing Phase (High Memory)

```bash
# Optimize for indexing
EMBED_BATCH=64        # Higher batch
N_GPU_LAYERS=0        # CPU-only (free GPU RAM)
DB_INSERT_BATCH=500   # Larger batches
```

### Query Phase (Lower Memory)

```bash
# Optimize for queries
EMBED_BATCH=16        # Lower batch
N_GPU_LAYERS=24       # Use GPU
DB_INSERT_BATCH=100   # Not used during queries
TOP_K=3               # Fewer chunks
```

## Escalation Path

If memory issues persist:

1. **Profile memory usage**:
   ```bash
   # Python memory profiler
   pip install memory_profiler
   python -m memory_profiler rag_low_level_m1_16gb_verbose.py
   ```

2. **Check for memory leaks**:
   ```bash
   # Monitor over time
   while true; do
       ps aux --sort=-%mem | head -5
       sleep 5
   done
   ```

3. **Upgrade system**:
   - Add more RAM (16GB → 32GB)
   - Upgrade GPU (8GB → 16GB VRAM)
   - Add swap space (temporary)

4. **Use remote resources**:
   - Deploy to cloud with more resources
   - Use RunPod for GPU inference
   - Separate database to dedicated server

## Related Runbooks

- [Database Failure](database-failure.md)
- [vLLM Server Crash](vllm-crash.md)

## Change Log

- 2026-01-07: Initial version
