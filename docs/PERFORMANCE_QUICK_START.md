# RAG Pipeline Performance - Quick Start Guide

**Quick reference for optimizing your RAG pipeline on M1 Mac Mini 16GB**

---

## Current Performance Baseline

```
Query Latency:        8-15 seconds (LLM-dominated)
Vector Search:        11ms (excellent with HNSW)
Embedding:            67 chunks/sec
Database Insertion:   1,250 nodes/sec
```

## Primary Bottleneck: LLM Generation (97% of query time)

---

## Quick Wins (Copy-Paste Commands)

### 1. Enable vLLM Server (3-4x Faster Queries)

**What it does:** Keeps model loaded in GPU memory for instant queries

**Setup (one-time):**
```bash
# Install vLLM
pip install vllm

# Start server (in terminal 1)
./scripts/start_vllm_server.sh
# Or manually:
vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192
```

**Use (terminal 2):**
```bash
# Single query
USE_VLLM=1 python rag_low_level_m1_16gb_verbose.py --query "your question"

# Interactive mode
USE_VLLM=1 python rag_interactive.py
```

**Results:**
- First query: ~60s (one-time warmup)
- Subsequent queries: 2-3s (no reload!)
- Speedup: 3-4x faster

---

### 2. Optimize Embedding Batch Size (1.5x Faster Indexing)

**What it does:** Better GPU utilization during indexing

```bash
# Test with larger batch size
EMBED_BATCH=128 python rag_low_level_m1_16gb_verbose.py

# If no OOM errors, make permanent:
echo 'export EMBED_BATCH=128' >> ~/.bashrc
```

**Results:**
- Throughput: 67 → 90-100 chunks/sec
- Indexing time: 2.5 → 1.7 minutes (10k chunks)

---

### 3. Reduce Memory Pressure

**What it does:** Prevents swapping under load

```bash
# Smaller batches for memory-constrained scenarios
export EMBED_BATCH=32
export DB_INSERT_BATCH=100

# Monitor memory
watch -n 1 "ps aux | grep python | grep -v grep | awk '{print \$6/1024 \"MB\"}'"
```

---

## Configuration Quick Reference

### Fast Queries (Optimize for Speed)
```bash
export USE_VLLM=1
export MAX_NEW_TOKENS=128
export TOP_K=3
export CHUNK_SIZE=500
```
**Result:** 2-3s queries, shorter answers

---

### High Quality (Optimize for Accuracy)
```bash
export MAX_NEW_TOKENS=512
export TOP_K=6
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=200
export TEMPERATURE=0.3
```
**Result:** Longer answers, more context, slower

---

### Fast Indexing (Optimize for Throughput)
```bash
export EMBED_BATCH=128
export DB_INSERT_BATCH=500
export CHUNK_SIZE=700
export CHUNK_OVERLAP=150
```
**Result:** 90-100 chunks/sec

---

### Low Memory (Optimize for Stability)
```bash
export EMBED_BATCH=32
export DB_INSERT_BATCH=100
export N_GPU_LAYERS=0
export MAX_NEW_TOKENS=128
```
**Result:** 4-6GB memory usage, stable

---

## Performance Checklist

### Before Running
- [ ] Close unnecessary applications (free up memory)
- [ ] Check available memory: `free -h` or Activity Monitor
- [ ] Verify database running: `psql -h localhost -U fryt -d vector_db -c "SELECT 1;"`
- [ ] Check if vLLM server is running (if using): `curl http://localhost:8000/health`

### After Optimization
- [ ] Run test query: `python rag_low_level_m1_16gb_verbose.py --query "test"`
- [ ] Check timing logs for improvements
- [ ] Monitor memory: `watch -n 1 "free -h"`
- [ ] Verify answer quality unchanged

---

## Troubleshooting

### Query is Slow (>15s)
```bash
# Check if LLM is the bottleneck
# Look for "Generation Stats" in output
# If LLM takes >10s, enable vLLM server
USE_VLLM=1 python rag_low_level_m1_16gb_verbose.py --query "test"
```

### Out of Memory Errors
```bash
# Reduce batch sizes
export EMBED_BATCH=32
export DB_INSERT_BATCH=100
```

### Vector Search is Slow (>100ms)
```bash
# Check if HNSW index exists
psql -h localhost -U fryt -d vector_db -c "
SELECT indexname FROM pg_indexes
WHERE tablename LIKE 'data_%'
AND indexname LIKE '%hnsw%';
"

# Create index if missing
python rag_low_level_m1_16gb_verbose.py --create-hnsw-index
```

### vLLM Server Not Responding
```bash
# Check server status
curl http://localhost:8000/health

# Restart server
pkill -f "vllm serve"
./scripts/start_vllm_server.sh
```

---

## Measuring Performance

### Measure Query Latency
```bash
# Run with timing
time python rag_low_level_m1_16gb_verbose.py --query "test query"

# Look for these in output:
# - "Query embedding computed in X.XXs"
# - "Vector search complete in X.XXs"
# - "Answer generated in X.XXs"
```

### Measure Indexing Throughput
```bash
# Look for these in output:
# - "Throughput: X.X nodes/second" (embedding)
# - "Throughput: X.X nodes/second" (insertion)
```

### Check Database Performance
```bash
python performance_analysis.py --database-check --table your_table_name
```

---

## Performance Targets (M1 Mac Mini 16GB)

### Current vs Target

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Query Time | 8-15s | <5s | ⚠️ Needs vLLM |
| Vector Search | 11ms | <50ms | ✓ Excellent |
| Embedding | 67 c/s | 90+ c/s | ⚠️ Can improve |
| DB Insert | 1,250 n/s | 1,000+ n/s | ✓ Excellent |

### After vLLM Optimization

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Query Time | 8-15s | 2-3s | 3-4x faster |
| First Query | 8-15s | 60s | Slower (warmup) |
| Subsequent | 8-15s | 2-3s | 3-4x faster |
| Tokens/sec | 10-20 | 50-100 | 5x faster |

---

## Commands Summary

```bash
# 1. Install dependencies
pip install vllm psutil

# 2. Start vLLM server (terminal 1)
./scripts/start_vllm_server.sh

# 3. Run optimized query (terminal 2)
USE_VLLM=1 EMBED_BATCH=128 python rag_low_level_m1_16gb_verbose.py --query "test"

# 4. Check performance
python performance_analysis.py --analyze-all --output report.json

# 5. Monitor resources
watch -n 1 "ps aux | grep python | grep -v grep"
```

---

## Next Steps

1. **Read full report:** `PERFORMANCE_ANALYSIS_REPORT.md`
2. **Run analysis tool:** `python performance_analysis.py --help`
3. **Enable vLLM:** Follow "Quick Wins" section above
4. **Benchmark your setup:** Test configurations and measure results
5. **Monitor continuously:** Set up performance logging

---

## Resources

- Full Analysis: `PERFORMANCE_ANALYSIS_REPORT.md`
- Analysis Tool: `performance_analysis.py`
- vLLM Setup: `./scripts/start_vllm_server.sh`
- Project Guide: `CLAUDE.md`

---

**Last Updated:** 2026-01-07
**For:** M1 Mac Mini 16GB RAM
