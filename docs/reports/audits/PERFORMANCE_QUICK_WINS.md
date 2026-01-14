# Performance Quick Wins - RAG Pipeline

**Date**: 2026-01-09 | **Target Audience**: Developers & Operators

---

## TL;DR - Critical Findings

**PRIMARY BOTTLENECK**: LLM generation (8s out of 8.2s total = 97% of query time)

**QUICK FIX**: Deploy vLLM server → **3.1x faster queries** (8.2s → 2.6s)

**OVERALL RATING**: B+ (Good foundation, one critical bottleneck)

---

## Top 5 Optimizations by ROI

```
┌──────────────────────────────┬─────────┬─────────┬─────────┐
│ Optimization                 │ Speedup │ Effort  │ ROI     │
├──────────────────────────────┼─────────┼─────────┼─────────┤
│ 1. vLLM Server (GPU)         │   3.1x  │  2 hrs  │  ⭐⭐⭐  │
│ 2. MLX Embedding (M1)        │   7.5x  │  1 hr   │  ⭐⭐⭐  │
│ 3. Increase EMBED_BATCH      │   1.5x  │  1 min  │  ⭐⭐⭐  │
│ 4. Enable Semantic Cache     │   1.3x  │  5 min  │  ⭐⭐   │
│ 5. Connection Pooling (API)  │   5x*   │  2 hrs  │  ⭐⭐   │
└──────────────────────────────┴─────────┴─────────┴─────────┘
* Context-dependent (high-load scenarios)
```

---

## 1-Minute Quick Wins

### Query Performance (PRIMARY ISSUE)

**Current**: 8.2s per query
**Target**: <3s per query

```bash
# OPTION A: vLLM Server (Requires GPU) - RECOMMENDED
./scripts/start_vllm_server.sh
export USE_VLLM=1

# Result: 8.2s → 2.6s (3.1x faster)
```

```bash
# OPTION B: Reduce Token Length (No GPU needed)
export MAX_NEW_TOKENS=128  # vs default 256

# Result: 8s → 4s (2x faster)
# Trade-off: Shorter answers
```

### Indexing Performance

```bash
# Increase embedding batch size (all platforms)
export EMBED_BATCH=64  # vs default 32

# Result: 150s → 75s for 10k chunks (2x faster)
# Memory: +100-200MB
```

```bash
# MLX Backend (M1 Mac ONLY)
pip install mlx mlx-embedding-models
export EMBED_BACKEND=mlx

# Result: 150s → 20s for 10k chunks (7.5x faster)
# Already integrated in code!
```

### Caching

```bash
# Enable semantic query cache
export ENABLE_SEMANTIC_CACHE=1
export SEMANTIC_CACHE_THRESHOLD=0.92

# Result: 30-40% faster for repetitive queries
# Memory: ~50KB per cached query
```

---

## Performance Baselines

### M1 Mac Mini 16GB

**Indexing (10,000 chunks)**:
- Document loading: 20s
- Chunking: 60s
- Embedding: 150s ← Bottleneck
- Database: 8s
- **Total: 238s (4 min)**

**With MLX**: 108s (1.8 min) - **2.2x faster**

**Query**:
- Embedding: 50ms
- Vector search: 100ms
- LLM generation: 8000ms ← Bottleneck
- **Total: 8160ms**

**With vLLM (GPU)**: 2660ms - **3.1x faster**

### RTX 4090 (GPU Server)

**Indexing**: 103s - **2.3x faster than M1**
**Query**: 2610ms - **3.1x faster than M1**

---

## Critical Checks

### 1. Is HNSW Index Present?

```bash
python scripts/benchmarking_performance_analysis.py --database-check

# Look for: "index_type": "HNSW"
# Without HNSW: 500-2000ms vector search
# With HNSW:     50-100ms vector search (10-20x faster)
```

**Fix if missing**:
```bash
python rag_low_level_m1_16gb_verbose.py --create-hnsw-index
```

### 2. Memory Pressure?

```bash
# Check available RAM
python scripts/benchmarking_performance_analysis.py --system-resources

# Healthy: 4-6GB available (on 16GB system)
# Warning: <3GB available → reduce batch sizes
```

**Fix if low memory**:
```bash
export EMBED_BATCH=32
export DB_INSERT_BATCH=100
```

### 3. GPU Utilization?

```bash
# M1 Mac: Check for MPS
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True

# NVIDIA: Check for CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## Configuration Cheat Sheet

### Development (M1 Mac)

```bash
# .env
EMBED_MODEL=BAAI/bge-small-en
EMBED_BATCH=64
EMBED_BACKEND=mlx        # 7.5x faster embedding
CHUNK_SIZE=700
CHUNK_OVERLAP=150
TOP_K=4
MAX_NEW_TOKENS=256
USE_VLLM=0               # CPU-based llama.cpp
```

### Production (GPU Server)

```bash
# .env
EMBED_MODEL=BAAI/bge-small-en
EMBED_BATCH=128          # Higher batch for GPU
EMBED_BACKEND=huggingface
CHUNK_SIZE=700
CHUNK_OVERLAP=150
TOP_K=4
MAX_NEW_TOKENS=256
USE_VLLM=1               # GPU-accelerated vLLM ⭐
ENABLE_SEMANTIC_CACHE=1  # Enable caching ⭐
```

---

## When to Use Each Optimization

### Always Enable

- ✅ HNSW index (10-20x faster vector search)
- ✅ Embedding batch size ≥64 (1.5-2x faster)
- ✅ Memory management (`gc.collect()` - already in code)

### Enable for Production

- ✅ vLLM server (3-4x faster queries) - **HIGHEST IMPACT**
- ✅ Semantic caching (30-40% faster for repetitive queries)
- ✅ Connection pooling (5x for high-load APIs)

### Enable for M1 Mac

- ✅ MLX backend (5-20x faster embedding) - **Apple Silicon ONLY**
- ✅ MPS GPU acceleration (already enabled)

### Skip Unless Needed

- ❌ Async pipeline (complex, only for batch workloads)
- ❌ Distributed embedding (overkill for <100k chunks)
- ❌ Query reranking (quality vs speed trade-off)

---

## Performance Budget Compliance

### Indexing Budget: 10k chunks in <2 min

```
Current (M1):    238s ❌ MISS (4 min)
With MLX:        108s ⚠  CLOSE (1.8 min)
With GPU:        103s ⚠  CLOSE (1.7 min)

Status: Need MLX or GPU to meet target
```

### Query Budget: <3s end-to-end

```
Current (M1):    8.2s  ❌ MISS
With vLLM:       2.6s  ✅ MEETS TARGET

Status: Need vLLM (GPU) to meet target
```

### Memory Budget: <10GB peak (on 16GB system)

```
Current:         10GB  ✅ MEETS TARGET
With optimizations: 10GB  ✅ STABLE

Status: No memory issues
```

---

## Common Pitfalls

### 1. No Vector Index

**Symptom**: Vector search takes 500-2000ms
**Fix**: `python rag_low_level_m1_16gb_verbose.py --create-hnsw-index`
**Impact**: 10-20x faster retrieval

### 2. Small Batch Size

**Symptom**: Embedding takes >3 minutes for 10k chunks
**Fix**: `export EMBED_BATCH=64`
**Impact**: 1.5-2x faster embedding

### 3. CPU-Only LLM

**Symptom**: Queries take 8-15 seconds
**Fix**: Deploy vLLM server + `USE_VLLM=1`
**Impact**: 3-4x faster queries

### 4. Missing MLX on M1

**Symptom**: Embedding uses PyTorch (slower)
**Fix**: `pip install mlx mlx-embedding-models && export EMBED_BACKEND=mlx`
**Impact**: 5-20x faster embedding (M1 only)

### 5. Context Window Overflow

**Symptom**: Error "context window exceeded"
**Fix**: Reduce `CHUNK_SIZE` or `TOP_K`
```bash
export CHUNK_SIZE=500  # vs 700
export TOP_K=3         # vs 4
```

---

## Monitoring Commands

```bash
# Full system analysis
python scripts/benchmarking_performance_analysis.py --analyze-all

# Database check (indexes, query speed)
python scripts/benchmarking_performance_analysis.py --database-check

# Embedding benchmark
python scripts/benchmarking_performance_analysis.py --embedding-benchmark

# Performance dashboard
python scripts/generate_performance_dashboard.py

# Regression tests
pytest tests/test_performance_regression.py -v
```

---

## Quick Diagnostic

```bash
# 1. Check current performance
time python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "What are the main findings?"

# 2. Check GPU availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()}')"

# 3. Check database indexes
psql -d vector_db -c "
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE indexname LIKE '%hnsw%' OR indexname LIKE '%ivfflat%';"

# 4. Check memory
python -c "import psutil; m=psutil.virtual_memory(); print(f'Used: {m.percent}%, Available: {m.available/1e9:.1f}GB')"
```

---

## Getting Help

**Full audit report**: `/Users/frytos/code/llamaIndex-local-rag/PERFORMANCE_ENGINEERING_AUDIT.md`

**Documentation**:
- Performance quick start: `docs/PERFORMANCE_QUICK_START.md`
- Environment variables: `docs/ENVIRONMENT_VARIABLES.md`
- vLLM setup: `docs/VLLM_SERVER_GUIDE.md`

**Scripts**:
- Analysis: `scripts/benchmarking_performance_analysis.py`
- Benchmarks: `scripts/run_comprehensive_benchmark.py`
- Dashboard: `scripts/generate_performance_dashboard.py`

---

**Last Updated**: 2026-01-09
**Performance Engineering Audit**: Complete
**Overall Rating**: B+ (Good foundation, clear optimization path)
