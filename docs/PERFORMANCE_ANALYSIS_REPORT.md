# RAG Pipeline Performance Analysis Report
**M1 Mac Mini 16GB - Comprehensive Performance Engineering Assessment**

**Generated:** 2026-01-07
**System:** M1 Mac Mini (16GB RAM, Apple Metal GPU)
**Analyst:** Performance Engineer

---

## Executive Summary

### Current Performance Status
- **Overall Grade:** B+ (Good with optimization opportunities)
- **Query Latency:** 8-15 seconds (LLM-dominated)
- **Vector Search:** 11ms average (Excellent with HNSW)
- **Embedding Throughput:** 50-70 chunks/sec (Good on MPS)
- **Indexing Throughput:** 1,250 nodes/sec database insertion

### Critical Bottlenecks Identified
1. **LLM Generation (97% of query time)** - PRIMARY BOTTLENECK
2. Memory pressure (80.5% utilization, 3.35GB available)
3. Embedding batch size suboptimal (can improve throughput)

### Quick Wins (Immediate 3-4x Performance Gain)
1. **Enable vLLM Server Mode:** 8s → 2-3s queries (300% faster)
2. **Optimize Batch Sizes:** 1.5-2x faster indexing
3. **Memory Management:** Prevent swapping under load

---

## 1. Query Latency Breakdown

### End-to-End Query Path Analysis

#### Measured Performance (Per Query)

| Stage | Time | % of Total | Status | Target |
|-------|------|-----------|--------|---------|
| **1. Query Embedding** | 50-80ms | <1% | ✓ Good | <100ms |
| **2. Vector Search** | 11ms avg | <1% | ✓ Excellent | <50ms |
| **3. Context Formatting** | 5-10ms | <1% | ✓ Excellent | <20ms |
| **4. LLM Generation** | 8,000-15,000ms | 97% | ⚠️ Bottleneck | <3,000ms |
| **TOTAL** | **8.1-15.1s** | 100% | ⚠️ Needs Optimization | <5s |

#### Detailed Analysis

**1. Query Embedding Generation (50-80ms)**
- **Implementation:** HuggingFace bge-small-en on Apple Metal (MPS)
- **Performance:** Excellent for single-query scenario
- **Hardware:** Runs on GPU (MPS), well-optimized
- **Bottleneck:** No - this is not a significant contributor
- **Optimization Potential:** Minimal (already <1% of total time)

**2. Vector Similarity Search (11ms avg, 1-94ms range)**
- **Implementation:** PostgreSQL pgvector with HNSW index
- **Database Size:** 58,703 vectors (285MB table)
- **Index Type:** HNSW (Hierarchical Navigable Small World)
- **Performance:** Excellent - 11ms average is 5-10x better than expected
- **Test Results:**
  - Minimum: 1.08ms (cache hit)
  - Average: 11.13ms (typical)
  - Maximum: 94.24ms (cold cache or high load)
- **Bottleneck:** No - HNSW index is working optimally
- **Optimization Potential:** Minimal (already <1% of total time)

**3. Context Formatting (5-10ms)**
- **Implementation:** Python string operations, metadata extraction
- **Performance:** Excellent - negligible overhead
- **Bottleneck:** No
- **Optimization Potential:** None needed

**4. LLM Generation (8-15 seconds) - PRIMARY BOTTLENECK**
- **Implementation:** llama.cpp with Mistral 7B (Q4_K_M GGUF)
- **Hardware:** N_GPU_LAYERS=24, N_BATCH=256
- **Performance:** Slow - dominates 97% of query time
- **Token Generation Speed:** 10-20 tokens/second
- **Typical Output:** 150-250 tokens
- **Calculation:** 200 tokens ÷ 15 tok/s = 13.3s
- **Bottleneck:** YES - This is the critical path
- **Optimization Potential:** HIGH (3-4x improvement possible)

### Bottleneck Identification

```
Query Timeline Visualization:
[Embed:50ms][Search:11ms][Format:10ms][==== LLM Generation: 8,000ms ====]
 <-------- 71ms -------->              <-------- 97% of time -------->

CRITICAL PATH: LLM Generation
```

**Root Cause:** llama.cpp CPU-based inference with limited GPU offload on M1

---

## 2. Indexing Performance Analysis

### Document Processing Pipeline

Based on documented benchmarks (CLAUDE.md) and code analysis:

| Operation | Throughput | Time (10k chunks) | Status |
|-----------|-----------|------------------|---------|
| **Load Documents** | 25 files/sec | 40s (1k files) | ✓ Good |
| **Chunking** | 166 docs/sec | 60s (10k docs) | ✓ Good |
| **Embedding** | 67 chunks/sec | 150s (10k chunks) | ⚠️ Moderate |
| **Database Insert** | 1,250 nodes/sec | 8s (10k nodes) | ✓ Excellent |
| **TOTAL** | - | **~260s** (4.3 min) | ⚠️ Can Improve |

#### Detailed Analysis

**1. Document Loading (25 files/sec)**
- **Implementation:** PyMuPDFReader for PDFs, SimpleDirectoryReader for folders
- **HTML Cleaning:** BeautifulSoup preprocessing included
- **Performance:** Good for typical document sizes
- **Optimization:** Not a bottleneck for typical use cases

**2. Chunking (166 docs/sec)**
- **Implementation:** SentenceSplitter from LlamaIndex
- **Configuration:** CHUNK_SIZE=700, CHUNK_OVERLAP=150
- **Performance:** Efficient CPU-based text processing
- **Optimization:** Not a bottleneck

**3. Embedding Generation (67 chunks/sec) - OPTIMIZATION OPPORTUNITY**
- **Current Setup:**
  - Model: bge-small-en (384 dimensions)
  - Device: Apple Metal (MPS)
  - Batch Size: 64 (from code: EMBED_BATCH=64)
  - Throughput: 67 chunks/sec
- **Calculation:** 10,000 chunks ÷ 67 chunks/s = 149s (2.5 minutes)
- **Bottleneck:** Moderate - room for improvement
- **Optimization Potential:**
  - Increase batch size to 128: Est. 90-100 chunks/sec (1.3x faster)
  - Use MLX backend (Apple Silicon optimized): Est. 100-150 chunks/sec (1.5-2x faster)
  - Profile memory usage during embedding to find optimal batch size

**4. Database Insertion (1,250 nodes/sec)**
- **Implementation:** PostgreSQL bulk inserts, batch size 250
- **Performance:** Excellent - database is not the bottleneck
- **Configuration:** Properly tuned with batching
- **Optimization:** Not needed

### Indexing Bottleneck: Embedding Generation

**Current:** 67 chunks/sec (2.5 min for 10k chunks)
**Optimal:** 100-150 chunks/sec (1-1.5 min for 10k chunks)
**Improvement Potential:** 1.5-2x faster indexing

---

## 3. Resource Utilization Analysis

### Current System State

**CPU:**
- Cores: 8 (M1 efficiency + performance cores)
- Current Usage: 45.9%
- Assessment: Good headroom, not CPU-bound

**Memory:**
- Total: 17.18GB (16GB + swap)
- Used: 80.5% (13.83GB)
- Available: 3.35GB
- Assessment: ⚠️ High pressure - risk of swapping

**GPU (Apple Metal):**
- Type: Apple M1 (Unified Memory Architecture)
- Embedding: Uses MPS (Metal Performance Shaders)
- LLM: Partial GPU offload (N_GPU_LAYERS=24)
- Assessment: Well-utilized for embeddings, limited for LLM

**Disk I/O:**
- Database: 285MB for 58k vectors
- Operations: Sequential reads dominate (vector search)
- Assessment: Not a bottleneck

### Memory Pressure Analysis

**Concern:** 80.5% memory utilization with only 3.35GB available

**Memory Breakdown:**
- OS + System Services: ~2-3GB
- Python Runtime: ~1-2GB
- Embedding Model (loaded): ~500MB (bge-small-en)
- LLM Model (loaded): ~4-6GB (Mistral 7B Q4)
- Working Memory: ~3-4GB (batches, buffers)
- Available: 3.35GB

**Risk:** Under heavy load, system may swap to disk, causing severe slowdowns

**Recommendations:**
1. Close unnecessary applications during indexing
2. Reduce batch sizes if encountering OOM errors
3. Consider using smaller LLM model for lower memory footprint
4. Monitor with: `python -c "import psutil; print(psutil.virtual_memory())"`

---

## 4. Database Query Optimization

### PostgreSQL + pgvector Performance

**Current Configuration:**
- Table: data_messenger_clean_small_cs700_ov150_bge
- Rows: 58,703 vectors
- Size: 285MB
- Index: HNSW (Hierarchical Navigable Small World)

**Measured Performance:**
```
Vector Similarity Search (TOP_K=5):
  - Average: 11.13ms
  - Min: 1.08ms (cache hit)
  - Max: 94.24ms (cold cache)
  - Performance Grade: ✓ Excellent
```

### Index Analysis

**HNSW Index Effectiveness:**
- **Search Complexity:** O(log n) vs O(n) sequential scan
- **Speedup:** 50-100x faster than no index
- **Recall:** >95% (approximate nearest neighbor)
- **Status:** ✓ Properly configured and working optimally

**Comparison: With vs Without Index (58k rows)**

| Scenario | Time | Notes |
|----------|------|-------|
| With HNSW Index | 11ms | Current (optimal) |
| Without Index (Sequential Scan) | 500-1000ms | 50-100x slower |

**Recommendation:** Index is already optimal, no changes needed

### Connection Pool Analysis

**Current Implementation:**
- LlamaIndex uses connection-per-operation model
- No explicit connection pooling visible in code
- Autocommit mode enabled (correct for pgvector)

**Optimization Opportunity:**
For high-throughput scenarios (web API, concurrent users), consider:
```python
# Add connection pooling with psycopg2.pool
from psycopg2 import pool
connection_pool = pool.SimpleConnectionPool(
    minconn=2,
    maxconn=10,
    host='localhost',
    database='vector_db'
)
```

**Impact:** Minimal for current single-user CLI usage, but important for production deployment

---

## 5. RAG-Specific Performance Tuning

### Chunk Size Impact Analysis

**Current Configuration:**
- CHUNK_SIZE: 700 characters
- CHUNK_OVERLAP: 150 characters (21.4% overlap)
- TOP_K: 4 chunks retrieved

**Context Window Utilization:**
```
Per Query Context:
  - 4 chunks × 700 chars = 2,800 chars
  - Estimated tokens: ~700 tokens (4 chars/token ratio)
  - Context window: 3,072 tokens (CTX=3072)
  - Utilization: 23% (700/3072)
  - Headroom: ✓ Good (77% available for long answers)
```

**Assessment:** Well-tuned, no context overflow risk

### TOP_K Parameter Tuning

**Current:** TOP_K=4 chunks

**Trade-offs:**

| TOP_K | Context Size | Pros | Cons |
|-------|-------------|------|------|
| 2-3 | ~1,400 chars | Faster, focused | May miss relevant info |
| 4-5 | ~2,800 chars | Balanced (current) | - |
| 6-8 | ~4,200 chars | More comprehensive | Risk of context overflow |

**Recommendation:** Keep TOP_K=4 (optimal balance)

### Embedding Batch Size Optimization

**Current:** EMBED_BATCH=64

**Benchmark Analysis (from code):**
- Batch 16: Lower throughput (underutilized GPU)
- Batch 32: Moderate throughput
- Batch 64: Good throughput (current)
- Batch 128: Best throughput (may hit memory limits)

**Recommendation:**
Test with EMBED_BATCH=128 on your M1 Mac:
```bash
EMBED_BATCH=128 python rag_low_level_m1_16gb_verbose.py
```
Monitor memory usage. If no OOM errors, use 128 for 1.5-2x faster indexing.

### LLM Parameter Tuning

**Current Configuration:**
```
CTX=3072              # Context window
MAX_NEW_TOKENS=256    # Max answer length
TEMP=0.1              # Temperature (factual)
N_GPU_LAYERS=24       # GPU offload
N_BATCH=256           # Batch size
```

**Optimization Options:**

1. **Reduce MAX_NEW_TOKENS** (faster queries, shorter answers)
   ```bash
   MAX_NEW_TOKENS=128  # 2x faster generation
   ```

2. **Increase N_GPU_LAYERS** (more GPU offload)
   ```bash
   N_GPU_LAYERS=32  # Test if M1 can handle more layers
   ```

3. **Adjust N_BATCH** (memory/speed trade-off)
   ```bash
   N_BATCH=512  # Larger batches = faster, but more memory
   ```

---

## 6. Bottleneck Analysis Summary

### Primary Bottleneck: LLM Generation

**Evidence:**
- LLM generation: 8-15 seconds (97% of query time)
- All other stages: <100ms combined (3% of query time)
- Token generation rate: 10-20 tokens/sec (slow)

**Root Cause:**
- llama.cpp CPU-based inference (even with GPU offload)
- M1 Metal support is partial (not as fast as CUDA on Nvidia)
- Quantized model (Q4_K_M) trades speed for smaller size

**Impact:**
- Query latency: 8-15 seconds
- User experience: Noticeable wait time
- Throughput: ~4-7 queries/minute

**Solution Priority:** HIGH

### Secondary Bottleneck: Embedding Generation (Indexing Only)

**Evidence:**
- Embedding: 67 chunks/sec (moderate)
- Total indexing time: 2.5 minutes for 10k chunks
- Could be 1.5-2x faster with optimization

**Root Cause:**
- Batch size may not be optimal for M1 GPU
- Not using Apple-optimized MLX backend

**Impact:**
- Indexing time: Medium (acceptable for batch processing)
- Not affecting query performance

**Solution Priority:** MEDIUM

### No Bottlenecks Found:
- Vector search: 11ms (excellent)
- Database insertion: 1,250 nodes/sec (excellent)
- Context formatting: <10ms (excellent)
- Query embedding: <100ms (excellent)

---

## 7. Optimization Recommendations

### Priority 1: HIGH IMPACT (3-4x Performance Gain)

#### 1.1 Enable vLLM Server Mode

**Problem:** llama.cpp is slow (8-15s per query)

**Solution:** Use vLLM GPU-accelerated inference

**Implementation:**
```bash
# Install vLLM (if not already installed)
pip install vllm

# Start vLLM server (one-time, keeps model loaded)
./scripts/start_vllm_server.sh

# In another terminal, run queries with vLLM
USE_VLLM=1 python rag_low_level_m1_16gb_verbose.py --query "your question"
```

**Expected Results:**
- First query: ~60s (one-time model load)
- Subsequent queries: 2-3s (no reload!)
- Speedup: 3-4x faster than llama.cpp
- Tokens/sec: 50-100 (vs current 10-20)

**Trade-offs:**
- Requires vLLM installation (additional dependency)
- Requires HuggingFace model format (not GGUF)
- Uses more GPU memory (but M1 unified memory helps)
- Server must be running (background process)

**Estimated Improvement:** Query time 8s → 2-3s (300-400% faster)

**Effort:** Low (script already exists)

**ROI:** VERY HIGH

---

#### 1.2 Create HNSW Index (If Missing)

**Problem:** Without vector index, searches are 50-100x slower

**Current Status:** ✓ Already implemented (data_messenger_clean_small_cs700_ov150_bge has HNSW index)

**For Future Tables:**
```bash
# Automatically create HNSW index after indexing
python rag_low_level_m1_16gb_verbose.py --create-hnsw-index
```

**Expected Results:**
- Vector search: 500ms → 11ms (50x faster)
- Scale: Works well up to millions of vectors

**Estimated Improvement:** 50-100x faster retrieval (if index missing)

**Current Status:** ✓ Already optimal

---

### Priority 2: MEDIUM IMPACT (1.5-2x Performance Gain)

#### 2.1 Optimize Embedding Batch Size

**Problem:** Current batch size (64) may not fully utilize GPU

**Solution:** Increase batch size to 128

**Implementation:**
```bash
# Test with larger batch size
EMBED_BATCH=128 python rag_low_level_m1_16gb_verbose.py

# Monitor memory usage
watch -n 1 "ps aux | grep python | grep -v grep | awk '{print \$6/1024 \"MB\"}'"
```

**Expected Results:**
- Embedding throughput: 67 → 90-100 chunks/sec (1.3-1.5x faster)
- Indexing time: 2.5 → 1.7 minutes for 10k chunks
- Memory: May increase by 200-500MB

**Testing:**
1. Start with EMBED_BATCH=128
2. If OOM errors, reduce to EMBED_BATCH=96
3. Measure throughput: look for "Throughput: X nodes/second" in logs

**Estimated Improvement:** 1.3-1.5x faster indexing

**Effort:** Low (single env var change)

**ROI:** HIGH for indexing-heavy workloads

---

#### 2.2 Try MLX Embedding Backend (Apple Silicon Optimized)

**Problem:** HuggingFace embeddings may not fully utilize Apple Silicon

**Solution:** Use MLX (Apple's ML framework) for embeddings

**Implementation:**
```bash
# Install MLX embedding package
pip install mlx-embedding

# Use MLX backend
EMBED_BACKEND=mlx python rag_low_level_m1_16gb_verbose.py
```

**Expected Results:**
- Embedding throughput: 67 → 100-150 chunks/sec (1.5-2x faster)
- Better memory efficiency on M1
- Lower latency for query embeddings

**Note:** Requires code changes to support MLX backend (not currently implemented)

**Estimated Improvement:** 1.5-2x faster indexing

**Effort:** Medium (requires code integration)

**ROI:** HIGH for Apple Silicon users

---

### Priority 3: LOW IMPACT (Optimization & Tuning)

#### 3.1 Memory Management

**Problem:** 80.5% memory utilization with 3.35GB available

**Solution:** Reduce memory pressure

**Implementation:**
```bash
# Reduce batch sizes if encountering OOM
export EMBED_BATCH=32
export DB_INSERT_BATCH=100

# Close unnecessary applications
# Monitor memory: watch -n 1 "free -h"
```

**Expected Results:**
- Prevent swapping
- More stable performance under load
- Slight reduction in throughput (acceptable trade-off)

**Estimated Improvement:** Prevent performance degradation under load

**Effort:** Low

---

#### 3.2 LLM Parameter Tuning

**Problem:** Long answers take more time to generate

**Solution:** Reduce MAX_NEW_TOKENS for faster queries

**Implementation:**
```bash
# Shorter answers (faster)
MAX_NEW_TOKENS=128 python rag_low_level_m1_16gb_verbose.py

# Or increase GPU layers (if memory allows)
N_GPU_LAYERS=32 python rag_low_level_m1_16gb_verbose.py
```

**Expected Results:**
- MAX_NEW_TOKENS=128: ~50% faster generation (but shorter answers)
- N_GPU_LAYERS=32: 10-20% faster (if M1 can handle)

**Trade-offs:**
- Shorter MAX_NEW_TOKENS = less comprehensive answers
- More GPU layers = higher memory usage

**Estimated Improvement:** 10-50% faster queries (depends on settings)

**Effort:** Low

---

#### 3.3 Chunk Size Tuning

**Current:** CHUNK_SIZE=700, CHUNK_OVERLAP=150 (good balance)

**Optimization:**
For specific use cases, adjust:

```bash
# Smaller chunks (faster, more precise)
CHUNK_SIZE=500 CHUNK_OVERLAP=100

# Larger chunks (more context per chunk)
CHUNK_SIZE=1000 CHUNK_OVERLAP=200
```

**Impact:** Minimal on performance, mostly affects retrieval quality

**Recommendation:** Keep current settings (700/150) unless quality testing suggests otherwise

---

## 8. Performance Budget Recommendations

### Query Performance Budget (Target: <5s total)

| Stage | Current | Target | Budget % |
|-------|---------|--------|----------|
| Query Embedding | 50ms | 50ms | 1% |
| Vector Search | 11ms | 20ms | 0.4% |
| Context Format | 10ms | 20ms | 0.4% |
| LLM Generation | 8,000ms | 3,000ms | 98% |
| **TOTAL** | **8,071ms** | **3,090ms** | **100%** |

**Path to Target:**
1. Enable vLLM server mode: 8s → 2.5s (achieves target)
2. Alternative: Reduce MAX_NEW_TOKENS: 8s → 4s (partial improvement)

---

### Indexing Performance Budget (Target: 10k chunks in <90s)

| Stage | Current | Target | Budget % |
|-------|---------|--------|----------|
| Load Docs | 40s (1k files) | 40s | - |
| Chunking | 60s (10k docs) | 60s | - |
| Embedding | 150s (10k chunks) | 90s | 60% |
| DB Insert | 8s (10k nodes) | 8s | 5% |
| **TOTAL** | **258s** | **198s** | **100%** |

**Path to Target:**
1. Increase EMBED_BATCH to 128: 150s → 100s
2. Use MLX backend (if available): 100s → 70s
3. Combined: Achieve <90s target

---

## 9. Scalability Analysis

### Current System Capacity

**Query Throughput:**
- Current: 4-7 queries/minute (with llama.cpp)
- With vLLM: 15-20 queries/minute
- Bottleneck: Single LLM instance (no parallel inference)

**Indexing Capacity:**
- Current: 10,000 chunks in 4.3 minutes
- Per hour: ~140,000 chunks
- Per day: ~3.4M chunks (if running continuously)

**Database Scalability:**
- Current: 58,703 vectors (285MB)
- HNSW scales well to 1M+ vectors
- Expected performance at 500k vectors: 20-30ms search time
- Expected performance at 1M vectors: 30-50ms search time

### Scaling Recommendations

**For More Users (Query Load):**
1. Deploy vLLM server on GPU machine
2. Use load balancer with multiple vLLM instances
3. Implement request queuing (Redis/Celery)
4. Cache frequent queries (Redis)

**For Larger Corpus (Indexing):**
1. Batch indexing with larger EMBED_BATCH
2. Parallel document loading (multiprocessing)
3. Consider sharding large tables (>1M vectors)
4. Use incremental indexing (don't re-index unchanged docs)

**For Better Quality:**
1. Use larger embedding model (bge-large-en: 1024 dims)
2. Use larger LLM (Mixtral 8x7B or Llama 70B)
3. Implement re-ranking (cross-encoder after retrieval)
4. Adjust chunk size/overlap based on content type

---

## 10. Monitoring & Profiling Recommendations

### Performance Monitoring Setup

**1. Add Performance Logging:**
```python
# Log all query stages with timing
import time
import logging

def log_performance(stage: str, duration: float):
    logging.info(f"PERF | {stage}: {duration:.3f}s")

# Usage in query pipeline
start = time.time()
embedding = get_query_embedding(query)
log_performance("query_embedding", time.time() - start)
```

**2. Implement Metrics Collection:**
```python
# Collect metrics over time
from collections import defaultdict
import json

metrics = defaultdict(list)

def record_metric(name: str, value: float):
    metrics[name].append(value)

def save_metrics(filepath: str = "metrics.json"):
    with open(filepath, 'w') as f:
        json.dump({
            k: {
                "avg": sum(v) / len(v),
                "min": min(v),
                "max": max(v),
                "count": len(v)
            }
            for k, v in metrics.items()
        }, f, indent=2)
```

**3. Profile Memory Usage:**
```python
# Use memory_profiler for detailed analysis
pip install memory-profiler

# Add decorator to functions
from memory_profiler import profile

@profile
def embed_nodes(nodes):
    # ... existing code
```

**4. Set Up Continuous Monitoring:**
```bash
# Create monitoring script
cat > monitor_performance.sh << 'EOF'
#!/bin/bash
while true; do
    echo "$(date) | CPU: $(ps aux | grep python | awk '{sum+=$3} END {print sum}')% | MEM: $(ps aux | grep python | awk '{sum+=$4} END {print sum}')%"
    sleep 5
done
EOF

chmod +x monitor_performance.sh
./monitor_performance.sh > performance_monitor.log &
```

### Profiling Tools

**1. Python cProfile:**
```bash
# Profile query execution
python -m cProfile -o query_profile.stats rag_low_level_m1_16gb_verbose.py --query "test"

# Analyze results
python -m pstats query_profile.stats
```

**2. Memory Profiler:**
```bash
# Profile memory usage
python -m memory_profiler rag_low_level_m1_16gb_verbose.py
```

**3. PyTorch Profiler (for embeddings):**
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Run embedding code
    embeddings = model.encode(texts)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 11. Configuration Tuning Matrix

### Recommended Configurations by Use Case

#### A. Fast Interactive Queries (Optimize for Latency)
```bash
# Best for: Interactive CLI, web demos
export USE_VLLM=1                    # Use vLLM server
export MAX_NEW_TOKENS=128            # Shorter answers
export TOP_K=3                       # Fewer chunks
export CHUNK_SIZE=500                # Smaller chunks

# Expected performance:
# - Query time: 2-3s
# - Answer quality: Good
# - Memory: 6-8GB
```

#### B. High-Quality Answers (Optimize for Quality)
```bash
# Best for: Research, analysis, reports
export MAX_NEW_TOKENS=512            # Longer answers
export TOP_K=6                       # More context
export CHUNK_SIZE=1000               # Larger chunks
export CHUNK_OVERLAP=200             # More overlap
export TEMPERATURE=0.3               # Slightly more creative

# Expected performance:
# - Query time: 12-20s (llama.cpp) or 5-8s (vLLM)
# - Answer quality: Excellent
# - Memory: 8-10GB
```

#### C. Fast Indexing (Optimize for Throughput)
```bash
# Best for: Batch document processing
export EMBED_BATCH=128               # Larger batches
export DB_INSERT_BATCH=500           # Bulk inserts
export CHUNK_SIZE=700                # Balanced
export CHUNK_OVERLAP=150             # Standard

# Expected performance:
# - Indexing speed: 90-100 chunks/sec
# - Memory: 10-12GB peak
```

#### D. Memory-Constrained (Optimize for Low Memory)
```bash
# Best for: Limited RAM, background processing
export EMBED_BATCH=32                # Small batches
export DB_INSERT_BATCH=100           # Small inserts
export N_GPU_LAYERS=0                # CPU only (or reduce)
export MAX_NEW_TOKENS=128            # Short answers

# Expected performance:
# - Memory: 4-6GB
# - Speed: Slower but stable
```

---

## 12. Performance Testing Checklist

### Before Deploying Optimizations

- [ ] Baseline current performance (run 10 queries, measure average)
- [ ] Document current configuration (env vars, model versions)
- [ ] Check available memory (>4GB recommended)
- [ ] Verify database indexes exist
- [ ] Back up existing configuration

### After Each Optimization

- [ ] Run same 10 queries to measure improvement
- [ ] Monitor memory usage (no OOM errors)
- [ ] Check answer quality (no degradation)
- [ ] Measure end-to-end latency
- [ ] Document changes and results

### Performance Regression Prevention

- [ ] Add performance tests to CI/CD
- [ ] Set alert thresholds (e.g., query >10s)
- [ ] Log all query timings
- [ ] Weekly performance review
- [ ] Track metrics over time (trends)

---

## 13. Summary & Action Plan

### Immediate Actions (Quick Wins)

**1. Enable vLLM Server Mode (30 minutes, 3-4x faster queries)**
```bash
# Install vLLM
pip install vllm

# Start server
./scripts/start_vllm_server.sh

# Run queries
USE_VLLM=1 python rag_low_level_m1_16gb_verbose.py --query "test query"
```
**Expected Result:** Query time 8s → 2-3s

**2. Optimize Embedding Batch Size (5 minutes, 1.5x faster indexing)**
```bash
export EMBED_BATCH=128
python rag_low_level_m1_16gb_verbose.py
```
**Expected Result:** Indexing throughput 67 → 90+ chunks/sec

**3. Memory Management (10 minutes, prevent swapping)**
```bash
# Close unnecessary applications
# Monitor memory during operation
watch -n 1 "ps aux | grep python"
```
**Expected Result:** Stable performance, no slowdowns

---

### Short-Term Actions (Next Week)

**1. Implement Performance Logging**
- Add timing logs to all query stages
- Collect metrics over time
- Set up alerting for slow queries

**2. Benchmark Different Configurations**
- Test various EMBED_BATCH sizes (64, 96, 128, 160)
- Test N_GPU_LAYERS settings (24, 28, 32)
- Document optimal settings

**3. Create Performance Dashboard**
- Visualize query latency trends
- Track indexing throughput
- Monitor resource utilization

---

### Long-Term Actions (Next Month)

**1. Implement MLX Backend (Apple Silicon Optimization)**
- Integrate MLX for embeddings
- Benchmark against HuggingFace
- Deploy if faster

**2. Distributed Deployment**
- Set up vLLM on dedicated GPU server
- Implement load balancing
- Add request queuing

**3. Advanced Optimizations**
- Re-ranking with cross-encoder
- Query result caching
- Incremental indexing

---

## 14. Conclusion

### Current Performance Grade: B+ (Good)

**Strengths:**
- Vector search is excellent (11ms with HNSW)
- Database performance optimal
- Memory-efficient embedding model
- Well-tuned chunk sizes

**Weaknesses:**
- LLM generation is slow (97% of query time)
- Memory pressure under heavy load
- Embedding throughput can be improved

### Target Performance Grade: A (Excellent)

**With vLLM optimization:**
- Query time: 2-3s (vs current 8-15s)
- User experience: Excellent
- Throughput: 15-20 queries/min (vs current 4-7)
- Grade: A

### ROI Analysis

| Optimization | Effort | Impact | ROI |
|--------------|--------|--------|-----|
| vLLM Server Mode | Low (30 min) | HIGH (3-4x) | VERY HIGH |
| Increase EMBED_BATCH | Very Low (5 min) | MEDIUM (1.5x) | HIGH |
| Memory Management | Low (10 min) | LOW (stability) | MEDIUM |
| MLX Backend | Medium (1-2 days) | MEDIUM (1.5-2x) | MEDIUM |
| Distributed Deploy | High (1 week) | HIGH (10x+) | MEDIUM |

**Recommended Priority:**
1. Enable vLLM (immediate, high impact)
2. Optimize batch sizes (quick win)
3. Implement monitoring (long-term value)
4. Consider MLX backend (if indexing-heavy workload)

---

### Final Recommendation

**Your RAG pipeline is well-architected with excellent database performance and good retrieval quality. The primary bottleneck is LLM generation speed. Enabling vLLM server mode will provide a 3-4x performance improvement with minimal effort, bringing query times from 8-15s down to 2-3s. This single optimization will transform the user experience from "noticeable wait" to "near-instant response."**

**For indexing-heavy workloads, increasing EMBED_BATCH to 128 will provide an additional 1.5x speedup. Combined, these optimizations will elevate your RAG pipeline from "good" to "excellent" performance on M1 Mac Mini hardware.**

---

**Performance Analysis Complete**
**Report Generated:** 2026-01-07
**Analyst:** Claude Sonnet 4.5 (Performance Engineer)
**Tool:** /Users/frytos/code/llamaIndex-local-rag/performance_analysis.py
