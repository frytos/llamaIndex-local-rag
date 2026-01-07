# 🚀 START HERE - After Embedding Completes

**Current Status**: Waiting for embedding to finish (~12-15 minutes remaining)

When your embedding completes, you have **2 options**:

---

## Option 1: Quick Start (Copy/Paste Commands) ⚡

Open the file and follow step-by-step:
```bash
cat POST_EMBEDDING_PLAN.md
```

Or use the interactive script:
```bash
source QUICK_COMMANDS.sh
run_all  # Runs everything automatically
```

---

## Option 2: Manual Execution (Step-by-Step) 📋

### Step 1: Install MLX (2 min)
```bash
pip install mlx mlx-embedding-models rank-bm25
python -c "import mlx.core as mx; print('✓ MLX installed')"
```

### Step 2: Benchmark (5 min)
```bash
python scripts/benchmark_embeddings.py --compare --model BAAI/bge-large-en-v1.5
```

**Decision point**: If speedup is 5-20x → proceed. If not → troubleshoot.

### Step 3: Apply HNSW Index (3 min)
```bash
./scripts/apply_hnsw.sh inbox_clean
```

### Step 4: Test MLX on Small Subset (2 min)
```bash
EMBED_BACKEND=mlx EMBED_BATCH=64 CHUNK_SIZE=300 CHUNK_OVERLAP=100 \
  PDF_PATH=data/inbox_small PGTABLE=test_mlx_small RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only
```

### Step 5: Full Re-index with MLX (5-8 min)
**Only if benchmark looks good!**
```bash
EMBED_BACKEND=mlx EMBED_BATCH=64 EXTRACT_CHAT_METADATA=1 \
  CHUNK_SIZE=300 CHUNK_OVERLAP=100 \
  PDF_PATH=data/inbox_clean PGTABLE=inbox_mlx_optimized RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only
```

### Step 6: Apply HNSW to New Table (3 min)
```bash
./scripts/apply_hnsw.sh inbox_mlx_optimized
```

### Step 7: Test Production Query
```bash
PGTABLE=inbox_mlx_optimized HYBRID_ALPHA=0.5 ENABLE_FILTERS=1 \
  MMR_THRESHOLD=0.7 TOP_K=6 LOG_FULL_CHUNKS=1 COLORIZE_CHUNKS=1 \
  python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

---

## Expected Results 🎯

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Indexing time | 45-50 min | 5-8 min | **6-9x faster** |
| Query speed | ~300ms | ~60-150ms | **2-5x faster** |
| Query embedding | ~12ms | ~8ms | **1.5x faster** |

---

## Files Created for You

- **POST_EMBEDDING_PLAN.md** - Detailed step-by-step plan
- **QUICK_COMMANDS.sh** - Interactive commands
- **mlx_embedding.py** - MLX backend implementation
- **reranker.py** - Cross-encoder reranking
- **query_cache.py** - Query embedding cache
- **scripts/benchmark_embeddings.py** - Performance benchmarking
- **scripts/compare_models.py** - Model comparison
- **scripts/apply_hnsw.sh** - HNSW index application

---

## Quick Health Check

Before starting, verify embedding completed successfully:

```bash
# Check process is done
ps aux | grep rag_low_level_m1_16gb_verbose

# Verify row count
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "SELECT COUNT(*) FROM inbox_clean;"
# Should show: 47651
```

---

## Troubleshooting

**MLX won't install?**
- Requires Python 3.11+
- macOS only (Apple Silicon)
- Run: `python --version`

**Benchmark fails?**
- Check: `python -c "import mlx.core"`
- If fails: System will auto-fallback to HuggingFace

**HNSW fails?**
- Requires pgvector >= 0.5.0
- Check: `PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"`

---

## What Now?

**Immediately after embedding finishes:**
```bash
# Option A: Automated (recommended)
source QUICK_COMMANDS.sh && run_all

# Option B: Manual
pip install mlx mlx-embedding-models rank-bm25
python scripts/benchmark_embeddings.py --compare
# ... follow POST_EMBEDDING_PLAN.md
```

---

## Need Help?

- **Detailed plan**: `cat POST_EMBEDDING_PLAN.md`
- **Quick commands**: `source QUICK_COMMANDS.sh && show_usage`
- **Test utilities**: `python reranker.py --test`

---

**🎯 Goal**: Go from 45-minute indexing → 5-8 minutes with MLX + HNSW optimization

**Ready?** Wait for embedding to complete, then run Option 1 or 2 above!
