# Post-Embedding Execution Plan

**Created**: 2024-12-19 06:50
**Status**: Ready to execute when embedding completes

---

## Prerequisites Check

Before starting, verify embedding is complete:
```bash
# Check if script finished (no process running)
ps aux | grep rag_low_level_m1_16gb_verbose

# Verify row count (should be 47,651)
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "SELECT COUNT(*) FROM inbox_clean;"
```

---

## Phase 1: Install MLX (2 minutes)

```bash
# Navigate to project
cd /Users/frytos/code/llamaIndex-local-rag

# Activate venv
source .venv/bin/activate

# Install MLX dependencies
pip install mlx mlx-embedding-models rank-bm25

# Verify installation
python -c "import mlx.core as mx; print(f'✓ MLX {mx.__version__}')"
python -c "from mlx_embedding_models.embedding import EmbeddingModel; print('✓ mlx-embedding-models')"
python -c "from rank_bm25 import BM25Okapi; print('✓ rank-bm25')"
```

**Expected output**: All three checks pass with ✓

---

## Phase 2: Benchmark MLX vs HuggingFace (5 minutes)

```bash
# Compare both backends with your model
python scripts/benchmark_embeddings.py \
  --compare \
  --model BAAI/bge-large-en-v1.5 \
  --batch-sizes "16,32,64,128"
```

**Expected results**:
- MLX should be 5-20x faster than HuggingFace
- Cosine similarity > 0.999 (embeddings are consistent)
- See exact throughput for your M1 Mac Mini

**Save the output** - you'll want to reference these numbers!

---

## Phase 3: Apply HNSW Index (3-5 minutes)

```bash
# Apply HNSW index to existing table for faster queries
./scripts/apply_hnsw.sh inbox_clean
```

When prompted, type `y` and press Enter.

**Expected results**:
- Index creation: 2-5 minutes
- Index size: ~50-100 MB
- Future queries: 2-5x faster

---

## Phase 4: Test Improvements (5 minutes)

### 4.1 Test Query Speed (with HNSW)
```bash
# Time a query with new HNSW index
time python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "What did Elena say about Morocco?"
```

### 4.2 Test Reranking
```bash
# Test cross-encoder reranking
python reranker.py \
  --query "What did we discuss about traveling?" \
  --top-n 3
```

### 4.3 Test Query Cache
```bash
# First run (no cache)
time python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "participant:EB Morocco"

# Second run (should be instant with cache)
time python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "participant:EB Morocco"
```

---

## Phase 5: Test MLX on Small Subset (2 minutes)

```bash
# Index small test set with MLX
EMBED_BACKEND=mlx \
EMBED_BATCH=64 \
CHUNK_SIZE=300 \
CHUNK_OVERLAP=100 \
PDF_PATH=data/inbox_small \
PGTABLE=test_mlx_small \
RESET_TABLE=1 \
python rag_low_level_m1_16gb_verbose.py --index-only
```

**Expected results**:
- 50 conversations indexed in ~30 seconds
- No errors or warnings
- "Expected speedup: 5-20x" message shows

### Query the test index
```bash
PGTABLE=test_mlx_small \
ENABLE_FILTERS=1 \
HYBRID_ALPHA=0.5 \
python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "participant:EB Morocco"
```

---

## Phase 6: Compare Models (Optional, 10 minutes)

Test if smaller model (bge-small) maintains quality:

```bash
# Compare all three models
python scripts/compare_models.py --backend mlx
```

**Decision point**:
- If bge-small quality is acceptable → 3x faster + 1/3 storage
- If not → stick with bge-large

---

## Phase 7: Full Re-index with MLX (5-8 minutes)

**ONLY DO THIS if benchmark results look good!**

```bash
# Backup current table (optional but recommended)
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "CREATE TABLE inbox_clean_backup AS SELECT * FROM inbox_clean;"

# Re-index with MLX
EMBED_BACKEND=mlx \
EMBED_BATCH=64 \
EXTRACT_CHAT_METADATA=1 \
CHUNK_SIZE=300 \
CHUNK_OVERLAP=100 \
PDF_PATH=data/inbox_clean \
PGTABLE=inbox_mlx_optimized \
RESET_TABLE=1 \
python rag_low_level_m1_16gb_verbose.py --index-only
```

**Expected results**:
- 47,651 chunks in 5-8 minutes (vs your 45-50 minutes)
- ~150-400 chunks/sec throughput
- Peak memory < 8 GB

### Apply HNSW to new table
```bash
./scripts/apply_hnsw.sh inbox_mlx_optimized
```

---

## Phase 8: Production Query Test

Test the optimized setup:

```bash
# Query with all features enabled
PGTABLE=inbox_mlx_optimized \
HYBRID_ALPHA=0.5 \
ENABLE_FILTERS=1 \
MMR_THRESHOLD=0.7 \
TOP_K=6 \
LOG_FULL_CHUNKS=1 \
COLORIZE_CHUNKS=1 \
python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --interactive
```

**Test queries**:
1. `participant:EB after:2024-06-01 Morocco`
2. `What did we discuss about travel?`
3. `Find messages about food`

---

## Success Criteria Checklist

- [ ] MLX installed successfully
- [ ] Benchmark shows 5-20x speedup
- [ ] Embedding consistency > 0.999
- [ ] HNSW index applied (queries 2-5x faster)
- [ ] Reranking works
- [ ] Query cache works
- [ ] Small subset indexed with MLX (no errors)
- [ ] Full corpus re-indexed in 5-8 minutes (optional)
- [ ] Queries work correctly with new index

---

## Troubleshooting

### MLX Installation Issues
```bash
# If import fails, check Python version
python --version  # Should be 3.11+

# If MLX unavailable, will auto-fallback to HuggingFace
# Check logs for: "Falling back to HuggingFace backend"
```

### HNSW Index Issues
```bash
# If index creation fails, check pgvector version
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Requires pgvector >= 0.5.0 for HNSW
```

### Query Errors After MLX
```bash
# If queries fail, verify table exists
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "SELECT tablename FROM pg_tables WHERE tablename LIKE 'inbox%';"
```

---

## Performance Comparison Table

Fill this in as you test:

| Metric | Before (HF) | After (MLX) | Speedup |
|--------|-------------|-------------|---------|
| Indexing time (47k chunks) | 45-50 min | ___ min | ___x |
| Query embedding latency | ___ ms | ___ ms | ___x |
| Batch throughput | 21 ch/s | ___ ch/s | ___x |
| Query retrieval time | ___ ms | ___ ms | ___x |

---

## Next Steps After Completion

1. **Update PGTABLE default** to use optimized index
2. **Update EMBED_BACKEND** default to mlx in your env
3. **Consider using bge-small** if quality tests pass
4. **Integrate reranking** into query pipeline (optional)
5. **Enable query caching** for repeated queries (optional)

---

## Rollback Plan

If anything goes wrong:

```bash
# Restore original table (if backed up)
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "DROP TABLE IF EXISTS inbox_clean; \
   ALTER TABLE inbox_clean_backup RENAME TO inbox_clean;"

# Or just use original table
PGTABLE=inbox_clean python rag_low_level_m1_16gb_verbose.py --query-only
```

---

**Ready to execute**: Just wait for embedding to finish, then run Phase 1!
