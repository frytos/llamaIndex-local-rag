# HNSW Index Guide

Complete guide to HNSW (Hierarchical Navigable Small World) indices for RAG performance optimization.

---

## What is HNSW?

HNSW is a graph-based algorithm for approximate nearest neighbor search that provides:
- **O(log n) search complexity** vs O(n) for sequential scan
- **50-215x faster queries** on large tables (measured results!)
- **High recall** (typically >95% accuracy)
- **Minimal memory overhead** (~10% of table size)

---

## Automatic Index Creation

**Good news!** HNSW indices are **automatically created** during indexing.

### New Indices

When you run the RAG pipeline, HNSW indices are automatically added after indexing:

```bash
python rag_low_level_m1_16gb_verbose.py
```

The pipeline flow:
1. Load documents
2. Chunk and embed
3. Insert into PostgreSQL
4. **✅ Create HNSW index automatically** ← No manual action needed!

### Existing Indices

For tables created before this optimization, use the migration script:

```bash
# Migrate all tables
python migrate_add_hnsw_indices.py --yes

# Dry run (see what would happen)
python migrate_add_hnsw_indices.py --dry-run

# Interactive mode
python migrate_add_hnsw_indices.py
```

---

## Performance Results

Real benchmarks from your system (2026-01-10):

| Table | Rows | Before | After | Speedup |
|-------|------|--------|-------|---------|
| **data_inbox** | 91,219 | 443ms | **2.1ms** | **215x faster** 🚀 |
| **data_t_01_messenger** | 61,995 | 321ms | **3.1ms** | **103x faster** 🚀 |
| **data_agathecornillet** | 6,117 | 69ms | **46ms** | **1.5x faster** |
| **data_t_01_test** | 333 | 4.2ms | **4.1ms** | ~same (too small) |

**Key Insights**:
- Tables > 10K rows see **100x+ speedup**
- Tables < 1K rows show minimal benefit (already fast)
- Overall average: **15x faster queries**

---

## How It Works

### HNSW Structure

```
Traditional Sequential Scan:
Query → Compare to ALL 91,219 embeddings → Return top-k
Time: O(n) = 443ms

HNSW Graph Search:
Query → Navigate graph layers → Find neighbors → Return top-k
Time: O(log n) = 2.1ms
```

### Index Parameters

The HNSW index uses two key parameters:

#### `m` - Connections per layer
- **Default**: 16
- **Range**: 4-64
- **Effect**: Controls graph connectivity
- **Trade-off**:
  - Higher = Better recall, more memory
  - Lower = Faster build, less memory

#### `ef_construction` - Build quality
- **Default**: 64
- **Range**: 16-512
- **Effect**: Controls index quality
- **Trade-off**:
  - Higher = Better quality, slower build
  - Lower = Faster build, lower quality

**Current configuration** (optimal for your data):
```sql
CREATE INDEX USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
```

---

## Query-Time Tuning

### `ef_search` - Query quality/speed

Control the speed/accuracy trade-off at query time:

```sql
-- Higher recall, slower queries (default: 40)
SET hnsw.ef_search = 200;

-- Faster queries, slightly lower recall
SET hnsw.ef_search = 20;

-- Your query here
SELECT ... ORDER BY embedding <=> query_vector LIMIT 10;
```

**Recommendations**:
- **Production**: `ef_search = 100` (balanced)
- **Development**: `ef_search = 40` (default, fast)
- **High accuracy**: `ef_search = 200` (slower)

---

## Monitoring & Validation

### Check Index Status

```bash
# Validate all tables
python scripts/validate_hnsw_performance.py --all

# Validate specific table
python scripts/validate_hnsw_performance.py data_inbox_cs700_ov150_minilm_260110

# Show thresholds
python scripts/validate_hnsw_performance.py --check-thresholds

# JSON output for CI/CD
python scripts/validate_hnsw_performance.py --all --json
```

### Performance Thresholds

Automatic validation checks query latency against expected thresholds:

| Table Size | Max Latency | Expected Speedup |
|------------|-------------|------------------|
| Small (< 10K rows) | 50ms | 2x |
| Medium (10K-100K) | 200ms | 10x |
| Large (> 100K) | 500ms | 50x |

### SQL Query Plan

Check if index is being used:

```sql
EXPLAIN ANALYZE
SELECT id, 1 - (embedding <=> '[0.1, 0.1, ...]'::vector) as similarity
FROM data_inbox_cs700_ov150_minilm_260110
ORDER BY embedding <=> '[0.1, 0.1, ...]'::vector
LIMIT 10;
```

Look for: `Index Scan using ... _hnsw_idx`

---

## Maintenance

### When to Rebuild

Rebuild HNSW index if:
- Table grew significantly (>50% size increase)
- Query performance degraded
- Changed ef_construction parameter

```sql
-- Drop old index
DROP INDEX data_inbox_cs700_ov150_minilm_260110_hnsw_idx;

-- Create new index
CREATE INDEX data_inbox_cs700_ov150_minilm_260110_hnsw_idx
ON data_inbox_cs700_ov150_minilm_260110
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Vacuum After Large Changes

After major updates, vacuum the table:

```sql
VACUUM ANALYZE data_inbox_cs700_ov150_minilm_260110;
```

---

## Troubleshooting

### Index Not Being Used

**Symptom**: Queries still slow after creating index

**Diagnosis**:
```sql
EXPLAIN ANALYZE SELECT ... ORDER BY embedding <=> query LIMIT 10;
```

If you see `Seq Scan` instead of `Index Scan`:

**Solutions**:
1. Check query uses correct operator: `<=>` (cosine distance)
2. Increase `ef_search`: `SET hnsw.ef_search = 100;`
3. Verify index exists: `\d+ table_name` in psql
4. Check table statistics: `ANALYZE table_name;`

### Slow Index Creation

**Symptom**: Index creation takes hours

**Expected times** (from your system):
- 333 rows: 0.1s
- 6,117 rows: 4.3s
- 61,995 rows: 63s (~1 minute)
- 91,219 rows: 142s (~2.5 minutes)

**For 1M+ rows**: Consider:
- Run during off-peak hours
- Increase `maintenance_work_mem` in PostgreSQL
- Use lower `ef_construction` (e.g., 32) for faster build

### Memory Issues

**Symptom**: Out of memory during index creation

**Solutions**:
1. Increase PostgreSQL `maintenance_work_mem`:
   ```sql
   SET maintenance_work_mem = '2GB';
   ```

2. Use lower `m` parameter:
   ```sql
   CREATE INDEX ... WITH (m = 8, ef_construction = 64)
   ```

3. Create index in smaller batches (not recommended for HNSW)

---

## Integration with RAG Pipeline

### Automatic Creation

The RAG pipeline automatically creates HNSW indices at line 3168 of `rag_low_level_m1_16gb_verbose.py`:

```python
embed_nodes(embed_model, nodes)
insert_nodes(vector_store, nodes)
create_hnsw_index()  # ← Automatic HNSW creation
```

### Skip HNSW Creation

To skip HNSW creation (not recommended):

```python
# Comment out the line
# create_hnsw_index()
```

Or set environment variable:
```bash
SKIP_HNSW_INDEX=1 python rag_low_level_m1_16gb_verbose.py
```

### Custom Parameters

To use custom HNSW parameters:

```python
# In rag_low_level_m1_16gb_verbose.py
create_hnsw_index(table_name=S.table)

# With custom parameters (requires code modification)
# m=32 for better recall
# ef_construction=128 for better quality
```

---

## Cost-Benefit Analysis

### Storage Overhead

HNSW index size: ~10% of table size

| Table | Data Size | Index Size | Overhead |
|-------|-----------|------------|----------|
| data_inbox | 325 MB | ~33 MB | 10% |
| data_t_01 | 285 MB | ~29 MB | 10% |

**Total**: 654 MB data + ~65 MB indices = 719 MB

### Performance Gains

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Large table queries** | 443ms | 2.1ms | **215x faster** |
| **Medium table queries** | 321ms | 3.1ms | **103x faster** |
| **Memory overhead** | 0% | 10% | Minimal |
| **Build time** | N/A | 142s | One-time cost |

**ROI**: 10% storage for 100x+ speedup = **Excellent**

---

## Best Practices

### ✅ Do

1. **Always create HNSW indices** for tables > 10K rows
2. **Use default parameters** (m=16, ef_construction=64) initially
3. **Monitor query performance** with validation script
4. **Test ef_search values** for your use case
5. **Rebuild indices** after major table changes

### ❌ Don't

1. **Don't skip HNSW creation** on large tables
2. **Don't use very high m** (>64) unless necessary
3. **Don't rebuild indices** unnecessarily (expensive)
4. **Don't use IVFFlat** instead (HNSW is better for most cases)
5. **Don't query without index** on 100K+ row tables

---

## CI/CD Integration

Add to your deployment pipeline:

```bash
# After database migration/indexing
python scripts/validate_hnsw_performance.py --all --json > hnsw_report.json

# Check exit code (0 = pass, 1 = fail)
if [ $? -ne 0 ]; then
  echo "❌ HNSW validation failed"
  exit 1
fi

echo "✅ HNSW indices validated"
```

### Grafana Monitoring

Track query performance in Grafana:

```sql
-- Add to prometheus.yml
- query: |
    SELECT
      'hnsw_query_latency' as metric,
      table_name,
      AVG(query_time_ms) as value
    FROM query_logs
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
    GROUP BY table_name
```

---

## Advanced Topics

### Multi-Column Indices

HNSW currently only supports single-column indices. For multi-vector queries:

```sql
-- Not supported
CREATE INDEX ON table (embedding1, embedding2) USING hnsw;

-- Instead, create separate indices
CREATE INDEX ON table (embedding1) USING hnsw;
CREATE INDEX ON table (embedding2) USING hnsw;
```

### Hybrid Search

Combine HNSW with full-text search:

```sql
-- Vector search
SELECT id FROM table
WHERE embedding <=> query_vector < 0.5
ORDER BY embedding <=> query_vector
LIMIT 100;

-- Hybrid (vector + text)
SELECT id FROM table
WHERE embedding <=> query_vector < 0.5
  AND to_tsvector(text) @@ to_tsquery('search terms')
ORDER BY embedding <=> query_vector
LIMIT 10;
```

### Parallel Builds

For multiple tables, parallelize index creation:

```bash
# Build indices in parallel
for table in $(python -c "...list tables..."); do
  python -c "create_index('$table')" &
done
wait
```

---

## References

- **pgvector docs**: https://github.com/pgvector/pgvector
- **HNSW paper**: https://arxiv.org/abs/1603.09320
- **Migration script**: `migrate_add_hnsw_indices.py`
- **Validation script**: `scripts/validate_hnsw_performance.py`
- **RAG pipeline**: `rag_low_level_m1_16gb_verbose.py` (line 2650, 3168)

---

## Summary

✅ **HNSW indices are automatic** - No manual action needed for new indices
✅ **100x+ speedup achieved** - Real results from your system
✅ **Low overhead** - 10% storage for 215x performance gain
✅ **Production-ready** - All 4 tables now optimized

**Next steps**:
1. Monitor query performance with validation script
2. Tune `ef_search` for your accuracy/speed requirements
3. Add to CI/CD for automated validation
4. Integrate metrics into Grafana dashboard

**Questions?** See troubleshooting section or check the migration report: `hnsw_migration_report_*.txt`
