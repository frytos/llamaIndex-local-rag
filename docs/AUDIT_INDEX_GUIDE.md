# RAG Index Audit Guide

Quick reference for auditing RAG index health and quality.

---

## Quick Start

```bash
# List all tables
python audit_index.py

# Audit specific table
python audit_index.py <table_name>

# Example
python audit_index.py data_inbox_cs700_ov150_minilm_260110
```

---

## What It Checks

### 1. ✅ Index Existence & Size
- Table exists in database
- Row count (number of chunks)
- Storage size on disk

### 2. 🔧 Configuration Consistency
- All chunks have same configuration
- No mixed chunk sizes
- Metadata completeness
- Model and dimension consistency

### 3. 📊 Chunk Quality
- Size distribution (min/max/avg/median)
- Standard deviation
- Chunks per document
- Sample chunk previews

### 4. 🧬 Embedding Health
- Embedding dimensions
- Model used
- No null embeddings
- Dimension consistency

### 5. ⚡ Query Performance
- Test query execution
- Retrieval latency
- Similarity score distribution

---

## Output Format

```
================================================================================
RAG INDEX AUDIT REPORT: table_name
================================================================================
Generated: 2026-01-10 14:58:03

## SUMMARY
Status: ✅ HEALTHY / ⚠️ WARNINGS / ❌ ISSUES
Rows: 6,117
Storage Size: 42 MB

## CONFIGURATION
✅ Consistent configuration across all chunks
  - Chunk Size: 700
  - Overlap: 150
  - Model: BAAI/bge-m3
  - Dimensions: 1024

## CHUNK STATISTICS
  - Average Size: 1,411 characters
  - Median Size: 1,458 characters
  - Range: 0 - 3,827 characters
  - Std Dev: 342 characters
  - Chunks per Document: 117.6 avg

## EMBEDDING HEALTH
✅ All chunks have embeddings
  - Dimensions: 1024
✅ Embedding dimensions consistent

## QUERY PERFORMANCE
Test retrieval (dummy query):
  - Latency: 0.042s
  - Top Similarity: -0.02
  - Avg Similarity: -0.021
  ⚠️ Low similarity scores detected (expected for dummy query)

## RECOMMENDATIONS
  ✅ Index is healthy and ready for production use
```

---

## Status Indicators

| Symbol | Meaning | Action Required |
|--------|---------|-----------------|
| ✅ | Healthy | No action needed |
| ⚠️ | Warning | Review recommendations |
| ❌ | Error | Fix immediately |
| 💡 | Tip | Consider optimization |
| 🔧 | Fix | Apply suggested fix |
| ⚡ | Performance | Optimization available |

---

## Common Issues & Fixes

### ⚠️ Mixed Configurations

**Symptom**:
```
⚠️ WARNING: Multiple configurations detected
- Config 1: cs=500, ov=100 (800 rows)
- Config 2: cs=700, ov=150 (434 rows)
```

**Cause**: Indexed same table with different chunk sizes/overlap.

**Fix**:
```bash
RESET_TABLE=1 PGTABLE=table_name python rag_low_level_m1_16gb_verbose.py
```

### ⚠️ Missing Metadata

**Symptom**:
```
⚠️ WARNING: Legacy index (no metadata)
Cannot verify chunk configuration.
```

**Cause**: Index created before metadata tracking was added.

**Fix**: Re-index to add metadata tracking:
```bash
RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
```

### ❌ Null Embeddings

**Symptom**:
```
❌ 150 chunks missing embeddings
```

**Cause**: Indexing failed partway through.

**Fix**: Re-run indexing to complete:
```bash
python rag_low_level_m1_16gb_verbose.py
```

### ⚡ Slow Query Performance

**Symptom**:
```
⚡ Slow query performance - consider adding vector index
Latency: 1.2s
```

**Cause**: Large index (>10K chunks) without vector index.

**Fix**: Add HNSW index for faster similarity search:
```sql
CREATE INDEX ON table_name
USING hnsw (embedding vector_cosine_ops);
```

Or via psql:
```bash
source .env
psql -h $PGHOST -U $PGUSER -d $DB_NAME -c \
  "CREATE INDEX ON table_name USING hnsw (embedding vector_cosine_ops);"
```

---

## Quality Thresholds

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| **Config consistency** | 1 config | - | Multiple configs |
| **Avg similarity** | > 0.5 | 0.3-0.5 | < 0.3 |
| **Null embeddings** | 0 | - | > 0 |
| **Chunk variance** | < 50% | 50-100% | > 100% |
| **Query latency** | < 100ms | 100-500ms | > 500ms |

---

## Interpreting Results

### Chunk Size Analysis

**Actual chunk size vs configured chunk size**:

| Ratio | Interpretation | Action |
|-------|----------------|--------|
| **0.8-1.2x** | Normal for documents | None |
| **1.5-2.5x** | Normal for chat data | None (messages preserve context) |
| **> 3x** | Oversized chunks | Decrease CHUNK_SIZE |
| **< 0.5x** | Undersized chunks | Check data quality |

**Example**:
- Configured: `CHUNK_SIZE=700`
- Actual: `1,411 chars average`
- Ratio: **2.0x** → Normal for conversational data ✅

### Query Latency

| Index Size | Expected Latency | Threshold |
|------------|------------------|-----------|
| < 1K chunks | < 10ms | Fast |
| 1K - 10K | 10-100ms | Good |
| 10K - 100K | 100-500ms | Acceptable |
| > 100K | 500ms+ | Add HNSW index |

### Similarity Scores

**Note**: Dummy query scores are meaningless. Test with real queries:

```bash
# Test retrieval quality
PGTABLE=table_name \
python rag_low_level_m1_16gb_verbose.py --query-only \
--query "What did we discuss about travel plans?"
```

Expected similarity scores for **real queries**:
- **> 0.7**: Excellent match
- **0.5 - 0.7**: Good match
- **0.3 - 0.5**: Moderate match
- **< 0.3**: Poor match (may need better embedding model or smaller chunks)

---

## Batch Audit

Audit all tables at once:

```bash
# List all tables
python audit_index.py > tables.txt

# Audit each (manual)
for table in $(python audit_index.py | grep -E '^\s+[0-9]+\.' | awk '{print $2}'); do
  echo "Auditing $table..."
  python audit_index.py "$table"
done
```

---

## Report Files

Each audit generates two files:

1. **Console output**: Immediate results
2. **Text file**: `audit_report_<table>_<timestamp>.txt`

Example:
```
audit_report_data_inbox_cs700_ov150_minilm_260110_20260110_145818.txt
```

---

## Integration with RAG Pipeline

### Before Indexing

Check if table already exists:
```bash
python audit_index.py
```

### After Indexing

Validate new index:
```bash
python audit_index.py <new_table_name>
```

### Regular Monitoring

Run audit weekly/monthly to check index health:
```bash
# Add to cron
0 0 * * 0 cd /path/to/project && python audit_index.py table_name >> audit.log
```

---

## Advanced Usage

### Custom Database Credentials

```bash
# Override .env settings
PGHOST=remote.db.com \
PGUSER=custom_user \
PGPASSWORD=secret \
DB_NAME=prod_db \
python audit_index.py table_name
```

### Export Results to JSON

```python
# Modify audit_index.py to output JSON
import json

# Add to main() after generating report
results = {
    "table": table_name,
    "status": "healthy",
    "row_count": table_info['row_count'],
    "size": table_info['size'],
    # ... add more fields
}

with open(f"audit_{table_name}.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Troubleshooting

### Connection Failed

```
❌ Failed to connect to database: connection refused
```

**Fix**:
1. Check PostgreSQL is running: `pg_isready -h localhost -p 5432`
2. Verify credentials in `.env`
3. Test connection: `psql -h $PGHOST -U $PGUSER -d $DB_NAME`

### Table Not Found

```
❌ Table 'my_table' not found
```

**Fix**:
1. List available tables: `python audit_index.py`
2. Check table name spelling (case-sensitive)
3. Verify you're connected to correct database

### Vector Extension Missing

```
❌ function vector_dims(vector) does not exist
```

**Fix**: Enable pgvector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## See Also

- **Comprehensive Audit Report**: `COMPREHENSIVE_INDEX_AUDIT.md`
- **Performance Tuning**: `docs/PERFORMANCE_QUICK_START.md`
- **Environment Variables**: `docs/ENVIRONMENT_VARIABLES.md`
- **RAG Pipeline**: `rag_low_level_m1_16gb_verbose.py`

---

## Script Location

- **Script**: `/Users/frytos/code/llamaIndex-local-rag/audit_index.py`
- **Requirements**: `psycopg2`, `numpy`, `python-dotenv`
- **Python**: 3.11+

Install dependencies:
```bash
pip install psycopg2-binary numpy python-dotenv
```
