---
description: Audit RAG index health and configuration
---

# Audit RAG Index

Perform a comprehensive audit of a RAG index to check health, consistency, and quality.

## Usage

```
/audit-index [table_name]
```

## What It Checks

### 1. Index Existence & Size
- Table exists in database
- Row count (number of chunks)
- Estimated storage size

### 2. Configuration Consistency
- All chunks have same configuration
- No mixed chunk sizes
- Metadata completeness

### 3. Chunk Quality
- Size distribution (min/max/avg)
- Overlap ratio
- Sample chunk preview

### 4. Embedding Health
- Embedding dimensions
- Model used
- No null embeddings

### 5. Query Performance
- Test query execution
- Similarity score distribution
- Retrieval latency

## Output Format

```markdown
# Index Audit Report: my_index

## Summary
- Status: ✅ Healthy / ⚠️ Warnings / ❌ Issues
- Rows: 1,234
- Configuration: cs=700, ov=150

## Checks

### Configuration Consistency
✅ All chunks use same configuration
- Chunk Size: 700
- Overlap: 150
- Model: BAAI/bge-small-en

### Chunk Statistics
- Total: 1,234 chunks
- Avg Size: 650 chars
- Min/Max: 45 / 720 chars
- Chunks per doc: 8.2 avg

### Embedding Health
✅ All embeddings present
- Dimensions: 384
- No null values

### Test Query Results
Query: "test retrieval"
- Top score: 0.72 (Good)
- Avg score: 0.58
- Latency: 0.12s

## Recommendations
1. Consider increasing chunk size for more context
2. Index is healthy, ready for production
```

## SQL Queries Used

```sql
-- Row count
SELECT COUNT(*) FROM "table_name";

-- Configuration check
SELECT DISTINCT
  metadata_->>'_chunk_size',
  metadata_->>'_chunk_overlap',
  metadata_->>'_embed_model'
FROM "table_name";

-- Chunk size stats
SELECT
  AVG(LENGTH(text))::int as avg_size,
  MIN(LENGTH(text)) as min_size,
  MAX(LENGTH(text)) as max_size,
  STDDEV(LENGTH(text))::int as stddev
FROM "table_name";

-- Null embedding check
SELECT COUNT(*) FROM "table_name" WHERE embedding IS NULL;
```

## Quality Thresholds

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Config consistency | 1 config | - | Multiple |
| Avg similarity | > 0.5 | 0.3-0.5 | < 0.3 |
| Null embeddings | 0 | - | > 0 |
| Chunk variance | < 50% | 50-100% | > 100% |

## Common Issues Detected

### Mixed Index
```
⚠️ WARNING: Multiple configurations detected
- Config 1: cs=500, ov=100 (800 rows)
- Config 2: cs=700, ov=150 (434 rows)

FIX: Run with RESET_TABLE=1 to rebuild with consistent config
```

### Missing Metadata
```
⚠️ WARNING: Legacy index (no metadata)
Cannot verify chunk configuration.

FIX: Re-index to add metadata tracking
```

### Poor Retrieval Quality
```
⚠️ WARNING: Low similarity scores
Average score: 0.28 (threshold: 0.4)

FIX: Try smaller chunks or different embedding model
```

## Follow-up Commands

After audit, you may want to:
- `/run-rag --reset` - Rebuild index
- `/optimize-rag` - Get optimization suggestions
- `/compare-chunks` - Test different configurations
