# Chunk Size Management Guide

## The Problem: Mixed Indexes

When you change `CHUNK_SIZE` without rebuilding the index, you create a **mixed index** - a single table containing chunks from multiple configurations. This causes:

1. **Retrieval returns old chunks**: Your queries still find chunks from the previous configuration
2. **Chunk size appears to do nothing**: Because the vector DB is searching through the old chunks
3. **Unreliable results**: Mix of different chunk sizes leads to inconsistent retrieval

## Why This Happens

```bash
# Run 1: chunk_size=700
CHUNK_SIZE=700 python rag_low_level_m1_16gb_verbose.py
# → Creates chunks of ~700 chars, stores in table 'ethical-slut_paper'

# Run 2: chunk_size=500 (NO RESET!)
CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py
# → Creates chunks of ~500 chars, ADDS to same table
# → Table now has BOTH 700-char and 500-char chunks
# → Retrieval may return either type!
```

## The Fix: Three Approaches

### Option 1: Reset Table (Recommended for Iteration)

```bash
# Always reset when changing chunk parameters
RESET_TABLE=1 CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py
```

**Pros:**
- Clean index with consistent configuration
- Simple and straightforward

**Cons:**
- Must re-index (takes time)
- Can't compare results across configurations

### Option 2: Configuration-Specific Table Names (Recommended for Experiments)

```bash
# Each configuration gets its own table
PGTABLE=ethical-slut_cs700_ov150 CHUNK_SIZE=700 CHUNK_OVERLAP=150 python rag_low_level_m1_16gb_verbose.py

PGTABLE=ethical-slut_cs500_ov100 CHUNK_SIZE=500 CHUNK_OVERLAP=100 python rag_low_level_m1_16gb_verbose.py

# Query each configuration
PGTABLE=ethical-slut_cs700_ov150 python rag_low_level_m1_16gb_verbose.py --query-only --query "Your question"
PGTABLE=ethical-slut_cs500_ov100 python rag_low_level_m1_16gb_verbose.py --query-only --query "Your question"
```

**Pros:**
- Keep multiple configurations side-by-side
- Easy to compare results
- No accidental mixing

**Cons:**
- Uses more disk space (one table per config)

### Option 3: Query-Only Mode

```bash
# Index once
CHUNK_SIZE=700 python rag_low_level_m1_16gb_verbose.py

# Query without re-indexing
python rag_low_level_m1_16gb_verbose.py --query-only --query "Your question"
```

**Note:** In query-only mode, `CHUNK_SIZE` doesn't affect results - you're querying whatever was indexed.

## How to Verify Configuration

The script now automatically:

1. **Stores configuration in metadata**: Every chunk records its `chunk_size`, `chunk_overlap`, and `embed_model`
2. **Detects mismatches**: Warns when you try to add chunks with different parameters
3. **Blocks mixed indexes**: Exits with error if configuration mismatch detected (unless you use RESET_TABLE=1)
4. **Shows chunk config in logs**: Retrieval logs show which configuration produced each chunk

Example output:
```
📄 Retrieved Chunks (these will be sent to LLM):

  1. Similarity: 0.7234 | Source: 131 [cs=700, ov=150]
     Text: "The concept of jealousy is explored..."
```

## Recommended Chunk Sizes

Based on your embedding model (`BAAI/bge-small-en`) and LLM context window (3072 tokens):

### Conservative (Safe Default)
```bash
CHUNK_SIZE=1400        # ~350 tokens
CHUNK_OVERLAP=280      # ~20% overlap
TOP_K=4                # 4 chunks = ~1400 tokens
```

### Smaller Chunks (More Precise)
```bash
CHUNK_SIZE=800         # ~200 tokens
CHUNK_OVERLAP=160      # ~20% overlap
TOP_K=6                # 6 chunks = ~1200 tokens
```

### Larger Chunks (More Context)
```bash
CHUNK_SIZE=2000        # ~500 tokens
CHUNK_OVERLAP=400      # ~20% overlap
TOP_K=3                # 3 chunks = ~1500 tokens
```

**Important Limits:**
- **Embedding model max**: `BAAI/bge-small-en` typically handles ~512 tokens max
  - Chunks larger than this get truncated during embedding
  - Stay well below this limit (use ~350 tokens / 1400 chars max)
- **LLM context window**: 3072 tokens total
  - Must fit: system prompt + retrieved chunks + query + answer
  - Budget ~1500-2000 tokens for retrieved chunks

## Experimentation Workflow

### Step 1: Choose Configurations to Test
```bash
# Define your test matrix
configs=(
  "cs1400_ov280"
  "cs1000_ov200"
  "cs800_ov160"
)
```

### Step 2: Index Each Configuration
```bash
for config in "${configs[@]}"; do
  # Extract params from config name
  cs=$(echo $config | sed 's/cs\([0-9]*\).*/\1/')
  ov=$(echo $config | sed 's/.*_ov\([0-9]*\)/\1/')

  echo "Indexing $config..."
  PGTABLE="ethical-slut_$config" \
  CHUNK_SIZE=$cs \
  CHUNK_OVERLAP=$ov \
  python rag_low_level_m1_16gb_verbose.py
done
```

### Step 3: Query Each Configuration
```bash
# Same question across all configs
question="What is the main thesis about jealousy?"

for config in "${configs[@]}"; do
  echo -e "\n=== Testing $config ==="
  LOG_QUERIES=1 \
  PGTABLE="ethical-slut_$config" \
  python rag_low_level_m1_16gb_verbose.py \
    --query-only \
    --query "$question"
done
```

### Step 4: Compare Results
```bash
# Query logs are saved to query_logs/{table_name}/
# Compare similarity scores, answer quality, etc.

for config in "${configs[@]}"; do
  echo "=== $config ==="
  latest_log=$(ls -t query_logs/ethical-slut_$config/*.json | head -1)
  jq -r '.retrieval.quality_metrics' "$latest_log"
done
```

## Common Mistakes

### ❌ Mistake 1: Changing CHUNK_SIZE without RESET_TABLE
```bash
# Wrong: Creates mixed index
CHUNK_SIZE=700 python rag_low_level_m1_16gb_verbose.py
CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py  # ❌ Adds to same table!
```

### ✅ Fix: Reset or use different table
```bash
# Option A: Reset
RESET_TABLE=1 CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py

# Option B: Different table
PGTABLE=ethical-slut_cs500_ov100 CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py
```

### ❌ Mistake 2: Setting CHUNK_SIZE in query-only mode
```bash
# Wrong: CHUNK_SIZE doesn't affect queries!
CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py --query-only
# This queries whatever chunk size was used during indexing
```

### ✅ Fix: Query the right table
```bash
# You must query the table that was indexed with chunk_size=500
PGTABLE=ethical-slut_cs500_ov100 python rag_low_level_m1_16gb_verbose.py --query-only
```

### ❌ Mistake 3: Chunk size too large for embedding model
```bash
# Wrong: Exceeds model's max sequence length
CHUNK_SIZE=5000  # ~1250 tokens - will be truncated!
```

### ✅ Fix: Stay within limits
```bash
# Keep chunks under ~350 tokens (~1400 chars) for bge-small-en
CHUNK_SIZE=1400
```

## Debugging Mixed Indexes

### Check if you have a mixed index:
```bash
# The script will automatically detect and warn:
python rag_low_level_m1_16gb_verbose.py
```

Example warning:
```
❌ CONFIGURATION MISMATCH DETECTED!
  Current config: chunk_size=500, overlap=100, model=BAAI/bge-small-en
  Existing index: chunk_size=700, overlap=150, model=BAAI/bge-small-en

  ❌ Proceeding will create a MIXED INDEX
```

### Fix a mixed index:
```bash
# Option 1: Reset and rebuild
RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py

# Option 2: Start fresh with new table name
PGTABLE=ethical-slut_cs500_ov100_clean CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py
```

## Summary

**Golden Rules:**
1. **RESET_TABLE=1** when changing chunk parameters (unless using different table names)
2. **Use config-specific table names** when experimenting (e.g., `PGTABLE=doc_cs700_ov150`)
3. **CHUNK_SIZE only affects indexing**, not querying
4. **Stay within limits**: ~350 tokens (~1400 chars) max for bge-small-en
5. **The script will warn you** - read the warnings!

**For experiments:**
- Use different table names for each configuration
- Keep logs with `LOG_QUERIES=1`
- Compare results using the same questions
- Don't trust results from mixed indexes
