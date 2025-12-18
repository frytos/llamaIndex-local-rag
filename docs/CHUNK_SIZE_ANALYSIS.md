# Comprehensive Analysis: RAG Chunk Size Issue & Benchmark Implementation

**Date:** 2025-12-17
**Document:** Analysis of chunk size ineffectiveness and solution design
**Status:** Ready for implementation

---

## Executive Summary

**Problem:** Changing `CHUNK_SIZE` environment variable doesn't affect retrieved results because new chunks are appended to existing tables, and retrieval queries all chunks without filtering.

**Root Cause:** Default incremental indexing behavior + no run-specific metadata + no query-time filtering.

**Solution:** Use table naming pattern to isolate each chunk configuration (e.g., `ethical-slut_cs900_ov150`).

**Bonus:** Create benchmark script to systematically compare chunk configurations.

---

## Problem Analysis

### Root Cause

When you change `CHUNK_SIZE` without setting `RESET_TABLE=1`, the system performs **incremental indexing** - adding new chunks to the existing table alongside old chunks. During retrieval, the vector search queries ALL chunks (old and new) without any filtering, so old chunks from previous runs continue to appear in results.

### Evidence from Code

**Location: `rag_low_level_m1_16gb_verbose.py:1350-1354`**
```python
existing_rows = count_rows()
if existing_rows and existing_rows > 0 and not S.reset_table:
    log.warning(f"Table '{S.table}' already contains {existing_rows} rows")
    log.warning("  Proceeding will add MORE rows (incremental indexing)")
```

This code explicitly allows incremental indexing by default (`RESET_TABLE=0`).

**Location: `rag_low_level_m1_16gb_verbose.py:689-695` (Retrieval)**
```python
vsq = VectorStoreQuery(
    query_embedding=q_emb,
    similarity_top_k=self._similarity_top_k,
    mode="default",
)
res = self._vector_store.query(vsq)
```

The retrieval performs raw vector similarity search with **no metadata filtering** - it queries all chunks regardless of when they were indexed or what chunk size was used.

**Location: `rag_low_level_m1_16gb_verbose.py:952` (Metadata Storage)**
```python
n.metadata = src_doc.metadata  # Only copies document metadata
```

No run-specific metadata is stored (no `chunk_size`, `run_id`, `timestamp`, etc.), making it impossible to distinguish between chunks from different indexing runs.

### Why Your Logs Show Identical Chunks

Your query logs show the exact same chunk (page 133, ~1034 chars) being retrieved across runs with different chunk sizes (100, 500, 900, 5000). This happens because:

1. The first run created chunks and stored them
2. Subsequent runs with different chunk sizes **added more chunks** to the same table
3. The old chunk scored highly and kept appearing in top-4 results
4. No mechanism exists to filter out old chunks or prefer current chunks

---

## Proposed Solutions

### Option A: Include Chunk Parameters in Table Name (RECOMMENDED)

**Pros:**
- Automatic isolation - each configuration gets its own clean index
- No code changes required - just change how you set `PGTABLE`
- Easy to compare different chunk strategies side-by-side
- Clean, predictable behavior

**Cons:**
- Multiple tables in database
- Manual cleanup if experimenting with many configs

**Implementation:**
```bash
# Instead of:
export PGTABLE="ethical-slut_paper"

# Use:
export PGTABLE="ethical-slut_cs${CHUNK_SIZE}_ov${CHUNK_OVERLAP}"
```

Example table names:
- `ethical-slut_cs900_ov150`
- `ethical-slut_cs500_ov100`
- `ethical-slut_cs1400_ov300`

---

### Option B: Always Reset Table When Experimenting

**Pros:**
- No code changes needed
- Simple to understand
- Works with existing code

**Cons:**
- Must remember to set `RESET_TABLE=1` every time
- Loses previous indexes (can't compare old vs new)
- Re-indexes even when not needed

**Implementation:**
```bash
export RESET_TABLE=1
python rag_low_level_m1_16gb_verbose.py
```

**Verification:** Look for log line:
```
RESET_TABLE=1 -> Dropping table 'ethical-slut_paper' if it exists.
```

---

### Option C: Store Index Signature in Metadata + Filter During Retrieval

**Pros:**
- Single table can contain multiple run configurations
- Can query specific runs or compare across runs
- Most flexible for experimentation

**Cons:**
- Requires code changes in 2-3 locations
- More complex to implement and maintain
- PGVectorStore metadata filtering support needs verification

**Implementation Details:**

1. **Store run signature in metadata** (`build_nodes()` - line ~952)
   ```python
   n.metadata = src_doc.metadata.copy()
   n.metadata["index_id"] = f"cs{S.chunk_size}_ov{S.chunk_overlap}_{S.embed_model.model_name}"
   n.metadata["indexed_at"] = datetime.now().isoformat()
   ```

2. **Pass current index_id to retriever** (modify `VectorDBRetriever.__init__()`)
   ```python
   self._current_index_id = current_index_id
   ```

3. **Add metadata filter to query** (`_retrieve()` - line ~689)
   ```python
   from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

   filters = MetadataFilters(filters=[
       ExactMatchFilter(key="index_id", value=self._current_index_id)
   ])

   vsq = VectorStoreQuery(
       query_embedding=q_emb,
       similarity_top_k=self._similarity_top_k,
       mode="default",
       filters=filters,  # Add this
   )
   ```

**Files to modify:**
- `rag_low_level_m1_16gb_verbose.py:952` (build_nodes)
- `rag_low_level_m1_16gb_verbose.py:670` (VectorDBRetriever.__init__)
- `rag_low_level_m1_16gb_verbose.py:689` (VectorStoreQuery construction)

---

## Additional Issue: Embedding Model Token Limits

**Problem:** You tested `CHUNK_SIZE=5000`. If using `BAAI/bge-small-en`, the model typically has a max sequence length of **512 tokens** (~2000 characters).

**Impact:** Chunks larger than ~2000 characters will be **silently truncated** during embedding, making large chunk sizes ineffective.

**Recommendation:**
- Keep `CHUNK_SIZE` between 700-1800 characters (175-450 tokens)
- For longer context, increase `TOP_K` instead of chunk size

---

## Recommended Chunk Size Strategy

Based on your setup (3072 context window, top_k=4):

| Scenario | Chunk Size (chars) | Chunk Overlap (chars) | Reasoning |
|----------|-------------------|----------------------|-----------|
| **Precise factual Q&A** | 700-1000 | 100-150 | Small chunks for precise matching |
| **Balanced (default)** | 1200-1500 | 200-300 | Medium chunks with good context |
| **Narrative/story context** | 1400-1800 | 250-400 | Larger chunks preserve narrative flow |

For "Ethical Slut" (narrative book), I recommend starting with:
- `CHUNK_SIZE=1400`
- `CHUNK_OVERLAP=280` (20% overlap)
- `TOP_K=4`

---

## Implementation Plan

### Part 1: Fix Chunk Size Issue (Option A - Table Naming Pattern)

**No code changes required** - just change how you set the `PGTABLE` environment variable.

**Before (problematic):**
```bash
export PGTABLE="ethical-slut_paper"
export CHUNK_SIZE=900
python rag_low_level_m1_16gb_verbose.py
```

**After (correct):**
```bash
export PGTABLE="ethical-slut_cs${CHUNK_SIZE}_ov${CHUNK_OVERLAP}"
python rag_low_level_m1_16gb_verbose.py
```

**Example table names:**
- `ethical-slut_cs700_ov150`
- `ethical-slut_cs900_ov150`
- `ethical-slut_cs1400_ov280`

**Verification:** After running, check PostgreSQL to see the actual table created:
```sql
\dt data_ethical*
```

### Part 2: Create Benchmark Script

**New file:** `benchmark_chunks.py`

**Purpose:** Systematically test different chunk configurations across the same questions, track metrics, and produce comparative analysis.

#### Architecture

```
1. Load configuration (questions + chunk configs)
2. For each chunk configuration:
   a. Generate table name: {doc_base}_cs{size}_ov{overlap}
   b. Check if table exists (skip indexing if --no-reindex)
   c. Index document if needed
   d. Run all test questions
   e. Collect metrics (scores, timing, answer quality)
3. Generate comparison report (JSON + CSV + Markdown)
```

#### Key Components

**Input:**
- Questions file (JSON) or CLI args
- Chunk configurations to test
- Document path
- Flags: `--reindex`, `--cleanup-tables`

**Output (in `benchmark_results/{timestamp}_{doc}/`):**
- `results.json` - Complete metrics for all configs and queries
- `results.csv` - Tabular data for spreadsheet analysis
- `comparison.md` - Human-readable report with recommendations
- `queries/` - Individual query logs per configuration

**Metrics Tracked:**
- Retrieval quality (best/avg/worst similarity scores)
- Retrieval time
- Generation time & tokens/sec
- Answer length
- Score consistency across questions
- Indexing stats (chunks created, time taken)

#### Implementation Details

**Reuse from main script:**
```python
from rag_low_level_m1_16gb_verbose import (
    Settings,
    build_embed_model,
    build_llm,
    load_documents,
    chunk_documents,
    build_nodes,
    embed_nodes,
    make_vector_store,
    insert_nodes,
    VectorDBRetriever,
    count_rows,
)
```

**Table naming function:**
```python
def get_benchmark_table_name(pdf_path: str, chunk_size: int, chunk_overlap: int) -> str:
    """ethical-slut_cs900_ov150"""
    doc_base = Path(pdf_path).stem.lower().replace(" ", "-")
    return f"{doc_base}_cs{chunk_size}_ov{chunk_overlap}"
```

**Configuration override:**
```python
# For each benchmark config
S.chunk_size = config['chunk_size']
S.chunk_overlap = config['chunk_overlap']
S.table = get_benchmark_table_name(pdf_path, chunk_size, chunk_overlap)
```

#### Example Usage

**Quick test with CLI questions:**
```bash
python benchmark_chunks.py \
  --doc data/ethical-slut.pdf \
  --questions "Where does jealousy come from?" \
             "What are the main principles?" \
  --configs "[(700,150),(900,150),(1400,280)]" \
  --output benchmark_results/quick_test
```

**Comprehensive benchmark with questions file:**
```bash
python benchmark_chunks.py \
  --doc data/ethical-slut.pdf \
  --questions-file test_questions.json \
  --configs "[(500,100),(700,150),(900,150),(1200,200)]" \
  --output benchmark_results/comprehensive
```

**Query existing indexes (fast):**
```bash
python benchmark_chunks.py \
  --doc data/ethical-slut.pdf \
  --questions-file test_questions.json \
  --configs "[(700,150),(900,150)]" \
  --no-reindex
```

#### Expected Runtime

For ethical-slut.pdf (305 pages, 2.77 MB):
- Indexing one config: ~2-3 minutes
- Running 5 queries: ~5-7 minutes per config
- **Total for 4 configs**: ~30-40 minutes
- **With --no-reindex**: ~20-30 minutes (queries only)

#### Sample Output Structure

```
benchmark_results/
└── 20251217_155030_ethical-slut/
    ├── config.json              # Benchmark metadata
    ├── results.json             # Complete results
    ├── results.csv              # Tabular export
    ├── comparison.md            # Human-readable report
    └── queries/
        ├── cs700_ov150/
        │   ├── q1_20251217_155032.json
        │   └── q2_20251217_155033.json
        └── cs900_ov150/
            ├── q1_20251217_155120.json
            └── q2_20251217_155121.json
```

**comparison.md snippet:**
```markdown
# RAG Benchmark Results: ethical-slut

| Configuration | Avg Score | Avg Retrieval | Avg Generation | Avg Answer Length |
|---------------|-----------|---------------|----------------|-------------------|
| cs700_ov150   | 0.8523    | 0.14s         | 58.12s         | 567 chars         |
| cs900_ov150   | 0.8234    | 0.13s         | 64.45s         | 612 chars         |

## Recommendations

**Best for Quality**: cs700_ov150 (highest avg score: 0.8523)
**Best for Speed**: cs1200_ov200 (fastest: 0.10s retrieval)
**Balanced**: cs900_ov150 (good quality + moderate speed)
```

### Part 3: Verification Steps

1. **Verify table naming works:**
   ```bash
   export PGTABLE="test_cs900_ov150"
   export CHUNK_SIZE=900
   export CHUNK_OVERLAP=150
   python rag_low_level_m1_16gb_verbose.py --query-only -q "test"
   ```

   Check PostgreSQL: `\dt data_test*` should show `data_test_cs900_ov150`

2. **Test benchmark script with 2 configs + 2 questions:**
   ```bash
   python benchmark_chunks.py \
     --doc data/ethical-slut.pdf \
     --questions "Where does jealousy come from?" "What are the main principles?" \
     --configs "[(700,150),(900,150)]" \
     --output benchmark_results/test_run
   ```

3. **Review outputs:**
   - Check `benchmark_results/test_run/results.json` structure
   - Open `comparison.md` to verify report quality
   - Import `results.csv` into spreadsheet

4. **Scale up:**
   - Add more questions (5-10 total)
   - Test 4-6 chunk configurations
   - Run comprehensive benchmark

### Files to Create/Modify

**New Files:**
- `benchmark_chunks.py` - Main benchmark script
- `test_questions.json` - Example questions file
- `benchmark_results/` - Output directory (auto-created)

**No modifications to existing files needed!**

**Critical files for reference:**
- `rag_low_level_m1_16gb_verbose.py` - Import core functions
- `query_logs/` - Existing log format to match

### Recommended Test Configurations

Based on your 3072 context window and narrative content:

| Config | Chunk Size | Overlap | Use Case |
|--------|------------|---------|----------|
| Precise | 700 | 150 | Factual Q&A |
| Balanced | 900 | 150 | General purpose |
| Narrative | 1400 | 280 | Story/context questions |
| Large | 1800 | 360 | Maximum context |

---

## Appendix: Detailed Code Analysis

### A. How Indexing Works (Line-by-Line)

**File:** `rag_low_level_m1_16gb_verbose.py`

#### 1. RESET_TABLE Logic (Lines 498-513)

```python
def reset_table_if_requested():
    if not S.reset_table:
        log.info("RESET_TABLE=0 -> keeping existing table (may duplicate on re-ingest).")
        return

    conn = db_conn()
    conn.autocommit = True
    with conn.cursor() as c:
        log.warning(f"RESET_TABLE=1 -> Dropping table '{S.table}' if it exists.")
        c.execute(f'DROP TABLE IF EXISTS "{S.table}";')
    conn.close()
```

**Key Points:**
- Defaults to `RESET_TABLE=0` (keep existing data)
- Drops entire table with `DROP TABLE IF EXISTS`
- Table name is `S.table` but PGVectorStore creates `data_{S.table}`

#### 2. Existing Row Check (Lines 1350-1354)

```python
existing_rows = count_rows()
if existing_rows and existing_rows > 0 and not S.reset_table:
    log.warning(f"Table '{S.table}' already contains {existing_rows} rows")
    log.warning("  Proceeding will add MORE rows (incremental indexing)")
```

**Why This Causes Problems:**
- Warns user but proceeds anyway
- New chunks get appended to old chunks
- No way to distinguish which run created which chunks

#### 3. Chunking (Lines 901-910)

```python
splitter = SentenceSplitter(chunk_size=S.chunk_size, chunk_overlap=S.chunk_overlap)

for doc_idx, doc in enumerate(docs):
    cs = splitter.split_text(doc.text)
    chunks.extend(cs)
    doc_idxs.extend([doc_idx] * len(cs))
```

**Configuration:**
- `S.chunk_size` from `CHUNK_SIZE` env var (default: 700)
- `S.chunk_overlap` from `CHUNK_OVERLAP` env var (default: 150)
- Uses `SentenceSplitter` from LlamaIndex

#### 4. Metadata Storage (Lines 952)

```python
n.metadata = src_doc.metadata
```

**What's Missing:**
- No `chunk_size` stored
- No `chunk_overlap` stored
- No `run_id` or `timestamp`
- Only inherits document metadata (page_label, source)

**This is why we can't filter by chunk configuration during retrieval.**

#### 5. Retrieval Query (Lines 689-695)

```python
vsq = VectorStoreQuery(
    query_embedding=q_emb,
    similarity_top_k=self._similarity_top_k,
    mode="default",
)
res = self._vector_store.query(vsq)
```

**No Filtering:**
- Only has `query_embedding` and `similarity_top_k`
- No `filters` parameter (no metadata filtering)
- Queries ALL chunks in table regardless of run

### B. Why Same Chunks Appear Across Runs

**Evidence from Query Logs:**

Across runs with different `CHUNK_SIZE` values (100, 500, 900, 5000), the retrieved chunks were byte-for-byte identical:
- Same source page (e.g., page 133)
- Same ~1034 character length
- Same similarity scores

**Explanation:**

1. **First run:** Creates chunks with `CHUNK_SIZE=900`, stores them in `ethical-slut_paper` table
2. **Second run (CHUNK_SIZE=500):**
   - Doesn't reset table (RESET_TABLE=0)
   - Creates NEW chunks with size 500
   - Appends them to existing table
   - Table now has BOTH 900-char and 500-char chunks
3. **Query:**
   - Vector search queries ALL chunks
   - Old 900-char chunk scores highly (contains more context)
   - Old chunk appears in top-4 results
   - Looks like chunk size "didn't change"

**The Fix:**

Use separate tables per configuration:
```bash
export PGTABLE="ethical-slut_cs${CHUNK_SIZE}_ov${CHUNK_OVERLAP}"
```

Now each configuration gets its own isolated table:
- `data_ethical-slut_cs900_ov150`
- `data_ethical-slut_cs500_ov100`
- No mixing between runs

### C. Table Naming Gotcha

**What You Set:**
```bash
export PGTABLE="my_table"
```

**What Gets Created:**
```
data_my_table
```

**Why:** PGVectorStore automatically adds `data_` prefix (see line 521):
```python
actual_table = f"data_{S.table}"
```

**Verification:**
```bash
psql -U fryt -d vector_db -c "\dt data_*"
```

### D. Embedding Model Token Limits

**Model:** `BAAI/bge-small-en`
**Max Sequence Length:** ~512 tokens (~2000 characters)

**Impact of Large Chunks:**
- `CHUNK_SIZE=5000` creates ~5000-char chunks
- Embedding model truncates to ~2000 chars
- Only first 40% of chunk gets embedded
- Similarity scores become less meaningful

**Recommendation:**
- Keep `CHUNK_SIZE` ≤ 1800 characters
- Or use model with larger context (e.g., `bge-large`)

### E. Complete Indexing Flow

```
main() [line 1312]
  │
  ├─ count_rows() [1350]
  │  └─ SELECT COUNT(*) FROM "data_{table}"
  │     Result: 1234 rows (from previous run!)
  │
  ├─ reset_table_if_requested() [1356]
  │  └─ if RESET_TABLE=1: DROP TABLE
  │     if RESET_TABLE=0: keep existing rows
  │
  ├─ load_documents() [1365]
  │  └─ PyMuPDFReader → Document[]
  │
  ├─ chunk_documents() [1366]
  │  └─ SentenceSplitter(chunk_size, chunk_overlap)
  │     → chunks[] (NEW chunks from current run)
  │
  ├─ build_nodes() [1367]
  │  └─ TextNode(text=chunk, metadata=doc.metadata)
  │     Missing: chunk_size, run_id in metadata
  │
  ├─ embed_nodes() [1369]
  │  └─ embed_model.get_text_embedding_batch()
  │     → node.embedding = [0.123, -0.456, ...]
  │
  └─ insert_nodes() [1370]
     └─ vector_store.add(nodes)
        → INSERT INTO "data_{table}"
        → Appends to existing 1234 rows
        → Total: 1234 + NEW rows
```

**Problem:** Old chunks (1234) + new chunks (NEW) mixed in same table.

### F. Retrieval Flow

```
query_engine.query("Where does jealousy come from?")
  │
  ├─ VectorDBRetriever._retrieve() [670]
  │  │
  │  ├─ Embed query [686]
  │  │  └─ embed_model.get_query_embedding(query_str)
  │  │
  │  ├─ Vector search [689-695]
  │  │  └─ VectorStoreQuery(query_embedding, similarity_top_k=4)
  │  │     → Searches ALL rows in table
  │  │     → No filtering by metadata
  │  │     → Returns top-4 most similar chunks
  │  │     → Could be from ANY previous run
  │  │
  │  └─ Return NodeWithScore[] [702-705]
  │
  └─ LLM synthesis [from query engine]
     └─ prompt = context + query + "Answer:"
        → LLM generates answer from retrieved context
```

**No Run Isolation:** Queries don't know which chunks came from which run.

### G. Alternative Solutions (Not Chosen)

#### Option B: Always Reset Table

**Pros:**
- Simple, no code changes
- Works immediately

**Cons:**
- Must remember `RESET_TABLE=1` every time
- Can't compare configurations side-by-side
- Loses previous indexes

**Implementation:**
```bash
export RESET_TABLE=1
python rag_low_level_m1_16gb_verbose.py
```

#### Option C: Metadata Filtering

**Pros:**
- Single table can hold multiple runs
- Can query specific runs or compare
- Most flexible

**Cons:**
- Requires code changes in 3 places
- More complex
- Need to verify PGVectorStore filtering support

**Implementation:**

1. **Store run ID in metadata** (line ~952):
```python
n.metadata = src_doc.metadata.copy()
n.metadata["index_id"] = f"cs{S.chunk_size}_ov{S.chunk_overlap}"
n.metadata["indexed_at"] = datetime.now().isoformat()
```

2. **Add filter to query** (line ~689):
```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

filters = MetadataFilters(filters=[
    ExactMatchFilter(key="index_id", value=current_index_id)
])

vsq = VectorStoreQuery(
    query_embedding=q_emb,
    similarity_top_k=self._similarity_top_k,
    mode="default",
    filters=filters,  # NEW
)
```

3. **Pass index_id to retriever** (modify constructor):
```python
retriever = VectorDBRetriever(
    vector_store=vector_store,
    embed_model=embed_model,
    similarity_top_k=4,
    current_index_id=f"cs{S.chunk_size}_ov{S.chunk_overlap}",  # NEW
)
```

**Why We Didn't Choose This:**
- Option A (table naming) achieves the same isolation
- No code changes needed
- Simpler to understand and verify
- Table naming is more explicit (can see configs in `\dt` output)

---

## Summary

**The Issue:** Incremental indexing + no query filtering = mixed chunk sizes in results

**The Solution:** Encode chunk config in table name → automatic isolation

**The Benefit:** Clean, predictable behavior without code changes

**Next Step:** Create benchmark script to systematically compare configurations
