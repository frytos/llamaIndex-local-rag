# Advanced Retrieval Features Guide

## Overview

Three powerful retrieval enhancements for chat logs and documents:

1. **Hybrid Search** - Combines BM25 keyword matching with semantic vector search
2. **Metadata Filtering** - Filter by participant names, dates, and more
3. **MMR (Maximal Marginal Relevance)** - Ensures diverse, non-redundant results

---

## 1. Hybrid Search (BM25 + Semantic)

### What It Does
- **BM25**: Traditional keyword matching (exact terms, word frequency)
- **Vector Search**: Semantic understanding (meaning, context)
- **Hybrid**: Combines both for best of both worlds

### Configuration

```bash
HYBRID_ALPHA=0.5  # Balance between BM25 and vector
```

**Values:**
- `0.0` = Pure BM25 (keyword only)
- `0.3` = Favor keywords (good for technical terms, names)
- `0.5` = Balanced hybrid (recommended for chat logs)
- `0.7` = Favor semantics (good for conceptual queries)
- `1.0` = Pure vector (semantic only, default)

### When to Use

**Use BM25-weighted (alpha < 0.7) for:**
- Searching for specific names: "What did Elena say about Paris?"
- Technical terms: "Find messages mentioning PostgreSQL"
- Exact phrases: "Who said 'amazing'"
- Rare words that semantic models might miss

**Use semantic-weighted (alpha > 0.5) for:**
- Conceptual queries: "What did we discuss about relationships?"
- Paraphrased queries: "When did we talk about meeting up?"
- Understanding intent vs exact words

### Example

```bash
# Balanced hybrid search
HYBRID_ALPHA=0.5 python rag_low_level_m1_16gb_verbose.py \
  --query-only --interactive \
  --query "Find messages about traveling to Europe"
```

### Requirements

```bash
pip install rank-bm25
```

---

## 2. Metadata Filtering

### What It Does
Automatically extracts and filters by:
- **Participants**: Filter messages from specific people
- **Date ranges**: Find messages within time periods
- **Message counts**: See conversation density

### Configuration

```bash
ENABLE_FILTERS=1  # Enable metadata filtering (default: enabled)
EXTRACT_CHAT_METADATA=1  # Extract participant/date info (default: enabled)
```

### Query Syntax

**Filter by participant:**
```
"What did we discuss about travel participant:EB"
"participant:Arnaud vacation plans"
"from:Elena messages about food"
```

**Filter by date range:**
```
"What happened after:2024-06-01"
"Messages before:2023-12-31"
"Conversations after:2024-01-01 before:2024-06-30"
```

**Combine filters:**
```
"participant:Elena after:2024-06-01 travel plans"
"from:Arnaud before:2024-01-01 meeting"
```

### Example

```bash
# Index with metadata extraction
EXTRACT_CHAT_METADATA=1 RESET_TABLE=1 \
  CHUNK_SIZE=300 CHUNK_OVERLAP=100 \
  PDF_PATH=data/inbox_clean \
  PGTABLE=inbox_cs300_filtered \
  python rag_low_level_m1_16gb_verbose.py

# Query with filters
ENABLE_FILTERS=1 python rag_low_level_m1_16gb_verbose.py \
  --query-only --interactive \
  --query "participant:EB after:2024-06-01 What did Elena say about events?"
```

### Output Format

Chunks will show metadata:
```
Score: 0.8750 | Source: inbox_clean | Participants: EB, Arnaud Grd | 2024-06-18 to 2024-06-20
```

---

## 3. MMR (Maximal Marginal Relevance)

### What It Does
Prevents retrieving similar/redundant chunks:
- Finds relevant chunks
- Penalizes chunks too similar to already-selected ones
- Ensures diversity in conversation coverage

### Configuration

```bash
MMR_THRESHOLD=0.7  # Enable MMR with lambda parameter
```

**Values:**
- `0.0` = Disabled (default)
- `0.3` = Maximum diversity (very different chunks)
- `0.5` = Balanced diversity
- `0.7` = Moderate diversity (recommended)
- `0.9` = Minimal diversity (mostly relevance)

### When to Use

**Use MMR (0.5-0.7) when:**
- Chat logs have repetitive messages
- You want broad conversation coverage
- Query returns many chunks from same conversation
- You need diverse perspectives

**Disable MMR (0.0) when:**
- You want all details from same conversation thread
- Deep-diving into specific topic
- Chunks are naturally diverse

### Example

```bash
# Enable MMR for diverse results
MMR_THRESHOLD=0.7 TOP_K=6 \
  python rag_low_level_m1_16gb_verbose.py \
  --query-only --interactive \
  --query "What did we discuss about future plans?"
```

**Output:**
```
🎲 Applying MMR (Maximal Marginal Relevance) for diversity:
  • Lambda: 0.70 (0=max diversity, 1=max relevance)
  • Selected 6 diverse results
```

---

## Combined Usage Examples

### Example 1: Chat Log Power Search

```bash
# Index with all features
EXTRACT_CHAT_METADATA=1 \
CHUNK_SIZE=300 \
CHUNK_OVERLAP=100 \
RESET_TABLE=1 \
PDF_PATH=data/inbox_clean \
PGTABLE=inbox_advanced \
python rag_low_level_m1_16gb_verbose.py

# Query with all features enabled
HYBRID_ALPHA=0.5 \
ENABLE_FILTERS=1 \
MMR_THRESHOLD=0.7 \
TOP_K=6 \
LOG_FULL_CHUNKS=1 \
COLORIZE_CHUNKS=1 \
python rag_low_level_m1_16gb_verbose.py \
  --query-only --interactive
```

### Example 2: Find Specific Person's Messages

```bash
# Query: What did Elena say about events in June 2024?
ENABLE_FILTERS=1 HYBRID_ALPHA=0.4 \
  python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "participant:EB after:2024-06-01 before:2024-06-30 events"
```

### Example 3: Broad Topic Discovery

```bash
# Get diverse chunks about a topic
HYBRID_ALPHA=0.6 MMR_THRESHOLD=0.7 TOP_K=8 \
  python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "What topics did we discuss over the years?"
```

### Example 4: Name-Based Search

```bash
# Find exact name mentions (keyword-heavy)
HYBRID_ALPHA=0.3 \
  python rag_low_level_m1_16gb_verbose.py \
  --query-only \
  --query "Who mentioned Paris?"
```

---

## Recommended Settings for Chat Logs

### For Instagram/WhatsApp/Messenger

```bash
# Indexing settings
EXTRACT_CHAT_METADATA=1      # Extract participants/dates
CHUNK_SIZE=300               # 3-5 messages per chunk
CHUNK_OVERLAP=100            # 33% overlap for context
RESET_TABLE=1                # Clean slate

# Retrieval settings
HYBRID_ALPHA=0.5             # Balanced hybrid
ENABLE_FILTERS=1             # Enable filtering
MMR_THRESHOLD=0.7            # Diverse results
TOP_K=6                      # More chunks for context

# Display settings
LOG_FULL_CHUNKS=1            # See complete messages
COLORIZE_CHUNKS=1            # Color-code participants
```

### Complete Command

```bash
# Index
EXTRACT_CHAT_METADATA=1 CHUNK_SIZE=300 CHUNK_OVERLAP=100 RESET_TABLE=1 \
  PDF_PATH=data/your_chat.html \
  PGTABLE=chat_advanced \
  python rag_low_level_m1_16gb_verbose.py

# Query
HYBRID_ALPHA=0.5 ENABLE_FILTERS=1 MMR_THRESHOLD=0.7 TOP_K=6 \
  LOG_FULL_CHUNKS=1 COLORIZE_CHUNKS=1 \
  python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

---

## Troubleshooting

### "rank-bm25 not installed"
```bash
pip install rank-bm25
```
Or the system falls back to pure vector search automatically.

### Filters not working
Make sure you:
1. Re-indexed with `EXTRACT_CHAT_METADATA=1`
2. Used `RESET_TABLE=1` if table already existed
3. Check chunk has chat metadata: look for "Participants:" in output

### MMR too slow
- Reduce `TOP_K` (fewer candidates to diversify)
- Increase `MMR_THRESHOLD` closer to 1.0 (less diversity computation)

### Hybrid search finds wrong results
- Adjust `HYBRID_ALPHA`:
  - Lower (0.3-0.4) for more keyword emphasis
  - Higher (0.7-0.8) for more semantic emphasis

---

## Performance Impact

| Feature | Index Time | Query Time | Memory |
|---------|-----------|------------|--------|
| Metadata Extraction | +5% | +0% | +10% |
| Hybrid Search (BM25) | +0% | +50-100% | +50% (loads corpus) |
| MMR Diversity | +0% | +20-40% | +10% |

**Notes:**
- BM25 loads entire corpus into memory on first query
- For large datasets (>10K chunks), BM25 init takes 5-10 seconds
- MMR only processes top candidates (efficient)

---

## Quick Reference

```bash
# Environment Variables
HYBRID_ALPHA=0.5              # 0=BM25, 1=vector
ENABLE_FILTERS=1              # Enable metadata filtering
MMR_THRESHOLD=0.7             # Enable MMR diversity
EXTRACT_CHAT_METADATA=1       # Extract participant/date info

# Query Syntax
participant:Name              # Filter by participant
from:Name                     # Alias for participant
after:YYYY-MM-DD             # Messages after date
before:YYYY-MM-DD            # Messages before date
since:YYYY-MM-DD             # Alias for after
```

---

## Next Steps

1. **Re-index your chat logs** with metadata extraction:
   ```bash
   EXTRACT_CHAT_METADATA=1 RESET_TABLE=1 CHUNK_SIZE=300 CHUNK_OVERLAP=100 \
     PDF_PATH=data/inbox_clean \
     PGTABLE=inbox_advanced \
     python rag_low_level_m1_16gb_verbose.py
   ```

2. **Install BM25** for hybrid search:
   ```bash
   pip install rank-bm25
   ```

3. **Try interactive queries** with all features:
   ```bash
   HYBRID_ALPHA=0.5 ENABLE_FILTERS=1 MMR_THRESHOLD=0.7 \
     LOG_FULL_CHUNKS=1 COLORIZE_CHUNKS=1 \
     python rag_low_level_m1_16gb_verbose.py --query-only --interactive
   ```

4. **Experiment with different alpha values** to see the difference:
   - Try pure BM25: `HYBRID_ALPHA=0.0`
   - Try balanced: `HYBRID_ALPHA=0.5`
   - Try pure semantic: `HYBRID_ALPHA=1.0`

Enjoy your enhanced RAG system! 🚀
