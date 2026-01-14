# Dimension Selector - Streamlit UI Enhancement

**Date**: 2026-01-09
**Status**: ✅ Complete

## What Was Added

Enhanced the Streamlit web UI (`rag_web.py`) to **automatically detect and match embedding dimensions** between indexed tables and query models.

## Key Features

### 1. Automatic Dimension Detection
- Detects the embedding dimension from PostgreSQL vector column type
- Shows dimension in table listings: `vector(384)`, `vector(768)`, `vector(1024)`
- Extracts and displays the model used for indexing

### 2. Smart Model Selection
- **Filters compatible models** based on table dimension
- Only shows models that match the table's embedding dimension
- Prevents dimension mismatch errors before querying

### 3. Visual Warnings
```
🚨 DIMENSION MISMATCH!
Table has 384 dimensions but selected model has 768 dimensions
This will cause an error. Please select a compatible model above.
```

### 4. Index Info Display
When selecting an index, you now see:
- **Chunks**: Total number of chunks
- **Chunk Size**: Chunking configuration
- **Embedding Dim**: Vector dimension (384, 768, 1024)
- **Indexed with**: Original model name

## Changes Made

### File: `rag_web.py`

#### 1. Enhanced `list_vector_tables()` (lines 171-244)
- Added dimension detection from PostgreSQL column metadata
- Extracts `embed_model` from chunk metadata
- Returns dimension info for each table

#### 2. Updated `page_query()` (lines 722-820)
- Shows table dimension prominently
- Filters embedding models by compatibility
- Displays dimension mismatch warnings
- Blocks queries if dimensions don't match

#### 3. Updated `run_query()` (lines 830-854)
- Accepts embedding model parameters
- Updates settings before querying
- Shows full traceback on errors

#### 4. Enhanced `page_view_indexes()` (lines 932-942)
- Shows "Embed Dim" column in table view

## How It Works

### Before (Problem)
```
User: Queries with bge-base (768 dims)
Table: Indexed with bge-small (384 dims)
Result: ValueError: expected 384 dimensions, not 768 ❌
```

### After (Solution)
```
1. User selects table → UI detects: 384 dimensions
2. UI filters models → Only shows 384-dim models ✓
3. User selects bge-small (384 dims)
4. Query succeeds ✓
```

## Usage

### Launch Streamlit UI
```bash
streamlit run rag_web.py
```

### Query Page Workflow
1. **Select Index** - Choose from available tables
   - See dimension info: `inbox_conversations (10687 chunks, dim=384)`

2. **Query Settings** - UI automatically:
   - Detects table dimension (384)
   - Shows only compatible models
   - Warns if you select incompatible model

3. **Ask Question** - Query runs with correct model

## Model-Dimension Mapping

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast, English |
| `bge-small-en` | 384 | Recommended, English |
| `paraphrase-multilingual-MiniLM` | 384 | Fast, Multilingual |
| `bge-base-en-v1.5` | 768 | Better quality, English |
| `bge-large-en-v1.5` | 1024 | Best quality, English |
| `bge-m3` | 1024 | Best, Multilingual (100+ languages) |

## Backend Selection

The UI also lets you choose:
- **huggingface** - Standard PyTorch backend
- **mlx** - Apple Silicon optimized (9x faster for bge-m3)

## Benefits

✅ **Prevents errors** - Catches dimension mismatches before querying
✅ **User-friendly** - Clear warnings and automatic filtering
✅ **Transparent** - Shows what model was used for indexing
✅ **Fast** - Detects dimensions via SQL query (milliseconds)

## Testing

Test the dimension selector:

```bash
# 1. Index with different models
export EMBED_MODEL=BAAI/bge-small-en-v1.5 EMBED_DIM=384
python rag_low_level_m1_16gb_verbose.py  # Creates 384-dim table

export EMBED_MODEL=BAAI/bge-base-en-v1.5 EMBED_DIM=768
python rag_low_level_m1_16gb_verbose.py  # Creates 768-dim table

# 2. Launch UI
streamlit run rag_web.py

# 3. Go to Query page
# - Select 384-dim table → See only 384-dim models
# - Select 768-dim table → See only 768-dim models
# - Try selecting wrong model → See warning
```

## Next Steps

Consider adding:
- [ ] Dimension converter (re-index to different dimension)
- [ ] Model recommendation based on use case
- [ ] Dimension comparison tool (show which is better)
- [ ] Batch dimension checker for all tables

## Related Files

- `rag_web.py` - Enhanced Streamlit UI
- `rag_low_level_m1_16gb_verbose.py` - Core RAG pipeline
- `diagnose_dimension_mismatch.py` - CLI diagnostic tool
- `debug_hybrid_search.py` - Hybrid search debugging
