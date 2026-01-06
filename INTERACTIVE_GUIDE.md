# Interactive CLI Guide - MLX-Optimized RAG

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch interactive CLI
python rag_interactive.py
```

## Features

### 1. Index New Document(s)
**Best for**: First-time indexing of documents or folders

**Workflow**:
1. Select document or folder from `data/` directory
2. Configure chunking parameters (presets available)
3. Select embedding model and backend
   - **MLX backend** (recommended for M1/M2/M3 Macs): 5-20x faster indexing
   - **HuggingFace backend**: Standard option, works on all platforms
4. Confirm table name and start indexing

**Example Output** (with MLX):
```
  Document: data/inbox_clean/
  Backend: MLX
  Batch size: 64

  ⚡ MLX acceleration enabled!
  Expected speedup: 5-20x vs standard backend
```

---

### 2. Query Existing Index
**Best for**: Asking questions on already-indexed data

**Workflow**:
1. Select from list of existing indexes
2. Configure TOP_K (retrieval count)
3. Enter interactive query mode
4. Type queries or use filters:
   - `participant:Name search terms`
   - `after:2024-01-01 search terms`
   - `before:2024-12-31 search terms`

---

### 3. Index + Query (Full Pipeline)
**Best for**: Quick start - index and immediately query

**Workflow**:
Combines options 1 and 2 in a single flow.

---

### 4. View Existing Indexes
**Best for**: Checking what's already indexed

**Shows**:
- Table name
- Number of chunks
- Chunking configuration (chunk_size, overlap)
- Index signature

---

### 5. Quick Re-index / MLX Optimization ⚡
**Best for**:
- Re-indexing with different chunking parameters
- **Migrating existing indexes to MLX backend for speedup**
- Testing different embedding models

**Workflow**:
1. Select document or folder (same as before or new)
2. Configure new parameters:
   - Different chunk size/overlap
   - **Switch to MLX backend** for dramatic speedup
   - Try different embedding model
3. Auto-generates table name with `_mlx` suffix if using MLX
4. Option to query immediately after indexing

**Example Use Case**:
```
Scenario: You have "inbox_clean" indexed with HuggingFace (50 minutes)
Goal: Re-index with MLX backend (5-8 minutes)

Steps:
1. Choose option 5 (Quick re-index / MLX optimization)
2. Select: data/inbox_clean/
3. Keep same chunking: Fine-grained (300/60)
4. Select: bge-small-en model
5. When prompted: "Use MLX backend?" → Yes
6. Table name: inbox_clean_cs300_ov60_mlx
7. Confirm and watch it index 6-9x faster!
```

---

## MLX Backend Details

### When to Use MLX
- ✅ You have an M1/M2/M3 Mac
- ✅ Indexing large document collections (1000+ chunks)
- ✅ Need to re-index frequently
- ✅ Want faster experimentation cycles

### Performance Comparison

| Scenario | HuggingFace (MPS) | MLX | Speedup |
|----------|-------------------|-----|---------|
| 1,000 chunks | ~50 seconds | ~6 seconds | **8x** |
| 10,000 chunks | ~8 minutes | ~1 minute | **8x** |
| 47,000 chunks | ~45-50 minutes | ~5-8 minutes | **6-9x** |

### Automatic Optimizations with MLX
When you select MLX backend, the CLI automatically:
- ✓ Increases batch size from 32 → 64
- ✓ Enables Metal GPU acceleration
- ✓ Uses optimized MLX models
- ✓ Enables chat metadata extraction
- ✓ Adds `_mlx` suffix to table name

### MLX Installation
If MLX is not detected, you'll see:
```
Note: MLX not installed. Using HuggingFace backend.
Install MLX for 5-20x speedup: pip install mlx mlx-embedding-models
```

To install:
```bash
pip install mlx mlx-embedding-models rank-bm25
```

---

## Chunking Presets

| Preset | Size | Overlap | Best For |
|--------|------|---------|----------|
| Ultra-fine | 100 | 20 | Chat logs, tweets, short messages |
| Fine-grained | 300 | 60 | Q&A with specific facts |
| Balanced (Recommended) | 700 | 150 | General-purpose documents |
| Contextual | 1200 | 240 | Summaries, complex topics |
| Large context | 2000 | 400 | Lengthy explanations |
| Custom | - | - | Enter your own values |

---

## Embedding Models

| Model | Dimensions | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fastest | Good | General text |
| bge-small-en | 384 | Fast | Better | Balanced (recommended) |
| bge-base-en | 768 | Medium | Very Good | Higher quality needs |
| bge-large-en | 1024 | Slower | Best | Maximum quality |

**Note**: All models work with both HuggingFace and MLX backends.

---

## Example Workflows

### Workflow 1: First-Time Indexing with MLX
```
1. python rag_interactive.py
2. Choose: 1 (Index new document)
3. Select: data/inbox_clean/
4. Chunking: Fine-grained (300/60)
5. Model: bge-small-en
6. MLX prompt: Yes (Use MLX backend)
7. Table: inbox_clean_cs300_ov60 (suggested)
8. Confirm and watch fast indexing!
```

### Workflow 2: Migrating Existing Index to MLX
```
1. python rag_interactive.py
2. Choose: 5 (Quick re-index / MLX optimization)
3. Select: Same document as before
4. Chunking: Same as before (or try new settings)
5. Model: Same as before (or try new model)
6. MLX prompt: Yes (Use MLX backend)
7. Table: old_name_mlx (auto-suggested)
8. Confirm and compare speed!
```

### Workflow 3: Testing Different Chunking
```
1. python rag_interactive.py
2. Choose: 5 (Quick re-index / MLX optimization)
3. Select: Your document
4. Try different chunking presets:
   - Ultra-fine (100/20) for chat logs
   - Balanced (700/150) for mixed content
   - Large context (2000/400) for essays
5. Use MLX for fast iteration
6. Query each index to compare results
```

---

## Advanced Features

### Hybrid Search
All queries use hybrid search by default:
- BM25 keyword matching
- Vector semantic search
- Configurable via `HYBRID_ALPHA` environment variable

### Metadata Filtering
When indexing chat logs or dated documents, use filters in queries:
```
participant:Alice meeting agenda
after:2024-06-01 project updates
before:2024-12-31 quarterly review
```

### HNSW Indexing
Automatically enabled for faster vector similarity search.

---

## Troubleshooting

### MLX Not Detected
```bash
# Check Python version (needs 3.11+)
python --version

# Install MLX
pip install mlx mlx-embedding-models

# Verify installation
python -c "import mlx.core as mx; print('MLX OK')"
```

### Database Connection Error
```bash
# Start PostgreSQL (if using Docker)
docker-compose up -d

# Verify connection
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "SELECT version();"
```

### Memory Issues
- Reduce `EMBED_BATCH` size (default: 64 for MLX, 32 for HuggingFace)
- Use smaller embedding model (all-MiniLM-L6-v2 or bge-small-en)
- Process documents in smaller batches

---

## Environment Variables (Advanced)

Override defaults by setting before running:

```bash
# Embedding configuration
export EMBED_BACKEND=mlx          # mlx or huggingface
export EMBED_MODEL=BAAI/bge-small-en
export EMBED_DIM=384
export EMBED_BATCH=64

# Chunking
export CHUNK_SIZE=300
export CHUNK_OVERLAP=60

# Retrieval
export TOP_K=4
export HYBRID_ALPHA=0.5           # 0.0=BM25 only, 1.0=vector only
export ENABLE_FILTERS=1

# Database
export PGHOST=localhost
export PGPORT=5432
export PGUSER=fryt
export PGPASSWORD=frytos
export DB_NAME=vector_db

# Then run
python rag_interactive.py
```

---

## Next Steps

After indexing with MLX:
1. **Benchmark**: Compare indexing times (HuggingFace vs MLX)
2. **Query**: Test retrieval quality on both indexes
3. **Optimize**: Fine-tune chunking parameters for your use case
4. **Scale**: Index larger document collections with confidence

For programmatic use, see `rag_low_level_m1_16gb_verbose.py` with environment variables.

---

## Support

- Documentation: `CLAUDE.md`
- MLX Setup: `START_HERE.md`, `POST_EMBEDDING_PLAN.md`
- Quick Commands: `QUICK_COMMANDS.sh`
