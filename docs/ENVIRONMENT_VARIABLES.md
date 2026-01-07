# Environment Variables Reference

Complete guide to all configurable environment variables for the RAG pipeline.

---

## Quick Start

```bash
# Minimal indexing (fast)
EMBED_BACKEND=mlx PDF_PATH=data/myfile.pdf PGTABLE=myindex python rag_low_level_m1_16gb_verbose.py --index-only

# Minimal querying
PGTABLE=myindex python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

---

## Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_NAME` | `vector_db` | PostgreSQL database name |
| `PGHOST` | `localhost` | Database host address |
| `PGPORT` | `5432` | Database port |
| `PGUSER` | `fryt` | Database username |
| `PGPASSWORD` | `frytos` | Database password |
| `PGTABLE` | `llama2_paper` | Table name for vector store |

**Example:**
```bash
PGHOST=localhost PGPORT=5432 PGUSER=fryt PGPASSWORD=frytos DB_NAME=vector_db
```

---

## Document & Index Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_PATH` | `data/llama2.pdf` | File or folder path to index |
| `RESET_TABLE` | `0` | Drop table before indexing (`0` or `1`) |
| `RESET_DB` | `0` | Drop entire database (`0` or `1`) ⚠️ DANGEROUS! |
| `EXTRACT_CHAT_METADATA` | `1` | Extract metadata from chat logs (`0` or `1`) |

**Examples:**
```bash
# Index a single file
PDF_PATH=data/document.pdf

# Index a folder
PDF_PATH=data/inbox_clean

# Reset table before indexing (recommended during development)
RESET_TABLE=1

# Extract chat metadata (participant, dates, message counts)
EXTRACT_CHAT_METADATA=1
```

---

## Chunking Configuration

**Critical for RAG quality!** Chunking determines how documents are split into searchable pieces.

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `700` | Characters per chunk (100-2000) |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks (typically 15-25% of chunk_size) |

### Chunking Presets

| Preset | CHUNK_SIZE | CHUNK_OVERLAP | Best For |
|--------|------------|---------------|----------|
| **Ultra-fine** | 100 | 20 | Chat logs, tweets, short messages |
| **Fine-grained** | 300 | 60 | Q&A with specific facts |
| **Balanced** ⭐ | 700 | 150 | General-purpose documents (recommended) |
| **Contextual** | 1200 | 240 | Summaries, complex topics |
| **Large context** | 2000 | 400 | Lengthy explanations, essays |

**Examples:**
```bash
# Ultra-fine (chat logs)
CHUNK_SIZE=100 CHUNK_OVERLAP=20

# Balanced (recommended)
CHUNK_SIZE=700 CHUNK_OVERLAP=150

# Large context (essays, long-form content)
CHUNK_SIZE=2000 CHUNK_OVERLAP=400
```

### Chunking Guidelines

**Overlap ratio:**
- 10-15%: Minimal, fast indexing
- 15-25%: Balanced (recommended)
- 25-30%: Maximum context preservation

**Trade-offs:**
- **Small chunks** (100-300): Precise retrieval but less context
- **Medium chunks** (500-800): Balanced (best for most cases)
- **Large chunks** (1000-2000): More context but less precise

---

## Embedding Configuration

Controls how text is converted to vector representations.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `BAAI/bge-small-en` | HuggingFace model name |
| `EMBED_DIM` | `384` | Vector dimensions (384/768/1024) |
| `EMBED_BATCH` | `32` | Batch size for embedding |
| `EMBED_BACKEND` | `huggingface` | Backend: `huggingface` \| `mlx` |

### Model Options

| Model | EMBED_DIM | Speed | Quality | MLX Support | Best For |
|-------|-----------|-------|---------|-------------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fastest | Good | ⚠️ Limited | Quick prototyping |
| `BAAI/bge-small-en` ⭐ | 384 | Fast | Very Good | ✅ Excellent | General use (recommended) |
| `BAAI/bge-base-en` | 768 | Medium | Excellent | ✅ Good | Higher quality needs |
| `BAAI/bge-large-en-v1.5` | 1024 | Slow | Best | ⚠️ Poor | Maximum quality |

### Backend Options

| Backend | Speed | Hardware | Best For |
|---------|-------|----------|----------|
| `huggingface` | Baseline | Any (CPU/GPU/MPS) | Compatibility |
| `mlx` ⚡ | **5-20x faster** | Apple Silicon only | M1/M2/M3 Macs |

### Recommended Configurations

**Fast indexing with MLX (recommended for M1/M2/M3):**
```bash
EMBED_BACKEND=mlx
EMBED_MODEL=BAAI/bge-small-en
EMBED_DIM=384
EMBED_BATCH=64
```

**Maximum quality:**
```bash
EMBED_BACKEND=mlx
EMBED_MODEL=BAAI/bge-large-en-v1.5
EMBED_DIM=1024
EMBED_BATCH=32
```

**Balanced (no MLX):**
```bash
EMBED_BACKEND=huggingface
EMBED_MODEL=BAAI/bge-small-en
EMBED_DIM=384
EMBED_BATCH=32
```

### Batch Size Guidelines

| Backend | Model Size | Recommended EMBED_BATCH | Memory |
|---------|------------|------------------------|--------|
| HuggingFace | Small (384d) | 32-64 | 8GB+ |
| HuggingFace | Large (1024d) | 16-32 | 16GB+ |
| MLX | Small (384d) | 128-256 | 8GB+ |
| MLX | Large (1024d) | 64-128 | 16GB+ |

---

## Retrieval Configuration

Controls how chunks are retrieved during queries.

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | `4` | Number of chunks to retrieve (2-10) |
| `HYBRID_ALPHA` | `1.0` | Search mode: `0.0`=BM25, `0.5`=hybrid, `1.0`=vector |
| `ENABLE_FILTERS` | `1` | Enable metadata filtering (`0` or `1`) |
| `MMR_THRESHOLD` | `0.0` | Diversity: `0`=disabled, `0.5`=balanced, `1.0`=max |

### Search Mode (HYBRID_ALPHA)

| Value | Mode | Description |
|-------|------|-------------|
| `0.0` | Pure BM25 | Keyword-only search (like grep) |
| `0.3` | BM25-heavy | Mostly keywords with some semantics |
| `0.5` | Balanced hybrid | Best of both worlds ⭐ |
| `0.7` | Vector-heavy | Mostly semantic with some keywords |
| `1.0` | Pure vector | Semantic-only search (default) |

**Examples:**
```bash
# Pure vector search (default)
HYBRID_ALPHA=1.0

# Balanced hybrid (recommended)
HYBRID_ALPHA=0.5

# Keyword-only search
HYBRID_ALPHA=0.0
```

### Metadata Filtering

When `ENABLE_FILTERS=1`, you can filter by chat metadata in queries:

```bash
# Query syntax examples:
participant:Alice meeting agenda
after:2024-06-01 project updates
before:2024-12-31 quarterly review
participant:Bob after:2024-01-01 travel plans
```

### MMR (Maximum Marginal Relevance)

Controls diversity in retrieved results:

| Value | Behavior |
|-------|----------|
| `0.0` | Disabled (default) - returns most similar chunks |
| `0.5` | Balanced - mix of relevance and diversity |
| `1.0` | Maximum relevance - focus on query match |

**Example:**
```bash
# Enable diversity (avoid repetitive chunks)
MMR_THRESHOLD=0.5
```

---

## LLM Configuration

Controls the local LLM (Mistral 7B via llama.cpp).

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_URL` | Mistral 7B Q4_K_M | URL to download GGUF model |
| `MODEL_PATH` | _(empty)_ | Local path to GGUF model (overrides URL) |
| `TEMP` | `0.1` | Temperature (0.0=deterministic, 1.0=creative) |
| `MAX_NEW_TOKENS` | `256` | Maximum tokens to generate |
| `CTX` | `3072` | Context window size |
| `N_GPU_LAYERS` | `16` | Layers to offload to Metal GPU (0-32) |
| `N_BATCH` | `128` | Batch size for prompt processing |

### Apple Silicon Tuning

| Hardware | N_GPU_LAYERS | N_BATCH | CTX | Memory Usage |
|----------|--------------|---------|-----|--------------|
| M1/M2 8GB | 8 | 64 | 2048 | Conservative |
| M1/M2 16GB | 16 | 128 | 3072 | Balanced ⭐ |
| M1/M2 16GB | 24 | 256 | 8192 | Aggressive |
| M1 Pro/Max 32GB | 32 | 512 | 8192 | Maximum |

**Examples:**
```bash
# Conservative (8GB Mac)
N_GPU_LAYERS=8 N_BATCH=64 CTX=2048

# Balanced (16GB Mac, recommended)
N_GPU_LAYERS=16 N_BATCH=128 CTX=3072

# Aggressive (16GB Mac, fast inference)
N_GPU_LAYERS=24 N_BATCH=256 CTX=8192
```

### Temperature Guidelines

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.0-0.2` | Deterministic | Factual Q&A, retrieval tasks |
| `0.3-0.5` | Slightly creative | Summaries, explanations |
| `0.6-0.8` | Creative | Brainstorming, creative writing |
| `0.9-1.0` | Very creative | Story generation, poetry |

---

## Query Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QUESTION` | _(default question)_ | Default query text |

**Example:**
```bash
QUESTION="What did Alice say about the Morocco trip?" python rag_low_level_m1_16gb_verbose.py --query-only
```

---

## Logging & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_FULL_CHUNKS` | `0` | Log full chunk content (`0` or `1`) |
| `COLORIZE_CHUNKS` | `0` | Colorize chunk output (`0` or `1`) |
| `LOG_QUERIES` | `0` | Save queries to JSON log (`0` or `1`) |
| `DB_INSERT_BATCH` | `250` | Database insert batch size |

**Examples:**
```bash
# Debug mode with full chunk logging
LOG_LEVEL=DEBUG LOG_FULL_CHUNKS=1

# Colorized output (better terminal readability)
COLORIZE_CHUNKS=1

# Save all queries to query_logs/
LOG_QUERIES=1
```

---

## Complete Usage Examples

### 1. Fast MLX Indexing (Recommended)

```bash
EMBED_BACKEND=mlx \
EMBED_MODEL=BAAI/bge-small-en \
EMBED_DIM=384 \
EMBED_BATCH=64 \
CHUNK_SIZE=700 \
CHUNK_OVERLAP=150 \
PDF_PATH=data/inbox_clean \
PGTABLE=inbox_mlx_fast \
RESET_TABLE=1 \
EXTRACT_CHAT_METADATA=1 \
python rag_low_level_m1_16gb_verbose.py --index-only
```

**Performance:** ~6-10 minutes for 47K chunks

---

### 2. High-Quality Indexing

```bash
EMBED_BACKEND=mlx \
EMBED_MODEL=BAAI/bge-large-en-v1.5 \
EMBED_DIM=1024 \
EMBED_BATCH=32 \
CHUNK_SIZE=700 \
CHUNK_OVERLAP=150 \
PDF_PATH=data/inbox_clean \
PGTABLE=inbox_mlx_quality \
RESET_TABLE=1 \
python rag_low_level_m1_16gb_verbose.py --index-only
```

**Performance:** ~40-45 minutes for 47K chunks (higher quality)

---

### 3. Ultra-Fine Chunking (Chat Logs)

```bash
EMBED_BACKEND=mlx \
EMBED_MODEL=BAAI/bge-small-en \
EMBED_DIM=384 \
EMBED_BATCH=128 \
CHUNK_SIZE=100 \
CHUNK_OVERLAP=20 \
EXTRACT_CHAT_METADATA=1 \
PDF_PATH=data/inbox_clean \
PGTABLE=inbox_ultrafine \
RESET_TABLE=1 \
python rag_low_level_m1_16gb_verbose.py --index-only
```

**Performance:** ~8-12 minutes for 47K chunks (many small chunks)

---

### 4. Query with Hybrid Search + Filters

```bash
PGTABLE=inbox_mlx_fast \
TOP_K=6 \
HYBRID_ALPHA=0.5 \
ENABLE_FILTERS=1 \
MMR_THRESHOLD=0.5 \
LOG_FULL_CHUNKS=1 \
COLORIZE_CHUNKS=1 \
python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

Then query with filters:
```
> participant:Alice Morocco trip
> after:2024-06-01 project updates
```

---

### 5. Production Query (Fast LLM)

```bash
PGTABLE=inbox_mlx_fast \
TOP_K=4 \
HYBRID_ALPHA=0.5 \
N_GPU_LAYERS=24 \
N_BATCH=256 \
CTX=8192 \
TEMP=0.1 \
MAX_NEW_TOKENS=512 \
python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

---

### 6. Debug Mode (Troubleshooting)

```bash
LOG_LEVEL=DEBUG \
LOG_FULL_CHUNKS=1 \
COLORIZE_CHUNKS=1 \
LOG_QUERIES=1 \
PGTABLE=inbox_mlx_fast \
python rag_low_level_m1_16gb_verbose.py --query-only --query "test query"
```

---

## Performance Optimization Cheatsheet

### For Fastest Indexing:
```bash
EMBED_BACKEND=mlx
EMBED_MODEL=BAAI/bge-small-en
EMBED_BATCH=128
CHUNK_SIZE=700
```

### For Best Quality:
```bash
EMBED_MODEL=BAAI/bge-large-en-v1.5
EMBED_DIM=1024
CHUNK_SIZE=700
TOP_K=6
HYBRID_ALPHA=0.5
```

### For Chat Logs:
```bash
CHUNK_SIZE=100
CHUNK_OVERLAP=20
EXTRACT_CHAT_METADATA=1
ENABLE_FILTERS=1
```

### For Fast LLM Inference:
```bash
N_GPU_LAYERS=24
N_BATCH=256
CTX=8192
TEMP=0.1
```

---

## Tips & Tricks

### 1. Table Naming Convention
Include config in table names for easy tracking:
```bash
PGTABLE=inbox_cs700_ov150_bge_small_mlx
```

### 2. Iterative Development
Use `RESET_TABLE=1` during development:
```bash
RESET_TABLE=1  # Avoids duplicate rows
```

### 3. Test Before Full Index
Index a small subset first:
```bash
PDF_PATH=data/inbox_small PGTABLE=test_small
```

### 4. Monitor Memory
For 16GB Macs, stay conservative with batch sizes:
```bash
EMBED_BATCH=64 N_BATCH=128
```

### 5. Save Queries for Analysis
```bash
LOG_QUERIES=1  # Saves to query_logs/ directory
```

---

## Troubleshooting

### Slow Embedding
```bash
# Check if MLX is being used
EMBED_BACKEND=mlx EMBED_MODEL=BAAI/bge-small-en

# Increase batch size
EMBED_BATCH=64  # or 128 for MLX
```

### Out of Memory
```bash
# Reduce batch sizes
EMBED_BATCH=32
N_BATCH=64

# Reduce GPU layers
N_GPU_LAYERS=8
```

### Poor Retrieval Quality
```bash
# Try smaller chunks
CHUNK_SIZE=300 CHUNK_OVERLAP=60

# Use hybrid search
HYBRID_ALPHA=0.5

# Increase TOP_K
TOP_K=8
```

### Context Window Overflow
```bash
# Reduce chunk size
CHUNK_SIZE=500

# Reduce TOP_K
TOP_K=3

# Increase context window
CTX=8192
```

---

## Quick Reference Table

| Task | Key Variables |
|------|---------------|
| Fast indexing | `EMBED_BACKEND=mlx EMBED_BATCH=64` |
| Quality indexing | `EMBED_MODEL=BAAI/bge-large-en-v1.5 CHUNK_SIZE=700` |
| Chat logs | `CHUNK_SIZE=100 EXTRACT_CHAT_METADATA=1` |
| Hybrid search | `HYBRID_ALPHA=0.5 ENABLE_FILTERS=1` |
| Fast LLM | `N_GPU_LAYERS=24 N_BATCH=256` |
| Debug | `LOG_LEVEL=DEBUG LOG_FULL_CHUNKS=1` |

---

## See Also

- `CLAUDE.md` - Full project documentation
- `INTERACTIVE_GUIDE.md` - Interactive CLI guide
- `FIXES_APPLIED.md` - Recent fixes and improvements
- `START_HERE.md` - Quick start guide
