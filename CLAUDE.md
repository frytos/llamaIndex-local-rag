# Local RAG Pipeline - Claude Development Guide

**Last Updated**: January 2026 | **Version**: 2.0.0

## Project Overview

A local Retrieval-Augmented Generation (RAG) pipeline optimized for both M1 Mac (16GB) and GPU servers (RTX 4090). Supports indexing documents (PDF, HTML, code files, etc.) into PostgreSQL+pgvector and querying with local LLM (Mistral 7B via llama.cpp or vLLM for GPU acceleration).

## Tech Stack

- **Runtime**: Python 3.11+
- **Framework**: LlamaIndex
- **Vector Store**: PostgreSQL + pgvector + **HNSW indices** (100x+ faster queries)
- **Embeddings**: HuggingFace (bge-small-en, all-MiniLM-L6-v2)
- **LLM**: llama.cpp (GGUF) or vLLM (AWQ) for Mistral 7B
- **GPU**: Apple Metal (MPS) on Mac, CUDA on NVIDIA GPUs
- **Web UI**: Streamlit + Plotly

## Quick Start Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run interactive CLI
python rag_interactive.py

# Run main pipeline (HNSW indices automatically created)
python rag_low_level_m1_16gb_verbose.py

# Launch web UI
streamlit run rag_web.py

# Run with custom config
CHUNK_SIZE=500 CHUNK_OVERLAP=100 PDF_PATH=data/myfile.pdf PGTABLE=myindex python rag_low_level_m1_16gb_verbose.py

# Add HNSW indices to existing tables
python migrate_add_hnsw_indices.py --yes

# Validate query performance
python scripts/validate_hnsw_performance.py --all
```

## File Structure

```
llamaIndex-local-rag/
├── rag_low_level_m1_16gb_verbose.py  # Main RAG pipeline (core logic)
├── rag_interactive.py                 # CLI menu interface
├── rag_web.py                         # Streamlit web UI
├── vllm_client.py                     # vLLM OpenAI-compatible client
├── vllm_wrapper.py                    # vLLM high-level wrapper
├── reranker.py                        # Query reranking utilities
├── query_cache.py                     # Query caching system
├── performance_analysis.py            # Performance benchmarking
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── CLAUDE.md                          # This file - developer guide
│
├── config/                            # Configuration files
│   ├── .env.example                   # Environment variables template
│   ├── docker-compose.yml             # Docker setup
│   ├── pytest.ini                     # Test configuration
│   ├── requirements_vllm.txt          # vLLM dependencies
│   ├── runpod_config.env              # RunPod configuration
│   └── runpod_vllm_config.env         # RunPod vLLM configuration
│
├── docs/                              # Documentation
│   ├── START_HERE.md                  # Getting started guide
│   ├── RUNPOD_FINAL_SETUP.md          # RunPod deployment guide
│   ├── VLLM_SERVER_GUIDE.md           # vLLM server setup
│   ├── PERFORMANCE_QUICK_START.md     # Performance tuning
│   ├── ENVIRONMENT_VARIABLES.md       # Config reference
│   ├── GRAFANA_MCP_SETUP.md           # Grafana MCP integration
│   └── *.md                           # Other guides
│
├── scripts/                           # Utility scripts
│   ├── README.md                      # Scripts documentation
│   ├── check_dependencies.py          # Dependency checker
│   ├── start_vllm_server.sh           # Start vLLM server
│   ├── quick_start_optimized.sh       # Quick start script
│   ├── quick_start_vllm.sh            # vLLM quick start
│   ├── deploy_runpod.sh               # Deploy to RunPod
│   ├── tensorboard_embeddings.py      # Embedding visualization
│   ├── chainlit_app.py                # Alternative chat UI
│   └── [other deployment, benchmark, visualization scripts]
│
├── data/                              # Documents to index (gitignored)
├── logs/                              # Log files (gitignored)
├── benchmarks/                        # Performance results (gitignored)
├── query_logs/                        # Query history (gitignored)
├── results/                           # Analysis results (gitignored)
├── archive/                           # Old versions/experiments
├── tests/                             # Unit tests
├── utils/                             # Shared utilities
└── .claude/                           # Claude Code config
    ├── settings.local.json
    ├── agents/
    └── commands/
```

## Key Configuration (Environment Variables)

### Document & Index
```bash
PDF_PATH=data/document.pdf    # File or folder to index
PGTABLE=my_index              # PostgreSQL table name
RESET_TABLE=1                 # Drop table before indexing (0 or 1)
```

### Chunking (RAG Quality Tuning)
```bash
CHUNK_SIZE=700                # Characters per chunk (100-2000)
CHUNK_OVERLAP=150             # Overlap between chunks
```

### Embedding Model
```bash
EMBED_MODEL=BAAI/bge-small-en # Model name
EMBED_DIM=384                  # Vector dimensions
EMBED_BATCH=64                 # Batch size for embedding
```

**Multilingual Support**: For multilingual content (French, English, Spanish, etc.):
- Use `EMBED_MODEL=BAAI/bge-m3` with `EMBED_DIM=1024` for best quality (100+ languages)
- **Performance Boost on Apple Silicon**: Use `EMBED_BACKEND=mlx` for 9x faster loading and 3.7x faster embedding with bge-m3 (93 texts/sec vs 25 texts/sec with PyTorch)
- Use `EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with `EMBED_DIM=384` for faster processing

### LLM Configuration
```bash
CTX=3072                      # Context window size
MAX_NEW_TOKENS=256            # Max generation length
TEMP=0.1                      # Temperature (0.0-1.0)
N_GPU_LAYERS=24               # Layers to offload to GPU
N_BATCH=256                   # LLM batch size
```

### Retrieval
```bash
TOP_K=4                       # Number of chunks to retrieve
```

### Database
```bash
PGHOST=localhost
PGPORT=5432
PGUSER=your_database_user
PGPASSWORD=your_database_password
DB_NAME=vector_db
```

**Note**: Set these in `.env` file (copy from `config/.env.example`) or export them.

## Core Functions (rag_low_level_m1_16gb_verbose.py)

### Document Processing
- `load_documents(path)` - Load files/folders
- `clean_html_content(html)` - Strip HTML tags
- `chunk_documents(docs)` - Split into chunks
- `build_nodes(docs, chunks, doc_idxs)` - Create TextNodes with metadata

### Embedding & Storage
- `build_embed_model()` - Load HuggingFace model (auto-detects MPS)
- `embed_nodes(model, nodes)` - Compute embeddings
- `make_vector_store()` - Connect to pgvector
- `insert_nodes(store, nodes)` - Store in database

### Query & Retrieval
- `VectorDBRetriever` - Custom retriever with similarity scoring
- `build_llm()` - Load Mistral via llama.cpp
- `run_query(engine, question)` - Execute RAG query

### Database Utilities
- `ensure_db_exists()` - Create database
- `ensure_pgvector_extension()` - Enable pgvector
- `count_rows()` - Get row count
- `check_index_configuration()` - Detect config mismatches

## Code Style Guidelines

### Python Conventions
- **Docstrings**: Google style
- **Type hints**: Use for function signatures
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Line length**: 100 characters max

### Example Pattern
```python
def process_document(
    doc_path: str,
    chunk_size: int = 700,
    chunk_overlap: int = 150,
) -> List[TextNode]:
    """Process a document into chunked nodes.

    Args:
        doc_path: Path to document file or folder
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between adjacent chunks

    Returns:
        List of TextNode objects with embeddings
    """
    docs = load_documents(doc_path)
    chunks, doc_idxs = chunk_documents(docs)
    return build_nodes(docs, chunks, doc_idxs)
```

## Common Pitfalls

### Context Window Overflow
```python
# Problem: Chunks too large + TOP_K too high = token overflow
# Solution: Reduce chunk_size or TOP_K
CTX=8192 TOP_K=3 CHUNK_SIZE=500
```

### Mixed Index Detection
```python
# Problem: Indexing with different chunk sizes into same table
# Solution: Use RESET_TABLE=1 or different table name
PGTABLE=doc_cs500_ov100  # Include config in name
```

### GPU Memory (M1 16GB)
```python
# Optimal settings for M1 16GB
N_GPU_LAYERS=24   # ~75% of layers on GPU
N_BATCH=256       # Reasonable batch size
EMBED_BATCH=64    # For embeddings
```

## RAG Quality Tuning

### Chunk Size Guidelines
| Size | Use Case | Trade-off |
|------|----------|-----------|
| 100-300 | Q&A, chat logs | Precise but loses context |
| 500-800 | General documents | Balanced (recommended) |
| 1000-2000 | Long-form content | More context but less precise |

### Overlap Ratio
- **10-15%**: Minimal, fast indexing
- **15-25%**: Balanced (recommended)
- **25-30%**: Maximum context preservation

### Chat & Conversational Data

For chat logs and conversational data (Facebook Messenger, Slack, etc.):

**Optimized Settings**:
```bash
CHUNK_SIZE=300                # Fits 3-5 messages per chunk
CHUNK_OVERLAP=50              # ~15% overlap
EMBED_MODEL=BAAI/bge-m3       # Multilingual support
EMBED_DIM=1024                # Match bge-m3 dimensions
EMBED_BACKEND=mlx             # 9x faster on Apple Silicon (recommended for M1/M2/M3)
TOP_K=6                       # More chunks for context
```

**Enable Metadata Extraction** (enabled by default):
```bash
EXTRACT_CHAT_METADATA=1       # Parse FB Messenger metadata
EXTRACT_ENHANCED_METADATA=1   # Extract rich metadata
EXTRACT_ENTITIES=1            # Detect names, places
```

**Enables queries like**:
- "What did Quentin say about Léo?"
- "Messages from March 2022"
- "Conversations with Alice"

**HTML Cleaning**: Messenger HTML exports are automatically cleaned (scripts, styles, CSS classes removed).

## Testing

```bash
# Test database connection (load credentials from .env)
source .env
psql -h $PGHOST -U $PGUSER -d $DB_NAME -c "SELECT version();"

# Test embedding model
python -c "from rag_low_level_m1_16gb_verbose import build_embed_model; m = build_embed_model(); print('OK')"

# Test full pipeline (query-only)
python rag_low_level_m1_16gb_verbose.py --query-only --query "test"
```

## Troubleshooting

### Issue: "Transaction aborted" error
**Solution**: Database connection not using autocommit
```python
conn = psycopg2.connect(...)
conn.autocommit = True
```

### Issue: Embeddings as strings (not arrays)
**Solution**: Parse with json.loads in visualization code
```python
import json
emb = json.loads(embedding_string)
```

### Issue: Slow embedding on CPU
**Solution**: Verify MPS is detected
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

### Issue: Context window overflow
**Solution**: Increase CTX or reduce TOP_K/CHUNK_SIZE
```bash
CTX=8192 TOP_K=3 CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py
```

## Performance Benchmarks (M1 Mac Mini 16GB)

| Operation | Time | Throughput |
|-----------|------|------------|
| Load 1000 HTML files | ~40s | 25 files/s |
| Chunk 1000 docs | ~6s | 166 docs/s |
| Embed 10000 chunks | ~150s | 67 chunks/s |
| Insert 10000 nodes | ~8s | 1250 nodes/s |
| **Query (with HNSW)** | **~2-3ms** | **🚀 100x+ faster** |
| Query (generation) | ~5-15s | ~10 tokens/s |

### HNSW Index Performance (Real Results)

**Measured speedups** from HNSW index optimization (2026-01-10):

| Table Size | Before | After | Speedup |
|-----------|--------|-------|---------|
| 91K chunks | 443ms | **2.1ms** | **215x faster** 🚀 |
| 62K chunks | 321ms | **3.1ms** | **103x faster** 🚀 |
| 6K chunks | 69ms | **46ms** | **1.5x faster** |

**Key findings**:
- HNSW indices are **automatically created** during indexing
- Tables > 10K chunks see **100x+ speedup**
- 10% storage overhead for massive performance gain
- See `docs/HNSW_INDEX_GUIDE.md` for details

## Available Slash Commands

- `/run-rag` - Run RAG pipeline with parameters
- `/optimize-rag` - Analyze and suggest optimizations
- `/audit-index` - Check index health and consistency
- `/compare-chunks` - Compare different chunk configurations

### Performance Tracking Commands

- **View Dashboard** - `python scripts/generate_performance_dashboard.py`
- **Update Baselines** - `python scripts/update_baselines.py --dry-run`
- **Performance Report** - `python scripts/generate_performance_report.py --format markdown`
- **Run Benchmark** - `python scripts/run_comprehensive_benchmark.py --mode quick`

**Quick commands:**
```bash
# Run tests with performance tracking
ENABLE_PERFORMANCE_RECORDING=1 pytest tests/test_performance_regression.py -v

# Generate dashboard (30 days)
python scripts/generate_performance_dashboard.py

# Check for baseline updates
python scripts/update_baselines.py --dry-run
```
