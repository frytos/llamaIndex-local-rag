# LlamaIndex Local RAG

A production-ready Retrieval-Augmented Generation (RAG) system that runs entirely locally on your machine. Optimized for Apple Silicon (M1/M2/M3) Macs with 16GB RAM, but works on any compatible hardware.

## Features

- **100% Local & Private**: No external API calls - all processing happens on your machine
- **Vector Database**: PostgreSQL with pgvector extension for scalable vector storage
- **Local LLM**: llama.cpp for efficient local inference (Mistral 7B Instruct)
- **Multiple Document Formats**: PDF, DOCX, TXT, and Markdown support
- **Extensive Logging**: Learn how RAG works with detailed pipeline logging
- **Docker Integration**: Easy PostgreSQL + pgvector setup
- **Optimized for Apple Silicon**: Metal GPU acceleration for faster inference

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Docker & Docker Compose
- 16GB RAM recommended (8GB minimum)
- ~5GB disk space for models and database

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd llamaIndex-local-rag
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start PostgreSQL with pgvector:
```bash
docker-compose up -d
```

5. Run the RAG pipeline:
```bash
RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
```

## Usage

### Basic Usage

The script processes a PDF document, creates embeddings, stores them in PostgreSQL, and answers questions:

```bash
# First run - index the document and answer question
RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py

# Subsequent runs - query existing index
python rag_low_level_m1_16gb_verbose.py
```

### Environment Variables

Configure the pipeline using environment variables:

#### Database Configuration
```bash
DB_NAME=vector_db           # PostgreSQL database name
PGHOST=localhost            # Database host
PGPORT=5432                 # Database port
PGUSER=fryt                 # Database user
PGPASSWORD=frytos           # Database password
PGTABLE=llama2_paper        # Table name for vectors
```

#### Document & Behavior
```bash
PDF_PATH=data/llama2.pdf    # Path to PDF file
RESET_TABLE=1               # Drop table before ingestion (1) or append (0)
RESET_DB=0                  # Drop entire database (1) - USE WITH CAUTION
```

#### RAG Quality Tuning
```bash
CHUNK_SIZE=900              # Characters per chunk (smaller = more precise, larger = more context)
CHUNK_OVERLAP=120           # Character overlap between chunks (helps preserve context)
TOP_K=4                     # Number of similar chunks to retrieve
```

#### Embedding Configuration
```bash
EMBED_MODEL=BAAI/bge-small-en   # HuggingFace embedding model
EMBED_DIM=384                    # Embedding dimension (must match model)
EMBED_BATCH=16                   # Batch size for embedding computation
```

#### LLM Configuration
```bash
MODEL_PATH=                 # Local GGUF model path (if already downloaded)
MODEL_URL=https://...       # URL to download GGUF model (default: Mistral 7B Q4_K_M)
TEMP=0.1                    # Temperature (0.0 = deterministic, 1.0 = creative)
MAX_NEW_TOKENS=256          # Maximum tokens in answer
CTX=3072                    # Context window size
N_GPU_LAYERS=16             # Layers offloaded to GPU (higher = faster, more VRAM)
N_BATCH=128                 # Batch size for LLM processing
```

#### Query
```bash
QUESTION="What are the key findings?"  # Question to ask
```

#### Logging
```bash
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
```

### Example Configurations

**Fast but less accurate** (good for testing):
```bash
CHUNK_SIZE=1200 CHUNK_OVERLAP=100 TOP_K=2 TEMP=0.3 python rag_low_level_m1_16gb_verbose.py
```

**Balanced** (default settings):
```bash
CHUNK_SIZE=900 CHUNK_OVERLAP=120 TOP_K=4 TEMP=0.1 python rag_low_level_m1_16gb_verbose.py
```

**High quality** (slower but more accurate):
```bash
CHUNK_SIZE=600 CHUNK_OVERLAP=150 TOP_K=6 TEMP=0.05 python rag_low_level_m1_16gb_verbose.py
```

**Using your own PDF**:
```bash
PDF_PATH=my_document.pdf PGTABLE=my_document RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
```

**Custom question**:
```bash
QUESTION="What is the main conclusion?" python rag_low_level_m1_16gb_verbose.py
```

## Architecture

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  PyMuPDFReader  │  Load pages
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ SentenceSplitter │  Chunk text (900 chars, 120 overlap)
└────────┬─────────┘
         │
         ▼
┌────────────────────┐
│ HuggingFace Embed  │  BAAI/bge-small-en → 384-dim vectors
└─────────┬──────────┘
          │
          ▼
┌──────────────────────┐
│ PostgreSQL + pgvector│  Store embeddings & metadata
└──────────┬───────────┘
           │
    ┌──────▼──────┐
    │   Query     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Retriever  │  Vector similarity search (top-4)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ LlamaCPP LLM│  Mistral 7B generates answer
    └──────┬──────┘
           │
           ▼
       ┌────────┐
       │ Answer │
       └────────┘
```

## Hardware Requirements

### Minimum
- CPU: Any modern x86_64 or ARM64 processor
- RAM: 8GB (may swap during LLM inference)
- Storage: 5GB free space
- GPU: Optional (CPU-only mode works)

### Recommended (M1/M2/M3 Mac)
- CPU: Apple Silicon (M1 or newer)
- RAM: 16GB
- Storage: 10GB free space
- GPU: Metal acceleration (automatic on Apple Silicon)

### Performance Expectations

On M1 Mac with 16GB RAM:
- **Embedding**: ~50 chunks/second
- **LLM Inference**: ~15-20 tokens/second
- **Full Pipeline** (13MB PDF, 68 pages): ~2-3 minutes

On Linux/Windows with 16GB RAM and RTX 3060:
- **Embedding**: ~100 chunks/second
- **LLM Inference**: ~25-35 tokens/second
- **Full Pipeline**: ~1-2 minutes

## Project Structure

```
llamaIndex-local-rag/
├── rag_low_level_m1_16gb_verbose.py  # Main RAG pipeline
├── rag_minimal_local.py              # Lightweight file-based RAG (no DB)
├── docker-compose.yml                # PostgreSQL + pgvector setup
├── db-init/
│   └── 001-pgvector.sql             # Database initialization
├── data/
│   └── llama2.pdf                   # Sample document
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusions
└── README.md                         # This file
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'llama_index'"
Install dependencies:
```bash
pip install -r requirements.txt
```

### "psycopg2.OperationalError: could not connect to server"
Start PostgreSQL:
```bash
docker-compose up -d
# Wait 10 seconds for startup
docker-compose ps  # Should show "Up" status
```

### "FileNotFoundError: Missing data/llama2.pdf"
Download a sample PDF or provide your own:
```bash
PDF_PATH=/path/to/your/document.pdf python rag_low_level_m1_16gb_verbose.py
```

### LLM download fails or is too slow
Download manually and point to local path:
```bash
# Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
MODEL_PATH=/path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf python rag_low_level_m1_16gb_verbose.py
```

### Out of memory errors
Reduce memory usage:
```bash
N_GPU_LAYERS=8 N_BATCH=64 EMBED_BATCH=8 python rag_low_level_m1_16gb_verbose.py
```

### Slow performance on Apple Silicon
Increase GPU layers (if you have enough RAM):
```bash
N_GPU_LAYERS=32 python rag_low_level_m1_16gb_verbose.py
```

## RAG Quality Tuning Guide

### Chunk Size (`CHUNK_SIZE`)
- **Smaller (400-700)**: More precise retrieval, but may miss context
- **Medium (800-1000)**: Balanced precision and context
- **Larger (1200-1500)**: More context, but less precise matching

### Chunk Overlap (`CHUNK_OVERLAP`)
- **Low (50-100)**: Faster processing, may break sentences
- **Medium (120-200)**: Good balance, preserves most context
- **High (250-400)**: Maximum context preservation, slower

### Top-K (`TOP_K`)
- **Low (2-3)**: Fast, focused answers, may miss relevant info
- **Medium (4-6)**: Balanced coverage
- **High (8-10)**: Comprehensive but may include noise

### Temperature (`TEMP`)
- **0.0-0.1**: Deterministic, factual (best for RAG)
- **0.3-0.5**: Slightly more varied
- **0.7-1.0**: Creative but may hallucinate

## Advanced Usage

### Using Different Models

**Smaller/Faster Models** (for 8GB RAM):
```bash
MODEL_URL=https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Larger/Better Models** (for 32GB+ RAM):
```bash
MODEL_URL=https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
N_GPU_LAYERS=0  # Use CPU for larger models
```

### Multiple Documents

Index multiple documents in separate tables:
```bash
# Index first document
PDF_PATH=doc1.pdf PGTABLE=doc1 RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py

# Index second document
PDF_PATH=doc2.pdf PGTABLE=doc2 RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py

# Query specific document
PGTABLE=doc1 QUESTION="..." python rag_low_level_m1_16gb_verbose.py
```

## Development

### Running Tests
```bash
# TODO: Add tests
pytest tests/
```

### Code Style
```bash
black rag_low_level_m1_16gb_verbose.py
ruff check rag_low_level_m1_16gb_verbose.py
```

## Contributing

Contributions welcome! Areas for improvement:
- Interactive query mode (REPL)
- Web UI (Gradio/Streamlit)
- Support for more document formats
- Incremental indexing (avoid re-indexing unchanged docs)
- Automated RAG quality evaluation
- Multi-document querying
- Document management (list/delete indexed docs)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [LlamaIndex](https://www.llamaindex.ai/)
- Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for local inference
- Embeddings from [BAAI BGE models](https://huggingface.co/BAAI)
- Vector storage with [pgvector](https://github.com/pgvector/pgvector)

## Support

For issues, questions, or contributions, please open an issue on GitHub.
