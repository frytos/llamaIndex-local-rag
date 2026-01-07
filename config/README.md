# Configuration Directory

Configuration files for the Local RAG Pipeline.

## Files

### Environment Configuration

- **`.env.example`** - Template for environment variables
  ```bash
  cp .env.example ../.env
  # Edit ../.env with your actual values
  ```

### Docker Configuration

- **`docker-compose.yml`** - PostgreSQL + pgvector setup
  ```bash
  docker-compose -f config/docker-compose.yml up -d
  ```

### Python Dependencies

- **`../requirements.txt`** (in root) - Production dependencies (pinned versions)
  ```bash
  pip install -r requirements.txt
  ```

- **`../requirements-dev.txt`** - Development tools (testing, linting, security)
  ```bash
  pip install -r requirements-dev.txt
  ```

- **`../requirements-optional.txt`** - Optional features (Web UI, MLX, hybrid search)
  ```bash
  pip install -r requirements-optional.txt
  ```

- **`requirements_vllm.txt`** - vLLM GPU backend (NVIDIA GPUs, 10-15x faster)
  ```bash
  pip install -r config/requirements_vllm.txt
  ```

### Testing Configuration

- **`pytest.ini`** - Pytest configuration
  ```bash
  pytest -c config/pytest.ini
  ```

### RunPod Configuration

- **`runpod_config.env`** - RunPod environment variables
- **`runpod_vllm_config.env`** - RunPod with vLLM configuration

## Environment Variables

The `.env.example` file is a **comprehensive template** documenting all 50+ environment variables used in the RAG pipeline. It includes:

### 12 Configuration Sections

1. **Database Configuration** - PostgreSQL connection (REQUIRED)
2. **Document Processing** - Input paths, table names, reset options
3. **Chunking Configuration** - RAG quality tuning (CHUNK_SIZE, CHUNK_OVERLAP)
4. **Embedding Configuration** - Models, backends (MLX/HuggingFace/PyTorch), batch sizes
5. **LLM Backend Selection** - Choose llama.cpp (GGUF) or vLLM (GPU)
6. **LLM Generation Settings** - Temperature, context window, max tokens
7. **Retrieval Configuration** - TOP_K, hybrid search, MMR diversity
8. **Performance Tuning** - Batch sizes, GPU layers, memory optimization
9. **Logging Configuration** - Log levels, verbosity, query logging
10. **Optional Features** - Default questions, metadata extraction
11. **External Dependencies** - HuggingFace cache, model paths
12. **Development Settings** - Debug flags, resource monitoring

### Quick Start

```bash
# 1. Copy template to project root
cp config/.env.example .env

# 2. Edit REQUIRED variables
nano .env
# Set: PGUSER, PGPASSWORD, PDF_PATH

# 3. Load environment
source .env
# OR
export $(cat .env | xargs)

# 4. Run pipeline
python rag_low_level_m1_16gb_verbose.py
```

### Key Variables by Use Case

**Minimal Setup (CPU-only):**
```bash
PGUSER=myuser
PGPASSWORD=mypass
PDF_PATH=data/document.pdf
```

**Optimized for M1 Mac (16GB):**
```bash
EMBED_BACKEND=mlx
N_GPU_LAYERS=24
N_BATCH=256
CTX=8192
```

**Optimized for NVIDIA GPU (RTX 4090):**
```bash
USE_VLLM=1
VLLM_MODEL=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
EMBED_BACKEND=torch
EMBED_BATCH=256
N_GPU_LAYERS=99
```

**Debugging:**
```bash
LOG_LEVEL=DEBUG
LOG_FULL_CHUNKS=1
LOG_QUERIES=1
```

### Variable Documentation

Every variable in `.env.example` includes:
- **Description** - What it does
- **Valid values/ranges** - Acceptable inputs
- **Default values** - What's used if not set
- **Required vs optional** - Necessity level
- **Performance implications** - Impact on speed/memory
- **Platform-specific guidelines** - M1 vs NVIDIA GPU recommendations

See `.env.example` for complete documentation of all 50+ variables.

## Usage Examples

### Local Development

```bash
# 1. Copy environment template
cp config/.env.example .env

# 2. Edit .env with your settings
nano .env

# 3. Start PostgreSQL
docker-compose -f config/docker-compose.yml up -d

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run RAG pipeline
python rag_low_level_m1_16gb_verbose.py
```

### GPU Server (vLLM)

```bash
# 1. Install vLLM dependencies
pip install -r config/requirements_vllm.txt

# 2. Use vLLM configuration
source config/runpod_vllm_config.env

# 3. Start vLLM server
./scripts/start_vllm_server.sh

# 4. Run with vLLM backend
export LLM_BACKEND=vllm
python rag_low_level_m1_16gb_verbose.py
```

## Configuration Best Practices

1. **Never commit `.env` files** - Use `.env.example` as template
2. **Use separate tables** for different configurations
3. **Document custom settings** in project-specific files
4. **Test configuration** before production deployment
