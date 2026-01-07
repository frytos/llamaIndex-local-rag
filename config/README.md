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

- **`../requirements.txt`** (in root) - Main dependencies
  ```bash
  pip install -r requirements.txt
  ```

- **`requirements_vllm.txt`** - vLLM GPU backend (optional)
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

Copy `.env.example` to `../.env` (project root) and configure:

### Required Variables

```bash
# Database
PGHOST=localhost
PGPORT=5432
PGUSER=your_db_user
PGPASSWORD=your_db_password
DB_NAME=vector_db

# Documents
PDF_PATH=data/your_documents
PGTABLE=your_index_name
```

### Optional Variables

```bash
# Chunking
CHUNK_SIZE=700
CHUNK_OVERLAP=150

# Retrieval
TOP_K=4

# LLM Backend
LLM_BACKEND=llamacpp  # or vllm
N_GPU_LAYERS=24       # for llamacpp GPU offload
```

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
