# External Integrations

**Analysis Date:** 2026-01-15

## APIs & External Services

**Cloud GPU Services:**
- **RunPod** - GPU pod management API
  - SDK/Client: `runpod` Python package, `utils/runpod_manager.py`
  - Auth: `RUNPOD_API_KEY` environment variable
  - Endpoints used: Pod creation, status monitoring, SSH tunneling
  - Documentation: `docs/RUNPOD_FINAL_SETUP.md`, `docs/RUNPOD_DEPLOYMENT_GUIDE.md`

**Model Sources:**
- **Hugging Face Hub** - Model downloads and embeddings
  - Library: huggingface-hub 0.36.0 (`requirements.txt` line 58)
  - Cache: `HF_HOME` environment variable
  - Models: BAAI/bge-small-en, BAAI/bge-m3 (multilingual), Mistral 7B
  - Download: TheBloke GGUF/AWQ quantized models

**LLM API Endpoints** (Optional):
- **OpenAI-Compatible API** - vLLM server endpoint
  - Base URL: `VLLM_API_BASE` environment variable (supports HTTPS proxy)
  - Port: `VLLM_PORT=8000` (default)
  - Client: `vllm_client.py` - LlamaIndex OpenAI compatibility layer
  - Alternative to local inference for GPU acceleration

## Data Storage

**Databases:**
- **PostgreSQL 16 + pgvector** - Primary data store
  - Connection: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `DB_NAME`
  - Docker Image: pgvector/pgvector:pg16 (`config/docker-compose.yml` line 4)
  - Extension: pgvector auto-enabled
  - HNSW Indices: Automatically created for >10K chunks (100x+ performance boost)
  - Connection Pooling: asyncpg with configurable pool size

**File Storage:**
- Local filesystem - Document storage
  - Location: `data/` directory (gitignored)
  - Formats: PDF, HTML, DOCX, TXT, Markdown

**Caching:**
- diskcache 5.6.3 - Disk-based semantic caching (`requirements.txt` line 130)
  - Enables 10-100x speedup for similar queries
  - Location: Configurable via environment

## Monitoring & Observability

**Prometheus Stack** (Docker Compose):
- **Prometheus** - Time-series metrics collection (`config/docker-compose.yml` line 40)
  - Port: 9090
  - Configuration: `monitoring/prometheus.yml`
  - Retention: 30 days

- **Grafana** - Visualization and dashboards (`config/docker-compose.yml` line 82)
  - Port: 3000
  - Auth: `GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD`
  - Service Account Token: `GRAFANA_SERVICE_ACCOUNT_TOKEN`
  - Dashboards: `grafana/dashboards/`

- **Alertmanager** - Alert routing (`config/docker-compose.yml` line 66)
  - Port: 9093
  - Configuration: `monitoring/alertmanager.yml`

- **PostgreSQL Exporter** - Database metrics (`config/docker-compose.yml` line 25)
  - Port: 9187
  - Exports: Query performance, connections, index stats

- **Node Exporter** - Host system metrics (`config/docker-compose.yml` line 109)
  - Port: 9100

- **cAdvisor** - Container resource metrics (`config/docker-compose.yml` line 128)
  - Port: 8080

**Metrics & Performance:**
- prometheus-client 0.23.1 - Prometheus integration (`requirements.txt` line 199)
- Custom metrics: `utils/metrics.py`
- Performance tracking: `utils/performance_history.py`

## Infrastructure & Deployment

**Docker Support:**
- **Docker Compose** - Multi-container orchestration (`config/docker-compose.yml` - 210 lines)
- **Containers**: PostgreSQL, Prometheus, Grafana, Alertmanager, Node Exporter, cAdvisor, PostgreSQL Exporter, Backup service

**Backup Services:**
- Daily PostgreSQL backups (`config/docker-compose.yml` line 147)
- 7-day retention policy
- Location: `../backups/`

## Cloud Deployment Platforms

**RunPod GPU Cloud:**
- Full integration via RunPod API
- Config: `config/runpod_config.env`, `config/runpod_vllm_config.env`
- Scripts: `scripts/deploy_runpod.sh`, `scripts/deploy_and_init_runpod.py`
- Health monitoring: `utils/runpod_health.py`
- SSH tunneling: `utils/ssh_tunnel.py`
- Pod management: `utils/runpod_manager.py`

## Alternative UI Frameworks

**Chainlit** (Optional):
- Alternative chat UI - `scripts/chainlit_app.py`
- Configuration: `.chainlit/config.toml`
- Installation: Commented in `requirements-optional.txt` line 67

## Document Source Integrations

**Facebook Messenger** (Optional):
- JSON export processing
- Metadata extraction: `EXTRACT_CHAT_METADATA=1`
- Parser: HTML cleaning, participant extraction (`utils/metadata_extractor.py`)

## Inference Backends

**Local Inference:**
- **llama.cpp** (GGUF format) - CPU + GPU - Default
  - Model: Mistral 7B Instruct (Q4_K_M quantized, ~4GB)
  - GPU layers: 24 (M1 Mac optimized)
  - URL: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/

- **vLLM** (HuggingFace models) - GPU only - 3-4x faster
  - Models: AWQ quantized (4-bit), Mistral 7B, Llama 2
  - GPU memory: 80-90% utilization
  - Tensor parallelism: Multi-GPU support
  - Requires: NVIDIA CUDA 11.8+, Linux/WSL2

## Advanced Features

**Query Enhancement:**
- **HyDE** (Hypothetical Document Embeddings) - Optional retrieval enhancement
  - Enabled: `ENABLE_HYDE=1`
  - Hypotheses: 1-3 configurable

- **Hybrid Search** - Vector + keyword (BM25)
  - Library: rank-bm25>=0.2.2 (`requirements-optional.txt` line 47)
  - Alpha weight: 0.0-1.0 (0.5=balanced)

- **Semantic Caching** - Query result caching
  - Library: scikit-learn (cosine similarity) (`requirements.txt` line 59)
  - Storage: diskcache (disk-based) (`requirements.txt` line 130)

- **Query Reranking** - Result reordering
  - Library: sentence-transformers (`requirements.txt` line 53)
  - Utility: `utils/reranker.py`

## Environment Configuration

**Development:**
- Required env vars: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `DB_NAME`
- Secrets location: `.env` file (gitignored), `config/.env.example` template
- Mock/stub services: PostgreSQL via Docker

**Production:**
- Secrets management: Environment variables only
- Database: PostgreSQL with pgvector extension
- Deployment: Docker Compose or RunPod GPU cloud

---

*Integration audit: 2026-01-15*
*Update when adding/removing external services*
