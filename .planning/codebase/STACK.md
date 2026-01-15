# Technology Stack

**Analysis Date:** 2026-01-15

## Languages

**Primary:**
- Python 3.11+ - All application code (`pyproject.toml`, `requirements.txt`)

**Secondary:**
- Shell scripts - Deployment automation (`scripts/*.sh`)
- YAML - Docker and configuration (`config/docker-compose.yml`)

## Runtime

**Environment:**
- Python 3.11.9 (verified in environment)
- Package Manager: pip (Poetry-compatible via pyproject.toml)
- Virtual Environment: Python venv (`.venv/`)

**Package Manager:**
- pip with pinned versions
- Lockfile: `requirements.txt`, `requirements-optional.txt`, `requirements-dev.txt`

## Frameworks

**Core:**
- LlamaIndex 0.14.10 - RAG orchestration framework (`requirements.txt` lines 26-27)

**Testing:**
- pytest 8.0+ - Test framework (`requirements-dev.txt` line 17, `pyproject.toml` line 44)
- pytest-cov 5.0+ - Coverage reporting (`requirements-dev.txt` line 18)
- pytest-asyncio 0.23+ - Async test support (`requirements-dev.txt` line 19)

**Build/Dev:**
- Black 24.0+ - Code formatter (`requirements-dev.txt` line 24, `pyproject.toml` line 41)
- Ruff 0.6+ - Linter (`requirements-dev.txt` line 25, `pyproject.toml` line 42)
- MyPy 1.11+ - Type checker (`requirements-dev.txt` line 26, `pyproject.toml` line 43)

## Key Dependencies

**Critical:**
- llama-cpp-python 0.3.16 - Local LLM inference (`requirements.txt` line 42)
- sentence-transformers 5.2.0 - Embeddings (`requirements.txt` line 53)
- psycopg2-binary 2.9.11 - PostgreSQL adapter (`requirements.txt` line 66)
- pgvector 0.4.2 - Vector similarity (`requirements.txt` line 67)

**Infrastructure:**
- PostgreSQL 16 with pgvector extension - Vector storage (`config/docker-compose.yml` line 4)
- Streamlit 1.28+ - Web UI framework (`requirements.txt` line 33)
- Plotly 5.18+ - Interactive charts (`requirements.txt` line 34)
- PyMuPDF 1.26.7 - PDF parsing (`requirements.txt` line 74)

**ML/AI:**
- PyTorch 2.9.1 - Deep learning backend (`requirements.txt` line 54)
- HuggingFace Transformers 4.57.3 - Model hub (`requirements.txt` line 55)
- scikit-learn 1.8.0 - ML utilities (reranking, semantic cache) (`requirements.txt` line 59)
- MLX 0.20.0 (optional) - Apple Silicon acceleration (9x speedup) (`requirements-optional.txt` line 35)

**Performance:**
- vLLM 0.13+ (optional) - GPU acceleration (15x faster) (`config/requirements_vllm.txt` line 24)
- diskcache 5.6.3 - Disk-based caching (`requirements.txt` line 130)

## Configuration

**Environment:**
- .env file loading via python-dotenv 1.2.1 (`requirements.txt` line 113)
- Auto-loaded from project root or `config/` directory (`rag_low_level_m1_16gb_verbose.py` lines 36-43)
- Template: `config/.env.example` (575 lines of configuration)

**Build:**
- `pyproject.toml` - Build, packaging, tool config (258 lines)
- `Makefile` - Development commands (82 lines)
- `.pre-commit-config.yaml` - Git hook automation

## Platform Requirements

**Development:**
- macOS/Linux/Windows (cross-platform)
- Apple Silicon optimization: Metal GPU (MPS) acceleration
- NVIDIA GPU support: CUDA for vLLM backend

**Production:**
- Docker containers - PostgreSQL, Prometheus, Grafana (`config/docker-compose.yml`)
- RunPod GPU cloud - Deployment templates and automation
- 16GB RAM recommended (8GB minimum)
- ~5GB disk space for models and database

---

*Stack analysis: 2026-01-15*
*Update after major dependency changes*
