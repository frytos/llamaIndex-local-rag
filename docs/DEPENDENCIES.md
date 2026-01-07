# Dependency Management Guide

**Last Updated**: January 2026

This document explains the dependency structure and installation options for the Local RAG Pipeline project.

## Overview

The project uses a modular dependency structure with three main requirement files:

1. **requirements.txt** - Production dependencies (required)
2. **requirements-dev.txt** - Development tools (optional)
3. **requirements-optional.txt** - Optional features (optional)
4. **config/requirements_vllm.txt** - vLLM GPU backend (optional)

## Requirements Files

### 1. requirements.txt (Production - Required)

Core dependencies needed to run the RAG pipeline in production. All versions are pinned with `==` for reproducibility.

**Contains:**
- LlamaIndex framework and extensions
- llama.cpp for local LLM inference
- Sentence transformers and embeddings
- PyTorch (CPU/Metal support)
- PostgreSQL + pgvector
- Document processing (PDF, DOCX, HTML)
- Core utilities

**Installation:**
```bash
pip install -r requirements.txt
```

**Size:** ~107 packages, ~3-4GB installed

### 2. requirements-dev.txt (Development - Optional)

Development tools for testing, linting, formatting, and quality assurance. Uses `>=` for flexibility.

**Contains:**
- Testing: pytest, pytest-cov, pytest-asyncio
- Linting: black, ruff, isort, pylint, flake8
- Type checking: mypy with type stubs
- Security: pip-audit, bandit, safety
- Documentation: sphinx
- Profiling: memory-profiler, line-profiler, py-spy
- Development utilities: ipython, ipdb, pre-commit

**Installation:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

**Use cases:**
- Running tests
- Code formatting and linting
- Security audits
- Performance profiling
- Contributing to the project

### 3. requirements-optional.txt (Optional Features)

Optional features and alternative backends for enhanced performance and functionality.

**Contains:**

#### Web UI Components
- `streamlit>=1.28.0` - Web interface
- `plotly>=5.18.0` - Interactive visualizations
- `umap-learn>=0.5.4` - Dimensionality reduction

#### Apple Silicon Optimization (M1/M2/M3)
- `mlx>=0.20.0` - Apple MLX framework
- `mlx-embedding-models>=0.0.11` - MLX embeddings
- **Performance:** 5-20x faster embedding generation

#### GPU Acceleration (NVIDIA)
- `vllm>=0.13.0` - GPU-accelerated LLM inference
- **Performance:** 10-15x faster than llama.cpp CPU
- **Requirements:** CUDA 11.8+, 8GB+ VRAM

#### Advanced Retrieval
- `rank-bm25>=0.2.2` - Hybrid search (BM25 + semantic)

#### Alternative UIs
- `chainlit>=1.0.0` - ChatGPT-like interface
- `nomic>=2.0.0` - Embedding visualization

**Installation:**
```bash
# Full installation
pip install -r requirements.txt -r requirements-optional.txt

# Individual features
pip install streamlit plotly umap-learn  # Web UI only
pip install mlx mlx-embedding-models     # Apple Silicon optimization
pip install vllm                         # GPU acceleration
pip install rank-bm25                    # Hybrid search
```

### 4. config/requirements_vllm.txt (vLLM Backend - Optional)

Specialized requirements file for vLLM GPU backend with detailed usage instructions.

**Installation:**
```bash
pip install -r config/requirements_vllm.txt
```

**Requirements:**
- CUDA 11.8 or higher
- NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- Linux or WSL2 (Windows native not supported)

**Performance Comparison (RTX 4090):**
- vLLM: ~150 tokens/sec
- llama.cpp CPU: ~10 tokens/sec
- Speedup: 15x

## Installation Scenarios

### Minimal Setup (Production Only)
```bash
# Basic RAG pipeline
pip install -r requirements.txt
```

### Developer Setup
```bash
# Production + development tools
pip install -r requirements.txt -r requirements-dev.txt
```

### Full Featured Setup
```bash
# Everything (production + dev + optional features)
pip install -r requirements.txt -r requirements-dev.txt -r requirements-optional.txt
```

### Use Case Specific

#### Web UI User
```bash
pip install -r requirements.txt
pip install streamlit plotly umap-learn
```

#### Apple Silicon Mac User
```bash
pip install -r requirements.txt
pip install mlx mlx-embedding-models rank-bm25
```

#### GPU Accelerated (NVIDIA)
```bash
pip install -r requirements.txt
pip install -r config/requirements_vllm.txt
```

#### Contributor/Developer
```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install streamlit plotly mlx mlx-embedding-models  # Optional features
```

## Version Management

### Production (requirements.txt)
- **Pinning Strategy:** Exact versions with `==`
- **Reason:** Reproducibility, stability, security
- **Update Policy:** Manual updates after testing

### Development (requirements-dev.txt)
- **Pinning Strategy:** Minimum versions with `>=`
- **Reason:** Flexibility, latest bug fixes
- **Update Policy:** Can use latest versions

### Optional (requirements-optional.txt)
- **Pinning Strategy:** Minimum versions with `>=`
- **Reason:** Latest features, compatibility
- **Update Policy:** User's choice

## Dependency Groups

### Core LlamaIndex Framework
```
llama-index==0.14.10
llama-index-core==0.14.10
llama-index-embeddings-huggingface==0.6.1
llama-index-llms-llama-cpp==0.5.1
llama-index-vector-stores-postgres==0.7.2
```

### Machine Learning Stack
```
torch==2.9.1                  # PyTorch
sentence-transformers==5.2.0  # Embeddings
transformers==4.57.3          # HuggingFace
scikit-learn==1.8.0           # ML utilities
scipy==1.16.3                 # Scientific computing
numpy==2.3.5                  # Numerical operations
```

### Database Stack
```
psycopg2-binary==2.9.11       # PostgreSQL adapter
pgvector==0.4.2               # Vector similarity
SQLAlchemy==2.0.45            # ORM
asyncpg==0.31.0               # Async PostgreSQL
```

### Document Processing
```
PyMuPDF==1.26.7               # PDF processing
pypdf==6.4.2                  # Alternative PDF
python-docx==1.2.0            # Word documents
beautifulsoup4==4.14.3        # HTML parsing
lxml==6.0.2                   # XML/HTML parser
```

### LLM Backends
```
llama_cpp_python==0.3.16      # CPU/Metal inference
vllm>=0.13.0                  # GPU inference (optional)
mlx>=0.20.0                   # Apple Silicon (optional)
```

## Updating Dependencies

### Update Production Dependencies
```bash
# Check for updates
pip list --outdated

# Update specific package
pip install --upgrade package-name==new-version

# Regenerate requirements.txt
pip freeze > requirements-frozen.txt
# Review changes and update requirements.txt manually
```

### Security Updates
```bash
# Scan for vulnerabilities
pip-audit

# Update vulnerable packages
pip install --upgrade vulnerable-package
```

### Testing After Updates
```bash
# Run test suite
pytest

# Check for breaking changes
python rag_low_level_m1_16gb_verbose.py --query-only

# Verify web UI
streamlit run rag_web.py
```

## Common Issues

### "ModuleNotFoundError"
```bash
# Install missing dependencies
pip install -r requirements.txt

# For web UI features
pip install -r requirements-optional.txt
```

### "No module named 'mlx'"
```bash
# Apple Silicon optimization (optional)
pip install mlx mlx-embedding-models
```

### "No module named 'vllm'"
```bash
# GPU acceleration (optional, NVIDIA only)
pip install -r config/requirements_vllm.txt
```

### "pytest not found"
```bash
# Development dependencies
pip install -r requirements-dev.txt
```

### Version Conflicts
```bash
# Create fresh environment
python3 -m venv .venv-fresh
source .venv-fresh/bin/activate
pip install -r requirements.txt

# Check for conflicts
pip check
```

## Best Practices

### 1. Use Virtual Environments
```bash
# Always use virtual environments
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Pin Production Versions
- Production uses exact versions (`==`)
- Ensures reproducible deployments
- Prevents unexpected breakage

### 3. Document Custom Requirements
If you add custom dependencies:
```bash
# Add to appropriate file with comment
echo "my-package==1.0.0  # Custom feature X" >> requirements-optional.txt
```

### 4. Regular Security Audits
```bash
# Weekly security checks
pip-audit

# Update vulnerable packages promptly
```

### 5. Test Before Committing
```bash
# Run tests after dependency changes
pytest

# Verify functionality
python rag_low_level_m1_16gb_verbose.py --query-only
```

## Troubleshooting

### Slow Installation
```bash
# Use pip cache
pip install --cache-dir ~/.cache/pip -r requirements.txt

# Use binary wheels
pip install --only-binary :all: -r requirements.txt
```

### Build Failures
```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt-get install build-essential python3-dev

# Install build dependencies (macOS)
xcode-select --install
```

### Out of Memory During Installation
```bash
# Install in smaller batches
pip install llama-index torch
pip install -r requirements.txt
```

## Additional Resources

- **Security Audit:** See `SECURITY_AUDIT.md`
- **Contributing Guide:** See `CONTRIBUTING.md`
- **vLLM Setup:** See `config/requirements_vllm.txt`
- **Performance Optimization:** See `docs/PERFORMANCE_QUICK_START.md`
- **Development Setup:** See `CONTRIBUTING.md`

## Support

For dependency-related issues:
1. Check this document
2. Search existing GitHub issues
3. Open a new issue with:
   - Python version (`python --version`)
   - OS and version
   - Output of `pip list`
   - Error message
