# Development Guide

**Local RAG Pipeline - Developer Setup and Workflow**

This guide will help you set up your development environment and contribute effectively to the project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Debugging](#debugging)
- [Performance](#performance)
- [Documentation](#documentation)
- [Useful Commands](#useful-commands)

## Prerequisites

### Required Software

- **Python 3.11 or higher** ([Download](https://www.python.org/downloads/))
- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop))
- **Git** ([Download](https://git-scm.com/downloads))

### Optional but Recommended

- **VS Code** with Python extension
- **GitHub CLI** (`gh`) for PR management
- **Pre-commit** hooks for code quality

### System Requirements

- **RAM**: 16GB recommended (8GB minimum)
- **Disk**: 10GB free space for models and data
- **GPU**: Optional but recommended (Apple Metal or NVIDIA CUDA)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd llamaIndex-local-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Check dependencies
python scripts/check_dependencies.py
```

### 2. Install Dependencies

```bash
# Production dependencies (required)
pip install -r requirements.txt

# Development tools (for contributing)
pip install -r requirements-dev.txt

# Optional features (Web UI, MLX, etc.)
pip install -r requirements-optional.txt

# vLLM GPU backend (NVIDIA GPUs only)
pip install -r config/requirements_vllm.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp config/.env.example .env

# Edit .env and set required variables
nano .env

# Required variables:
# PGUSER=your_database_user
# PGPASSWORD=your_database_password
# PDF_PATH=data/your_documents
```

### 4. Start PostgreSQL

```bash
# Start PostgreSQL with pgvector
docker-compose -f config/docker-compose.yml up -d

# Verify connection (use your actual credentials)
source .env
psql -h $PGHOST -U $PGUSER -d $DB_NAME -c "SELECT version();"
```

### 5. Install Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Test hooks (optional)
pre-commit run --all-files
```

### 6. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
```

### 7. Run the Pipeline

```bash
# Index documents and run query
RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py

# Query existing index
python rag_low_level_m1_16gb_verbose.py --query-only --query "your question"

# Interactive mode
python rag_interactive.py

# Web UI
streamlit run rag_web.py
```

## Project Structure

See [REPOSITORY_ORGANIZATION.md](REPOSITORY_ORGANIZATION.md) for complete details.

**Key Directories:**
- `config/` - Configuration files and templates
- `docs/` - Documentation (guides, API docs)
- `scripts/` - Utility scripts (deployment, benchmarks)
- `tests/` - Unit and integration tests
- `utils/` - Shared utility modules

**Key Files:**
- `rag_low_level_m1_16gb_verbose.py` - Main RAG pipeline
- `rag_interactive.py` - CLI interface
- `rag_web.py` - Streamlit web UI
- `vllm_client.py` - vLLM OpenAI-compatible client
- `CLAUDE.md` - Detailed developer reference

## Development Workflow

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Create bugfix branch
git checkout -b bugfix/issue-description

# Create docs branch
git checkout -b docs/documentation-update
```

### Making Changes

1. **Make your changes** in appropriate files
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run quality checks** (see below)
5. **Commit with clear messages**

### Commit Messages

Follow conventional commit format:

```
feat: add new embedding model support
fix: correct context window calculation
docs: update environment variables guide
test: add tests for query caching
refactor: reorganize utility modules
```

### Pre-commit Hooks

Hooks run automatically before each commit:

```bash
# Manually run all hooks
pre-commit run --all-files

# Skip hooks (when necessary)
git commit --no-verify -m "message"

# Update hook versions
pre-commit autoupdate
```

**What hooks check:**
- Code formatting (Black)
- Linting (Ruff)
- Type checking (MyPy)
- Security issues (Bandit)
- YAML/JSON syntax
- Large files
- Trailing whitespace

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Run tests matching pattern
pytest -k "test_embedding"

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=. --cov-report=term --cov-report=html

# Run fast tests only
pytest -m "not slow"

# Run GPU tests only
pytest -m gpu
```

### Writing Tests

```python
# tests/test_my_feature.py
import pytest

def test_basic_functionality():
    """Test basic functionality."""
    result = your_function(input_data)
    assert result == expected_output

@pytest.mark.slow
def test_expensive_operation():
    """Mark slow tests for optional skipping."""
    pass

@pytest.mark.gpu
def test_gpu_feature():
    """Mark GPU tests to run conditionally."""
    pytest.importorskip("torch")
    pass
```

### Coverage Requirements

- **Minimum**: 70% overall coverage
- **Target**: 80%+ for new code
- **Critical paths**: 90%+ (RAG pipeline core)

## Code Quality

### Formatting

```bash
# Format all files
black .

# Check formatting
black --check .
```

### Linting

```bash
# Lint all files
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking

```bash
# Type check all files
mypy .

# Check specific files
mypy rag_low_level_m1_16gb_verbose.py
```

### Security

```bash
# Security scan
bandit -r .

# Dependency audit
pip-audit
```

## Debugging

### Dependency Checker

```bash
python scripts/check_dependencies.py
```

### Logging

```bash
export LOG_LEVEL=DEBUG
export LOG_FULL_CHUNKS=1
export LOG_QUERIES=1
```

### Common Issues

**PostgreSQL Connection:**
```bash
docker-compose -f config/docker-compose.yml up -d
```

**GPU Detection:**
```python
import torch
print(torch.cuda.is_available())      # NVIDIA
print(torch.backends.mps.is_available())  # Apple
```

## Performance

### Benchmarking

```bash
python scripts/benchmarking_performance_analysis.py
python scripts/system_analyze_resources.py
```

### Platform-Specific Optimization

**M1 Mac:**
```bash
EMBED_BACKEND=mlx N_GPU_LAYERS=24 CTX=8192
```

**RTX 4090:**
```bash
USE_VLLM=1 EMBED_BACKEND=torch N_GPU_LAYERS=99
```

## Documentation

### When to Update

- Adding new features
- Changing configuration
- Fixing user-facing bugs
- Improving setup

### Where to Update

- Core changes: `CLAUDE.md`, `README.md`
- New scripts: `scripts/README.md`
- Config options: `config/.env.example`
- Architecture: `REPOSITORY_ORGANIZATION.md`

## Useful Commands

```bash
# Setup
python scripts/check_dependencies.py
pre-commit install

# Quality
black .
ruff check --fix .
mypy .
pytest --cov=.

# Database
docker-compose -f config/docker-compose.yml up -d

# Pipeline
python rag_low_level_m1_16gb_verbose.py
python rag_interactive.py
streamlit run rag_web.py

# Deployment
./scripts/deploy_runpod.sh
./scripts/start_vllm_server.sh
```

## Getting Help

- Documentation: `docs/START_HERE.md`
- Developer Guide: `CLAUDE.md`
- Contributing: `CONTRIBUTING.md`
- Organization: `REPOSITORY_ORGANIZATION.md`

Happy coding! 🚀
