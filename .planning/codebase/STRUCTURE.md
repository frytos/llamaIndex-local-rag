# Codebase Structure

**Analysis Date:** 2026-01-15

## Directory Layout

```
llamaIndex-local-rag/
├── rag_low_level_m1_16gb_verbose.py  # Main pipeline (3277 lines)
├── rag_interactive.py                 # CLI menu interface
├── rag_web.py                         # Streamlit web UI (2085 lines)
├── rag_web_enhanced.py                # Enhanced web UI (3319 lines)
├── rag_web_backend.py                 # Backend API server
├── vllm_wrapper.py                    # GPU LLM integration
├── vllm_client.py                     # OpenAI-compatible client
├── core/                              # Configuration module
├── config/                            # Static configuration
├── utils/                             # Shared utilities (22 modules)
├── tests/                             # Test suite (310+ tests, 30.94% coverage)
├── scripts/                           # Utility scripts (80+)
├── examples/                          # Integration demonstrations
├── docs/                              # Documentation (40+ guides)
├── data/                              # User documents (gitignored)
├── logs/                              # Execution logs (gitignored)
├── benchmarks/                        # Performance results (gitignored)
├── query_logs/                        # Query history (gitignored)
└── archive/                           # Legacy/experimentation
```

## Directory Purposes

**Core Entry Points:**
- Purpose: Main application entry points
- Contains: `rag_low_level_m1_16gb_verbose.py` (main), `rag_interactive.py` (CLI), `rag_web.py` (Web UI)
- Key files: Main pipeline orchestrator, interactive menu, Streamlit UI
- Subdirectories: None (flat structure at root)

**core/**
- Purpose: Configuration module
- Contains: `config.py` - Settings dataclass with validation
- Key files: `core/config.py` - Centralized configuration with validation
- Subdirectories: None

**config/**
- Purpose: Static configuration files
- Contains: `.env.example`, `docker-compose.yml`, `requirements_*.txt`, `constants.py`
- Key files: `constants.py` (frozen dataclasses), `.env.example` (575 lines)
- Subdirectories: None

**utils/**
- Purpose: Shared utility services
- Contains: 22 specialized modules (naming, reranking, caching, metadata, platform detection)
- Key files: `naming.py`, `reranker.py`, `query_expansion.py`, `query_cache.py`, `metadata_extractor.py`, `mlx_embedding.py`
- Subdirectories: None (flat structure)

**tests/**
- Purpose: Comprehensive test suite
- Contains: 43+ test modules, `conftest.py` (fixtures), test data
- Key files: `conftest.py` (707 lines), `test_core_config.py`, `test_database_integration.py`
- Subdirectories: `test_data/` (fixtures)

**scripts/**
- Purpose: Utility scripts (80+)
- Contains: Deployment, benchmarking, data cleaning, visualization
- Key files: `deploy_runpod.sh`, `benchmark_embeddings.py`, `audit_index.py`, `chainlit_app.py`
- Subdirectories: None (flat structure)

**examples/**
- Purpose: Integration demonstrations
- Contains: Demo files for semantic cache, HyDE, metadata extraction, conversation memory
- Key files: `semantic_cache_demo.py`, `hyde_example.py`, `metadata_extraction_demo.py`
- Subdirectories: None

**docs/**
- Purpose: Documentation (40+ guides)
- Contains: Architecture diagrams, project explanation, advanced retrieval, performance guides
- Key files: `PROJECT_EXPLANATION.md`, `ARCHITECTURE_DIAGRAMS.md`, `ADVANCED_RETRIEVAL.md`, `PERFORMANCE_QUICK_START.md`
- Subdirectories: None (flat structure)

## Key File Locations

**Entry Points:**
- `rag_low_level_m1_16gb_verbose.py` - Main RAG pipeline (CLI entry)
- `rag_interactive.py` - Interactive menu interface
- `rag_web.py` - Streamlit web UI
- `rag_web_backend.py` - Backend API server

**Configuration:**
- `core/config.py` - Settings dataclass with validation
- `config/constants.py` - Frozen constant defaults
- `config/.env.example` - Environment variables template (575 lines)
- `.env` - Local environment variables (gitignored)

**Core Logic:**
- `rag_low_level_m1_16gb_verbose.py` - Complete RAG pipeline
- `vllm_wrapper.py` - GPU LLM integration
- `vllm_client.py` - OpenAI-compatible client

**Testing:**
- `tests/conftest.py` - Shared fixtures (707 lines)
- `tests/test_*.py` - 43+ test modules
- `config/pytest.ini` - Pytest configuration

**Documentation:**
- `README.md` - Main project README (734 lines)
- `CLAUDE.md` - Developer guide (this file)
- `docs/*.md` - 40+ documentation guides

## Naming Conventions

**Files:**
- snake_case for modules: `rag_low_level_m1_16gb_verbose.py`, `rag_interactive.py`, `rag_web.py`
- Descriptive names including context: `vllm_wrapper.py`, `vllm_client.py`, `metadata_extractor.py`
- Test files: `test_*.py` prefix (e.g., `test_core_config.py`, `test_chunking.py`)

**Directories:**
- Lowercase: `core/`, `config/`, `utils/`, `tests/`, `scripts/`, `docs/`, `examples/`
- Purpose-driven: Clear intent (e.g., `scripts/`, `benchmarks/`, `metrics/`)

**Special Patterns:**
- `__init__.py` for package exports
- `.py` extension for Python modules
- `.sh` extension for shell scripts

## Where to Add New Code

**New Feature:**
- Primary code: `rag_low_level_m1_16gb_verbose.py` (if core pipeline), or new file at root
- Tests: `tests/test_<feature>.py`
- Config if needed: `core/config.py` or `config/constants.py`

**New Utility:**
- Implementation: `utils/<utility_name>.py`
- Tests: `tests/test_<utility>.py`
- Import in main: Add to imports in `rag_low_level_m1_16gb_verbose.py`

**New Script:**
- Implementation: `scripts/<script_name>.py`
- Documentation: Add usage to `scripts/README.md`
- Make executable: `chmod +x scripts/<script_name>.py`

**New Documentation:**
- Implementation: `docs/<doc_name>.md`
- Link from README: Add to main `README.md` documentation section

**Utilities:**
- Shared helpers: `utils/<helper_name>.py`
- Type definitions: `core/config.py` or `config/constants.py`

## Special Directories

**data/**
- Purpose: User documents for indexing
- Source: User-provided (PDFs, HTML, DOCX, TXT, Markdown)
- Committed: No (in .gitignore)

**logs/**
- Purpose: Execution logs
- Source: Generated by application
- Committed: No (in .gitignore)

**query_logs/**
- Purpose: Query history and metrics
- Source: Generated by RAG pipeline
- Committed: No (in .gitignore)

**benchmarks/**
- Purpose: Performance results
- Source: Generated by benchmark scripts
- Committed: No (in .gitignore)

**.venv/**
- Purpose: Python virtual environment
- Source: Created by `python -m venv .venv`
- Committed: No (in .gitignore)

---

*Structure analysis: 2026-01-15*
*Update when directory structure changes*
