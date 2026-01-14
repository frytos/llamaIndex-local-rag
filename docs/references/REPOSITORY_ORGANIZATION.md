# Repository Organization Guide

**Last Updated**: January 2026

This document describes the repository structure, naming conventions, and organization principles.

## Directory Structure

```
llamaIndex-local-rag/
├── config/              # Configuration files
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── tests/               # Unit tests
├── utils/               # Shared utilities
├── data/                # Documents (gitignored)
├── logs/                # Log files (gitignored)
├── benchmarks/          # Performance results (gitignored)
├── archive/             # Old versions
└── [Python modules]     # Core scripts in root
```

## Naming Conventions

### Documentation Files
- **Format**: `UPPERCASE_WITH_UNDERSCORES.md`
- **Examples**: `PERFORMANCE_QUICK_START.md`, `RUNPOD_FINAL_SETUP.md`
- **Exceptions**: `README.md` (standard), tool-specific guides like `chainlit.md` → `CHAINLIT_GUIDE.md`

### Shell Scripts
- **Format**: `lowercase_with_underscores.sh`
- **Functional Prefixes**:
  - `deploy_*` - Deployment scripts
  - `index_*` - Indexing operations
  - `database_*` - Database operations
  - `system_*` - System utilities
  - `monitoring_*` - Monitoring tools
  - `config_*` - Configuration scripts
  - `helper_*` - Helper utilities
  - `quick_start_*` - Quick start scripts

### Python Scripts
- **Format**: `snake_case.py`
- **Examples**: `rag_interactive.py`, `vllm_client.py`, `check_dependencies.py`
- **Consistent**: All Python files use snake_case

### Config Files
- **Format**: `lowercase_with_underscores` or standard names
- **Examples**: `.env.example`, `docker-compose.yml`, `pytest.ini`

## File Organization Principles

### 1. Separation of Concerns
- **Core logic** in root Python files
- **Configuration** in `config/`
- **Documentation** in `docs/`
- **Utilities** in `scripts/`
- **Tests** in `tests/`

### 2. Discoverability
- Each major directory has a `README.md`
- Scripts have functional prefixes for easy identification
- Documentation is alphabetically sortable by topic

### 3. Maintainability
- Related files are grouped together
- Clear naming indicates purpose
- Deprecated files moved to `archive/`

### 4. Git-Friendly
- Generated files (logs, benchmarks) are gitignored
- Configuration templates (.env.example) are committed
- Actual secrets (.env) are never committed

## Key Directories

### `config/`
Configuration files and templates:
- `.env.example` - Environment variable template
- `docker-compose.yml` - PostgreSQL setup
- `pytest.ini` - Test configuration
- `requirements_vllm.txt` - Optional vLLM dependencies
- `runpod_*.env` - RunPod configurations

### `docs/`
All project documentation:
- Getting started guides
- Deployment instructions
- Performance tuning
- API references
- Troubleshooting

### `scripts/`
Organized by function:
- **Deployment**: RunPod, vLLM server control
- **Indexing**: Different embedding models
- **Benchmarking**: Performance testing
- **Visualization**: TensorBoard, Atlas
- **Data Processing**: Clean and prepare data
- **System**: Memory, monitoring

### `tests/`
Unit and integration tests:
- `test_*.py` - Test files
- `__init__.py` - Package initialization
- Follows pytest conventions

### `utils/`
Shared utility modules:
- Common functions used across scripts
- Helper utilities
- Naming conventions

## Scripts Reference

### Deployment Scripts
```bash
scripts/deploy_runpod.sh              # Deploy to RunPod
scripts/start_vllm_server.sh          # Start vLLM server
scripts/quick_start_vllm.sh           # Quick start with vLLM
scripts/quick_start_optimized.sh      # Optimized quick start
```

### Indexing Scripts
```bash
scripts/index_bge_small.sh            # Index with BGE-small
scripts/index_multilingual_e5.sh      # Index with E5 model
```

### Utility Scripts
```bash
scripts/check_dependencies.py         # Check all dependencies
scripts/database_apply_hnsw.sh        # Apply HNSW indexing
scripts/system_free_memory.sh         # Free system memory
scripts/monitoring_query.sh           # Monitor queries
```

## Documentation Reference

### Getting Started
- `docs/START_HERE.md` - Main entry point
- `docs/ENVIRONMENT_VARIABLES.md` - Configuration guide
- `config/README.md` - Configuration overview

### Deployment
- `docs/RUNPOD_FINAL_SETUP.md` - RunPod deployment
- `docs/VLLM_SERVER_GUIDE.md` - vLLM setup
- `docs/GITHUB_TOKEN_SETUP.md` - GitHub integration

### Performance
- `docs/PERFORMANCE_QUICK_START.md` - Quick tuning guide
- `docs/RAG_OPTIMIZATION_GUIDE.md` - RAG optimization
- `docs/SCALABILITY_ANALYSIS.md` - Scaling guide

### Development
- `CLAUDE.md` - Developer guide
- `README.md` - Project overview
- `scripts/README.md` - Scripts documentation

## Backward Compatibility

### Symlinks
- `docker-compose.yml` → `config/docker-compose.yml`

### Script Renames
Old scripts have been renamed with functional prefixes:
- `QUICK_COMMANDS.sh` → `helper_quick_commands.sh`
- `apply_hnsw.sh` → `database_apply_hnsw.sh`
- `free_memory.sh` → `system_free_memory.sh`

## Best Practices

### Adding New Files

1. **Determine category**: config, docs, script, test, or core module
2. **Follow naming convention**: Lowercase with underscores for scripts, UPPERCASE for major docs
3. **Add functional prefix**: For scripts, use appropriate prefix (deploy_, index_, etc.)
4. **Update README**: Add entry to relevant README.md
5. **Document purpose**: Add docstring or header comment

### Modifying Structure

1. **Update all READMEs**: Keep navigation current
2. **Update references**: Search for old paths in documentation
3. **Test imports**: Verify Python imports still work
4. **Update CLAUDE.md**: Reflect changes in developer guide
5. **Create symlinks**: If breaking backward compatibility

### Removing Files

1. **Move to archive/**: Don't delete, preserve history
2. **Document reason**: Add note in archive/README.md
3. **Update references**: Remove from documentation
4. **Check dependencies**: Ensure no scripts depend on it

## Migration Notes

### From Previous Structure

The repository was reorganized in January 2026:
- Documentation consolidated from root to `docs/`
- Configuration files moved to `config/`
- Scripts renamed with functional prefixes
- README files added to all major directories
- Naming conventions standardized

All functionality remains the same, just better organized.

## Quick Commands

```bash
# Check dependencies
python scripts/check_dependencies.py

# View scripts documentation
cat scripts/README.md

# View documentation index
cat docs/README.md

# View config guide
cat config/README.md

# Deploy to RunPod
./scripts/deploy_runpod.sh

# Start vLLM server
./scripts/start_vllm_server.sh
```

## Questions?

- **Repository structure**: See this file
- **Developer guide**: See `CLAUDE.md`
- **Getting started**: See `docs/START_HERE.md`
- **Configuration**: See `config/README.md`
- **Scripts**: See `scripts/README.md`
