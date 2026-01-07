# Contributing to Local RAG Pipeline

Thank you for your interest in contributing! This project is focused on building a high-performance, privacy-first local RAG system optimized for Apple Silicon and NVIDIA GPUs.

---

## 🚀 Quick Start for Contributors

### 1. Setup Development Environment

```bash
# Clone the repository
git clone <repo-url>
cd llamaIndex-local-rag

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (includes testing, linting, security tools)
pip install -r requirements-dev.txt

# Optional: Install all optional features (web UI, MLX, vLLM, etc.)
pip install -r requirements-optional.txt

# Copy environment template
cp .env.example .env
# Edit .env with your database credentials

# Start PostgreSQL
docker-compose up -d
```

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Run specific test file
pytest tests/test_naming_utils.py -v
```

### 3. Code Quality Checks

```bash
# Format code
black . --line-length 100

# Lint code
ruff check .

# Type check
mypy rag_*.py --ignore-missing-imports

# Security scan
pip-audit
```

---

## 📋 Development Guidelines

### Code Style

- **Python Version:** 3.11+
- **Line Length:** 100 characters
- **Formatting:** black with default settings
- **Docstrings:** Google style
- **Type Hints:** Required for all public functions

**Example:**
```python
def process_document(doc_path: str, chunk_size: int = 700) -> List[TextNode]:
    \"\"\"Process a document into chunked nodes.

    Args:
        doc_path: Path to document file
        chunk_size: Target chunk size in characters

    Returns:
        List of TextNode objects with embeddings

    Raises:
        ValueError: If doc_path doesn't exist
    \"\"\"
    # Implementation here
```

### Testing Requirements

- **Coverage Target:** 20% minimum (goal: 60%)
- **Test Naming:** `test_*.py` in `tests/` directory
- **All new features:** Must include tests
- **All bug fixes:** Should include regression test

**Running Tests:**
```bash
# Before committing
pytest -v

# Check coverage
pytest --cov=. --cov-report=term-missing
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add MLX embedding backend support
fix: correct TypeError in vLLM client
docs: update README with installation steps
test: add chunking validation tests
refactor: extract database utilities to module
perf: optimize GPU layer offloading for M1
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

---

## 🔧 Project Structure

```
llamaIndex-local-rag/
├── rag_low_level_m1_16gb_verbose.py  # Main pipeline (being modularized)
├── rag_interactive.py                 # CLI menu interface
├── rag_web.py                         # Streamlit web UI
├── utils/                             # Shared utilities
│   ├── naming.py                      # Table/model name generation
│   └── database.py                    # Database utilities (planned)
├── tests/                             # Test suite
│   ├── test_naming_utils.py
│   ├── test_config.py
│   ├── test_database.py
│   └── ...
├── scripts/                           # Utility scripts
└── docs/                              # Documentation
```

---

## 🎯 Areas for Contribution

### High Priority
- [ ] Increase test coverage (currently 11%, target 20%)
- [ ] Modularize main file (2,734 lines → modules)
- [ ] Add integration tests
- [ ] Improve error messages

### Medium Priority
- [ ] Add support for more document formats
- [ ] Integrate reranker module
- [ ] Add query result caching
- [ ] Performance benchmarking suite

### Documentation
- [ ] Add more code examples
- [ ] Create video tutorials
- [ ] Improve troubleshooting guide
- [ ] Add architecture diagrams

---

## 🐛 Reporting Issues

### Before Reporting
1. Check existing issues
2. Try with latest version
3. Review troubleshooting guide

### Issue Template
```markdown
**Description:**
Clear description of the issue

**Steps to Reproduce:**
1. Step one
2. Step two

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: macOS 14.2 / Linux / Windows
- Python: 3.11.9
- Hardware: M1 Mac Mini 16GB / RTX 4090
- Dependencies: (run `pip list | grep -E "(llama|torch|pgvector)"`)

**Logs:**
```
Paste relevant error logs here
```
```

---

## 🔀 Pull Request Process

### Before Submitting PR

1. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests:**
   ```bash
   # Add tests for your changes
   pytest tests/test_your_feature.py -v
   ```

3. **Run quality checks:**
   ```bash
   pytest                      # All tests pass
   black . --line-length 100   # Format code
   ruff check .                # No lint errors
   pip-audit                   # No vulnerabilities
   ```

4. **Update documentation:**
   - Update README.md if adding features
   - Update CLAUDE.md for development changes
   - Add docstrings to new functions

### PR Template

```markdown
## Description
What does this PR do?

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added/updated tests
- [ ] All tests passing
- [ ] Coverage not decreased

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No security issues introduced
```

### Review Process
1. Automated CI/CD checks must pass
2. Code review by maintainer
3. Merge to main

---

## 💡 Development Tips

### Running Locally

```bash
# Load environment
source .venv/bin/activate
source .env

# Run with optimizations
source /tmp/m1_optimized.env

# Test query
time python3 rag_low_level_m1_16gb_verbose.py --query-only --query "test"
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Log full chunks
export LOG_FULL_CHUNKS=1

# Run with Python debugger
python3 -m pdb rag_low_level_m1_16gb_verbose.py
```

### Performance Profiling

```bash
# Profile code
python3 -m cProfile -o profile.stats rag_low_level_m1_16gb_verbose.py

# Analyze
python3 -m pstats profile.stats

# Memory profiling
pip install memory-profiler
python3 -m memory_profiler rag_low_level_m1_16gb_verbose.py
```

---

## 🏗️ Architecture Decisions

### Why PostgreSQL + pgvector?
- Mature, reliable database
- ACID compliance
- JSON support for metadata
- Good performance with HNSW indexing

### Why Multiple LLM Backends?
- Flexibility: llama.cpp (CPU), vLLM (GPU)
- Performance: vLLM 15x faster than llama.cpp
- Compatibility: Works on M1 and NVIDIA

### Why MLX for M1?
- Apple Silicon optimized
- 5-20x faster than HuggingFace
- Lower memory footprint

---

## 📊 Performance Benchmarks

If adding performance-related changes, include benchmarks:

```python
import time

def benchmark_function():
    start = time.perf_counter()
    # Your code here
    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed:.3f}s")
```

**Report format:**
```markdown
### Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Query time | 15s | 5s | 3x faster |
```

---

## 🤝 Code of Conduct

- Be respectful and constructive
- Welcome beginners and help them learn
- Focus on technical merit
- Assume good intent

---

## 📞 Getting Help

- **Documentation:** Start with `AUDIT_EXECUTIVE_SUMMARY.md`
- **Issues:** Check existing issues first
- **Questions:** Open a discussion issue

---

## ✅ First Contribution Ideas

### Good First Issues
- [ ] Add more test cases (increase coverage)
- [ ] Fix typos in documentation
- [ ] Improve error messages
- [ ] Add code examples to README

### Small Improvements
- [ ] Add support for new document format
- [ ] Optimize batch size for specific GPU
- [ ] Add configuration validation tests
- [ ] Update dependencies

### Larger Features
- [ ] Integrate reranker module
- [ ] Add FastAPI wrapper
- [ ] Implement query result caching
- [ ] Multi-document search

---

## 📚 Resources

- **Main Documentation:** `README.md`
- **Developer Guide:** `CLAUDE.md`
- **Performance Guide:** `PERFORMANCE_QUICK_START.md`
- **Audit Reports:** `AUDIT_EXECUTIVE_SUMMARY.md`

---

**Thank you for contributing to making local RAG better!**
