# Testing Patterns

**Analysis Date:** 2026-01-15

## Test Framework

**Runner:**
- pytest 8.0+ (`requirements-dev.txt` line 17, `pyproject.toml` line 44)
- Config: `config/pytest.ini` and `pyproject.toml` lines 173-198

**Assertion Library:**
- pytest built-in expect
- Matchers: Standard assertions

**Run Commands:**
```bash
pytest                                    # Run all tests
pytest -m "not slow"                      # Skip slow tests
pytest -m integration                     # Only integration tests
pytest --cov-report=html                  # With coverage report
pytest -v                                 # Verbose output
pytest tests/test_chunking.py            # Single file
```

## Test File Organization

**Location:**
- Test files in `tests/` directory (separate from source)
- 43+ test modules organized by functionality

**Naming:**
- Test files: `test_*.py` (e.g., `test_core_config.py`, `test_chunking.py`)
- Test classes: `Test*` prefix (e.g., `TestSettings`, `TestPostgreSQLConnection`)
- Test functions: `test_*` prefix (e.g., `test_default_chunk_size()`)

**Structure:**
```
tests/
├── conftest.py                           # Shared fixtures (707 lines)
├── test_core_config.py                   # Configuration tests
├── test_chunking.py                      # Chunking logic
├── test_embedding.py                     # Embedding models
├── test_database.py                      # Basic DB operations
├── test_database_integration.py          # Comprehensive DB integration
├── test_retrieval.py                     # Retrieval logic
├── test_performance.py                   # Performance benchmarking
├── test_e2e_pipeline.py                  # End-to-end workflows
└── [35+ more test files]
```

## Test Structure

**Suite Organization:**
```python
import pytest
from unittest.mock import Mock, patch

class TestModuleName:
    def test_success_case(self):
        # arrange
        # act
        # assert

    def test_error_case(self):
        # test code
```

**Patterns:**
- Class-based organization for related tests
- Arrange/Act/Assert pattern in complex tests
- One assertion focus per test (but multiple expects OK)

## Mocking

**Framework:**
- unittest.mock.MagicMock and Mock objects
- pytest monkeypatch for environment variables

**Patterns:**
```python
from unittest.mock import Mock, patch

@patch('psycopg2.connect')
def test_connection_with_valid_credentials(self, mock_connect):
    mock_conn = Mock()
    mock_connect.return_value = mock_conn
    # Test logic...
```

**What to Mock:**
- External dependencies (LLMs, embeddings, database)
- File system operations (when testing logic, not I/O)
- Network calls
- Time/dates (use freezegun if needed)

**What NOT to Mock:**
- Pure functions, utilities
- Internal business logic
- Configuration objects (use real Settings)

## Fixtures and Factories

**Test Data:**
```python
# From conftest.py
@pytest.fixture
def mock_embedding():
    """Generate random embedding vector."""
    return [random.random() for _ in range(384)]

@pytest.fixture
def mock_text_node():
    """Create TextNode with embedding."""
    return TextNode(
        text="Sample text",
        embedding=[0.1] * 384,
        metadata={"source": "test"}
    )
```

**Location:**
- Shared fixtures: `tests/conftest.py` (707 lines)
- Factory functions: In test file or conftest
- Test data: `tests/test_data/` directory

**Parametrized Fixtures:**
- `chunk_sizes` - Tests with sizes: 100, 500, 1000, 2000
- `chunk_overlaps` - Tests with overlaps: 0, 50, 100, 200
- `embed_models` - Tests with different models

## Coverage

**Requirements:**
- Target: 30% minimum (`pyproject.toml` line 226)
- Actual: 30.94% (from README.md badge)
- 310+ passing tests

**Configuration:**
- pytest-cov plugin (`requirements-dev.txt` line 18)
- HTML reports in `htmlcov/`
- Term missing format

**View Coverage:**
```bash
pytest --cov-report=html
open htmlcov/index.html
```

**Excluded from Coverage:**
- Test files themselves (`pyproject.toml` line 205)
- Scripts in `scripts/` directory
- `conftest.py` fixtures
- Archive and old code
- `__repr__` methods
- `raise NotImplementedError`
- Main guard blocks (`if __name__ == "__main__"`)

## Test Types

**Unit Tests:**
- Scope: Test single function/class in isolation
- Mocking: Mock all external dependencies
- Speed: Fast (<1s per test)
- Examples: `test_core_config.py`, `test_chunking.py`

**Integration Tests:**
- Scope: Test multiple modules together
- Mocking: Mock only external services (database, LLM)
- Setup: May require test database
- Examples: `test_database_integration.py`, `test_retrieval_direct.py`
- Marker: `@pytest.mark.integration`

**E2E Tests:**
- Scope: Test full user workflows
- Mocking: Minimal (test real integration)
- Examples: `test_e2e_pipeline.py`

**Performance Tests:**
- Scope: Measure speed and throughput
- Examples: `test_performance.py`, `test_performance_regression.py`
- Marker: `@pytest.mark.slow` or `@pytest.mark.benchmark`

## Common Patterns

**Async Testing:**
```python
import pytest

@pytest.mark.asyncio
async def test_async_operation(self):
    result = await async_function()
    assert result == expected
```

**Error Testing:**
```python
def test_throws_on_invalid_input(self):
    with pytest.raises(ValueError, match="error message"):
        function_call()
```

**Snapshot Testing:**
- Not used in this codebase
- Prefer explicit assertions for clarity

## Test Markers

**Custom Markers** (from `config/pytest.ini`):
- `@pytest.mark.slow` - Slow tests (skip with `-m "not slow"`)
- `@pytest.mark.integration` - Integration tests requiring external services
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.benchmark` - Benchmark tests
- `@pytest.mark.regression` - Regression tests

**Usage:**
```python
@pytest.mark.slow
def test_large_document_processing(self):
    """This test takes >5 seconds."""

@pytest.mark.integration
def test_database_connection(self):
    """Requires actual PostgreSQL instance."""
```

## Linting & Code Quality

**Pre-commit Hooks** (`.pre-commit-config.yaml`):
- Black: Code formatting (24.10.0)
- Ruff: Linting + formatting (v0.8.4) with auto-fix
- MyPy: Type checking (v1.13.0)
- Bandit: Security checks (1.7.10)
- interrogate: Docstring coverage (1.7.0) - 50% minimum
- Markdown formatting: mdformat
- Shell validation: shellcheck
- YAML/JSON/TOML: Syntax validation

---

*Testing analysis: 2026-01-15*
*Update when test patterns change*
