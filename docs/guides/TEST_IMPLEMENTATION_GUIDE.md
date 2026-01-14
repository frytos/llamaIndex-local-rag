# Test Implementation Guide
**Practical Templates & Patterns for Adding Tests**

This guide provides copy-paste templates for quickly adding tests to untested modules.

---

## Quick Reference: Test Template Selection

| Module Type | Template | Example File |
|-------------|----------|--------------|
| Utility function | [Template 1](#template-1-utility-functions) | test_naming.py |
| Class with state | [Template 2](#template-2-stateful-classes) | test_query_cache.py |
| Integration | [Template 3](#template-3-integration-tests) | test_database_integration.py |
| API/Web | [Template 4](#template-4-web-api-tests) | test_web_backend.py |
| Performance | [Template 5](#template-5-performance-tests) | test_performance.py |

---

## Template 1: Utility Functions

**Use for:** Pure functions, transformations, validators

```python
"""Tests for utils/MODULE_NAME.py"""

import pytest
from utils.MODULE_NAME import function_to_test


class TestFunctionName:
    """Test function_to_test functionality."""

    def test_basic_functionality(self):
        """Test basic happy path."""
        result = function_to_test("input")
        assert result == "expected"

    def test_edge_case_empty_input(self):
        """Test handling of empty input."""
        result = function_to_test("")
        assert result == "default_value"

    def test_edge_case_none(self):
        """Test handling of None."""
        with pytest.raises(ValueError, match="cannot be None"):
            function_to_test(None)

    @pytest.mark.parametrize("input,expected", [
        ("case1", "result1"),
        ("case2", "result2"),
        ("case3", "result3"),
    ])
    def test_multiple_cases(self, input, expected):
        """Test multiple input cases."""
        assert function_to_test(input) == expected

    def test_error_handling(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError) as exc_info:
            function_to_test("invalid")
        assert "Invalid input" in str(exc_info.value)
```

**Example: tests/test_query_expansion.py**

```python
"""Tests for query expansion functionality."""

import pytest
from utils.query_expansion import (
    expand_query,
    generate_synonyms,
    extract_keywords,
)


class TestQueryExpansion:
    """Test query expansion utilities."""

    def test_expand_query_basic(self):
        """Test basic query expansion."""
        query = "What is machine learning?"
        expanded = expand_query(query)

        assert len(expanded) > len(query)
        assert "machine learning" in expanded.lower()
        assert "ML" in expanded or "artificial intelligence" in expanded

    def test_expand_query_empty(self):
        """Test expansion with empty query."""
        result = expand_query("")
        assert result == ""

    def test_generate_synonyms(self):
        """Test synonym generation."""
        word = "database"
        synonyms = generate_synonyms(word)

        assert isinstance(synonyms, list)
        assert len(synonyms) > 0
        assert "db" in synonyms or "datastore" in synonyms

    @pytest.mark.parametrize("query,expected_keywords", [
        ("Python programming", ["python", "programming"]),
        ("RAG system", ["rag", "system"]),
        ("vector database", ["vector", "database"]),
    ])
    def test_extract_keywords(self, query, expected_keywords):
        """Test keyword extraction."""
        keywords = extract_keywords(query)
        assert all(kw in keywords for kw in expected_keywords)
```

---

## Template 2: Stateful Classes

**Use for:** Classes with initialization, state management, caching

```python
"""Tests for utils/MODULE_NAME.py"""

import pytest
from unittest.mock import Mock, patch
from utils.MODULE_NAME import ClassName


@pytest.fixture
def instance():
    """Create instance for testing."""
    return ClassName(param1="value1", param2="value2")


@pytest.fixture
def mock_dependency():
    """Mock external dependency."""
    mock = Mock()
    mock.method.return_value = "mocked_result"
    return mock


class TestClassInitialization:
    """Test class initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        obj = ClassName()
        assert obj.param1 is not None
        assert obj.param2 is not None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        obj = ClassName(param1="custom", param2=42)
        assert obj.param1 == "custom"
        assert obj.param2 == 42

    def test_init_validation(self):
        """Test parameter validation on init."""
        with pytest.raises(ValueError, match="Invalid param"):
            ClassName(param1="invalid")


class TestClassMethods:
    """Test class methods."""

    def test_method_basic(self, instance):
        """Test basic method call."""
        result = instance.method_name("input")
        assert result == "expected"

    def test_method_state_change(self, instance):
        """Test that method changes state correctly."""
        instance.method_that_changes_state()
        assert instance.state_variable == "new_value"

    @patch('utils.MODULE_NAME.external_function')
    def test_method_with_external_call(self, mock_external, instance):
        """Test method that calls external function."""
        mock_external.return_value = "mocked"
        result = instance.method_using_external()

        mock_external.assert_called_once()
        assert result == "mocked"


class TestClassEdgeCases:
    """Test edge cases and error handling."""

    def test_method_with_none(self, instance):
        """Test method with None input."""
        with pytest.raises(ValueError):
            instance.method_name(None)

    def test_method_with_empty_list(self, instance):
        """Test method with empty list."""
        result = instance.method_name([])
        assert result == []

    def test_concurrent_access(self, instance):
        """Test thread-safety of methods."""
        # Use threading if needed
        pass
```

**Example: tests/test_query_cache.py**

```python
"""Tests for query caching functionality."""

import pytest
import time
from unittest.mock import Mock, patch
from utils.query_cache import QueryCache, CacheEntry


@pytest.fixture
def cache():
    """Create QueryCache instance for testing."""
    return QueryCache(ttl=60, max_size=100)


@pytest.fixture
def mock_embedder():
    """Mock embedding function."""
    mock = Mock()
    mock.embed.return_value = [0.1, 0.2, 0.3]
    return mock


class TestQueryCacheInit:
    """Test QueryCache initialization."""

    def test_init_with_defaults(self):
        """Test cache with default parameters."""
        cache = QueryCache()
        assert cache.ttl > 0
        assert cache.max_size > 0

    def test_init_with_custom_params(self):
        """Test cache with custom parameters."""
        cache = QueryCache(ttl=300, max_size=500)
        assert cache.ttl == 300
        assert cache.max_size == 500

    def test_init_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="TTL must be positive"):
            QueryCache(ttl=-1)


class TestCacheOperations:
    """Test cache get/set operations."""

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent_query")
        assert result is None

    def test_cache_hit(self, cache):
        """Test cache hit returns cached value."""
        cache.set("test_query", "cached_result")
        result = cache.get("test_query")
        assert result == "cached_result"

    def test_cache_expiration(self, cache):
        """Test cache entry expires after TTL."""
        cache = QueryCache(ttl=1)  # 1 second TTL
        cache.set("test_query", "cached_result")

        # Should be cached
        assert cache.get("test_query") == "cached_result"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("test_query") is None

    def test_cache_eviction(self, cache):
        """Test cache evicts oldest entry when full."""
        cache = QueryCache(max_size=3)

        # Fill cache
        cache.set("query1", "result1")
        cache.set("query2", "result2")
        cache.set("query3", "result3")

        # Add one more (should evict oldest)
        cache.set("query4", "result4")

        # query1 should be evicted
        assert cache.get("query1") is None
        assert cache.get("query4") == "result4"

    def test_cache_clear(self, cache):
        """Test cache clearing."""
        cache.set("query1", "result1")
        cache.set("query2", "result2")
        cache.clear()

        assert cache.get("query1") is None
        assert cache.get("query2") is None


class TestCacheMetrics:
    """Test cache metrics and statistics."""

    def test_hit_rate_calculation(self, cache):
        """Test cache hit rate calculation."""
        cache.set("query1", "result1")

        # 1 hit, 0 misses
        cache.get("query1")
        assert cache.hit_rate() == 1.0

        # 1 hit, 1 miss
        cache.get("nonexistent")
        assert cache.hit_rate() == 0.5

    def test_cache_size(self, cache):
        """Test cache size tracking."""
        assert cache.size() == 0

        cache.set("query1", "result1")
        assert cache.size() == 1

        cache.set("query2", "result2")
        assert cache.size() == 2
```

---

## Template 3: Integration Tests

**Use for:** Database, file I/O, external services

```python
"""Integration tests for MODULE_NAME."""

import pytest
from pathlib import Path
import psycopg2
from utils.MODULE_NAME import function_to_test


@pytest.fixture(scope="module")
def test_database():
    """Set up test database."""
    # Setup
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_pass"
    )
    conn.autocommit = True

    yield conn

    # Teardown
    conn.close()


@pytest.fixture
def temp_test_table(test_database):
    """Create temporary test table."""
    cursor = test_database.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding vector(384)
        )
    """)

    yield "test_embeddings"

    # Cleanup
    cursor.execute("DROP TABLE IF EXISTS test_embeddings")
    cursor.close()


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests with real database."""

    def test_database_connection(self, test_database):
        """Test database connection works."""
        cursor = test_database.cursor()
        cursor.execute("SELECT version()")
        result = cursor.fetchone()
        assert result is not None
        assert "PostgreSQL" in result[0]

    def test_insert_and_retrieve(self, test_database, temp_test_table):
        """Test inserting and retrieving data."""
        cursor = test_database.cursor()

        # Insert
        cursor.execute(
            f"INSERT INTO {temp_test_table} (text, embedding) VALUES (%s, %s)",
            ("test text", [0.1] * 384)
        )

        # Retrieve
        cursor.execute(f"SELECT text FROM {temp_test_table}")
        result = cursor.fetchone()

        assert result[0] == "test text"

    def test_vector_similarity_search(self, test_database, temp_test_table):
        """Test pgvector similarity search."""
        cursor = test_database.cursor()

        # Insert test vectors
        cursor.execute(
            f"INSERT INTO {temp_test_table} (text, embedding) VALUES (%s, %s)",
            ("document 1", [0.1] * 384)
        )

        # Query with similarity
        query_vec = [0.1] * 384
        cursor.execute(f"""
            SELECT text, embedding <=> %s as distance
            FROM {temp_test_table}
            ORDER BY distance
            LIMIT 1
        """, (query_vec,))

        result = cursor.fetchone()
        assert result[0] == "document 1"
        assert result[1] < 0.1  # Should be very similar


@pytest.mark.integration
class TestFileSystemIntegration:
    """Integration tests with file system."""

    def test_read_write_file(self, tmp_path):
        """Test reading and writing files."""
        test_file = tmp_path / "test.txt"
        content = "test content"

        # Write
        test_file.write_text(content)

        # Read
        assert test_file.read_text() == content

    def test_process_large_file(self, tmp_path):
        """Test processing large files."""
        large_file = tmp_path / "large.txt"
        large_content = "line\n" * 100000

        large_file.write_text(large_content)

        # Process file
        line_count = sum(1 for _ in open(large_file))
        assert line_count == 100000
```

**Example: tests/test_integration_mlx.py**

```python
"""Integration tests for MLX embedding on M1 Macs."""

import pytest
import torch
from utils.mlx_embedding import MLXEmbeddings
from utils.platform_detection import has_mps, has_mlx


@pytest.fixture
def mlx_embedder():
    """Create MLX embedder for testing."""
    if not has_mlx():
        pytest.skip("MLX not available")
    return MLXEmbeddings(model_name="BAAI/bge-small-en")


@pytest.mark.integration
@pytest.mark.skipif(not has_mlx(), reason="MLX not available")
class TestMLXEmbeddingIntegration:
    """Integration tests for MLX embedding."""

    def test_mlx_device_detection(self, mlx_embedder):
        """Test that MLX uses MPS device on M1."""
        assert mlx_embedder.device == "mps"

    def test_mlx_model_loading(self, mlx_embedder):
        """Test MLX model loads without errors."""
        assert mlx_embedder.model is not None
        assert mlx_embedder.model_name == "BAAI/bge-small-en"

    def test_mlx_single_embedding(self, mlx_embedder):
        """Test embedding single query."""
        query = "What is Retrieval-Augmented Generation?"
        embedding = mlx_embedder.embed_query(query)

        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
        assert not any(x == 0 for x in embedding)  # Should not be zeros

    def test_mlx_batch_embedding(self, mlx_embedder):
        """Test batch embedding multiple documents."""
        docs = [
            "Document 1 about RAG systems",
            "Document 2 about vector databases",
            "Document 3 about LLMs",
        ]
        embeddings = mlx_embedder.embed_documents(docs)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

        # Embeddings should be different
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    def test_mlx_performance_vs_cpu(self, mlx_embedder):
        """Test that MLX is faster than CPU."""
        import time
        from sentence_transformers import SentenceTransformer

        text = "Test embedding performance" * 10
        texts = [text] * 10

        # MLX timing
        start = time.time()
        mlx_embedder.embed_documents(texts)
        mlx_time = time.time() - start

        # CPU timing
        cpu_model = SentenceTransformer("BAAI/bge-small-en", device="cpu")
        start = time.time()
        cpu_model.encode(texts)
        cpu_time = time.time() - start

        # MLX should be faster (allow 20% margin)
        assert mlx_time < cpu_time * 0.8, f"MLX ({mlx_time:.2f}s) not faster than CPU ({cpu_time:.2f}s)"

    def test_mlx_memory_usage(self, mlx_embedder):
        """Test MLX memory usage is reasonable."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Embed large batch
        texts = ["Test text " * 100] * 100
        mlx_embedder.embed_documents(texts)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use more than 2GB
        assert memory_increase < 2000, f"Memory usage too high: {memory_increase:.0f}MB"
```

---

## Template 4: Web API Tests

**Use for:** REST APIs, web endpoints, Streamlit apps

```python
"""Tests for web API endpoints."""

import pytest
from fastapi.testclient import TestClient
from app import app  # Your FastAPI/Flask app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authentication headers."""
    return {"Authorization": "Bearer test_token"}


class TestAPIEndpoints:
    """Test API endpoint functionality."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_query_endpoint_success(self, client):
        """Test successful query."""
        payload = {"query": "What is RAG?"}
        response = client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_endpoint_validation(self, client):
        """Test query validation."""
        # Empty query
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

        # Missing query field
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_endpoint_authentication(self, client, auth_headers):
        """Test authenticated endpoint."""
        payload = {"query": "test"}

        # Without auth
        response = client.post("/query", json=payload)
        assert response.status_code == 401

        # With auth
        response = client.post("/query", json=payload, headers=auth_headers)
        assert response.status_code == 200

    def test_upload_document(self, client, tmp_path):
        """Test document upload endpoint."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test.pdf", f, "application/pdf")}
            )

        assert response.status_code == 200
        assert response.json()["status"] == "uploaded"

    def test_rate_limiting(self, client):
        """Test API rate limiting."""
        # Send many requests
        responses = [client.get("/query") for _ in range(100)]

        # Some should be rate limited
        rate_limited = [r for r in responses if r.status_code == 429]
        assert len(rate_limited) > 0
```

**Example: tests/test_web_backend.py**

```python
"""Tests for RAG web backend API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from rag_web_backend import app, query_engine


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_query_engine():
    """Mock the query engine."""
    mock = Mock()
    mock.query.return_value = Mock(
        response="Test answer",
        source_nodes=[
            Mock(text="Source 1", score=0.95),
            Mock(text="Source 2", score=0.87),
        ]
    )
    return mock


class TestWebBackendHealth:
    """Test backend health endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        assert "version" in response.json()

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "database" in data
        assert "model" in data


class TestWebBackendQuery:
    """Test query endpoint."""

    @patch('rag_web_backend.query_engine')
    def test_query_success(self, mock_engine, client):
        """Test successful query."""
        mock_engine.query.return_value = Mock(
            response="RAG is Retrieval-Augmented Generation",
            source_nodes=[Mock(text="Source", score=0.9)]
        )

        response = client.post("/query", json={
            "query": "What is RAG?",
            "top_k": 3
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

    def test_query_validation(self, client):
        """Test query input validation."""
        # Empty query
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

        # Query too long
        long_query = "a" * 10000
        response = client.post("/query", json={"query": long_query})
        assert response.status_code == 422

        # Invalid top_k
        response = client.post("/query", json={"query": "test", "top_k": -1})
        assert response.status_code == 422

    def test_query_error_handling(self, client):
        """Test query error handling."""
        with patch('rag_web_backend.query_engine.query', side_effect=Exception("DB error")):
            response = client.post("/query", json={"query": "test"})
            assert response.status_code == 500
            assert "error" in response.json()


class TestWebBackendUpload:
    """Test document upload endpoint."""

    def test_upload_pdf_success(self, client, tmp_path):
        """Test uploading PDF file."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("PDF content")

        with open(pdf_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test.pdf", f, "application/pdf")}
            )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_upload_invalid_file_type(self, client, tmp_path):
        """Test uploading invalid file type."""
        exe_file = tmp_path / "virus.exe"
        exe_file.write_bytes(b"executable")

        with open(exe_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("virus.exe", f, "application/x-msdownload")}
            )

        assert response.status_code == 400
        assert "not supported" in response.json()["error"].lower()

    def test_upload_file_too_large(self, client, tmp_path):
        """Test uploading file exceeding size limit."""
        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b"x" * (100 * 1024 * 1024))  # 100MB

        with open(large_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("large.pdf", f, "application/pdf")}
            )

        assert response.status_code == 413  # Payload Too Large
```

---

## Template 5: Performance Tests

**Use for:** Benchmarks, load tests, performance regression

```python
"""Performance tests for MODULE_NAME."""

import pytest
import time
from utils.MODULE_NAME import function_to_test


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""

    def test_function_performance(self, benchmark):
        """Benchmark function execution time."""
        result = benchmark(function_to_test, "input")
        assert result is not None

    def test_function_performance_large_input(self, benchmark):
        """Benchmark with large input."""
        large_input = "x" * 10000
        result = benchmark(function_to_test, large_input)
        assert result is not None

    def test_batch_processing_performance(self):
        """Test batch processing speed."""
        inputs = ["input" + str(i) for i in range(1000)]

        start = time.time()
        results = [function_to_test(inp) for inp in inputs]
        elapsed = time.time() - start

        # Should process 1000 items in under 1 second
        assert elapsed < 1.0
        assert len(results) == 1000

    def test_memory_usage(self):
        """Test memory usage stays reasonable."""
        import tracemalloc

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Perform operation
        large_input = "x" * 1000000
        function_to_test(large_input)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Check memory increase
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_increase = sum(stat.size_diff for stat in top_stats)

        # Should not increase memory by more than 100MB
        assert total_increase < 100 * 1024 * 1024


@pytest.mark.slow
class TestLoadPerformance:
    """Load and stress tests."""

    def test_concurrent_execution(self):
        """Test concurrent execution performance."""
        import concurrent.futures

        def process_item(i):
            return function_to_test(f"input_{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_item, i) for i in range(100)]
            results = [f.result() for f in futures]

        assert len(results) == 100
        assert all(r is not None for r in results)

    def test_sustained_load(self):
        """Test performance under sustained load."""
        duration = 10  # seconds
        start = time.time()
        count = 0

        while time.time() - start < duration:
            function_to_test("input")
            count += 1

        ops_per_second = count / duration
        print(f"Operations per second: {ops_per_second:.2f}")

        # Should handle at least 100 ops/second
        assert ops_per_second >= 100
```

---

## Running Tests: Quick Commands

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_query_cache.py -v

# Run specific test class
pytest tests/test_query_cache.py::TestCacheOperations -v

# Run specific test
pytest tests/test_query_cache.py::TestCacheOperations::test_cache_hit -v

# Run with coverage
pytest tests/ --cov=utils --cov-report=html

# Run only fast tests (skip slow/integration)
pytest tests/ -m "not slow and not integration"

# Run only integration tests
pytest tests/ -m integration

# Run tests in parallel
pytest tests/ -n auto

# Run with verbose output
pytest tests/ -vv --tb=short

# Run and stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Run tests matching pattern
pytest tests/ -k "cache"
```

---

## Coverage Commands

```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Show missing lines
pytest tests/ --cov=. --cov-report=term-missing

# Coverage for specific module
pytest tests/ --cov=utils.query_cache --cov-report=term-missing

# Fail if coverage below threshold
pytest tests/ --cov=. --cov-fail-under=50
```

---

## Debugging Tests

```bash
# Drop into debugger on failure
pytest tests/ --pdb

# Print output (disable capture)
pytest tests/ -s

# Very verbose output
pytest tests/ -vvv

# Show locals on failure
pytest tests/ -l

# Run with warnings
pytest tests/ -W all
```

---

## Best Practices Checklist

When writing tests, ensure:

- [ ] Test has clear, descriptive name
- [ ] Test tests ONE thing
- [ ] Assertions have failure messages
- [ ] Edge cases are covered (None, empty, huge input)
- [ ] Error cases are tested
- [ ] Fixtures are used for setup/teardown
- [ ] Mocks are used for expensive operations
- [ ] Integration tests use real dependencies
- [ ] Tests are independent (can run in any order)
- [ ] Tests clean up after themselves
- [ ] Performance tests have baseline assertions
- [ ] Markers are used (slow, integration, etc.)
- [ ] Docstrings explain WHAT is tested, not HOW

---

**Quick Start:**
1. Copy appropriate template above
2. Replace MODULE_NAME and function_to_test
3. Add test cases for your specific functionality
4. Run: `pytest tests/test_your_module.py -v`
5. Check coverage: `pytest --cov=utils.your_module --cov-report=term-missing`

**Need help?** See TESTING_QUALITY_AUDIT.md for comprehensive guide.
