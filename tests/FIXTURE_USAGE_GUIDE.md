# Test Fixtures Usage Guide

This guide explains how to use the shared test fixtures in `conftest.py` for testing the RAG pipeline.

## Overview

The `conftest.py` file provides **reusable test fixtures** that automatically set up mock data, services, and test environments. All fixtures are automatically available in any test file without imports.

## Quick Start

```python
# tests/test_my_feature.py
import pytest

def test_embedding_generation(mock_embed_model):
    """Test uses mock_embed_model fixture automatically."""
    text = "Sample text"
    embedding = mock_embed_model.get_text_embedding(text)
    assert len(embedding) == 384

def test_with_multiple_fixtures(mock_text_node, mock_vector_store):
    """Multiple fixtures can be injected together."""
    node = mock_text_node(text="Test chunk")
    mock_vector_store.add([node])
    assert len(mock_vector_store._nodes) == 1
```

## Available Fixtures

### 1. Mock Data Factories

#### `mock_embedding`
Generates fake 384-dimensional embedding vectors.

```python
def test_embeddings(mock_embedding):
    vec = mock_embedding(384, seed=42)
    assert len(vec) == 384
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6  # Normalized
```

#### `mock_document`
Creates test Document objects.

```python
def test_document_processing(mock_document):
    doc = mock_document(
        text="Test content",
        metadata={"source": "test.pdf", "page": 1}
    )
    assert doc.text == "Test content"
    assert doc.metadata["source"] == "test.pdf"
```

#### `mock_text_node`
Creates TextNode objects with embeddings.

```python
def test_text_nodes(mock_text_node):
    node = mock_text_node(
        text="Chunk content",
        doc_id="doc_123",
        node_id="node_456"
    )
    assert len(node.embedding) == 384
    assert node.metadata["document_id"] == "doc_123"
```

#### `mock_query_result`
Creates mock retrieval results.

```python
def test_query_results(mock_query_result):
    results = mock_query_result(
        num_results=3,
        scores=[0.95, 0.88, 0.75],
        texts=["Result 1", "Result 2", "Result 3"]
    )
    assert len(results) == 3
    assert results[0].score == 0.95
```

### 2. Mock Services

#### `mock_embed_model`
Mocked HuggingFaceEmbedding model.

```python
def test_embedding_model(mock_embed_model):
    # Single text
    emb = mock_embed_model.get_text_embedding("test")
    assert len(emb) == 384

    # Multiple texts
    embs = mock_embed_model.get_text_embeddings(["text1", "text2"])
    assert len(embs) == 2
```

#### `mock_llm`
Mocked LlamaCPP LLM.

```python
def test_llm_generation(mock_llm):
    response = mock_llm.complete("What is RAG?")
    assert response.text is not None

    # Streaming
    for chunk in mock_llm.stream_complete("Generate text"):
        assert hasattr(chunk, "delta")
```

#### `mock_vector_store`
Mocked PGVectorStore.

```python
def test_vector_store_operations(mock_vector_store, mock_text_node):
    # Add nodes
    nodes = [mock_text_node() for _ in range(5)]
    ids = mock_vector_store.add(nodes)
    assert len(ids) == 5

    # Query
    query_vec = [0.1] * 384
    result = mock_vector_store.query(query_vec, similarity_top_k=3)
    assert len(result.nodes) == 3

    # Delete
    mock_vector_store.delete(ids[0])
    assert len(mock_vector_store._nodes) == 4
```

#### `mock_db_connection`
Mocked psycopg2 database connection.

```python
def test_database_operations(mock_db_connection):
    cursor = mock_db_connection.cursor()
    cursor.execute("SELECT * FROM embeddings")
    rows = cursor.fetchall()
    assert isinstance(rows, list)
```

### 3. Test Data Fixtures

#### `sample_text`
Various text samples for testing.

```python
def test_text_processing(sample_text):
    short = sample_text["short"]
    html = sample_text["html"]
    code = sample_text["code"]

    assert len(short) < 100
    assert "<html>" in html
    assert "def " in code
```

#### `sample_pdf_path`
Temporary test PDF file.

```python
def test_pdf_loading(sample_pdf_path):
    assert sample_pdf_path.exists()
    assert sample_pdf_path.suffix == ".pdf"
```

#### `test_settings`
Mock Settings object with test values.

```python
def test_configuration(test_settings):
    assert test_settings.chunk_size == 500
    assert test_settings.embed_dim == 384
    assert test_settings.db_name == "test_vector_db"
```

#### `sample_metadata`
Sample metadata dictionaries.

```python
def test_metadata_handling(sample_metadata):
    pdf_meta = sample_metadata["pdf"]
    html_meta = sample_metadata["html"]

    assert "source" in pdf_meta
    assert "url" in html_meta
```

### 4. Cleanup Fixtures

#### `temp_dir`
Temporary directory (auto-cleaned).

```python
def test_file_operations(temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("content")
    assert test_file.exists()
    # Automatically cleaned up after test
```

#### `clean_env`
Clean environment variables.

```python
def test_with_clean_env(clean_env):
    import os
    # Environment is pre-set with test defaults
    assert os.getenv("CHUNK_SIZE") == "500"

    # You can modify for testing
    os.environ["CHUNK_SIZE"] = "1000"
```

#### `mock_query_logs_dir`
Temporary query logs directory.

```python
def test_logging(mock_query_logs_dir):
    log_file = mock_query_logs_dir / "query.json"
    log_file.write_text('{"query": "test"}')
    assert log_file.exists()
```

### 5. Parametrized Fixtures

These fixtures run tests multiple times with different values.

```python
def test_different_chunk_sizes(chunk_sizes):
    """Runs 4 times: 100, 500, 1000, 2000"""
    assert chunk_sizes in [100, 500, 1000, 2000]

def test_different_overlaps(chunk_overlaps):
    """Runs 4 times: 0, 50, 100, 200"""
    assert chunk_overlaps >= 0

def test_different_models(embed_models):
    """Runs 2 times with different embedding models"""
    assert "/" in embed_models
```

### 6. Session-Scoped Fixtures

Created once per test session for efficiency.

#### `test_data_dir`
Path to test data directory.

```python
def test_data_location(test_data_dir):
    assert test_data_dir.exists()
    assert test_data_dir.name == "test_data"
```

#### `sample_embeddings_cache`
Cached embeddings for faster tests.

```python
def test_cached_embeddings(sample_embeddings_cache):
    emb1 = sample_embeddings_cache("text1")
    emb2 = sample_embeddings_cache("text1")  # Uses cache
    assert emb1 == emb2
```

## Helper Functions

### `create_test_index`
Populate vector store with test data.

```python
def test_bulk_indexing(mock_vector_store, mock_text_node):
    from tests.conftest import create_test_index

    nodes = create_test_index(
        mock_vector_store,
        num_docs=10,
        chunks_per_doc=5,
        mock_text_node=mock_text_node
    )

    assert len(nodes) == 50  # 10 docs * 5 chunks
    assert len(mock_vector_store._nodes) == 50
```

## Complete Example: End-to-End RAG Test

```python
def test_complete_rag_pipeline(
    mock_document,
    mock_text_node,
    mock_embed_model,
    mock_vector_store,
    mock_llm,
):
    """Test complete RAG flow with mocked components."""

    # 1. Create document
    doc = mock_document(text="Document about RAG systems")

    # 2. Create chunks
    chunks = [
        mock_text_node(text=f"Chunk {i}: {doc.text[:30]}")
        for i in range(3)
    ]

    # 3. Generate embeddings
    for chunk in chunks:
        chunk.embedding = mock_embed_model.get_text_embedding(chunk.text)

    # 4. Index to vector store
    mock_vector_store.add(chunks)

    # 5. Query
    query = "What is RAG?"
    query_emb = mock_embed_model.get_text_embedding(query)
    results = mock_vector_store.query(query_emb, similarity_top_k=2)

    # 6. Generate response
    context = results.nodes[0].text
    response = mock_llm.complete(f"Answer based on: {context}")

    # Verify
    assert len(mock_vector_store._nodes) == 3
    assert len(results.nodes) == 2
    assert response.text is not None
```

## Best Practices

### 1. Use Fixtures Instead of Setup/Teardown

```python
# Bad
class TestMyFeature:
    def setup_method(self):
        self.store = create_mock_store()

    def teardown_method(self):
        self.store.cleanup()

# Good
def test_my_feature(mock_vector_store):
    # Store is automatically created and cleaned up
    pass
```

### 2. Combine Fixtures for Complex Tests

```python
def test_complex_scenario(
    mock_embed_model,
    mock_vector_store,
    mock_text_node,
    temp_dir,
):
    """Multiple fixtures work together seamlessly."""
    pass
```

### 3. Use Parametrized Fixtures for Comprehensive Testing

```python
def test_all_chunk_sizes(chunk_sizes, chunk_overlaps):
    """Tests all combinations of chunk sizes and overlaps."""
    # Runs 4 * 4 = 16 times
    assert chunk_overlaps < chunk_sizes
```

### 4. Scope Fixtures Appropriately

- **Function-scoped** (default): New instance per test
- **Module-scoped**: Shared across module tests
- **Session-scoped**: Created once for entire test session

```python
@pytest.fixture(scope="module")
def expensive_setup():
    """Created once per test module."""
    return setup_expensive_resource()
```

## Testing Tips

### Run Specific Fixture Tests

```bash
# All fixture tests
pytest tests/test_fixtures.py -v

# Specific test class
pytest tests/test_fixtures.py::TestMockEmbedding -v

# Specific test
pytest tests/test_fixtures.py::TestMockEmbedding::test_embedding_dimension -v
```

### Debug Fixture Issues

```bash
# Show fixture setup/teardown
pytest tests/test_my_feature.py --setup-show

# List available fixtures
pytest tests/ --fixtures
```

### Check Fixture Coverage

```bash
# Run with coverage
pytest tests/test_fixtures.py --cov=tests.conftest --cov-report=html
```

## Common Patterns

### Pattern 1: Test with Real and Mock Components

```python
def test_embedding_consistency(mock_text_node):
    """Test that mock embeddings are consistent."""
    node1 = mock_text_node(text="test", embedding=None)
    node2 = mock_text_node(text="test", embedding=None)

    # Same text should produce same embedding
    assert node1.embedding == node2.embedding
```

### Pattern 2: Test Configuration Variations

```python
def test_with_custom_config(test_settings, clean_env):
    """Test with modified configuration."""
    import os

    # Start with clean defaults
    assert os.getenv("CHUNK_SIZE") == "500"

    # Modify for test
    os.environ["CHUNK_SIZE"] = "1000"
    test_settings.chunk_size = 1000

    # Test with new config
    assert test_settings.chunk_size == 1000
```

### Pattern 3: Test Error Handling

```python
def test_error_handling(mock_vector_store):
    """Test error cases with mocks."""
    # Simulate error condition
    mock_vector_store.query = Mock(side_effect=Exception("Connection error"))

    with pytest.raises(Exception, match="Connection error"):
        mock_vector_store.query([0.1] * 384)
```

## Fixture Reference

| Fixture | Type | Scope | Description |
|---------|------|-------|-------------|
| `mock_embedding` | Factory | function | Generate embedding vectors |
| `mock_document` | Factory | function | Create Document objects |
| `mock_text_node` | Factory | function | Create TextNode with embeddings |
| `mock_query_result` | Factory | function | Create query results |
| `mock_embed_model` | Service | function | Mock embedding model |
| `mock_llm` | Service | function | Mock LLM |
| `mock_vector_store` | Service | function | Mock vector store |
| `mock_db_connection` | Service | function | Mock DB connection |
| `sample_text` | Data | function | Sample text strings |
| `sample_pdf_path` | Data | function | Test PDF file |
| `test_settings` | Data | function | Test configuration |
| `sample_metadata` | Data | function | Sample metadata |
| `temp_dir` | Cleanup | function | Temporary directory |
| `clean_env` | Cleanup | function | Clean environment |
| `mock_query_logs_dir` | Cleanup | function | Query logs directory |
| `chunk_sizes` | Parametrized | function | Chunk size variations |
| `chunk_overlaps` | Parametrized | function | Overlap variations |
| `embed_models` | Parametrized | function | Model variations |
| `test_data_dir` | Data | session | Test data directory |
| `sample_embeddings_cache` | Data | session | Cached embeddings |

## Further Reading

- [pytest fixtures documentation](https://docs.pytest.org/en/stable/explanation/fixtures.html)
- [conftest.py source code](/Users/frytos/code/llamaIndex-local-rag/tests/conftest.py)
- [test_fixtures.py examples](/Users/frytos/code/llamaIndex-local-rag/tests/test_fixtures.py)
