# RAG Improvements Test Suite

## Overview

Comprehensive test suite for RAG pipeline improvements located in `tests/test_rag_improvements.py`.

This test file validates four new modules:
1. **Reranking** (`utils/reranker.py`) - Cross-encoder reranking for better relevance
2. **Semantic Caching** (`utils/query_cache.py`) - Fast query result caching
3. **Query Expansion** (`utils/query_expansion.py`) - Query reformulation for better recall
4. **Metadata Extraction** (`utils/metadata_extractor.py`) - Rich document metadata

## Test Statistics

- **Total Tests**: 45
- **Test Classes**: 6 (TestReranker, TestQueryCache, TestSemanticQueryCache, TestQueryExpansion, TestMetadataExtraction, TestIntegration)
- **Code Coverage**: 
  - `query_cache.py`: 87.88%
  - `metadata_extractor.py`: 47.54%
  - `query_expansion.py`: 42.53%
  - `reranker.py`: 15.93% (requires sentence-transformers)

## Running Tests

### Basic Usage

```bash
# Run all tests with verbose output
pytest tests/test_rag_improvements.py -v

# Run with coverage report
pytest tests/test_rag_improvements.py --cov=utils --cov-report=html

# Run specific test class
pytest tests/test_rag_improvements.py::TestSemanticQueryCache -v

# Run specific test
pytest tests/test_rag_improvements.py::TestQueryCache::test_query_cache_initialization -v
```

### Filtered Test Runs

```bash
# Run only cache tests
pytest tests/test_rag_improvements.py -k "cache" -v

# Run only metadata tests
pytest tests/test_rag_improvements.py -k "metadata" -v

# Skip reranker tests (requires sentence-transformers)
pytest tests/test_rag_improvements.py -k "not reranker or graceful" -v

# Run integration tests only
pytest tests/test_rag_improvements.py::TestIntegration -v
```

### Advanced Options

```bash
# Stop at first failure
pytest tests/test_rag_improvements.py -x

# Show local variables on failure
pytest tests/test_rag_improvements.py -l

# Run with detailed output
pytest tests/test_rag_improvements.py -vv

# Run in parallel (requires pytest-xdist)
pytest tests/test_rag_improvements.py -n auto
```

## Test Categories

### 1. Reranker Tests (7 tests)

Tests cross-encoder reranking functionality:
- Initialization with default/custom models
- Reranking text lists and NodeWithScore objects
- Score updating and ranking
- Empty input handling
- Graceful degradation without dependencies

**Note**: 6 tests require `sentence-transformers` package. Install with:
```bash
pip install sentence-transformers
```

### 2. Query Cache Tests (6 tests)

Tests exact-match query caching:
- Cache initialization and directory management
- Cache hits and misses
- Multiple model support
- Statistics tracking
- Cache clearing

### 3. Semantic Query Cache Tests (11 tests)

Tests similarity-based query caching:
- Initialization with configurable threshold/max_size/TTL
- Exact and semantic similarity matching
- LRU eviction policy
- TTL expiration
- Disk persistence across sessions
- Enable/disable functionality
- Statistics (hits, misses, hit rate)

### 4. Query Expansion Tests (7 tests)

Tests query expansion/reformulation:
- Multiple expansion methods (llm, multi, keyword)
- Keyword-based expansion (no LLM required)
- Weighted query results
- Empty/invalid input handling
- Environment variable configuration

### 5. Metadata Extraction Tests (11 tests)

Tests enhanced metadata extraction:
- Structure metadata (headings, doc type)
- Semantic metadata (keywords, topics, entities)
- Technical metadata (code blocks, tables, equations)
- Quality signals (word count, reading level)
- Feature toggles and disabled behavior
- Integration with TextNode metadata

### 6. Integration Tests (3 tests)

Tests module integration:
- All modules import successfully
- Environment variables control features
- Combined workflow (cache + expansion + metadata)

## Test Fixtures

The test suite uses several pytest fixtures:

- `temp_cache_dir`: Temporary directory for cache testing
- `mock_node_with_score`: Mock NodeWithScore objects
- `sample_texts`: Sample text data for testing
- `sample_embedding`: Generate test embedding vectors

## Environment Variables

Tests validate environment variable controls:

```bash
# Query expansion
ENABLE_QUERY_EXPANSION=1
QUERY_EXPANSION_METHOD=llm|multi|keyword
QUERY_EXPANSION_COUNT=2

# Semantic caching
ENABLE_SEMANTIC_CACHE=1
SEMANTIC_CACHE_THRESHOLD=0.92
SEMANTIC_CACHE_MAX_SIZE=1000
SEMANTIC_CACHE_TTL=86400

# Metadata extraction
EXTRACT_ENHANCED_METADATA=1
EXTRACT_TOPICS=1
EXTRACT_ENTITIES=1
EXTRACT_CODE_BLOCKS=1
EXTRACT_TABLES=1

# Reranking
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Troubleshooting

### Reranker Tests Skipped

**Issue**: Tests show "sentence-transformers not available"

**Solution**: Install the package:
```bash
pip install sentence-transformers
```

### Coverage Too Low

**Issue**: Coverage report shows low overall percentage

**Solution**: Run focused coverage on tested modules:
```bash
pytest tests/test_rag_improvements.py \
  --cov=utils.reranker \
  --cov=utils.query_cache \
  --cov=utils.query_expansion \
  --cov=utils.metadata_extractor \
  --cov-report=term-missing
```

### Test Failures

**Issue**: Tests fail with import errors

**Solution**: Ensure you're running from project root:
```bash
cd /path/to/llamaIndex-local-rag
pytest tests/test_rag_improvements.py -v
```

### Slow Tests

**Issue**: Tests take too long

**Solution**: Skip LLM-based tests or run in parallel:
```bash
# Skip slow tests
pytest tests/test_rag_improvements.py -m "not slow" -v

# Run in parallel
pytest tests/test_rag_improvements.py -n auto
```

## Continuous Integration

Add to CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run RAG Improvements Tests
  run: |
    pytest tests/test_rag_improvements.py \
      --cov=utils \
      --cov-report=xml \
      --junitxml=junit.xml \
      -v
```

## Adding New Tests

When adding new features to RAG improvements:

1. Add test class following naming convention: `TestFeatureName`
2. Use descriptive test names: `test_feature_specific_behavior`
3. Include docstrings explaining what is tested
4. Use fixtures for common setup
5. Test both success and failure cases
6. Test environment variable controls
7. Test graceful degradation

Example:
```python
class TestNewFeature:
    """Test new feature functionality."""

    def test_feature_initialization(self):
        """Test feature can be initialized."""
        from utils.new_feature import NewFeature
        
        feature = NewFeature()
        assert feature is not None
        assert feature.enabled is True
```

## Related Documentation

- Implementation: `utils/reranker.py`, `utils/query_cache.py`, etc.
- Usage guides: `docs/SEMANTIC_CACHE_GUIDE.md`, `docs/METADATA_EXTRACTOR.md`
- Main test suite: `tests/test_e2e_pipeline.py`

## Contact

For questions or issues with tests:
1. Check test output for specific error messages
2. Review module documentation in `utils/`
3. Examine test implementation in `test_rag_improvements.py`
