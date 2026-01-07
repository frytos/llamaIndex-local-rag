"""
Comprehensive tests for RAG pipeline improvements.

This test suite validates all new improvements:
1. Reranking (utils/reranker.py)
2. Semantic Caching (utils/query_cache.py)
3. Query Expansion (utils/query_expansion.py)
4. Metadata Extraction (utils/metadata_extractor.py)

Tests cover:
- Individual module functionality
- Environment variable controls
- Graceful degradation when dependencies missing
- Integration between modules
- Edge cases and error handling

Usage:
    # Run all tests
    pytest tests/test_rag_improvements.py -v

    # Run specific test class
    pytest tests/test_rag_improvements.py::TestReranker -v

    # Run with coverage
    pytest tests/test_rag_improvements.py --cov=utils --cov-report=html
"""

import os
import sys
import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_node_with_score():
    """Create mock NodeWithScore object for testing."""
    def _create_node(text: str, score: float = 0.5):
        node = Mock()
        node.node = Mock()
        node.node.get_content = Mock(return_value=text)
        node.score = score
        return node
    return _create_node


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Elena mentioned she wants to visit Morocco next summer for the food.",
        "The weather in Morocco is nice this time of year.",
        "I love Moroccan food, especially tagine and couscous.",
        "Elena and I discussed travel plans yesterday, including Morocco.",
        "Morocco has beautiful architecture and vibrant markets.",
    ]


@pytest.fixture
def sample_embedding():
    """Create a sample 384-dim embedding vector."""
    def _create_embedding(seed: int = 42):
        np.random.seed(seed)
        vec = np.random.randn(384)
        return vec / np.linalg.norm(vec)  # Normalize
    return _create_embedding


# ============================================================================
# Test Reranker (utils/reranker.py)
# ============================================================================

class TestReranker:
    """Test cross-encoder reranking functionality."""

    def test_reranker_initialization(self):
        """Test Reranker can be initialized."""
        try:
            from utils.reranker import Reranker, CROSSENCODER_AVAILABLE

            if not CROSSENCODER_AVAILABLE:
                pytest.skip("sentence-transformers not available")

            reranker = Reranker()

            assert reranker is not None
            assert hasattr(reranker, 'model')
            assert hasattr(reranker, 'model_name')
            assert 'cross-encoder' in reranker.model_name.lower()

        except ImportError as e:
            pytest.fail(f"Failed to import Reranker: {e}")

    def test_reranker_with_custom_model(self):
        """Test Reranker initialization with custom model name."""
        try:
            from utils.reranker import Reranker, CROSSENCODER_AVAILABLE

            if not CROSSENCODER_AVAILABLE:
                pytest.skip("sentence-transformers not available")

            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            reranker = Reranker(model_name=model_name)

            assert reranker.model_name == model_name

        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_reranker_rerank_texts(self, sample_texts):
        """Test reranking text list."""
        try:
            from utils.reranker import Reranker, CROSSENCODER_AVAILABLE

            if not CROSSENCODER_AVAILABLE:
                pytest.skip("sentence-transformers not available")

            reranker = Reranker()
            query = "What did Elena say about Morocco?"

            results = reranker.rerank(query, sample_texts, top_n=2)

            # Check results
            assert len(results) == 2, "Should return top 2 results"
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "Results should be (index, score) tuples"

            # Check indices are valid
            for idx, score in results:
                assert 0 <= idx < len(sample_texts), f"Index {idx} out of range"
                assert isinstance(score, (float, np.floating)), f"Score should be float, got {type(score)}"

            # Check scores are descending
            scores = [score for _, score in results]
            assert scores[0] >= scores[1], "Scores should be in descending order"

            # First result should be most relevant (mentions Elena and Morocco)
            first_idx = results[0][0]
            first_text = sample_texts[first_idx]
            assert "elena" in first_text.lower(), "Top result should mention Elena"

        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_reranker_rerank_with_texts(self, sample_texts):
        """Test rerank_with_texts returns texts instead of indices."""
        try:
            from utils.reranker import Reranker, CROSSENCODER_AVAILABLE

            if not CROSSENCODER_AVAILABLE:
                pytest.skip("sentence-transformers not available")

            reranker = Reranker()
            query = "What did Elena say about Morocco?"

            results = reranker.rerank_with_texts(query, sample_texts, top_n=3)

            # Check results
            assert len(results) == 3, "Should return top 3 results"
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "Results should be (text, score) tuples"

            # Check texts are from original list
            for text, score in results:
                assert text in sample_texts, "Text should be from original list"
                assert isinstance(score, (float, np.floating)), "Score should be float"

        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_reranker_rerank_nodes(self, mock_node_with_score, sample_texts):
        """Test reranking NodeWithScore objects."""
        try:
            from utils.reranker import Reranker, CROSSENCODER_AVAILABLE

            if not CROSSENCODER_AVAILABLE:
                pytest.skip("sentence-transformers not available")

            reranker = Reranker()
            query = "What did Elena say about Morocco?"

            # Create mock nodes
            nodes = [mock_node_with_score(text, 0.5) for text in sample_texts]
            original_scores = [node.score for node in nodes]

            # Rerank
            reranked = reranker.rerank_nodes(query, nodes, top_k=3)

            # Check results
            assert len(reranked) == 3, "Should return top 3 nodes"
            assert all(hasattr(n, 'node') and hasattr(n, 'score') for n in reranked), "Should be NodeWithScore objects"

            # Check scores were updated (rerank scores should be different from original)
            reranked_scores = [node.score for node in reranked]
            assert reranked_scores != original_scores[:3], "Scores should be updated with rerank scores"

            # Check scores are descending
            for i in range(len(reranked_scores) - 1):
                assert reranked_scores[i] >= reranked_scores[i+1], "Scores should be in descending order"

        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_reranker_empty_input(self):
        """Test reranker with empty input."""
        try:
            from utils.reranker import Reranker, CROSSENCODER_AVAILABLE

            if not CROSSENCODER_AVAILABLE:
                pytest.skip("sentence-transformers not available")

            reranker = Reranker()
            query = "test query"

            # Empty list should return empty list
            results = reranker.rerank(query, [], top_n=3)
            assert results == [], "Empty input should return empty list"

            # Empty nodes should return empty list
            reranked = reranker.rerank_nodes(query, [], top_k=3)
            assert reranked == [], "Empty nodes should return empty list"

        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_reranker_graceful_degradation(self):
        """Test graceful degradation when sentence-transformers not available."""
        from utils.reranker import CROSSENCODER_AVAILABLE

        if CROSSENCODER_AVAILABLE:
            pytest.skip("sentence-transformers is available, cannot test degradation")

        # Should raise ImportError when trying to initialize
        with pytest.raises(ImportError, match="sentence-transformers not installed"):
            from utils.reranker import Reranker
            Reranker()


# ============================================================================
# Test Semantic Caching (utils/query_cache.py)
# ============================================================================

class TestQueryCache:
    """Test basic query cache (exact match)."""

    def test_query_cache_initialization(self, temp_cache_dir):
        """Test QueryCache initialization."""
        from utils.query_cache import QueryCache

        cache = QueryCache(cache_dir=temp_cache_dir)

        assert cache is not None
        assert cache.cache_dir == Path(temp_cache_dir)
        assert cache.cache_dir.exists()

    def test_query_cache_miss(self, temp_cache_dir):
        """Test cache miss returns None."""
        from utils.query_cache import QueryCache

        cache = QueryCache(cache_dir=temp_cache_dir)

        result = cache.get("test query", "test-model")
        assert result is None, "Cache miss should return None"

    def test_query_cache_set_and_get(self, temp_cache_dir):
        """Test caching and retrieving embedding."""
        from utils.query_cache import QueryCache

        cache = QueryCache(cache_dir=temp_cache_dir)

        query = "What is machine learning?"
        model = "bge-small-en"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Cache the embedding
        cache.set(query, model, embedding)

        # Retrieve it
        result = cache.get(query, model)

        assert result == embedding, "Cached embedding should match original"

    def test_query_cache_different_models(self, temp_cache_dir):
        """Test same query with different models are cached separately."""
        from utils.query_cache import QueryCache

        cache = QueryCache(cache_dir=temp_cache_dir)

        query = "test query"
        model1 = "model-1"
        model2 = "model-2"
        embedding1 = [1.0, 2.0]
        embedding2 = [3.0, 4.0]

        # Cache with two models
        cache.set(query, model1, embedding1)
        cache.set(query, model2, embedding2)

        # Retrieve and verify
        result1 = cache.get(query, model1)
        result2 = cache.get(query, model2)

        assert result1 == embedding1, "Model 1 embedding should match"
        assert result2 == embedding2, "Model 2 embedding should match"
        assert result1 != result2, "Embeddings should be different"

    def test_query_cache_stats(self, temp_cache_dir):
        """Test cache statistics."""
        from utils.query_cache import QueryCache

        cache = QueryCache(cache_dir=temp_cache_dir)

        # Cache some embeddings
        for i in range(5):
            cache.set(f"query {i}", "test-model", [float(i)])

        stats = cache.stats()

        assert stats['count'] == 5, "Should have 5 cached entries"
        assert stats['size_bytes'] > 0, "Should have non-zero size"
        assert 'cache_dir' in stats

    def test_query_cache_clear(self, temp_cache_dir):
        """Test clearing cache."""
        from utils.query_cache import QueryCache

        cache = QueryCache(cache_dir=temp_cache_dir)

        # Cache some embeddings
        for i in range(3):
            cache.set(f"query {i}", "test-model", [float(i)])

        assert cache.stats()['count'] == 3, "Should have 3 entries before clear"

        # Clear cache
        cache.clear()

        assert cache.stats()['count'] == 0, "Should have 0 entries after clear"


class TestSemanticQueryCache:
    """Test semantic query cache (similarity-based)."""

    def test_semantic_cache_initialization(self, temp_cache_dir):
        """Test SemanticQueryCache initialization."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            max_size=100,
            ttl=3600,
            cache_dir=temp_cache_dir,
        )

        assert cache is not None
        assert cache.similarity_threshold == 0.92
        assert cache.max_size == 100
        assert cache.ttl == 3600
        assert cache.enabled is True

    def test_semantic_cache_miss(self, temp_cache_dir, sample_embedding):
        """Test semantic cache miss."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(cache_dir=temp_cache_dir)

        query = "What is AI?"
        embedding = sample_embedding(42)

        result = cache.get_semantic(query, embedding)
        assert result is None, "Cache miss should return None"

    def test_semantic_cache_exact_match(self, temp_cache_dir, sample_embedding):
        """Test semantic cache with exact query match."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(cache_dir=temp_cache_dir)

        query = "What is machine learning?"
        embedding = sample_embedding(42)
        response = {"answer": "ML is a subset of AI...", "confidence": 0.95}

        # Cache the response
        cache.set_semantic(query, embedding, response)

        # Retrieve with same query and embedding
        result = cache.get_semantic(query, embedding)

        assert result is not None, "Should find cached result"
        assert result['answer'] == response['answer']
        assert result['confidence'] == response['confidence']

    def test_semantic_cache_similar_query(self, temp_cache_dir, sample_embedding):
        """Test semantic cache with similar query (above threshold)."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(
            similarity_threshold=0.90,
            cache_dir=temp_cache_dir
        )

        query1 = "What is machine learning?"
        query2 = "What is ML?"  # Similar question

        # Create similar embeddings (high similarity)
        # Use a smaller noise factor to ensure high similarity
        emb1 = sample_embedding(42)
        emb2 = emb1 + np.random.randn(384) * 0.01  # Very small noise
        emb2 = emb2 / np.linalg.norm(emb2)  # Renormalize

        # Verify similarity is above threshold
        similarity = float(np.dot(emb1, emb2))

        # If similarity is too low, use a copy with minimal noise
        if similarity < 0.90:
            # Use almost identical embedding
            emb2 = emb1 * 0.99 + np.random.randn(384) * 0.001
            emb2 = emb2 / np.linalg.norm(emb2)
            similarity = float(np.dot(emb1, emb2))

        assert similarity >= 0.90, f"Similarity {similarity} should be >= 0.90"

        # Cache first query
        response = {"answer": "Machine learning is...", "sources": ["doc1.pdf"]}
        cache.set_semantic(query1, emb1, response)

        # Try to retrieve with similar query
        result = cache.get_semantic(query2, emb2)

        assert result is not None, "Should find similar cached query"
        assert result['answer'] == response['answer']

    def test_semantic_cache_dissimilar_query(self, temp_cache_dir, sample_embedding):
        """Test semantic cache miss with dissimilar query."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(
            similarity_threshold=0.92,
            cache_dir=temp_cache_dir
        )

        query1 = "What is machine learning?"
        query2 = "How do I cook pasta?"  # Completely different

        emb1 = sample_embedding(42)
        emb2 = sample_embedding(99)  # Different seed = different embedding

        # Verify similarity is below threshold
        similarity = float(np.dot(emb1, emb2))
        assert similarity < 0.92, f"Similarity {similarity} should be < 0.92"

        # Cache first query
        response = {"answer": "ML is...", "confidence": 0.95}
        cache.set_semantic(query1, emb1, response)

        # Try to retrieve with dissimilar query
        result = cache.get_semantic(query2, emb2)

        assert result is None, "Should not find dissimilar query in cache"

    def test_semantic_cache_lru_eviction(self, temp_cache_dir, sample_embedding):
        """Test LRU eviction when max_size reached."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(
            max_size=3,
            cache_dir=temp_cache_dir
        )

        # Fill cache to max size
        for i in range(3):
            query = f"Query {i}"
            embedding = sample_embedding(100 + i)
            response = {"answer": f"Answer {i}"}
            cache.set_semantic(query, embedding, response)

        assert len(cache.cache) == 3, "Cache should be at max size"
        assert cache.evictions == 0, "No evictions yet"

        # Add one more (should trigger eviction)
        query_new = "Query that triggers eviction"
        embedding_new = sample_embedding(200)
        response_new = {"answer": "New answer"}
        cache.set_semantic(query_new, embedding_new, response_new)

        assert len(cache.cache) == 3, "Cache should still be at max size"
        assert cache.evictions == 1, "Should have 1 eviction"

    def test_semantic_cache_statistics(self, temp_cache_dir, sample_embedding):
        """Test cache statistics tracking."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(cache_dir=temp_cache_dir)

        query = "test query"
        embedding = sample_embedding(42)
        response = {"answer": "test answer"}

        # Cache a response
        cache.set_semantic(query, embedding, response)

        # Test cache hit
        result = cache.get_semantic(query, embedding)
        assert result is not None

        # Test cache miss
        result = cache.get_semantic("different query", sample_embedding(99))
        assert result is None

        # Check statistics
        stats = cache.stats()

        assert stats['count'] == 1, "Should have 1 cached entry"
        assert stats['hits'] == 1, "Should have 1 hit"
        assert stats['misses'] == 1, "Should have 1 miss"
        assert stats['hit_rate'] == 0.5, "Hit rate should be 50%"
        assert stats['evictions'] == 0, "Should have 0 evictions"

    def test_semantic_cache_ttl_expiration(self, temp_cache_dir, sample_embedding):
        """Test TTL-based expiration."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(
            ttl=1,  # 1 second TTL
            cache_dir=temp_cache_dir
        )

        query = "test query"
        embedding = sample_embedding(42)
        response = {"answer": "test answer"}

        # Cache a response
        cache.set_semantic(query, embedding, response)

        # Should be available immediately
        result = cache.get_semantic(query, embedding)
        assert result is not None, "Should find cached result immediately"

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        result = cache.get_semantic(query, embedding)
        assert result is None, "Should not find expired cache entry"

    def test_semantic_cache_persistence(self, temp_cache_dir, sample_embedding):
        """Test cache persistence across instances."""
        from utils.query_cache import SemanticQueryCache

        query = "test query"
        embedding = sample_embedding(42)
        response = {"answer": "test answer", "confidence": 0.95}

        # Create first instance and cache
        cache1 = SemanticQueryCache(cache_dir=temp_cache_dir)
        cache1.set_semantic(query, embedding, response)
        assert len(cache1.cache) == 1

        # Create new instance (should load from disk)
        cache2 = SemanticQueryCache(cache_dir=temp_cache_dir)
        assert len(cache2.cache) == 1, "Should load cached entry from disk"

        # Verify cached data
        result = cache2.get_semantic(query, embedding)
        assert result is not None
        assert result['answer'] == response['answer']

    def test_semantic_cache_disabled(self, temp_cache_dir, sample_embedding):
        """Test disabled cache."""
        from utils.query_cache import SemanticQueryCache

        # Create disabled cache
        cache = SemanticQueryCache(cache_dir=temp_cache_dir)
        cache.enabled = False

        query = "test query"
        embedding = sample_embedding(42)
        response = {"answer": "test"}

        # Try to cache (should do nothing)
        cache.set_semantic(query, embedding, response)
        assert len(cache.cache) == 0, "Disabled cache should not store"

        # Try to retrieve (should return None)
        result = cache.get_semantic(query, embedding)
        assert result is None, "Disabled cache should always return None"

    def test_semantic_cache_clear(self, temp_cache_dir, sample_embedding):
        """Test clearing semantic cache."""
        from utils.query_cache import SemanticQueryCache

        cache = SemanticQueryCache(cache_dir=temp_cache_dir)

        # Cache some entries
        for i in range(5):
            cache.set_semantic(
                f"query {i}",
                sample_embedding(100 + i),
                {"answer": f"answer {i}"}
            )

        assert len(cache.cache) == 5, "Should have 5 entries"

        # Clear cache
        cache.clear()

        assert len(cache.cache) == 0, "Cache should be empty"
        assert cache.stats()['count'] == 0


# ============================================================================
# Test Query Expansion (utils/query_expansion.py)
# ============================================================================

class TestQueryExpansion:
    """Test query expansion functionality."""

    def test_query_expander_initialization(self):
        """Test QueryExpander initialization."""
        from utils.query_expansion import QueryExpander

        expander = QueryExpander(method="keyword", expansion_count=2)

        assert expander is not None
        assert expander.method == "keyword"
        assert expander.expansion_count == 2

    def test_query_expander_keyword_method(self):
        """Test keyword-based expansion (no LLM needed)."""
        from utils.query_expansion import QueryExpander

        expander = QueryExpander(method="keyword", expansion_count=2)

        query = "What did Elena say about Morocco?"
        result = expander.expand(query)

        assert result.original == query
        assert result.method == "keyword"
        assert isinstance(result.expanded_queries, list)
        assert len(result.expanded_queries) <= 2
        assert 'metadata' in result.__dict__

    def test_query_expander_empty_query(self):
        """Test expansion with empty query."""
        from utils.query_expansion import QueryExpander

        expander = QueryExpander(method="keyword")

        result = expander.expand("")

        assert result.original == ""
        assert result.expanded_queries == []
        assert 'error' in result.metadata

    def test_query_expander_with_weights(self):
        """Test expand_with_weights method."""
        from utils.query_expansion import QueryExpander

        expander = QueryExpander(method="keyword", expansion_count=2)

        query = "What did Elena say about Morocco?"
        weights = expander.expand_with_weights(query)

        assert isinstance(weights, dict)
        assert query in weights, "Original query should be in weights"
        assert weights[query] == 1.0, "Original query should have weight 1.0"

        # All weights should be between 0.5 and 1.0
        for q, w in weights.items():
            assert 0.5 <= w <= 1.0, f"Weight {w} should be between 0.5 and 1.0"

    def test_query_expander_invalid_method(self):
        """Test initialization with invalid method (should fallback to llm)."""
        from utils.query_expansion import QueryExpander

        expander = QueryExpander(method="invalid_method")

        # Should fallback to "llm"
        assert expander.method == "llm"

    @patch.dict(os.environ, {"QUERY_EXPANSION_METHOD": "keyword", "QUERY_EXPANSION_COUNT": "3"})
    def test_query_expander_env_vars(self):
        """Test QueryExpander respects environment variables."""
        from utils.query_expansion import QueryExpander

        expander = QueryExpander()

        assert expander.method == "keyword"
        assert expander.expansion_count == 3

    def test_is_enabled_function(self):
        """Test is_enabled() function."""
        from utils.query_expansion import is_enabled

        # Test with environment variable
        with patch.dict(os.environ, {"ENABLE_QUERY_EXPANSION": "1"}):
            assert is_enabled() is True

        with patch.dict(os.environ, {"ENABLE_QUERY_EXPANSION": "0"}):
            assert is_enabled() is False


# ============================================================================
# Test Metadata Extraction (utils/metadata_extractor.py)
# ============================================================================

class TestMetadataExtraction:
    """Test enhanced metadata extraction."""

    def test_metadata_extractor_initialization(self):
        """Test DocumentMetadataExtractor initialization."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        assert extractor is not None
        assert hasattr(extractor, 'enabled')
        assert hasattr(extractor, 'extract_topics')
        assert hasattr(extractor, 'extract_entities')

    def test_extract_structure_metadata(self):
        """Test structure metadata extraction."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        text = """
# Introduction

This is a tutorial about Python programming.

## Step 1: Installation

First, install Python from python.org.

## Step 2: Hello World

Create your first program.
"""

        metadata = extractor.extract_structure_metadata(text, doc_format="md")

        assert metadata['format'] == "md"
        assert metadata['has_headings'] is True
        assert metadata['heading_count'] == 3
        assert 'doc_type' in metadata
        assert metadata['doc_type'] in ['tutorial', 'manual', 'general']

    def test_extract_semantic_metadata(self):
        """Test semantic metadata extraction."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        text = """
Python is a high-level programming language used for web development,
data science, and machine learning. Python frameworks like Django and
Flask are popular for building web applications. Python is also used
with tools like PyTorch and TensorFlow for deep learning.
"""

        metadata = extractor.extract_semantic_metadata(text)

        assert 'keywords' in metadata
        assert isinstance(metadata['keywords'], list)
        assert len(metadata['keywords']) > 0

        # Should detect Python as entity
        if 'entities' in metadata:
            entities_lower = [e.lower() for e in metadata['entities']]
            assert 'python' in entities_lower or 'django' in entities_lower

    def test_extract_technical_metadata_code(self):
        """Test technical metadata extraction for code."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        text = """
import numpy as np
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        return self.predict(X)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            self.forward(X)

def main():
    model = NeuralNetwork([784, 128, 10])
    print("Model initialized")
"""

        metadata = extractor.extract_technical_metadata(text, doc_format="py")

        assert metadata['has_code'] is True
        assert metadata['programming_language'] == "py"
        assert 'functions' in metadata
        assert 'classes' in metadata
        assert 'imports' in metadata

        # Check detected elements
        assert 'NeuralNetwork' in metadata['classes']
        assert 'main' in metadata['functions'] or 'forward' in metadata['functions']

    def test_extract_technical_metadata_tables(self):
        """Test detection of tables."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        text = """
Here are the results:

| Dataset | Accuracy | Time |
|---------|----------|------|
| CIFAR-10| 95.2%    | 2.5h |
| ImageNet| 87.3%    | 12h  |
"""

        metadata = extractor.extract_technical_metadata(text, doc_format="md")

        assert metadata['has_tables'] is True
        assert metadata['table_count'] > 0

    def test_extract_technical_metadata_equations(self):
        """Test detection of math equations."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        text = """
The learning rate is computed as:

$$\\alpha_t = \\alpha_0 \\cdot \\sqrt{1 - \\beta_2^t} / (1 - \\beta_1^t)$$

And the inline equation $E = mc^2$ shows energy-mass equivalence.
"""

        metadata = extractor.extract_technical_metadata(text, doc_format="md")

        assert metadata['has_equations'] is True
        assert metadata['equation_count'] >= 2

    def test_extract_quality_signals(self):
        """Test quality signals extraction."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        text = """
This is a test document. It has multiple sentences.
Each sentence is relatively short and easy to read.
The reading level should be moderate.
"""

        metadata = extractor.extract_quality_signals(text)

        assert 'word_count' in metadata
        assert 'sentence_count' in metadata
        assert 'avg_sentence_length' in metadata
        assert 'reading_level' in metadata
        assert 'char_count' in metadata

        assert metadata['word_count'] > 0
        assert metadata['sentence_count'] > 0
        assert metadata['reading_level'] in ['very_easy', 'easy', 'moderate', 'difficult', 'very_difficult']

    def test_extract_all_metadata(self):
        """Test extracting all metadata types."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        text = """
# Python Tutorial

Learn Python programming with examples.

## Introduction

Python is a versatile language used in web development and data science.

```python
def hello():
    print("Hello, World!")
```
"""

        metadata = extractor.extract_all_metadata(text, doc_format="md")

        # Should have all metadata types
        assert hasattr(metadata, 'structure')
        assert hasattr(metadata, 'semantic')
        assert hasattr(metadata, 'technical')
        assert hasattr(metadata, 'quality')

        # Structure should detect headings
        assert metadata.structure['has_headings'] is True

        # Technical should detect code
        assert metadata.technical['has_code'] is True

        # Quality should have metrics
        assert metadata.quality['word_count'] > 0

    def test_metadata_to_dict(self):
        """Test converting DocumentMetadata to flat dict."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        text = "# Test\n\nThis is a test document about Python programming."
        metadata = extractor.extract_all_metadata(text, doc_format="md")

        flat_dict = metadata.to_dict()

        assert isinstance(flat_dict, dict)
        # Check prefixes are applied
        assert any(k.startswith('struct_') for k in flat_dict.keys())
        assert any(k.startswith('qual_') for k in flat_dict.keys())

    def test_enhance_node_metadata_function(self):
        """Test enhance_node_metadata convenience function."""
        from utils.metadata_extractor import enhance_node_metadata

        text = "# Introduction\n\nThis is a test document."
        base_metadata = {"source": "test.md", "format": "md"}

        enhanced = enhance_node_metadata(text, base_metadata)

        assert isinstance(enhanced, dict)
        assert enhanced['source'] == "test.md"
        assert enhanced['format'] == "md"
        # Should have added metadata
        assert len(enhanced) > len(base_metadata)

    @patch.dict(os.environ, {"EXTRACT_ENHANCED_METADATA": "0"})
    def test_metadata_extraction_disabled(self):
        """Test metadata extraction when disabled."""
        from utils.metadata_extractor import DocumentMetadataExtractor

        extractor = DocumentMetadataExtractor()

        assert extractor.enabled is False

        text = "# Test\n\nSome content."
        metadata = extractor.extract_all_metadata(text, doc_format="md")

        # Should return empty metadata
        assert metadata.structure == {}
        assert metadata.semantic == {}
        assert metadata.technical == {}
        assert metadata.quality == {}


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration between modules."""

    def test_all_modules_import(self):
        """Test all modules can be imported successfully."""
        try:
            from utils import reranker
            from utils import query_cache
            from utils import query_expansion
            from utils import metadata_extractor

            assert reranker is not None
            assert query_cache is not None
            assert query_expansion is not None
            assert metadata_extractor is not None

        except ImportError as e:
            pytest.fail(f"Failed to import modules: {e}")

    def test_environment_variable_controls(self):
        """Test environment variables control features correctly."""

        # Test query expansion
        from utils.query_expansion import is_enabled as qe_enabled

        with patch.dict(os.environ, {"ENABLE_QUERY_EXPANSION": "1"}):
            assert qe_enabled() is True

        with patch.dict(os.environ, {"ENABLE_QUERY_EXPANSION": "0"}):
            assert qe_enabled() is False

        # Test semantic cache
        from utils.query_cache import SemanticQueryCache

        with patch.dict(os.environ, {"ENABLE_SEMANTIC_CACHE": "1"}):
            cache = SemanticQueryCache(cache_dir=tempfile.mkdtemp())
            assert cache.enabled is True

        with patch.dict(os.environ, {"ENABLE_SEMANTIC_CACHE": "0"}):
            cache = SemanticQueryCache(cache_dir=tempfile.mkdtemp())
            assert cache.enabled is False

    def test_combined_workflow(self, temp_cache_dir, sample_embedding):
        """Test combining multiple improvements in a workflow."""
        from utils.query_cache import SemanticQueryCache
        from utils.query_expansion import QueryExpander
        from utils.metadata_extractor import DocumentMetadataExtractor

        # Initialize components
        cache = SemanticQueryCache(cache_dir=temp_cache_dir)
        expander = QueryExpander(method="keyword", expansion_count=2)
        extractor = DocumentMetadataExtractor()

        # Test workflow
        query = "What is Python?"
        query_embedding = sample_embedding(42)

        # 1. Check cache
        cached = cache.get_semantic(query, query_embedding)
        assert cached is None, "Cache should be empty initially"

        # 2. Expand query
        expansion = expander.expand(query)
        assert expansion.original == query
        assert isinstance(expansion.expanded_queries, list)

        # 3. Extract metadata
        text = "# Python Tutorial\n\nPython is a programming language."
        metadata = extractor.extract_all_metadata(text, doc_format="md")
        assert metadata.structure['has_headings'] is True

        # 4. Cache response
        response = {"answer": "Python is a programming language", "metadata": metadata.to_dict()}
        cache.set_semantic(query, query_embedding, response)

        # 5. Verify cached
        cached = cache.get_semantic(query, query_embedding)
        assert cached is not None
        assert cached['answer'] == response['answer']


# ============================================================================
# Run tests with coverage
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=utils.reranker",
        "--cov=utils.query_cache",
        "--cov=utils.query_expansion",
        "--cov=utils.metadata_extractor",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
    ])
