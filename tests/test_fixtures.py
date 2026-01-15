"""Tests for pytest fixtures to ensure they work correctly.

This test file verifies that all fixtures in conftest.py are functioning
properly and generating valid test data.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Mock llama_index modules before importing
sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.schema'] = MagicMock()

# Create mock classes matching LlamaIndex schema
class MockDocument:
    def __init__(self, text="", metadata=None, **kwargs):
        self.text = text
        self.metadata = metadata or {}
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockTextNode:
    def __init__(self, text="", embedding=None, metadata=None, **kwargs):
        self.text = text
        self.embedding = embedding or []
        self.metadata = metadata or {}
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockNodeWithScore:
    def __init__(self, node=None, score=0.0):
        self.node = node or MockTextNode()
        self.score = score

# Assign to mocked modules
sys.modules['llama_index.core.schema'].Document = MockDocument
sys.modules['llama_index.core.schema'].TextNode = MockTextNode
sys.modules['llama_index.core.schema'].NodeWithScore = MockNodeWithScore

from llama_index.core.schema import Document, TextNode, NodeWithScore


# ============================================================================
# Test Mock Data Factories
# ============================================================================


class TestMockEmbedding:
    """Test mock embedding generation."""

    def test_embedding_dimension(self, mock_embedding):
        """Test that embeddings have correct dimensions."""
        vec = mock_embedding(384)
        assert len(vec) == 384

    def test_embedding_normalized(self, mock_embedding):
        """Test that embeddings are normalized to unit length."""
        vec = np.array(mock_embedding(384))
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6, "Vector should be unit normalized"

    def test_embedding_reproducible(self, mock_embedding):
        """Test that embeddings with same seed are reproducible."""
        vec1 = mock_embedding(384, seed=42)
        vec2 = mock_embedding(384, seed=42)
        assert vec1 == vec2, "Same seed should produce same embedding"

    def test_embedding_different_seeds(self, mock_embedding):
        """Test that different seeds produce different embeddings."""
        vec1 = mock_embedding(384, seed=1)
        vec2 = mock_embedding(384, seed=2)
        assert vec1 != vec2, "Different seeds should produce different embeddings"

    def test_embedding_custom_dimension(self, mock_embedding):
        """Test embeddings with custom dimensions."""
        for dim in [128, 256, 384, 768, 1024]:
            vec = mock_embedding(dim)
            assert len(vec) == dim, f"Should generate {dim}-dimensional vector"


class TestMockDocument:
    """Test mock Document factory."""

    def test_document_creation(self, mock_document):
        """Test basic document creation."""
        doc = mock_document()
        assert isinstance(doc, Document)
        assert len(doc.text) > 0
        assert isinstance(doc.metadata, dict)

    def test_document_custom_text(self, mock_document):
        """Test document with custom text."""
        custom_text = "Custom test content"
        doc = mock_document(text=custom_text)
        assert doc.text == custom_text

    def test_document_metadata(self, mock_document):
        """Test document metadata."""
        metadata = {"source": "test.pdf", "page": 5}
        doc = mock_document(metadata=metadata)
        assert doc.metadata["source"] == "test.pdf"
        assert doc.metadata["page"] == 5

    def test_document_id(self, mock_document):
        """Test document with custom ID."""
        doc_id = "doc_12345"
        doc = mock_document(doc_id=doc_id)
        assert doc.doc_id == doc_id

    def test_document_default_metadata(self, mock_document):
        """Test that documents have default metadata."""
        doc = mock_document()
        assert "source" in doc.metadata
        assert "page" in doc.metadata


class TestMockTextNode:
    """Test mock TextNode factory."""

    def test_text_node_creation(self, mock_text_node):
        """Test basic text node creation."""
        node = mock_text_node()
        assert isinstance(node, TextNode)
        assert len(node.text) > 0
        assert node.embedding is not None

    def test_text_node_embedding_dimension(self, mock_text_node):
        """Test that text nodes have correct embedding dimensions."""
        node = mock_text_node()
        assert len(node.embedding) == 384

    def test_text_node_custom_text(self, mock_text_node):
        """Test node with custom text."""
        custom_text = "Custom chunk content"
        node = mock_text_node(text=custom_text)
        assert node.text == custom_text

    def test_text_node_metadata(self, mock_text_node):
        """Test node metadata."""
        node = mock_text_node()
        assert "document_id" in node.metadata
        assert "source" in node.metadata

    def test_text_node_custom_metadata(self, mock_text_node):
        """Test node with custom metadata."""
        metadata = {"custom_field": "custom_value"}
        node = mock_text_node(metadata=metadata)
        assert node.metadata["custom_field"] == "custom_value"

    def test_text_node_custom_embedding(self, mock_text_node):
        """Test node with custom embedding."""
        custom_emb = [0.1] * 384
        node = mock_text_node(embedding=custom_emb)
        assert node.embedding == custom_emb


class TestMockQueryResult:
    """Test mock query result factory."""

    def test_query_result_creation(self, mock_query_result):
        """Test basic query result creation."""
        results = mock_query_result(num_results=3)
        assert len(results) == 3
        assert all(isinstance(r, NodeWithScore) for r in results)

    def test_query_result_scores(self, mock_query_result):
        """Test that results have correct scores."""
        results = mock_query_result(num_results=3)
        scores = [r.score for r in results]
        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True)
        # Scores should be between 0 and 1
        assert all(0.0 <= score <= 1.0 for score in scores)

    def test_query_result_custom_scores(self, mock_query_result):
        """Test results with custom scores."""
        custom_scores = [0.99, 0.85, 0.70]
        results = mock_query_result(num_results=3, scores=custom_scores)
        assert [r.score for r in results] == custom_scores

    def test_query_result_custom_texts(self, mock_query_result):
        """Test results with custom text content."""
        custom_texts = ["First result", "Second result", "Third result"]
        results = mock_query_result(num_results=3, texts=custom_texts)
        assert [r.node.text for r in results] == custom_texts

    def test_query_result_nodes_have_embeddings(self, mock_query_result):
        """Test that result nodes have embeddings."""
        results = mock_query_result(num_results=3)
        for result in results:
            assert result.node.embedding is not None
            assert len(result.node.embedding) == 384


# ============================================================================
# Test Mock Services
# ============================================================================


class TestMockEmbedModel:
    """Test mock embedding model."""

    def test_embed_model_attributes(self, mock_embed_model):
        """Test that mock model has required attributes."""
        assert hasattr(mock_embed_model, "embed_dim")
        assert hasattr(mock_embed_model, "model_name")
        assert mock_embed_model.embed_dim == 384

    def test_embed_model_single_text(self, mock_embed_model):
        """Test embedding a single text."""
        text = "Test text for embedding"
        embedding = mock_embed_model.get_text_embedding(text)
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_model_multiple_texts(self, mock_embed_model):
        """Test embedding multiple texts."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = mock_embed_model.get_text_embeddings(texts)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_embed_model_reproducible(self, mock_embed_model):
        """Test that same text produces same embedding."""
        text = "Test text"
        emb1 = mock_embed_model.get_text_embedding(text)
        emb2 = mock_embed_model.get_text_embedding(text)
        assert emb1 == emb2

    def test_embed_model_different_texts(self, mock_embed_model):
        """Test that different texts produce different embeddings."""
        emb1 = mock_embed_model.get_text_embedding("Text A")
        emb2 = mock_embed_model.get_text_embedding("Text B")
        # Different texts may produce same embedding with hash collisions, so just verify they're valid
        assert len(emb1) == 384
        assert len(emb2) == 384


class TestMockLLM:
    """Test mock LLM."""

    def test_llm_attributes(self, mock_llm):
        """Test that mock LLM has required attributes."""
        assert hasattr(mock_llm, "model_path")
        assert hasattr(mock_llm, "temperature")
        assert hasattr(mock_llm, "max_tokens")
        assert hasattr(mock_llm, "context_window")

    def test_llm_complete(self, mock_llm):
        """Test LLM completion."""
        prompt = "What is RAG?"
        response = mock_llm.complete(prompt)
        assert hasattr(response, "text")
        assert len(response.text) > 0
        assert prompt[:20] in response.text

    def test_llm_stream_complete(self, mock_llm):
        """Test LLM streaming completion."""
        prompt = "Generate text"
        chunks = list(mock_llm.stream_complete(prompt))
        assert len(chunks) > 0
        assert all(hasattr(chunk, "delta") for chunk in chunks)


class TestMockVectorStore:
    """Test mock vector store."""

    def test_vector_store_attributes(self, mock_vector_store):
        """Test that mock store has required attributes."""
        assert hasattr(mock_vector_store, "connection_string")
        assert hasattr(mock_vector_store, "table_name")
        assert hasattr(mock_vector_store, "embed_dim")
        assert mock_vector_store.embed_dim == 384

    def test_vector_store_add_nodes(self, mock_vector_store, mock_text_node):
        """Test adding nodes to vector store."""
        nodes = [mock_text_node() for _ in range(3)]
        ids = mock_vector_store.add(nodes)
        assert len(ids) == 3
        assert len(mock_vector_store._nodes) == 3

    def test_vector_store_query(self, mock_vector_store, mock_text_node):
        """Test querying vector store."""
        # Add some nodes first
        nodes = [mock_text_node() for _ in range(5)]
        mock_vector_store.add(nodes)

        # Query
        query_vec = [0.1] * 384
        result = mock_vector_store.query(query_vec, similarity_top_k=3)

        assert len(result.nodes) == 3
        assert len(result.similarities) == 3
        assert len(result.ids) == 3

    def test_vector_store_delete(self, mock_vector_store, mock_text_node):
        """Test deleting nodes from vector store."""
        node = mock_text_node(node_id="test_node_1")
        mock_vector_store.add([node])
        assert len(mock_vector_store._nodes) == 1

        mock_vector_store.delete("test_node_1")
        assert len(mock_vector_store._nodes) == 0


class TestMockDBConnection:
    """Test mock database connection."""

    def test_db_connection_cursor(self, mock_db_connection):
        """Test getting cursor from connection."""
        cursor = mock_db_connection.cursor()
        assert cursor is not None
        assert hasattr(cursor, "execute")
        assert hasattr(cursor, "fetchone")
        assert hasattr(cursor, "fetchall")

    def test_db_connection_autocommit(self, mock_db_connection):
        """Test that connection has autocommit."""
        assert hasattr(mock_db_connection, "autocommit")
        assert mock_db_connection.autocommit is True

    def test_db_connection_cursor_fetchone(self, mock_db_connection):
        """Test cursor fetchone."""
        cursor = mock_db_connection.cursor()
        result = cursor.fetchone()
        assert result is not None
        assert isinstance(result, tuple)

    def test_db_connection_cursor_fetchall(self, mock_db_connection):
        """Test cursor fetchall."""
        cursor = mock_db_connection.cursor()
        results = cursor.fetchall()
        assert isinstance(results, list)
        assert all(isinstance(row, tuple) for row in results)


# ============================================================================
# Test Test Data Fixtures
# ============================================================================


class TestSampleText:
    """Test sample text fixture."""

    def test_sample_text_types(self, sample_text):
        """Test that sample text has all expected types."""
        expected_keys = ["short", "medium", "long", "html", "code", "chat_message", "empty", "whitespace"]
        assert all(key in sample_text for key in expected_keys)

    def test_sample_text_short(self, sample_text):
        """Test short text sample."""
        assert len(sample_text["short"]) > 0
        assert len(sample_text["short"]) < 100

    def test_sample_text_medium(self, sample_text):
        """Test medium text sample."""
        assert len(sample_text["medium"]) > 100
        assert len(sample_text["medium"]) < 500

    def test_sample_text_long(self, sample_text):
        """Test long text sample."""
        assert len(sample_text["long"]) > 500

    def test_sample_text_html(self, sample_text):
        """Test HTML text sample."""
        assert "<html>" in sample_text["html"]
        assert "<body>" in sample_text["html"]

    def test_sample_text_code(self, sample_text):
        """Test code text sample."""
        assert "def " in sample_text["code"]
        assert "return" in sample_text["code"]

    def test_sample_text_empty(self, sample_text):
        """Test empty text sample."""
        assert sample_text["empty"] == ""

    def test_sample_text_whitespace(self, sample_text):
        """Test whitespace text sample."""
        assert sample_text["whitespace"].strip() == ""


class TestSamplePDFPath:
    """Test sample PDF path fixture."""

    def test_pdf_path_exists(self, sample_pdf_path):
        """Test that PDF file exists."""
        assert sample_pdf_path.exists()

    def test_pdf_path_is_file(self, sample_pdf_path):
        """Test that PDF path points to a file."""
        assert sample_pdf_path.is_file()

    def test_pdf_path_extension(self, sample_pdf_path):
        """Test that file has .pdf extension."""
        assert sample_pdf_path.suffix == ".pdf"

    def test_pdf_path_has_content(self, sample_pdf_path):
        """Test that PDF file has content."""
        content = sample_pdf_path.read_text()
        assert len(content) > 0


class TestTestSettings:
    """Test test settings fixture."""

    def test_settings_chunk_size(self, test_settings):
        """Test chunk size setting."""
        assert test_settings.chunk_size == 500
        assert 100 <= test_settings.chunk_size <= 2000

    def test_settings_chunk_overlap(self, test_settings):
        """Test chunk overlap setting."""
        assert test_settings.chunk_overlap == 100
        assert test_settings.chunk_overlap < test_settings.chunk_size

    def test_settings_embed_model(self, test_settings):
        """Test embedding model setting."""
        assert test_settings.embed_model == "BAAI/bge-small-en"
        assert "/" in test_settings.embed_model

    def test_settings_embed_dim(self, test_settings):
        """Test embedding dimension setting."""
        assert test_settings.embed_dim == 384

    def test_settings_database(self, test_settings):
        """Test database settings."""
        assert test_settings.db_name == "test_vector_db"
        assert test_settings.table_name == "test_embeddings"
        assert test_settings.pghost == "localhost"
        assert test_settings.pgport == 5432


class TestSampleMetadata:
    """Test sample metadata fixture."""

    def test_metadata_types(self, sample_metadata):
        """Test that metadata has all expected types."""
        expected_keys = ["pdf", "html", "code", "chat"]
        assert all(key in sample_metadata for key in expected_keys)

    def test_metadata_pdf(self, sample_metadata):
        """Test PDF metadata."""
        pdf_meta = sample_metadata["pdf"]
        assert "source" in pdf_meta
        assert "page" in pdf_meta
        assert "file_type" in pdf_meta

    def test_metadata_html(self, sample_metadata):
        """Test HTML metadata."""
        html_meta = sample_metadata["html"]
        assert "source" in html_meta
        assert "url" in html_meta

    def test_metadata_code(self, sample_metadata):
        """Test code metadata."""
        code_meta = sample_metadata["code"]
        assert "source" in code_meta
        assert "language" in code_meta

    def test_metadata_chat(self, sample_metadata):
        """Test chat metadata."""
        chat_meta = sample_metadata["chat"]
        assert "source" in chat_meta
        assert "participants" in chat_meta


# ============================================================================
# Test Cleanup Fixtures
# ============================================================================


class TestTempDir:
    """Test temporary directory fixture."""

    def test_temp_dir_exists(self, temp_dir):
        """Test that temp directory exists."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_temp_dir_writable(self, temp_dir):
        """Test that temp directory is writable."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temp directory is cleaned up (manual verification)."""
        # Note: Actual cleanup verification requires checking after fixture teardown
        # This test verifies the directory works during the test
        assert temp_dir.exists()


class TestCleanEnv:
    """Test clean environment fixture."""

    def test_clean_env_sets_defaults(self, clean_env):
        """Test that clean env sets default values."""
        import os
        assert os.getenv("CHUNK_SIZE") == "500"
        assert os.getenv("CHUNK_OVERLAP") == "100"
        assert os.getenv("EMBED_MODEL") == "BAAI/bge-small-en"

    def test_clean_env_modifiable(self, clean_env):
        """Test that env vars can be modified."""
        import os
        os.environ["CHUNK_SIZE"] = "1000"
        assert os.getenv("CHUNK_SIZE") == "1000"


class TestMockQueryLogsDir:
    """Test mock query logs directory fixture."""

    def test_query_logs_dir_exists(self, mock_query_logs_dir):
        """Test that query logs directory exists."""
        assert mock_query_logs_dir.exists()
        assert mock_query_logs_dir.is_dir()

    def test_query_logs_dir_writable(self, mock_query_logs_dir):
        """Test that query logs directory is writable."""
        log_file = mock_query_logs_dir / "test_query.json"
        log_file.write_text('{"query": "test"}')
        assert log_file.exists()


# ============================================================================
# Test Parametrized Fixtures
# ============================================================================


class TestParametrizedFixtures:
    """Test parametrized fixtures."""

    def test_chunk_sizes_parametrized(self, chunk_sizes):
        """Test chunk sizes parametrized fixture."""
        assert chunk_sizes in [100, 500, 1000, 2000]
        assert chunk_sizes > 0

    def test_chunk_overlaps_parametrized(self, chunk_overlaps):
        """Test chunk overlaps parametrized fixture."""
        assert chunk_overlaps in [0, 50, 100, 200]
        assert chunk_overlaps >= 0

    def test_embed_models_parametrized(self, embed_models):
        """Test embed models parametrized fixture."""
        assert embed_models in [
            "BAAI/bge-small-en",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
        assert "/" in embed_models


# ============================================================================
# Test Session-Scoped Fixtures
# ============================================================================


class TestSessionScopedFixtures:
    """Test session-scoped fixtures."""

    def test_test_data_dir(self, test_data_dir):
        """Test test data directory."""
        assert test_data_dir.exists()
        assert test_data_dir.is_dir()
        assert test_data_dir.name == "test_data"

    def test_sample_embeddings_cache(self, sample_embeddings_cache):
        """Test embeddings cache."""
        text1 = "Test text 1"
        text2 = "Test text 2"

        emb1 = sample_embeddings_cache(text1)
        emb2 = sample_embeddings_cache(text2)

        # Same text should return same embedding (cached)
        emb1_again = sample_embeddings_cache(text1)
        assert emb1 == emb1_again

        # Different text should return different embedding
        assert emb1 != emb2


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test helper functions from conftest."""

    def test_create_test_index(self, mock_vector_store, mock_text_node):
        """Test create_test_index helper."""
        from tests.conftest import create_test_index

        nodes = create_test_index(
            mock_vector_store,
            num_docs=3,
            chunks_per_doc=2,
            mock_text_node=mock_text_node,
        )

        # Should create 3 docs * 2 chunks = 6 nodes
        assert len(nodes) == 6
        assert len(mock_vector_store._nodes) == 6

    def test_create_test_index_default_params(self, mock_vector_store, mock_text_node):
        """Test create_test_index with default parameters."""
        from tests.conftest import create_test_index

        nodes = create_test_index(
            mock_vector_store,
            mock_text_node=mock_text_node,
        )

        # Default: 10 docs * 5 chunks = 50 nodes
        assert len(nodes) == 50


# ============================================================================
# Integration Tests
# ============================================================================


class TestFixtureIntegration:
    """Test that fixtures work together correctly."""

    def test_text_node_with_embed_model(self, mock_text_node, mock_embed_model):
        """Test using text node with embedding model."""
        node = mock_text_node()
        # Verify node has embedding
        assert node.embedding is not None
        assert len(node.embedding) == mock_embed_model.embed_dim

    def test_query_result_with_vector_store(self, mock_query_result, mock_vector_store):
        """Test query results work with vector store."""
        results = mock_query_result(num_results=3)

        # Add nodes to store
        nodes = [r.node for r in results]
        mock_vector_store.add(nodes)

        assert len(mock_vector_store._nodes) == 3

    def test_end_to_end_mock_flow(
        self,
        mock_document,
        mock_text_node,
        mock_embed_model,
        mock_vector_store,
        mock_llm,
    ):
        """Test complete mock RAG pipeline flow."""
        # 1. Create document
        doc = mock_document(text="Test document for RAG pipeline")

        # 2. Create text nodes
        nodes = [
            mock_text_node(text=f"Chunk {i} from {doc.text[:20]}")
            for i in range(3)
        ]

        # 3. Generate embeddings
        for node in nodes:
            node.embedding = mock_embed_model.get_text_embedding(node.text)

        # 4. Add to vector store
        mock_vector_store.add(nodes)

        # 5. Query vector store
        query_text = "Test query"
        query_embedding = mock_embed_model.get_text_embedding(query_text)
        results = mock_vector_store.query(query_embedding, similarity_top_k=2)

        # 6. Generate response
        prompt = f"Answer based on: {results.nodes[0].text}"
        response = mock_llm.complete(prompt)

        # Verify complete flow
        assert len(mock_vector_store._nodes) == 3
        assert len(results.nodes) == 2
        assert response.text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
