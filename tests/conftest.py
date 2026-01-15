"""Pytest shared fixtures for RAG pipeline testing.

This module provides reusable test fixtures including:
- Mock data factories (documents, nodes, embeddings)
- Mock services (embed models, LLMs, vector stores)
- Test data fixtures (sample text, paths, settings)
- Cleanup fixtures (temp directories, environment)

Usage:
    Fixtures are automatically available in all test files.
    Import conftest is not needed - pytest discovers it automatically.

Example:
    def test_something(mock_text_node, mock_embed_model):
        # Fixtures are injected automatically
        node = mock_text_node()
        embedding = mock_embed_model.get_text_embedding("test")
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
import numpy as np


# ============================================================================
# Mock Data Factories
# ============================================================================


@pytest.fixture
def mock_embedding():
    """Generate fake 384-dimensional embedding vectors.

    Returns:
        Callable that generates random embeddings

    Example:
        >>> emb_factory = mock_embedding
        >>> vec = emb_factory()
        >>> len(vec)
        384
    """

    def _generate_embedding(dim: int = 384, seed: Optional[int] = None) -> List[float]:
        """Generate a random embedding vector.

        Args:
            dim: Dimension of embedding (default: 384 for bge-small-en)
            seed: Random seed for reproducibility

        Returns:
            List of floats representing embedding
        """
        if seed is not None:
            np.random.seed(seed)
        # Generate random vector and normalize to unit length
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    return _generate_embedding


@pytest.fixture
def mock_document():
    """Create test Document objects.

    Returns:
        Callable that creates Document objects with customizable content

    Example:
        >>> doc_factory = mock_document
        >>> doc = doc_factory(text="Test content", metadata={"source": "test.pdf"})
    """
    from llama_index.core.schema import Document

    def _create_document(
        text: str = "Sample document text for testing purposes.",
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        """Create a Document object.

        Args:
            text: Document text content
            metadata: Optional metadata dict
            doc_id: Optional document ID

        Returns:
            Document object
        """
        if metadata is None:
            metadata = {
                "source": "test_document.pdf",
                "page": 1,
            }

        return Document(text=text, metadata=metadata, doc_id=doc_id or f"doc_{hash(text) % 1000}")

    return _create_document


@pytest.fixture
def mock_text_node(mock_embedding):
    """Create test TextNode objects with embeddings.

    Returns:
        Callable that creates TextNode objects

    Example:
        >>> node_factory = mock_text_node
        >>> node = node_factory(text="Test chunk")
        >>> len(node.embedding)
        384
    """
    from llama_index.core.schema import TextNode

    def _create_text_node(
        text: str = "Sample text chunk for testing.",
        doc_id: str = "test_doc_1",
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> TextNode:
        """Create a TextNode with embedding.

        Args:
            text: Node text content
            doc_id: Parent document ID
            node_id: Optional node ID
            metadata: Optional metadata dict
            embedding: Optional embedding vector (generated if None)

        Returns:
            TextNode object with embedding
        """
        if metadata is None:
            metadata = {
                "document_id": doc_id,
                "chunk_index": 0,
                "source": "test_document.pdf",
            }

        if embedding is None:
            embedding = mock_embedding(384, seed=42)

        node = TextNode(text=text, metadata=metadata, id_=node_id or f"node_{hash(text) % 1000}")
        node.embedding = embedding

        return node

    return _create_text_node


@pytest.fixture
def mock_query_result(mock_text_node):
    """Create mock retrieval results.

    Returns:
        Callable that creates list of NodeWithScore objects

    Example:
        >>> results_factory = mock_query_result
        >>> results = results_factory(num_results=3, scores=[0.95, 0.88, 0.75])
    """
    from llama_index.core.schema import NodeWithScore

    def _create_query_result(
        num_results: int = 3,
        scores: Optional[List[float]] = None,
        texts: Optional[List[str]] = None,
    ) -> List[NodeWithScore]:
        """Create mock query results.

        Args:
            num_results: Number of results to generate
            scores: Optional similarity scores (generated if None)
            texts: Optional text content for nodes

        Returns:
            List of NodeWithScore objects
        """
        if scores is None:
            # Generate decreasing scores from 0.95 to 0.7
            scores = [0.95 - (i * 0.08) for i in range(num_results)]

        if texts is None:
            texts = [
                f"Retrieved chunk {i+1} with relevant content for testing."
                for i in range(num_results)
            ]

        results = []
        for i in range(num_results):
            node = mock_text_node(
                text=texts[i] if i < len(texts) else f"Chunk {i+1}",
                doc_id=f"doc_{i+1}",
                node_id=f"node_{i+1}",
            )
            results.append(NodeWithScore(node=node, score=scores[i]))

        return results

    return _create_query_result


# ============================================================================
# Mock Services
# ============================================================================


@pytest.fixture
def mock_embed_model(mock_embedding):
    """Create a mocked HuggingFaceEmbedding model.

    Returns:
        Mock embedding model with get_text_embedding method

    Example:
        >>> model = mock_embed_model
        >>> embedding = model.get_text_embedding("test text")
        >>> len(embedding)
        384
    """
    mock_model = MagicMock()
    mock_model.embed_dim = 384
    mock_model.model_name = "BAAI/bge-small-en"

    def get_text_embedding(text: str) -> List[float]:
        """Generate embedding for text."""
        # Use text hash as seed for reproducibility
        seed = hash(text) % (2**32)
        return mock_embedding(384, seed=seed)

    def get_text_embeddings(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [get_text_embedding(text) for text in texts]

    mock_model.get_text_embedding = get_text_embedding
    mock_model.get_text_embeddings = get_text_embeddings

    return mock_model


@pytest.fixture
def mock_llm():
    """Create a mocked LlamaCPP LLM.

    Returns:
        Mock LLM with complete and stream methods

    Example:
        >>> llm = mock_llm()
        >>> response = llm.complete("What is RAG?")
        >>> response.text
        'Mock response: What is RAG?'
    """
    mock_llm_instance = MagicMock()
    mock_llm_instance.model_path = "/path/to/model.gguf"
    mock_llm_instance.temperature = 0.1
    mock_llm_instance.max_tokens = 256
    mock_llm_instance.context_window = 8192

    def complete(prompt: str, **kwargs) -> MagicMock:
        """Mock completion."""
        response = MagicMock()
        response.text = f"Mock response: {prompt[:50]}"
        response.additional_kwargs = {}
        return response

    def stream_complete(prompt: str, **kwargs):
        """Mock streaming completion."""
        for chunk in ["Mock ", "streaming ", "response"]:
            response = MagicMock()
            response.delta = chunk
            yield response

    mock_llm_instance.complete = complete
    mock_llm_instance.stream_complete = stream_complete

    return mock_llm_instance


@pytest.fixture
def mock_vector_store():
    """Create a mocked PGVectorStore.

    Returns:
        Mock vector store with add, query, and delete methods

    Example:
        >>> store = mock_vector_store()
        >>> store.add([node1, node2])
        >>> results = store.query(query_vec, similarity_top_k=3)
    """
    mock_store = MagicMock()
    mock_store.connection_string = "postgresql://localhost/test_db"
    mock_store.table_name = "test_embeddings"
    mock_store.embed_dim = 384
    mock_store.hybrid_search = False
    mock_store.text_search_config = "english"

    # Internal storage for testing
    mock_store._nodes = []

    def add(nodes: List) -> List[str]:
        """Mock add nodes."""
        mock_store._nodes.extend(nodes)
        return [node.node_id or f"node_{i}" for i, node in enumerate(nodes)]

    def query(
        query_embedding: List[float], similarity_top_k: int = 4, **kwargs
    ) -> MagicMock:
        """Mock query."""
        result = MagicMock()
        result.nodes = mock_store._nodes[:similarity_top_k]
        result.similarities = [0.95 - (i * 0.1) for i in range(similarity_top_k)]
        result.ids = [f"node_{i}" for i in range(similarity_top_k)]
        return result

    def delete(node_id: str) -> None:
        """Mock delete."""
        mock_store._nodes = [n for n in mock_store._nodes if n.node_id != node_id]

    mock_store.add = add
    mock_store.query = query
    mock_store.delete = delete

    return mock_store


@pytest.fixture
def mock_db_connection():
    """Create a mocked psycopg2 database connection.

    Returns:
        Mock connection with cursor, commit, rollback methods

    Example:
        >>> conn = mock_db_connection()
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT * FROM embeddings")
        >>> rows = cursor.fetchall()
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Mock cursor methods
    mock_cursor.fetchone.return_value = (1, "test_row")
    mock_cursor.fetchall.return_value = [(1, "row1"), (2, "row2")]
    mock_cursor.rowcount = 2
    mock_cursor.description = [("id",), ("data",)]

    mock_conn.cursor.return_value = mock_cursor
    mock_conn.commit.return_value = None
    mock_conn.rollback.return_value = None
    mock_conn.close.return_value = None
    mock_conn.autocommit = True

    return mock_conn


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_text():
    """Provide various sample text strings for testing.

    Returns:
        Dict with different types of sample text

    Example:
        >>> texts = sample_text()
        >>> texts["short"]
        'This is a short text.'
    """
    return {
        "short": "This is a short text for testing.",
        "medium": (
            "This is a medium-length text sample that contains multiple sentences. "
            "It provides enough content for testing chunking and embedding operations. "
            "The text should be long enough to demonstrate typical document processing."
        ),
        "long": (
            "This is a longer text sample designed for comprehensive testing. " * 20
            + "It contains repeated content to simulate real document processing scenarios. "
            + "This helps verify that the system can handle larger text inputs correctly. "
        ),
        "html": (
            "<html><body><h1>Title</h1><p>This is HTML content with <b>bold</b> "
            "and <i>italic</i> text.</p></body></html>"
        ),
        "code": (
            "def example_function(x: int) -> int:\n"
            '    """Example function for testing."""\n'
            "    return x * 2\n"
        ),
        "chat_message": "Alice: Hello, how are you?\nBob: I'm doing great, thanks!",
        "empty": "",
        "whitespace": "   \n\t   \n   ",
    }


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary test PDF file.

    Returns:
        Path to temporary PDF file

    Example:
        >>> pdf_path = sample_pdf_path()
        >>> pdf_path.exists()
        True
    """
    # Create a simple text file to simulate PDF
    # (actual PDF creation requires PyMuPDF, kept simple for testing)
    pdf_file = tmp_path / "test_document.pdf"
    pdf_file.write_text("Sample PDF content for testing purposes.")
    return pdf_file


@pytest.fixture
def test_settings():
    """Provide test configuration settings.

    Returns:
        Mock Settings object with test values

    Example:
        >>> settings = test_settings()
        >>> settings.chunk_size
        500
    """
    settings = MagicMock()
    settings.chunk_size = 500
    settings.chunk_overlap = 100
    settings.embed_model = "BAAI/bge-small-en"
    settings.embed_dim = 384
    settings.embed_batch = 32
    settings.top_k = 3
    settings.context_window = 4096
    settings.max_new_tokens = 256
    settings.temperature = 0.1
    settings.db_name = "test_vector_db"
    settings.table_name = "test_embeddings"
    settings.pghost = "localhost"
    settings.pgport = 5432
    settings.reset_table = False
    settings.hybrid_search = False
    settings.hybrid_alpha = 1.0
    return settings


@pytest.fixture
def sample_metadata():
    """Provide sample metadata dictionaries.

    Returns:
        Dict with various metadata examples

    Example:
        >>> metadata = sample_metadata()
        >>> metadata["pdf"]["source"]
        'document.pdf'
    """
    return {
        "pdf": {
            "source": "document.pdf",
            "page": 1,
            "file_type": "pdf",
            "created_at": "2024-01-01T00:00:00",
        },
        "html": {
            "source": "webpage.html",
            "url": "https://example.com/page",
            "file_type": "html",
            "title": "Example Page",
        },
        "code": {
            "source": "script.py",
            "language": "python",
            "file_type": "code",
            "lines": "1-50",
        },
        "chat": {
            "source": "messages.json",
            "conversation_id": "conv_123",
            "file_type": "chat",
            "participants": ["Alice", "Bob"],
        },
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files.

    Automatically cleaned up after test completes.

    Yields:
        Path to temporary directory

    Example:
        >>> def test_file_operations(temp_dir):
        ...     test_file = temp_dir / "test.txt"
        ...     test_file.write_text("content")
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def clean_env(monkeypatch):
    """Provide clean environment variables for tests.

    Sets test defaults and cleans up after test.

    Args:
        monkeypatch: pytest monkeypatch fixture

    Example:
        >>> def test_config(clean_env):
        ...     os.environ["CHUNK_SIZE"] = "600"
        ...     # Test runs with clean env
    """
    # Save original environment
    original_env = os.environ.copy()

    # Set test defaults
    test_env_vars = {
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "100",
        "EMBED_MODEL": "BAAI/bge-small-en",
        "EMBED_DIM": "384",
        "EMBED_BATCH": "32",
        "TOP_K": "3",
        "CTX": "4096",
        "MAX_NEW_TOKENS": "256",
        "TEMP": "0.1",
        "DB_NAME": "test_vector_db",
        "PGTABLE": "test_embeddings",
        "PGHOST": "localhost",
        "PGPORT": "5432",
        "RESET_TABLE": "0",
    }

    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)

    yield monkeypatch

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_query_logs_dir(tmp_path):
    """Create temporary query logs directory.

    Yields:
        Path to query logs directory

    Example:
        >>> def test_logging(mock_query_logs_dir):
        ...     log_file = mock_query_logs_dir / "query.json"
    """
    logs_dir = tmp_path / "query_logs"
    logs_dir.mkdir()
    yield logs_dir


# ============================================================================
# Parametrized Fixtures
# ============================================================================


@pytest.fixture(params=[100, 500, 1000, 2000])
def chunk_sizes(request):
    """Parametrized fixture for testing different chunk sizes.

    Example:
        >>> def test_chunking(chunk_sizes):
        ...     # Test runs 4 times with different chunk sizes
    """
    return request.param


@pytest.fixture(params=[0, 50, 100, 200])
def chunk_overlaps(request):
    """Parametrized fixture for testing different chunk overlaps.

    Example:
        >>> def test_overlap(chunk_overlaps):
        ...     # Test runs 4 times with different overlaps
    """
    return request.param


@pytest.fixture(params=["BAAI/bge-small-en", "sentence-transformers/all-MiniLM-L6-v2"])
def embed_models(request):
    """Parametrized fixture for testing different embedding models.

    Example:
        >>> def test_embeddings(embed_models):
        ...     # Test runs with different models
    """
    return request.param


# ============================================================================
# Session-Scoped Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory.

    Session-scoped: created once for entire test session.

    Returns:
        Path to tests/test_data directory
    """
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def sample_embeddings_cache():
    """Cache of sample embeddings for faster tests.

    Session-scoped to avoid regenerating embeddings for each test.

    Returns:
        Dict mapping text to embedding vectors
    """
    cache = {}

    def get_or_generate(text: str, dim: int = 384) -> List[float]:
        """Get cached embedding or generate new one."""
        if text not in cache:
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            vec = np.random.randn(dim)
            vec = vec / np.linalg.norm(vec)
            cache[text] = vec.tolist()
        return cache[text]

    return get_or_generate


# ============================================================================
# Helper Functions (not fixtures)
# ============================================================================


def create_test_index(
    vector_store,
    num_docs: int = 10,
    chunks_per_doc: int = 5,
    mock_text_node=None,
) -> List:
    """Helper to populate vector store with test data.

    Args:
        vector_store: Vector store instance
        num_docs: Number of documents to create
        chunks_per_doc: Chunks per document
        mock_text_node: Node factory fixture

    Returns:
        List of created nodes
    """
    nodes = []
    for doc_idx in range(num_docs):
        for chunk_idx in range(chunks_per_doc):
            node = mock_text_node(
                text=f"Document {doc_idx}, Chunk {chunk_idx}",
                doc_id=f"doc_{doc_idx}",
                node_id=f"node_{doc_idx}_{chunk_idx}",
            )
            nodes.append(node)

    vector_store.add(nodes)
    return nodes


# ============================================================================
# vLLM Fixtures (Week 1 - Day 1)
# ============================================================================


@pytest.fixture
def mock_vllm_server():
    """Mock vLLM server with health endpoint.

    Returns:
        MagicMock object simulating vLLM server

    Example:
        def test_vllm_connection(mock_vllm_server):
            assert mock_vllm_server.is_healthy
    """
    mock_server = MagicMock()
    mock_server.base_url = "http://localhost:8000/v1"
    mock_server.health_url = "http://localhost:8000/health"
    mock_server.is_healthy = True
    mock_server.model_name = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    return mock_server


@pytest.fixture
def mock_vllm_response():
    """Create mock vLLM API response.

    Returns:
        Callable that creates mock OpenAI-compatible responses

    Example:
        def test_vllm_generation(mock_vllm_response):
            response = mock_vllm_response(text="Test response")
            assert response.choices[0].message.content == "Test response"
    """
    def _create_response(text="Mock vLLM response", finish_reason="stop"):
        """Create a mock response object.

        Args:
            text: Response text content
            finish_reason: Completion finish reason

        Returns:
            MagicMock simulating OpenAI response
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = text
        response.choices[0].finish_reason = finish_reason
        return response

    return _create_response


@pytest.fixture
def mock_vllm_health_check():
    """Mock requests response for vLLM health check.

    Returns:
        MagicMock simulating requests.Response

    Example:
        with patch('requests.get', return_value=mock_vllm_health_check):
            # Health check will succeed
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    return mock_response


# ============================================================================
# File Validator Fixtures (Week 1 - Day 2)
# ============================================================================


@pytest.fixture
def sample_files(tmp_path):
    """Create sample files with correct/incorrect extensions.

    Returns:
        Dict mapping file types to Path objects

    Example:
        def test_validation(sample_files):
            pdf = sample_files['pdf_correct']
            assert pdf.exists()
    """
    files = {}

    # Real PDF with correct extension
    pdf_correct = tmp_path / "document.pdf"
    pdf_correct.write_bytes(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\nReal PDF content")
    files['pdf_correct'] = pdf_correct

    # JPEG with .pdf extension (misnamed)
    jpeg_as_pdf = tmp_path / "image.pdf"
    jpeg_as_pdf.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")
    files['jpeg_as_pdf'] = jpeg_as_pdf

    # PNG with correct extension
    png_correct = tmp_path / "photo.png"
    png_correct.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    files['png_correct'] = png_correct

    # GIF with correct extension
    gif_correct = tmp_path / "animation.gif"
    gif_correct.write_bytes(b"GIF89a\x01\x00\x01\x00")
    files['gif_correct'] = gif_correct

    # ZIP/DOCX with correct extension
    docx_correct = tmp_path / "document.docx"
    docx_correct.write_bytes(b"PK\x03\x04\x14\x00\x00\x00")
    files['docx_correct'] = docx_correct

    # Empty file
    empty_file = tmp_path / "empty.txt"
    empty_file.write_bytes(b"")
    files['empty'] = empty_file

    # Unknown file type
    unknown_file = tmp_path / "data.bin"
    unknown_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    files['unknown'] = unknown_file

    return files


@pytest.fixture
def corrupted_pdf(tmp_path):
    """Create a corrupted PDF file.

    Returns:
        Path to corrupted PDF

    Example:
        def test_corrupted(corrupted_pdf):
            # Should handle gracefully
            detect_file_type(corrupted_pdf)
    """
    pdf_file = tmp_path / "corrupted.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%\x00\x00\x00corrupt data")
    return pdf_file


@pytest.fixture
def test_directory_with_misnamed_files(tmp_path):
    """Create a test directory with correctly and incorrectly named files.

    Returns:
        Tuple of (directory_path, expected_misnamed_count)

    Example:
        def test_scan(test_directory_with_misnamed_files):
            dir_path, expected_count = test_directory_with_misnamed_files
            misnamed = scan_directory_for_misnamed_files(dir_path)
            assert len(misnamed) == expected_count
    """
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()

    # Correct files
    (test_dir / "doc1.pdf").write_bytes(b"%PDF-1.4\nContent")
    (test_dir / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (test_dir / "photo.jpg").write_bytes(b"\xFF\xD8\xFF\xE0JFIF")

    # Misnamed files (2 total)
    (test_dir / "fake_pdf.pdf").write_bytes(b"\xFF\xD8\xFF\xE0JFIF")  # JPEG as PDF
    (test_dir / "fake_image.png").write_bytes(b"%PDF-1.4\n")  # PDF as PNG

    # Create subdirectory with more files
    subdir = test_dir / "subfolder"
    subdir.mkdir()
    (subdir / "nested.pdf").write_bytes(b"%PDF-1.4\n")  # Correct
    (subdir / "wrong.jpg").write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG as JPEG (1 more misnamed)

    return test_dir, 3  # Total of 3 misnamed files


# ============================================================================
# Health Check Fixtures (Week 1 - Day 3)
# ============================================================================


@pytest.fixture
def mock_psutil():
    """Mock psutil for system resource checks.

    Returns:
        MagicMock simulating psutil module with CPU, memory, disk methods

    Example:
        def test_resources(mock_psutil):
            assert mock_psutil.cpu_percent.return_value == 50.0
    """
    with patch('utils.health_check.psutil') as mock:
        # CPU
        mock.cpu_percent.return_value = 50.0

        # Memory
        mock_memory = MagicMock()
        mock_memory.percent = 60.0
        mock_memory.available = 8 * 1024**3  # 8GB
        mock.virtual_memory.return_value = mock_memory

        # Disk
        mock_disk = MagicMock()
        mock_disk.percent = 70.0
        mock_disk.free = 50 * 1024**3  # 50GB
        mock.disk_usage.return_value = mock_disk

        # Boot time
        mock.boot_time.return_value = 1000000000.0

        yield mock


@pytest.fixture
def mock_db_connection_success():
    """Mock successful PostgreSQL connection.

    Returns:
        MagicMock simulating psycopg2 connection

    Example:
        with patch('psycopg2.connect', return_value=mock_db_connection_success):
            # Database check will succeed
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Mock cursor methods
    mock_cursor.fetchone.side_effect = [
        ("PostgreSQL 15.0",),  # version
        ("vector",),           # pgvector extension
        (5,),                  # table count
        ("100 MB",),           # database size
    ]

    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@pytest.fixture
def mock_torch_cuda():
    """Mock torch with CUDA available.

    Returns:
        MagicMock simulating torch module with CUDA

    Example:
        with patch('utils.health_check.torch', mock_torch_cuda):
            # GPU check will detect CUDA
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
    mock_torch.version.cuda = "12.1"
    mock_torch.backends.mps.is_available.return_value = False
    return mock_torch


@pytest.fixture
def mock_torch_mps():
    """Mock torch with MPS (Apple Silicon) available.

    Returns:
        MagicMock simulating torch module with MPS

    Example:
        with patch('utils.health_check.torch', mock_torch_mps):
            # GPU check will detect MPS
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0
    mock_torch.backends.mps.is_available.return_value = True
    return mock_torch


# ============================================================================
# Query Cache Fixtures (Week 1 - Day 4)
# ============================================================================


@pytest.fixture
def query_cache_dir(tmp_path):
    """Create temporary directory for query cache.

    Returns:
        Path to temporary cache directory

    Example:
        def test_cache(query_cache_dir):
            cache = QueryCache(cache_dir=query_cache_dir)
            # Cache isolated to test directory
    """
    cache_dir = tmp_path / "query_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def semantic_cache_dir(tmp_path):
    """Create temporary directory for semantic query cache.

    Returns:
        Path to temporary semantic cache directory

    Example:
        def test_semantic_cache(semantic_cache_dir):
            cache = SemanticQueryCache(cache_dir=semantic_cache_dir)
            # Cache isolated to test directory
    """
    cache_dir = tmp_path / "semantic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def sample_embeddings():
    """Generate sample embedding vectors for cache testing.

    Returns:
        Dict with pre-generated embeddings

    Example:
        def test_similarity(sample_embeddings):
            emb1 = sample_embeddings['query1']
            emb2 = sample_embeddings['similar']
            # Test similarity matching
    """
    import numpy as np

    # Create deterministic embeddings for testing
    np.random.seed(42)

    # Base query embedding
    query1 = np.random.randn(384)
    query1 = query1 / np.linalg.norm(query1)  # Normalize

    # Very similar query (99% similarity)
    similar = query1 + np.random.randn(384) * 0.01
    similar = similar / np.linalg.norm(similar)

    # Somewhat similar query (~85% similarity)
    somewhat_similar = query1 + np.random.randn(384) * 0.3
    somewhat_similar = somewhat_similar / np.linalg.norm(somewhat_similar)

    # Completely different query
    different = np.random.randn(384)
    different = different / np.linalg.norm(different)

    return {
        'query1': query1.tolist(),
        'similar': similar.tolist(),
        'somewhat_similar': somewhat_similar.tolist(),
        'different': different.tolist(),
    }


# ============================================================================
# MLX Embedding Fixtures (Week 1 - Day 5)
# ============================================================================


@pytest.fixture
def mock_mlx_model():
    """Mock MLX EmbeddingModel for testing.

    Returns:
        MagicMock simulating mlx-embedding-models EmbeddingModel

    Example:
        def test_mlx(mock_mlx_model):
            mock_mlx_model.encode.return_value = [[0.1] * 1024]
    """
    import numpy as np

    mock_model = MagicMock()

    # Mock encode to return valid embeddings
    def mock_encode(texts, show_progress=False):
        """Mock encode that returns embeddings based on input."""
        if not texts:
            return []

        # Return deterministic embeddings
        result = []
        for text in texts:
            if not text or not text.strip():
                # Empty text - model would still return embedding
                emb = np.random.randn(1024)
            else:
                # Use text hash for deterministic embeddings
                seed = hash(text) % 1000000
                np.random.seed(seed)
                emb = np.random.randn(1024)

            # Normalize
            emb = emb / np.linalg.norm(emb)
            result.append(emb)

        return result

    mock_model.encode.side_effect = mock_encode
    return mock_model


@pytest.fixture
def mock_mlx_embedding_model_class():
    """Mock MLX EmbeddingModel.from_registry class method.

    Returns:
        MagicMock for patching EmbeddingModel.from_registry

    Example:
        with patch('mlx_embedding_models.embedding.EmbeddingModel.from_registry',
                   return_value=mock_mlx_model):
            # MLXEmbedding will use mock model
    """
    return MagicMock()
