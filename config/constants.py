"""Configuration constants for RAG pipeline.

This module defines default values and thresholds used throughout
the RAG pipeline. These constants can be overridden via environment
variables in most cases.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    """Default chunking configuration parameters.

    These values represent the recommended defaults for document chunking
    in the RAG pipeline. They can be overridden via environment variables:
    - CHUNK_SIZE: Override DEFAULT_SIZE
    - CHUNK_OVERLAP: Override DEFAULT_OVERLAP
    """

    # Default chunk size in characters
    # Recommended range: 500-800 for balanced precision/context
    DEFAULT_SIZE: int = 700

    # Default overlap between chunks in characters
    # Recommended: 15-25% of chunk size
    DEFAULT_OVERLAP: int = 150

    # Minimum/maximum bounds for validation
    MIN_SIZE: int = 100
    MAX_SIZE: int = 2000
    MIN_OVERLAP: int = 0
    MAX_OVERLAP: int = 500


@dataclass(frozen=True)
class SimilarityThresholds:
    """Thresholds for similarity score interpretation.

    These thresholds help interpret cosine similarity scores
    (range 0.0-1.0) from vector retrieval:
    - 1.0: Identical text
    - 0.8+: Highly relevant
    - 0.6-0.8: Good relevance
    - 0.4-0.6: Moderate relevance
    - <0.4: Weak relevance (may produce poor answers)
    """

    # Excellent: Highly relevant results
    EXCELLENT: float = 0.8

    # Good: Relevant enough for quality answers
    GOOD: float = 0.6

    # Fair: Moderate relevance, answer may be vague
    FAIR: float = 0.4

    # Minimum threshold to consider retrieval useful
    MINIMUM: float = 0.3


@dataclass(frozen=True)
class LLMConfig:
    """Default LLM configuration parameters.

    These values are tuned for Mistral 7B on M1 Mac (16GB).
    Can be overridden via environment variables:
    - CTX: Override DEFAULT_CONTEXT_WINDOW
    - MAX_NEW_TOKENS: Override DEFAULT_MAX_TOKENS
    - TEMP: Override DEFAULT_TEMPERATURE
    - N_GPU_LAYERS: Override DEFAULT_GPU_LAYERS
    - N_BATCH: Override DEFAULT_BATCH_SIZE
    """

    # Context window size (tokens)
    DEFAULT_CONTEXT_WINDOW: int = 3072

    # Maximum tokens to generate
    DEFAULT_MAX_TOKENS: int = 256

    # Temperature for generation (0.0 = deterministic, 1.0 = creative)
    DEFAULT_TEMPERATURE: float = 0.1

    # Number of model layers to offload to GPU
    # 24 is optimal for M1 16GB with Mistral 7B
    DEFAULT_GPU_LAYERS: int = 24

    # Batch size for inference
    DEFAULT_BATCH_SIZE: int = 256

    # Temperature thresholds for classification
    TEMP_FACTUAL: float = 0.3    # Below this: factual/deterministic
    TEMP_BALANCED: float = 0.7   # Below this: balanced
    # Above TEMP_BALANCED: creative


@dataclass(frozen=True)
class RetrievalConfig:
    """Default retrieval configuration parameters.

    Can be overridden via environment variables:
    - TOP_K: Override DEFAULT_TOP_K
    - RERANK_TOP_K: Override DEFAULT_RERANK_TOP_K
    """

    # Default number of chunks to retrieve
    DEFAULT_TOP_K: int = 4

    # Default number of chunks after reranking
    DEFAULT_RERANK_TOP_K: int = 4

    # Minimum/maximum bounds
    MIN_TOP_K: int = 1
    MAX_TOP_K: int = 20


@dataclass(frozen=True)
class EmbeddingConfig:
    """Default embedding configuration parameters.

    Can be overridden via environment variables:
    - EMBED_MODEL: Override DEFAULT_MODEL
    - EMBED_DIM: Override DEFAULT_DIMENSION
    - EMBED_BATCH: Override DEFAULT_BATCH_SIZE
    """

    # Default embedding model
    DEFAULT_MODEL: str = "BAAI/bge-small-en"

    # Default embedding dimension (depends on model)
    DEFAULT_DIMENSION: int = 384

    # Default batch size for embedding
    DEFAULT_BATCH_SIZE: int = 64


@dataclass(frozen=True)
class DatabaseConfig:
    """Default database configuration parameters.

    Can be overridden via environment variables:
    - PGHOST, PGPORT, PGUSER, PGPASSWORD, DB_NAME
    """

    # Default connection parameters
    DEFAULT_HOST: str = "localhost"
    DEFAULT_PORT: str = "5432"
    DEFAULT_DB_NAME: str = "vector_db"

    # Connection pool settings
    DEFAULT_POOL_SIZE: int = 5
    DEFAULT_MAX_OVERFLOW: int = 10

    # Query timeout (seconds)
    DEFAULT_TIMEOUT: int = 30


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance-related constants and thresholds."""

    # Memory estimates (bytes)
    BYTES_PER_FLOAT32: int = 4

    # Progress bar thresholds
    PROGRESS_BAR_MIN_ITEMS: int = 10

    # Preview text length (characters)
    DEFAULT_PREVIEW_LENGTH: int = 150

    # Logging intervals
    LOG_EVERY_N_CHUNKS: int = 100
    LOG_EVERY_N_EMBEDDINGS: int = 50


# Export convenience instances
CHUNK = ChunkConfig()
SIMILARITY = SimilarityThresholds()
LLM = LLMConfig()
RETRIEVAL = RetrievalConfig()
EMBEDDING = EmbeddingConfig()
DATABASE = DatabaseConfig()
PERFORMANCE = PerformanceConfig()
