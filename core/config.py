"""Configuration management for RAG pipeline.

This module provides a centralized Settings class for managing all
configuration parameters. It replaces scattered environment variable
reads throughout the codebase with a single configuration object.

Usage:
    from core.config import Settings

    settings = Settings()
    print(f"Chunk size: {settings.chunk_size}")
    print(f"Database: {settings.db_name}")

Configuration can be overridden via environment variables (see .env.example).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.constants import (
    CHUNK,
    DATABASE,
    EMBEDDING,
    LLM,
    RETRIEVAL,
)


@dataclass
class Settings:
    """
    Centralized configuration for RAG pipeline.

    All settings can be overridden via environment variables.
    See config/.env.example for complete list of available options.

    Note: This is a refactored version extracted from the monolithic
    rag_low_level_m1_16gb_verbose.py. Future iterations will add
    validation methods and helper functions.
    """

    # === Database Configuration ===
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", DATABASE.DEFAULT_DB_NAME))
    host: str = field(default_factory=lambda: os.getenv("PGHOST", DATABASE.DEFAULT_HOST))
    port: str = field(default_factory=lambda: os.getenv("PGPORT", DATABASE.DEFAULT_PORT))
    user: Optional[str] = field(default_factory=lambda: os.getenv("PGUSER"))
    password: Optional[str] = field(default_factory=lambda: os.getenv("PGPASSWORD"))
    table: str = ""  # Will be auto-generated if not set via PGTABLE

    # === Input Configuration ===
    pdf_path: str = field(default_factory=lambda: os.getenv("PDF_PATH", "data/llama2.pdf"))

    # === Reset Behaviors ===
    # RESET_TABLE=1 is useful while iterating to avoid duplicate rows
    reset_table: bool = field(default_factory=lambda: os.getenv("RESET_TABLE", "0") == "1")
    # RESET_DB=1 is more nuclear; only use if you want a fresh DB
    reset_db: bool = field(default_factory=lambda: os.getenv("RESET_DB", "0") == "1")

    # === Chunking Configuration (RAG Quality Knobs) ===
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", str(CHUNK.DEFAULT_SIZE))))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", str(CHUNK.DEFAULT_OVERLAP))))

    # === Retrieval Configuration ===
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", str(RETRIEVAL.DEFAULT_TOP_K))))

    # Hybrid retrieval: 0.0=pure BM25, 0.5=balanced, 1.0=pure vector
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "1.0"))

    # Metadata filtering
    enable_filters: bool = os.getenv("ENABLE_FILTERS", "1") == "1"

    # MMR diversity: 0=disabled, 0.5=balanced, 1.0=max relevance
    mmr_threshold: float = float(os.getenv("MMR_THRESHOLD", "0.0"))

    # === Embedding Configuration ===
    embed_model_name: str = os.getenv("EMBED_MODEL", EMBEDDING.DEFAULT_MODEL)
    embed_dim: int = int(os.getenv("EMBED_DIM", str(EMBEDDING.DEFAULT_DIMENSION)))
    embed_batch: int = int(os.getenv("EMBED_BATCH", "32"))
    embed_backend: str = os.getenv("EMBED_BACKEND", "huggingface")  # or "mlx"

    # === LLM Configuration ===
    model_url: str = os.getenv(
        "MODEL_URL",
        "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    )
    model_path: str = os.getenv("MODEL_PATH", "")

    temperature: float = field(default_factory=lambda: float(os.getenv("TEMP", str(LLM.DEFAULT_TEMPERATURE))))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", str(LLM.DEFAULT_MAX_TOKENS))))
    context_window: int = field(default_factory=lambda: int(os.getenv("CTX", str(LLM.DEFAULT_CONTEXT_WINDOW))))

    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", str(LLM.DEFAULT_GPU_LAYERS)))
    n_batch: int = int(os.getenv("N_BATCH", str(LLM.DEFAULT_BATCH_SIZE)))

    # === Query Configuration ===
    question: str = os.getenv(
        "QUESTION",
        "Summarize the key safety-related training ideas described in this paper.",
    )

    # === Advanced Features ===
    # Reranking
    enable_reranking: bool = os.getenv("ENABLE_RERANKING", "0") == "1"
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "12"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", str(RETRIEVAL.DEFAULT_RERANK_TOP_K)))
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Query expansion
    enable_query_expansion: bool = os.getenv("ENABLE_QUERY_EXPANSION", "0") == "1"
    query_expansion_method: str = os.getenv("QUERY_EXPANSION_METHOD", "llm")
    query_expansion_count: int = int(os.getenv("QUERY_EXPANSION_COUNT", "2"))

    # Enhanced metadata extraction
    extract_enhanced_metadata: bool = os.getenv("EXTRACT_ENHANCED_METADATA", "0") == "1"

    # Semantic caching
    enable_semantic_cache: bool = os.getenv("ENABLE_SEMANTIC_CACHE", "0") == "1"
    semantic_cache_threshold: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
    semantic_cache_max_size: int = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "1000"))
    semantic_cache_ttl: int = int(os.getenv("SEMANTIC_CACHE_TTL", "86400"))

    def __post_init__(self):
        """Validate credentials are set after dataclass initialization."""
        if not self.user or not self.password:
            raise ValueError(
                "Database credentials not set!\n"
                "Required environment variables:\n"
                "  PGUSER=your_database_user\n"
                "  PGPASSWORD=your_database_password\n"
                "Set them in .env file or export them."
            )

    def validate(self) -> None:
        """
        Validate settings and provide helpful error messages.

        Raises:
            ValueError: If validation fails with actionable error messages.

        Note: Full validation logic will be ported in future iterations.
        Currently validates only basic constraints.
        """
        errors = []

        # Validate chunk configuration
        if self.chunk_size < CHUNK.MIN_SIZE or self.chunk_size > CHUNK.MAX_SIZE:
            errors.append(
                f"chunk_size must be between {CHUNK.MIN_SIZE} and {CHUNK.MAX_SIZE}, "
                f"got {self.chunk_size}"
            )

        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        # Validate retrieval configuration
        if self.top_k < RETRIEVAL.MIN_TOP_K or self.top_k > RETRIEVAL.MAX_TOP_K:
            errors.append(
                f"top_k must be between {RETRIEVAL.MIN_TOP_K} and {RETRIEVAL.MAX_TOP_K}, "
                f"got {self.top_k}"
            )

        # Validate temperature
        if self.temperature < 0.0 or self.temperature > 2.0:
            errors.append(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def connection_string(self) -> str:
        """
        Generate PostgreSQL connection string.

        Returns:
            Connection string in format: postgresql://user:pass@host:port/dbname
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"


# Singleton instance for backward compatibility
# TODO: Remove this once all code migrates to explicit Settings instantiation
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create the global Settings instance.

    Returns:
        The singleton Settings instance.

    Note: This function exists for backward compatibility during migration.
    New code should explicitly instantiate Settings() where needed.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
