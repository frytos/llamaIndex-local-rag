"""
MLX-optimized embeddings for Apple Silicon.

This module provides a LlamaIndex-compatible embedding class that uses
Apple's MLX framework for 5-20x faster embedding on M1/M2/M3 Macs.

Usage:
    EMBED_BACKEND=mlx python rag_low_level_m1_16gb_verbose.py
"""

from llama_index.core.embeddings import BaseEmbedding
from typing import List, Any, Optional
import logging

log = logging.getLogger(__name__)


class MLXEmbedding(BaseEmbedding):
    """
    MLX-optimized embeddings for Apple Silicon.

    Uses mlx-embedding-models for fast embedding on M1/M2/M3 Macs.
    Provides 5-20x speedup vs PyTorch on Apple Silicon.

    Args:
        model_name: HuggingFace model name (e.g., "BAAI/bge-large-en-v1.5")
        **kwargs: Additional arguments for BaseEmbedding
    """

    # Pydantic fields
    model: Optional[Any] = None
    _model_name: str = ""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        **kwargs: Any,
    ) -> None:
        # Initialize parent first with model field
        super().__init__(model=None, **kwargs)

        try:
            from mlx_embedding_models.embedding import EmbeddingModel
            import mlx.core as mx
        except ImportError:
            raise ImportError(
                "mlx-embedding-models not found. Install with:\n"
                "  pip install mlx mlx-embedding-models"
            )

        # Model name mapping (HuggingFace → MLX registry)
        model_map = {
            "BAAI/bge-small-en": "bge-small",
            "BAAI/bge-large-en-v1.5": "bge-large",
            "sentence-transformers/all-MiniLM-L6-v2": "minilm",
            "BAAI/bge-base-en": "bge-base",
            "BAAI/bge-base-en-v1.5": "bge-base",
        }

        mlx_model_name = model_map.get(model_name, model_name)

        log.info(f"Loading MLX embedding model: {mlx_model_name}")
        self.model = EmbeddingModel.from_registry(mlx_model_name)
        self._model_name = model_name

        # Pre-warm Metal GPU compilation (first inference is slow)
        log.debug("Pre-warming Metal GPU compilation...")
        _ = self.model.encode(["warmup"])
        log.debug("MLX model ready")

    @classmethod
    def class_name(cls) -> str:
        return "MLXEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a single query string.

        Args:
            query: Query text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Suppress MLX verbose output
        import os
        old_log_level = os.environ.get("TRANSFORMERS_VERBOSITY", None)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        try:
            embedding = self.model.encode([query], show_progress=False)[0]
        except TypeError:
            embedding = self.model.encode([query])[0]
        finally:
            if old_log_level is not None:
                os.environ["TRANSFORMERS_VERBOSITY"] = old_log_level
            elif "TRANSFORMERS_VERBOSITY" in os.environ:
                del os.environ["TRANSFORMERS_VERBOSITY"]

        return embedding.tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Suppress MLX verbose output
        import os
        old_log_level = os.environ.get("TRANSFORMERS_VERBOSITY", None)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        try:
            embedding = self.model.encode([text], show_progress=False)[0]
        except TypeError:
            embedding = self.model.encode([text])[0]
        finally:
            if old_log_level is not None:
                os.environ["TRANSFORMERS_VERBOSITY"] = old_log_level
            elif "TRANSFORMERS_VERBOSITY" in os.environ:
                del os.environ["TRANSFORMERS_VERBOSITY"]

        return embedding.tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple text strings (batch processing).

        This is the primary method for fast indexing. MLX handles
        batching efficiently on Metal GPU.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        # Suppress MLX's internal progress bars to keep our main progress bar visible
        import os
        old_log_level = os.environ.get("TRANSFORMERS_VERBOSITY", None)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        try:
            # MLX encode() returns numpy arrays, convert to list for each embedding
            embeddings = self.model.encode(texts, show_progress=False)
        except TypeError:
            # Fallback if show_progress parameter not supported
            embeddings = self.model.encode(texts)
        finally:
            # Restore original log level
            if old_log_level is not None:
                os.environ["TRANSFORMERS_VERBOSITY"] = old_log_level
            elif "TRANSFORMERS_VERBOSITY" in os.environ:
                del os.environ["TRANSFORMERS_VERBOSITY"]

        # Convert numpy arrays to lists - handle both 2D array and list of 1D arrays
        try:
            return embeddings.tolist()  # If it's a 2D numpy array
        except AttributeError:
            return [emb.tolist() for emb in embeddings]  # If it's a list of arrays

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Public batch embedding method for LlamaIndex compatibility.

        This method is called by LlamaIndex's embed_nodes() function.
        """
        return self._get_text_embeddings(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding (wraps sync method)."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding (wraps sync method)."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of _get_text_embeddings (wraps sync method)."""
        return self._get_text_embeddings(texts)
