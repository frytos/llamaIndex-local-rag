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
    _embed_dim: int = 384  # Default dimension, will be updated after model loads

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
            "BAAI/bge-m3": "bge-m3",  # Multilingual model (100+ languages)
        }

        # Dimension mapping for zero vector fallbacks
        dim_map = {
            "BAAI/bge-small-en": 384,
            "BAAI/bge-large-en-v1.5": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "BAAI/bge-base-en": 768,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-m3": 1024,
        }

        mlx_model_name = model_map.get(model_name, model_name)
        self._embed_dim = dim_map.get(model_name, 384)

        log.info(f"Loading MLX embedding model: {mlx_model_name}")
        self.model = EmbeddingModel.from_registry(mlx_model_name)
        self._model_name = model_name

        # Pre-warm Metal GPU compilation (first inference is slow)
        log.debug("Pre-warming Metal GPU compilation...")
        warmup_embedding = self.model.encode(["warmup"])

        # Detect actual embedding dimension from warmup
        if warmup_embedding is not None and len(warmup_embedding) > 0:
            try:
                if hasattr(warmup_embedding[0], 'shape'):
                    self._embed_dim = warmup_embedding[0].shape[0]
                elif hasattr(warmup_embedding[0], '__len__'):
                    self._embed_dim = len(warmup_embedding[0])
                log.debug(f"Detected embedding dimension: {self._embed_dim}")
            except Exception as e:
                log.warning(f"Could not detect embedding dimension: {e}, using default {self._embed_dim}")

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
        # Validate input query
        if not query or not isinstance(query, str):
            log.warning(f"Invalid query: {type(query)}, returning zero vector")
            return [0.0] * self._embed_dim

        # Clean query
        query = query.strip()
        if not query:
            log.warning("Empty query after stripping, returning zero vector")
            return [0.0] * self._embed_dim

        # Truncate very long queries
        if len(query) > 32000:
            log.warning(f"Query too long ({len(query)} chars), truncating")
            query = query[:32000]

        # Suppress MLX verbose output
        import os
        old_log_level = os.environ.get("TRANSFORMERS_VERBOSITY", None)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        try:
            try:
                embeddings = self.model.encode([query], show_progress=False)
            except TypeError:
                embeddings = self.model.encode([query])

            # Check if we got valid embeddings
            if embeddings is None or len(embeddings) == 0:
                log.error(f"MLX model returned empty result for query (len={len(query)})")
                return [0.0] * self._embed_dim

            embedding = embeddings[0]

        except IndexError as e:
            log.error(f"IndexError during query embedding (len={len(query)}): {e}")
            return [0.0] * self._embed_dim
        except Exception as e:
            log.error(f"Failed to embed query (len={len(query)}): {e}")
            return [0.0] * self._embed_dim
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
        # Validate input text
        if not text or not isinstance(text, str):
            log.warning(f"Invalid input text: {type(text)}, returning zero vector")
            # Return zero vector with proper dimensions (bge-m3 is 1024-dim)
            return [0.0] * self._embed_dim

        # Clean and validate text
        text = text.strip()
        if not text:
            log.warning("Empty text after stripping, returning zero vector")
            return [0.0] * self._embed_dim

        # Truncate very long texts (bge-m3 has 8192 token limit, ~32k chars)
        if len(text) > 32000:
            log.warning(f"Text too long ({len(text)} chars), truncating to 32000")
            text = text[:32000]

        # Suppress MLX verbose output
        import os
        old_log_level = os.environ.get("TRANSFORMERS_VERBOSITY", None)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        try:
            try:
                embeddings = self.model.encode([text], show_progress=False)
            except TypeError:
                embeddings = self.model.encode([text])

            # Check if we got valid embeddings
            if embeddings is None or len(embeddings) == 0:
                log.error(f"MLX model returned empty result for text (len={len(text)})")
                return [0.0] * self._embed_dim

            embedding = embeddings[0]

        except IndexError as e:
            log.error(f"IndexError during embedding (text len={len(text)}): {e}")
            # Log first 200 chars of problematic text for debugging
            log.debug(f"  Problematic text preview: {repr(text[:200])}")
            return [0.0] * self._embed_dim
        except Exception as e:
            log.error(f"Failed to embed text (len={len(text)}): {e}")
            log.debug(f"  Text preview: {repr(text[:200])}")
            return [0.0] * self._embed_dim
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
        # Validate and clean input texts
        cleaned_texts = []
        for i, text in enumerate(texts):
            # Check if text is valid
            if not text or not isinstance(text, str):
                log.warning(f"Batch item {i}: Invalid text type {type(text)}, using empty string")
                cleaned_texts.append("")
                continue

            # Clean text
            text = text.strip()

            # Truncate if too long (bge-m3 has 8192 token limit, ~32k chars)
            if len(text) > 32000:
                log.warning(f"Batch item {i}: Text too long ({len(text)} chars), truncating")
                text = text[:32000]

            cleaned_texts.append(text)

        # Suppress MLX's internal progress bars to keep our main progress bar visible
        import os
        old_log_level = os.environ.get("TRANSFORMERS_VERBOSITY", None)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        try:
            # MLX encode() returns numpy arrays, convert to list for each embedding
            try:
                embeddings = self.model.encode(cleaned_texts, show_progress=False)
            except TypeError:
                # Fallback if show_progress parameter not supported
                embeddings = self.model.encode(cleaned_texts)

            # Verify we got the expected number of embeddings
            if embeddings is None or len(embeddings) != len(texts):
                log.error(f"MLX batch embedding failed: expected {len(texts)} embeddings, got {len(embeddings) if embeddings is not None else 0}")
                # Fall back to one-by-one embedding
                log.warning("Falling back to individual embedding for this batch")
                result = []
                for i, text in enumerate(texts):
                    try:
                        result.append(self._get_text_embedding(text))
                    except Exception as e:
                        log.error(f"Failed to embed text {i}: {e}")
                        result.append([0.0] * self._embed_dim)
                return result

        except Exception as e:
            log.error(f"Batch embedding failed: {e}")
            # Fall back to one-by-one embedding
            log.warning("Falling back to individual embedding")
            result = []
            for i, text in enumerate(texts):
                try:
                    result.append(self._get_text_embedding(text))
                except Exception as e2:
                    log.error(f"Failed to embed text {i}: {e2}")
                    result.append([0.0] * self._embed_dim)
            return result
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
