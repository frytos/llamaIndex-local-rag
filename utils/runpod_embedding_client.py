"""
RunPod Embedding API Client

HTTP client for calling the embedding service running on RunPod GPU.
Handles batching, retries, and error handling.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


class RunPodEmbeddingClient:
    """
    HTTP client for RunPod embedding service.

    Features:
    - Batch API calls to reduce network overhead
    - Automatic retry with exponential backoff
    - Timeout handling
    - Progress tracking
    - Error handling with fallback support
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        timeout: int = 60,
        max_retries: int = 3,
        request_batch_size: int = 200
    ):
        """
        Initialize embedding client.

        Args:
            endpoint_url: Base URL of embedding service (e.g., "http://38.65.239.5:8001")
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: 60)
            max_retries: Max retry attempts for failed requests (default: 3)
            request_batch_size: Number of texts per HTTP request (default: 200)
        """
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.request_batch_size = request_batch_size

        # Create session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s delays
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]  # Renamed from method_whitelist in urllib3 2.0+
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        log.info(f"RunPod embedding client initialized: {endpoint_url}")

    def check_health(self) -> bool:
        """
        Check if embedding service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.endpoint_url}/health",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            is_healthy = (
                data.get("status") == "healthy" and
                data.get("model_loaded", False)
            )

            if is_healthy:
                log.info(f"✅ RunPod embedding service healthy (GPU: {data.get('gpu_available', False)})")
            else:
                log.warning(f"⚠️  RunPod embedding service unhealthy: {data}")

            return is_healthy

        except Exception as e:
            log.error(f"❌ Health check failed: {e}")
            return False

    def embed_texts(
        self,
        texts: List[str],
        model: str = "BAAI/bge-small-en",
        batch_size: int = 128,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Compute embeddings for a list of texts using RunPod GPU.

        Args:
            texts: List of text strings to embed
            model: Embedding model name
            batch_size: Batch size for GPU processing (sent to API)
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors (one per text)

        Raises:
            RuntimeError: If API call fails after retries
            ConnectionError: If cannot connect to RunPod service
        """
        if not texts:
            return []

        log.info(f"🚀 Embedding {len(texts)} texts via RunPod GPU...")
        log.info(f"   Model: {model}")
        log.info(f"   Batch size: {batch_size}")
        log.info(f"   Request batch size: {self.request_batch_size}")

        start_time = time.time()
        all_embeddings = []

        # Progress tracking
        if show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(texts), desc="Embedding", unit="texts")
            except ImportError:
                progress_bar = None
                log.warning("tqdm not available, progress bar disabled")
        else:
            progress_bar = None

        # Split into HTTP request batches
        for i in range(0, len(texts), self.request_batch_size):
            request_batch = texts[i:i + self.request_batch_size]

            try:
                # Call API
                batch_embeddings = self._embed_batch(request_batch, model, batch_size)
                all_embeddings.extend(batch_embeddings)

                # Update progress
                if progress_bar:
                    progress_bar.update(len(request_batch))

            except Exception as e:
                log.error(f"Failed to embed batch {i}-{i + len(request_batch)}: {e}")
                if progress_bar:
                    progress_bar.close()
                raise

        if progress_bar:
            progress_bar.close()

        # Calculate stats
        elapsed = time.time() - start_time
        throughput = len(texts) / elapsed if elapsed > 0 else 0

        log.info(f"✅ Embedded {len(texts)} texts in {elapsed:.2f}s ({throughput:.1f} texts/sec)")

        return all_embeddings

    def _embed_batch(
        self,
        texts: List[str],
        model: str,
        batch_size: int
    ) -> List[List[float]]:
        """
        Embed a single batch of texts via API call.

        Args:
            texts: Batch of texts to embed
            model: Embedding model name
            batch_size: GPU batch size (passed to API)

        Returns:
            List of embedding vectors

        Raises:
            requests.HTTPError: If API returns error status
            requests.Timeout: If request times out
            ConnectionError: If cannot connect to service
        """
        payload = {
            "texts": texts,
            "model": model,
            "batch_size": batch_size
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }

        try:
            response = self.session.post(
                f"{self.endpoint_url}/embed",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data["embeddings"]

            log.debug(f"Embedded {len(embeddings)} texts (dim={data['dimension']}, "
                     f"time={data['processing_time_ms']:.1f}ms, gpu={data['gpu_used']})")

            return embeddings

        except requests.Timeout:
            log.error(f"Request timeout after {self.timeout}s")
            raise ConnectionError(f"RunPod embedding service timeout (>{self.timeout}s)")

        except requests.ConnectionError as e:
            log.error(f"Connection error: {e}")
            raise ConnectionError(f"Cannot connect to RunPod embedding service: {e}")

        except requests.HTTPError as e:
            log.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"RunPod API error: {e.response.status_code} - {e.response.text}")

        except KeyError as e:
            log.error(f"Invalid response format: missing key {e}")
            raise RuntimeError(f"Invalid response from RunPod API: {e}")

    def close(self):
        """Close the HTTP session."""
        self.session.close()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client_from_env() -> Optional[RunPodEmbeddingClient]:
    """
    Create RunPod embedding client from environment variables.

    Environment variables:
    - RUNPOD_EMBEDDING_ENDPOINT: URL of embedding service
    - RUNPOD_EMBEDDING_API_KEY: API key for authentication

    Returns:
        RunPodEmbeddingClient if configured, None otherwise
    """
    endpoint = os.getenv("RUNPOD_EMBEDDING_ENDPOINT")
    api_key = os.getenv("RUNPOD_EMBEDDING_API_KEY")

    if not endpoint or not api_key:
        log.debug("RunPod embedding not configured (missing RUNPOD_EMBEDDING_ENDPOINT or RUNPOD_EMBEDDING_API_KEY)")
        return None

    return RunPodEmbeddingClient(endpoint, api_key)


def is_runpod_embedding_available() -> bool:
    """
    Check if RunPod embedding service is configured and available.

    Returns:
        True if service is configured, False otherwise
    """
    client = create_client_from_env()
    if client is None:
        return False

    try:
        return client.check_health()
    finally:
        client.close()
