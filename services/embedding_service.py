#!/usr/bin/env python3
"""
FastAPI Embedding Service for RunPod GPU

Provides HTTP API for computing embeddings using GPU acceleration.
Designed to run on RunPod with CUDA support.

Usage:
    uvicorn services.embedding_service:app --host 0.0.0.0 --port 8001

Endpoints:
    GET  /health - Health check
    POST /embed  - Compute embeddings for texts
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# API Key for authentication
API_KEY = os.getenv("RUNPOD_EMBEDDING_API_KEY", "")
if not API_KEY:
    log.warning("⚠️  RUNPOD_EMBEDDING_API_KEY not set - API will be UNSECURED!")

# Embedding model configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "huggingface")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))

# =============================================================================
# Pydantic Models
# =============================================================================

class EmbedRequest(BaseModel):
    """Request for embedding computation"""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1)
    model: str = Field(EMBED_MODEL, description="Embedding model name")
    batch_size: int = Field(EMBED_BATCH, description="Batch size for processing")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello world", "This is a test"],
                "model": "BAAI/bge-small-en",
                "batch_size": 128
            }
        }


class EmbedResponse(BaseModel):
    """Response with computed embeddings"""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model: str = Field(..., description="Model used")
    gpu_used: bool = Field(..., description="Whether GPU was used")

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "dimension": 384,
                "count": 2,
                "processing_time_ms": 125.5,
                "model": "BAAI/bge-small-en",
                "gpu_used": True
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    gpu_available: bool
    model_loaded: bool
    model_name: str


# =============================================================================
# Global State (Singleton Pattern)
# =============================================================================

class EmbeddingModelSingleton:
    """Singleton to hold the embedding model (loaded once at startup)"""
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None
        self.dimension = None

    def load_model(self, model_name: str = EMBED_MODEL):
        """Load embedding model with GPU support"""
        if self.model is not None and self.model_name == model_name:
            log.info(f"Model {model_name} already loaded")
            return

        log.info(f"Loading embedding model: {model_name}")
        start_time = time.time()

        # Force CUDA device for RunPod GPU
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
            log.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            log.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = "cpu"
            log.warning("⚠️  CUDA not available - falling back to CPU")

        # Load model with HuggingFace backend
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        self.model = HuggingFaceEmbedding(
            model_name=model_name,
            device=self.device
        )
        self.model_name = model_name

        # Get dimension from test embedding
        test_emb = self.model.get_text_embedding("test")
        self.dimension = len(test_emb)

        load_time = time.time() - start_time
        log.info(f"✅ Model loaded in {load_time:.2f}s")
        log.info(f"   Model: {model_name}")
        log.info(f"   Device: {self.device}")
        log.info(f"   Dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Compute embeddings for a list of texts"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not texts:
            return []

        log.info(f"Computing embeddings for {len(texts)} texts (batch_size={batch_size})")

        # Use batch method if available
        try:
            embeddings = self.model.get_text_embedding_batch(texts, show_progress=False)
            return embeddings
        except AttributeError:
            # Fallback to individual embedding
            log.warning("Batch method not available, using individual embedding")
            embeddings = [self.model.get_text_embedding(text) for text in texts]
            return embeddings


# Global singleton instance
embedding_singleton = EmbeddingModelSingleton()

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="RunPod Embedding Service",
    description="GPU-accelerated embedding computation service",
    version="1.0.0"
)

# =============================================================================
# Middleware
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    log.info(f"{request.method} {request.url.path} - {response.status_code} ({duration:.1f}ms)")
    return response

# =============================================================================
# Authentication
# =============================================================================

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header"""
    if not API_KEY:
        # No API key configured - allow all requests
        return

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import torch

    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        model_loaded=embedding_singleton.model is not None,
        model_name=embedding_singleton.model_name or "not_loaded"
    )


@app.post("/embed", response_model=EmbedResponse, dependencies=[])
async def embed_texts(request: EmbedRequest, x_api_key: Optional[str] = Header(None)):
    """
    Compute embeddings for a list of texts.

    Authentication: Include X-API-Key header with your API key.
    """
    # Verify API key
    verify_api_key(x_api_key)

    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")

    # Load model if not loaded or different model requested
    if embedding_singleton.model_name != request.model:
        try:
            embedding_singleton.load_model(request.model)
        except Exception as e:
            log.error(f"Failed to load model {request.model}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Compute embeddings
    start_time = time.time()
    try:
        embeddings = embedding_singleton.embed_texts(request.texts, request.batch_size)
    except Exception as e:
        log.error(f"Embedding computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    processing_time_ms = (time.time() - start_time) * 1000

    import torch
    gpu_used = torch.cuda.is_available() and embedding_singleton.device == "cuda"

    log.info(f"✅ Embedded {len(embeddings)} texts in {processing_time_ms:.1f}ms ({len(embeddings) / (processing_time_ms / 1000):.1f} texts/sec)")

    return EmbedResponse(
        embeddings=embeddings,
        dimension=embedding_singleton.dimension,
        count=len(embeddings),
        processing_time_ms=processing_time_ms,
        model=request.model,
        gpu_used=gpu_used
    )


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load embedding model at startup"""
    log.info("=" * 70)
    log.info("RunPod Embedding Service Starting...")
    log.info("=" * 70)

    # Load default model
    try:
        embedding_singleton.load_model(EMBED_MODEL)
        log.info("✅ Embedding service ready!")
    except Exception as e:
        log.error(f"❌ Failed to load embedding model: {e}")
        log.error("   Service will start but /embed endpoint will fail until model is loaded")

    log.info("=" * 70)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
