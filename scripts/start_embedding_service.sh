#!/bin/bash
#
# Start Embedding Service on RunPod
#
# This script starts the FastAPI embedding service on port 8001.
# It should be run on RunPod pods with GPU support.
#
# Usage:
#   ./start_embedding_service.sh
#
# Environment Variables:
#   RUNPOD_EMBEDDING_API_KEY - API key for authentication (required)
#   EMBED_MODEL - Embedding model name (default: BAAI/bge-small-en)
#   EMBED_BACKEND - Backend (default: huggingface)
#   PORT - Port to run on (default: 8001)
#

set -e  # Exit on error

# Default configuration
PORT=${PORT:-8001}
EMBED_MODEL=${EMBED_MODEL:-"BAAI/bge-small-en"}
EMBED_BACKEND=${EMBED_BACKEND:-"huggingface"}
RUNPOD_EMBEDDING_API_KEY=${RUNPOD_EMBEDDING_API_KEY:-""}

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Starting RunPod Embedding Service                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Port: $PORT"
echo "  Model: $EMBED_MODEL"
echo "  Backend: $EMBED_BACKEND"
echo "  API Key: ${RUNPOD_EMBEDDING_API_KEY:0:10}..." # Show first 10 chars only
echo ""

# Check if API key is set
if [ -z "$RUNPOD_EMBEDDING_API_KEY" ]; then
    echo "⚠️  WARNING: RUNPOD_EMBEDDING_API_KEY not set!"
    echo "   API will be UNSECURED. Set the environment variable to secure it."
    echo ""
fi

# Check if we're in the right directory
if [ ! -f "services/embedding_service.py" ]; then
    echo "❌ Error: services/embedding_service.py not found"
    echo "   Please run this script from the project root directory"
    exit 1
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import fastapi" 2>/dev/null || {
    echo "❌ Error: FastAPI not installed"
    echo "   Install with: pip install fastapi uvicorn[standard]"
    exit 1
}

python3 -c "import uvicorn" 2>/dev/null || {
    echo "❌ Error: uvicorn not installed"
    echo "   Install with: pip install uvicorn[standard]"
    exit 1
}

echo "✅ Dependencies OK"
echo ""

# Check CUDA availability
echo "Checking GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || {
    echo "⚠️  Warning: Could not check CUDA status"
}
echo ""

# Start the service
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Starting Embedding Service...                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Service will be available at: http://0.0.0.0:$PORT"
echo "Health check: http://0.0.0.0:$PORT/health"
echo "API endpoint: http://0.0.0.0:$PORT/embed"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Start uvicorn
python3 -m uvicorn services.embedding_service:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --no-access-log
