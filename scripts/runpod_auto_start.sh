#!/bin/bash
#
# RunPod Pod Automatic Startup Script
#
# This script runs automatically when a RunPod pod starts.
# It sets up the environment and starts all services.
#
# Services started:
# - PostgreSQL (port 5432)
# - Embedding API (port 8001)
# - vLLM Server (port 8000, if enabled)
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ RunPod Pod Automatic Startup                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Wait for GPU to be ready
echo "⏳ Waiting for GPU initialization..."
sleep 5

# Navigate to workspace
cd /workspace || cd /root

# Check if repo exists
if [ ! -d "llamaIndex-local-rag" ]; then
    echo "📦 Cloning repository..."
    git clone https://github.com/frytos/llamaIndex-local-rag.git
    cd llamaIndex-local-rag
else
    cd llamaIndex-local-rag
    echo "📦 Updating repository..."
    git pull origin main || echo "⚠️  Git pull failed, continuing with existing code"
fi

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install --quiet fastapi uvicorn[standard] requests || echo "⚠️  Dependency installation failed"

# Check if API key is set
if [ -z "$RUNPOD_EMBEDDING_API_KEY" ]; then
    echo "⚠️  WARNING: RUNPOD_EMBEDDING_API_KEY not set!"
    echo "   Embedding service will be UNSECURED"
    echo "   Set this environment variable in RunPod pod settings"
fi

# Start PostgreSQL (if not already running)
echo "🐘 Starting PostgreSQL..."
if ! pgrep -x "postgres" > /dev/null; then
    sudo service postgresql start || echo "⚠️  PostgreSQL start failed"
else
    echo "✅ PostgreSQL already running"
fi

# Start Embedding Service in background
echo "🚀 Starting Embedding Service on port 8001..."
export PORT=8001
export EMBED_MODEL=${EMBED_MODEL:-"BAAI/bge-small-en"}
export EMBED_BACKEND=${EMBED_BACKEND:-"huggingface"}

# Start the service in background with nohup
nohup python3 -m uvicorn services.embedding_service:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    --log-level info \
    > /workspace/embedding_service.log 2>&1 &

EMBEDDING_PID=$!
echo "✅ Embedding service started (PID: $EMBEDDING_PID)"
echo "   Logs: /workspace/embedding_service.log"

# Wait for service to be ready
echo "⏳ Waiting for embedding service to be ready..."
sleep 10

# Check if service is healthy
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Embedding service is healthy!"
    curl -s http://localhost:8001/health | python3 -m json.tool
else
    echo "⚠️  Embedding service health check failed"
    echo "   Check logs: tail -f /workspace/embedding_service.log"
fi

# Start vLLM if enabled
if [ "$USE_VLLM" = "1" ]; then
    echo "🤖 Starting vLLM Server on port 8000..."
    # Add vLLM startup command here if needed
    echo "   (vLLM startup not implemented yet)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ ✅ Pod Startup Complete                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Services:"
echo "  🐘 PostgreSQL:      port 5432"
echo "  🚀 Embedding API:   port 8001 (http://localhost:8001/health)"
echo "  🤖 vLLM Server:     port 8000 (if enabled)"
echo ""
echo "Logs:"
echo "  Embedding: tail -f /workspace/embedding_service.log"
echo ""
echo "To stop embedding service:"
echo "  kill $EMBEDDING_PID"
echo ""
