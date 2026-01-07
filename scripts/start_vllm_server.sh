#!/bin/bash
# Start vLLM Server on Runpod
# This keeps the model loaded in GPU memory for fast queries

set -e

echo "🚀 Starting vLLM Server on RTX 4090"
echo "===================================="
echo ""

# Configuration
MODEL="${VLLM_MODEL:-TheBloke/Mistral-7B-Instruct-v0.2-AWQ}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
GPU_MEM="${VLLM_GPU_MEMORY:-0.8}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Host: $HOST:$PORT"
echo "  GPU Memory: ${GPU_MEM} (80%)"
echo "  Max Length: $MAX_LEN tokens"
echo ""

# Check if vLLM is installed
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not installed!"
    echo "   Install with: pip install vllm"
    exit 1
fi

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT already in use!"
    echo "   Kill existing server with: pkill -f 'vllm serve'"
    echo "   Or use different port: export VLLM_PORT=8001"
    exit 1
fi

echo "Starting vLLM server (this takes ~60s one-time warmup)..."
echo "───────────────────────────────────────────────────────────"
echo ""

# Start vLLM server
vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --dtype float16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --tensor-parallel-size 1

# Note: This blocks. Run in background with &, or in separate terminal/tmux
