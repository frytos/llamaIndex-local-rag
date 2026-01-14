#!/bin/bash
# ============================================================================
# Manual Setup Completion Script
# ============================================================================
# Run this inside the pod to complete initialization if auto-setup failed
# ============================================================================

set -e

echo "=========================================================================="
echo "Completing RunPod Setup Manually"
echo "=========================================================================="
echo ""

# Activate virtual environment
echo "[1/3] Activating Python environment..."
source /workspace/rag-pipeline/.venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install Python dependencies
echo "[2/3] Installing Python dependencies (3-5 minutes)..."
echo "Installing main requirements..."
cd /workspace/rag-pipeline
pip install -r requirements.txt

echo ""
echo "Installing vLLM requirements..."
pip install -r config/requirements_vllm.txt

echo ""
echo "✅ All dependencies installed"
echo ""

# Check vLLM installed
echo "Verifying vLLM..."
if python3 -c "import vllm" 2>/dev/null; then
    echo "✅ vLLM is installed"
    vllm_version=$(python3 -c "import vllm; print(vllm.__version__)")
    echo "   Version: $vllm_version"
else
    echo "❌ vLLM not found"
    exit 1
fi

echo ""

# Create and start vLLM server
echo "[3/3] Starting vLLM server..."

cat > /workspace/rag-pipeline/start_vllm.sh << 'VLLM_SCRIPT'
#!/bin/bash
source /workspace/rag-pipeline/.venv/bin/activate

export VLLM_MODEL="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
export CUDA_VISIBLE_DEVICES=0

echo "Starting vLLM server..."
echo "Model: $VLLM_MODEL"
echo "Port: 8000"
echo ""

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    > /workspace/rag-pipeline/logs/vllm.log 2>&1 &

echo $! > /workspace/rag-pipeline/vllm.pid
echo "vLLM server started (PID: $(cat /workspace/rag-pipeline/vllm.pid))"
echo "Monitor with: tail -f /workspace/rag-pipeline/logs/vllm.log"
VLLM_SCRIPT

chmod +x /workspace/rag-pipeline/start_vllm.sh

# Create logs directory if missing
mkdir -p /workspace/rag-pipeline/logs

# Start vLLM
echo "Launching vLLM server..."
bash /workspace/rag-pipeline/start_vllm.sh

echo ""
echo "✅ vLLM starting (downloading model, will take 60-90 seconds)"
echo ""
echo "=========================================================================="
echo "SETUP COMPLETE"
echo "=========================================================================="
echo ""
echo "Monitor vLLM startup:"
echo "  tail -f /workspace/rag-pipeline/logs/vllm.log"
echo ""
echo "Test when ready (wait 90 seconds):"
echo "  curl http://localhost:8000/health"
echo ""
echo "Expected response: {\"status\":\"ok\"}"
echo ""
