#!/bin/bash
# Runpod Startup Script - Auto-setup RAG Pipeline
# This script runs automatically when the pod starts

set -e  # Exit on error

echo "🚀 RAG Pipeline Auto-Startup"
echo "=============================="
date
echo ""

# ==========================================
# Configuration
# ==========================================
REPO_URL="${REPO_URL:-https://github.com/frytos/llamaIndex-local-rag.git}"
# Note: This is a private repo. Set GH_TOKEN env var for authentication:
# REPO_URL="https://${GH_TOKEN}@github.com/frytos/llamaIndex-local-rag.git"
WORK_DIR="/workspace/rag-pipeline"
VENV_DIR="$WORK_DIR/.venv"

# ==========================================
# 1. System Info
# ==========================================
echo "📊 System Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ==========================================
# 2. Clone or Update Repository
# ==========================================
if [ -d "$WORK_DIR" ]; then
    echo "📂 Repository exists, pulling latest changes..."
    cd "$WORK_DIR"
    git pull || echo "⚠️  Git pull failed, continuing with existing code"
else
    echo "📥 Cloning repository..."
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi
echo "✅ Repository ready"
echo ""

# ==========================================
# 3. Python Environment
# ==========================================
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"
echo ""

# ==========================================
# 4. Install Dependencies
# ==========================================
echo "📦 Installing dependencies..."
echo "  This may take 2-4 minutes on first run..."
echo ""

echo "  [1/3] Upgrading pip..."
pip install --upgrade pip 2>&1 | grep -E "(Successfully|Requirement|Collecting)" || echo "    (in progress...)"

if [ -f "requirements.txt" ]; then
    echo ""
    echo "  [2/3] Installing requirements.txt..."
    echo "  ────────────────────────────────────"

    # Show progress for each package
    pip install -r requirements.txt 2>&1 | grep -E "(Successfully|Requirement|Collecting|Installing|Downloading)" | sed 's/^/    /'

    echo "  ────────────────────────────────────"
    echo "✅ Requirements installed"
else
    echo "⚠️  requirements.txt not found"
fi

# Install PyTorch with CUDA (if not already installed)
echo ""
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  [3/4] Installing PyTorch 2.4.0 + CUDA 12.4..."
    echo "  ────────────────────────────────────"
    echo "  ⏳ This is a large download (~2GB), please be patient..."

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124 2>&1 | \
        grep -E "(Successfully|Requirement|Collecting|Installing|Downloading|%)" | \
        sed 's/^/    /'

    echo "  ────────────────────────────────────"
    echo "✅ PyTorch installed"
else
    echo "  [3/4] PyTorch with CUDA already installed ✅"
fi
echo ""

# Install llama-cpp-python with CUDA support (critical for GPU acceleration)
echo "  [4/4] Installing llama-cpp-python with CUDA..."
echo "  ────────────────────────────────────"
echo "  🔨 Compiling llama.cpp with CUDA support (~3-5 minutes)..."
echo "  This ensures LLM runs on GPU (5-10x faster than CPU)"
echo ""

# Uninstall CPU-only version if exists
pip uninstall -y llama-cpp-python 2>/dev/null || true

# Install with CUDA support (compile from source)
CMAKE_ARGS="-DGGML_CUDA=on" \
FORCE_CMAKE=1 \
pip install llama-cpp-python==0.3.16 --force-reinstall --no-cache-dir 2>&1 | \
    grep -E "(Successfully|Found CUDA|GGML_CUDA|Building|Installing)" | \
    sed 's/^/    /'

echo ""
echo "  ────────────────────────────────────"
echo "✅ llama-cpp-python with CUDA installed"
echo ""

# ==========================================
# 5. Load Configuration
# ==========================================
if [ -f "runpod_config.env" ]; then
    echo "⚙️  Loading configuration..."
    source runpod_config.env
    echo "✅ Configuration loaded"
else
    echo "⚠️  runpod_config.env not found, using defaults"
    export EMBED_BACKEND=torch
    export N_GPU_LAYERS=99
    export N_BATCH=512
    export CTX=16384
fi
echo ""

# ==========================================
# 6. Setup PostgreSQL (if requested)
# ==========================================
if [ "${SETUP_POSTGRES:-0}" = "1" ]; then
    echo "🐘 Setting up PostgreSQL..."

    # Check if PostgreSQL is already installed
    if ! command -v psql &> /dev/null; then
        echo "  Installing PostgreSQL..."
        apt-get update -qq
        apt-get install -y -qq postgresql postgresql-contrib
    fi

    # Install pgvector extension (not in Ubuntu repos, compile from source)
    if [ ! -f "/usr/share/postgresql/14/extension/vector.control" ]; then
        echo "  Installing pgvector extension from source..."
        apt-get install -y -qq build-essential postgresql-server-dev-14 git

        cd /tmp
        git clone --branch v0.7.4 --depth 1 https://github.com/pgvector/pgvector.git
        cd pgvector
        make -s
        make install -s
        cd /workspace/rag-pipeline

        echo "  ✅ pgvector compiled and installed"
    fi

    # Start PostgreSQL
    service postgresql start || service postgresql restart

    # Create database and user
    # Note: In Docker containers, we're already root, so use 'su' instead of 'sudo' (which isn't installed)
    su - postgres -c "psql -c \"CREATE DATABASE ${DB_NAME:-vector_db};\"" 2>/dev/null || echo "  Database exists"
    su - postgres -c "psql -c \"CREATE USER ${PGUSER:-fryt} WITH PASSWORD '${PGPASSWORD:-frytos}';\"" 2>/dev/null || echo "  User exists"
    su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME:-vector_db} TO ${PGUSER:-fryt};\""
    su - postgres -c "psql -d ${DB_NAME:-vector_db} -c 'CREATE EXTENSION IF NOT EXISTS vector;'"

    echo "✅ PostgreSQL ready"
else
    echo "⏭️  Skipping PostgreSQL setup (set SETUP_POSTGRES=1 to enable)"
fi
echo ""

# ==========================================
# 7. Test GPU
# ==========================================
echo "🧪 Testing GPU + PyTorch..."
python3 << 'EOF'
import torch
import sys

try:
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"  ✅ PyTorch: {torch.__version__}")
    print(f"  ✅ CUDA: {torch.version.cuda}")
    print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"  ❌ GPU test failed: {e}")
    sys.exit(1)
EOF
echo ""

# ==========================================
# 8. Pre-download Models (optional)
# ==========================================
if [ "${DOWNLOAD_MODELS:-0}" = "1" ]; then
    echo "📥 Pre-downloading models..."
    python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

cache_dir = os.getenv('HF_HOME', '/workspace/huggingface_cache')
os.makedirs(cache_dir, exist_ok=True)

print("  Downloading BAAI/bge-small-en...")
snapshot_download('BAAI/bge-small-en', cache_dir=cache_dir)
print("  ✅ Model cached")
EOF
    echo "✅ Models downloaded"
else
    echo "⏭️  Skipping model pre-download (set DOWNLOAD_MODELS=1 to enable)"
fi
echo ""

# ==========================================
# 9. Run Initial Command (if specified)
# ==========================================
if [ -n "$RUN_COMMAND" ]; then
    echo "🎯 Running initial command: $RUN_COMMAND"
    eval "$RUN_COMMAND"
else
    echo "✅ Setup complete! No initial command specified."
fi
echo ""

# ==========================================
# 10. Summary
# ==========================================
echo "================================================"
echo "✅ RAG Pipeline Ready!"
echo "================================================"
echo ""
echo "Environment:"
echo "  Location: $WORK_DIR"
echo "  Python: $(which python3)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "Quick commands:"
echo "  cd $WORK_DIR"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 rag_low_level_m1_16gb_verbose.py --help"
echo ""
echo "To run a query:"
echo "  python3 rag_low_level_m1_16gb_verbose.py --query-only \\"
echo "    --query 'when did I go to New York'"
echo ""
echo "================================================"

# Keep container running (ALWAYS, to prevent restart loop)
# This is essential in Runpod when using Docker Command
echo "🔄 Keeping container alive (press Ctrl+C to exit)..."
echo "   SSH into the pod to use it: ssh <pod-address>"
echo ""

# Keep process running forever
tail -f /dev/null
