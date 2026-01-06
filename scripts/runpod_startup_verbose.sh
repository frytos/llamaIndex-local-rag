#!/bin/bash
# Runpod Startup Script - VERBOSE MODE
# This version shows ALL installation logs (no filtering)

set -e  # Exit on error

echo "🚀 RAG Pipeline Auto-Startup (VERBOSE MODE)"
echo "============================================"
date
echo ""

# ==========================================
# Configuration
# ==========================================
REPO_URL="${REPO_URL:-https://github.com/frytos/llamaIndex-local-rag.git}"
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
# 4. Install Dependencies (VERBOSE)
# ==========================================
echo "📦 Installing dependencies (VERBOSE MODE - ALL LOGS SHOWN)..."
echo "==============================================================="
echo ""

echo "⬆️  [1/3] Upgrading pip..."
echo "────────────────────────────────────────────────────────────"
pip install --upgrade pip
echo ""

if [ -f "requirements.txt" ]; then
    echo "📋 [2/3] Installing requirements.txt (FULL OUTPUT)..."
    echo "────────────────────────────────────────────────────────────"
    echo ""

    # Count total packages
    TOTAL_PACKAGES=$(grep -v "^#" requirements.txt | grep -v "^$" | wc -l | xargs)
    echo "  📦 Installing $TOTAL_PACKAGES packages from requirements.txt"
    echo ""

    # Install with full output
    pip install -v -r requirements.txt

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "✅ Requirements installed"
else
    echo "⚠️  requirements.txt not found"
fi

# Install PyTorch with CUDA
echo ""
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "🔥 [3/3] Installing PyTorch 2.4.0 + CUDA 12.4 (FULL OUTPUT)..."
    echo "────────────────────────────────────────────────────────────"
    echo "  📊 Package size: ~2GB"
    echo "  ⏱️  Expected time: 1-3 minutes"
    echo ""

    pip install -v torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "✅ PyTorch installed"
else
    echo "✅ [3/3] PyTorch with CUDA already installed"
fi
echo ""
echo "==============================================================="
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
fi
echo ""

# ==========================================
# 6. Setup PostgreSQL (if requested)
# ==========================================
if [ "${SETUP_POSTGRES:-0}" = "1" ]; then
    echo "🐘 Setting up PostgreSQL..."
    echo "────────────────────────────────────"

    if ! command -v psql &> /dev/null; then
        echo "  📥 Installing PostgreSQL (this may take 1-2 minutes)..."
        apt-get update
        apt-get install -y postgresql postgresql-contrib
    fi

    # Install pgvector extension (not in Ubuntu repos, compile from source)
    if [ ! -f "/usr/share/postgresql/14/extension/vector.control" ]; then
        echo "  🔨 Compiling pgvector extension from source (~30 seconds)..."
        apt-get install -y build-essential postgresql-server-dev-14 git

        cd /tmp
        git clone --branch v0.7.4 --depth 1 https://github.com/pgvector/pgvector.git
        cd pgvector
        make
        make install
        cd /workspace/rag-pipeline

        echo "  ✅ pgvector compiled and installed"
    fi

    echo "  🚀 Starting PostgreSQL..."
    service postgresql start || service postgresql restart

    echo "  🔧 Creating database and user..."
    # Note: In Docker containers, we're already root, so use 'su' instead of 'sudo' (which isn't installed)
    su - postgres -c "psql -c \"CREATE DATABASE ${DB_NAME:-vector_db};\"" 2>/dev/null || echo "  ℹ️  Database exists"
    su - postgres -c "psql -c \"CREATE USER ${PGUSER:-fryt} WITH PASSWORD '${PGPASSWORD:-frytos}';\"" 2>/dev/null || echo "  ℹ️  User exists"
    su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME:-vector_db} TO ${PGUSER:-fryt};\""

    echo "  🔌 Installing pgvector extension..."
    su - postgres -c "psql -d ${DB_NAME:-vector_db} -c 'CREATE EXTENSION IF NOT EXISTS vector;'"

    echo "────────────────────────────────────"
    echo "✅ PostgreSQL ready"
else
    echo "⏭️  Skipping PostgreSQL setup (set SETUP_POSTGRES=1 to enable)"
fi
echo ""

# ==========================================
# 7. Test GPU
# ==========================================
echo "🧪 Testing GPU + PyTorch..."
echo "────────────────────────────────────"
python3 << 'EOF'
import torch
import sys

try:
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"  ✅ PyTorch: {torch.__version__}")
    print(f"  ✅ CUDA: {torch.version.cuda}")
    print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  ✅ GPU Count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"  ❌ GPU test failed: {e}")
    sys.exit(1)
EOF
echo "────────────────────────────────────"
echo ""

# ==========================================
# 8. Pre-download Models (optional)
# ==========================================
if [ "${DOWNLOAD_MODELS:-0}" = "1" ]; then
    echo "📥 Pre-downloading models..."
    echo "────────────────────────────────────"
    python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

cache_dir = os.getenv('HF_HOME', '/workspace/huggingface_cache')
os.makedirs(cache_dir, exist_ok=True)

print("  📥 Downloading BAAI/bge-small-en (~133MB)...")
snapshot_download('BAAI/bge-small-en', cache_dir=cache_dir)
print("  ✅ Model cached")
EOF
    echo "────────────────────────────────────"
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
    echo "────────────────────────────────────"
    eval "$RUN_COMMAND"
    echo "────────────────────────────────────"
else
    echo "✅ Setup complete! No initial command specified."
fi
echo ""

# ==========================================
# 10. Summary
# ==========================================
echo "========================================================================"
echo "✅ RAG Pipeline Ready!"
echo "========================================================================"
echo ""
echo "📍 Location: $WORK_DIR"
echo "🐍 Python: $(which python3)"
echo "🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "💾 VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo ""
echo "🚀 Quick Start Commands:"
echo "────────────────────────────────────────────────────────────────────────"
echo "  cd $WORK_DIR"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 rag_low_level_m1_16gb_verbose.py --help"
echo ""
echo "📊 Run a test query:"
echo "  python3 rag_low_level_m1_16gb_verbose.py --query-only \\"
echo "    --query 'when did I go to New York'"
echo ""
echo "🔍 Index your data:"
echo "  python3 rag_low_level_m1_16gb_verbose.py"
echo "────────────────────────────────────────────────────────────────────────"
echo ""
echo "📝 Logs saved to: /tmp/runpod_setup.log"
echo ""

# Keep container running (if this is the main process)
if [ "${KEEP_ALIVE:-0}" = "1" ]; then
    echo "🔄 Keeping container alive..."
    tail -f /dev/null
fi
