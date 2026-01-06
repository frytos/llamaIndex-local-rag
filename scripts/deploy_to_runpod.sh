#!/bin/bash
# Quick deploy script for Runpod
# Run this after SSHing into your Runpod pod

set -e  # Exit on error

echo "🚀 Runpod RAG Pipeline Deployment Script"
echo "========================================="
echo ""

# ==========================================
# 1. Check GPU
# ==========================================
echo "📊 Step 1: Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ NVIDIA GPU detected"
else
    echo "❌ ERROR: No NVIDIA GPU found!"
    exit 1
fi
echo ""

# ==========================================
# 2. Install dependencies
# ==========================================
echo "📦 Step 2: Installing dependencies..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "  Installing Python packages..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt

    # Install PyTorch with CUDA support
    echo "  Installing PyTorch with CUDA..."
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo "✅ Dependencies installed"
else
    echo "❌ ERROR: requirements.txt not found"
    exit 1
fi
echo ""

# ==========================================
# 3. Load configuration
# ==========================================
echo "⚙️  Step 3: Loading configuration..."
if [ -f "runpod_config.env" ]; then
    source runpod_config.env
    echo "✅ Configuration loaded"
else
    echo "⚠️  Warning: runpod_config.env not found, using defaults"
    # Set minimal defaults
    export EMBED_BACKEND=torch
    export N_GPU_LAYERS=99
    export N_BATCH=512
fi
echo ""

# ==========================================
# 4. Setup PostgreSQL (optional)
# ==========================================
echo "🐘 Step 4: PostgreSQL setup..."
read -p "Do you want to setup local PostgreSQL? (y/N): " setup_postgres

if [[ $setup_postgres =~ ^[Yy]$ ]]; then
    echo "  Installing PostgreSQL..."
    apt-get update -qq
    apt-get install -y -qq postgresql postgresql-contrib

    echo "  Starting PostgreSQL..."
    service postgresql start

    echo "  Creating database..."
    sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME:-vector_db};" 2>/dev/null || echo "  Database already exists"
    sudo -u postgres psql -c "CREATE USER ${PGUSER:-fryt} WITH PASSWORD '${PGPASSWORD:-frytos}';" 2>/dev/null || echo "  User already exists"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME:-vector_db} TO ${PGUSER:-fryt};"

    echo "  Installing pgvector extension..."
    sudo -u postgres psql -d ${DB_NAME:-vector_db} -c "CREATE EXTENSION IF NOT EXISTS vector;"

    echo "✅ PostgreSQL setup complete"
else
    echo "⏭️  Skipping PostgreSQL setup (using external database)"
fi
echo ""

# ==========================================
# 5. Test GPU + PyTorch
# ==========================================
echo "🧪 Step 5: Testing GPU + PyTorch..."
python3 << EOF
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("✅ GPU test passed")
else:
    print("❌ ERROR: CUDA not available")
    exit(1)
EOF
echo ""

# ==========================================
# 6. Download models (cache)
# ==========================================
echo "📥 Step 6: Pre-downloading models..."
read -p "Download embedding model to cache? (Y/n): " download_models

if [[ ! $download_models =~ ^[Nn]$ ]]; then
    echo "  Downloading BAAI/bge-small-en..."
    python3 << EOF
from huggingface_hub import snapshot_download
import os
cache_dir = os.getenv('HF_HOME', '/workspace/huggingface_cache')
snapshot_download('BAAI/bge-small-en', cache_dir=cache_dir)
print("✅ Model downloaded")
EOF
else
    echo "⏭️  Skipping model download"
fi
echo ""

# ==========================================
# 7. Summary
# ==========================================
echo "✅ Setup Complete!"
echo "=================="
echo ""
echo "Environment:"
echo "  EMBED_BACKEND: ${EMBED_BACKEND:-torch}"
echo "  N_GPU_LAYERS: ${N_GPU_LAYERS:-99}"
echo "  N_BATCH: ${N_BATCH:-512}"
echo "  CTX: ${CTX:-16384}"
echo "  PGHOST: ${PGHOST:-localhost}"
echo "  DB_NAME: ${DB_NAME:-vector_db}"
echo ""
echo "Next steps:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Load config: source runpod_config.env"
echo "  3. Run indexing: python3 rag_low_level_m1_16gb_verbose.py"
echo "  4. Run query: python3 rag_low_level_m1_16gb_verbose.py --query-only --query 'your query'"
echo ""
echo "Troubleshooting:"
echo "  • Check GPU: nvidia-smi"
echo "  • Check PostgreSQL: service postgresql status"
echo "  • Check logs: tail -f *.log"
echo ""
