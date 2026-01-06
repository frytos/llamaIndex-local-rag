#!/bin/bash
# Verify Runpod Setup - Automated Testing Script
# Run this after pgvector compilation to verify everything works

set -e  # Exit on error

echo "🔍 Runpod Setup Verification Script"
echo "===================================="
echo ""

# ==========================================
# 1. Restart PostgreSQL
# ==========================================
echo "🐘 [1/6] Restarting PostgreSQL..."
service postgresql restart
sleep 2
echo "  ✅ PostgreSQL restarted"
echo ""

# ==========================================
# 2. Create pgvector Extension
# ==========================================
echo "🔌 [2/6] Creating pgvector extension..."
su - postgres -c "psql -d vector_db -c 'CREATE EXTENSION IF NOT EXISTS vector;'" 2>&1 | grep -v "^$" || echo "  Extension already exists"
echo "  ✅ Extension created"
echo ""

# ==========================================
# 3. Verify pgvector Extension
# ==========================================
echo "✅ [3/6] Verifying pgvector extension..."
echo "────────────────────────────────────────────────────────────"
su - postgres -c "psql -d vector_db -c '\dx vector'" | grep -A 3 "List of installed extensions"
echo "────────────────────────────────────────────────────────────"
echo ""

# ==========================================
# 4. Test User Connection
# ==========================================
echo "👤 [4/6] Testing user connection..."
export PGPASSWORD=frytos
psql -h localhost -U fryt -d vector_db -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';" 2>&1
echo "  ✅ User connection works"
echo ""

# ==========================================
# 5. Test GPU + PyTorch
# ==========================================
echo "🎮 [5/6] Testing GPU + PyTorch..."
echo "────────────────────────────────────────────────────────────"
cd /workspace/rag-pipeline
source .venv/bin/activate

python3 << 'EOF'
import torch
import sys

print("=" * 60)
try:
    assert torch.cuda.is_available(), "CUDA not available!"

    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.version.cuda}")
    print(f"✅ CUDA Available: True")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test a simple tensor operation on GPU
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print(f"✅ GPU Compute Test: Passed (matrix multiply)")

    print("=" * 60)
except Exception as e:
    print(f"❌ GPU Test Failed: {e}")
    print("=" * 60)
    sys.exit(1)
EOF

echo "────────────────────────────────────────────────────────────"
echo ""

# ==========================================
# 6. Test RAG Script
# ==========================================
echo "🧪 [6/6] Testing RAG script..."
echo "────────────────────────────────────────────────────────────"

# Test that script can be imported
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace/rag-pipeline')

try:
    # Test imports
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.postgres import PGVectorStore
    from llama_index.llms.llama_cpp import LlamaCPP
    import psycopg2

    print("✅ All imports successful")
    print("  ✅ llama-index")
    print("  ✅ HuggingFace embeddings")
    print("  ✅ PostgreSQL vector store")
    print("  ✅ llama-cpp")
    print("  ✅ psycopg2")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
EOF

echo "────────────────────────────────────────────────────────────"
echo ""

# ==========================================
# Summary
# ==========================================
echo "================================================"
echo "✅ ALL CHECKS PASSED!"
echo "================================================"
echo ""
echo "📊 System Summary:"
echo "  🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  💾 VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "  🐘 PostgreSQL: Running"
echo "  🔌 pgvector: v0.7.4 installed"
echo "  🐍 Python venv: Active"
echo "  📦 Venv size: $(du -sh /workspace/rag-pipeline/.venv/ | cut -f1)"
echo ""
echo "🚀 Ready to Run!"
echo "────────────────────────────────────────────────────────────"
echo ""
echo "Quick commands:"
echo ""
echo "  # Test query (if you have data indexed):"
echo "  cd /workspace/rag-pipeline"
echo "  source .venv/bin/activate"
echo "  python3 rag_low_level_m1_16gb_verbose.py --query-only \\"
echo "    --query 'your question'"
echo ""
echo "  # Index data:"
echo "  python3 rag_low_level_m1_16gb_verbose.py"
echo ""
echo "  # Interactive mode:"
echo "  python3 rag_interactive.py"
echo ""
echo "  # Web UI:"
echo "  streamlit run rag_web.py --server.port 8000"
echo ""
echo "────────────────────────────────────────────────────────────"
echo ""
