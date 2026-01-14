#!/bin/bash
# Quick backend performance test

source .venv/bin/activate
set -a && source .env && set +a

echo "=========================================="
echo "🔬 Backend Performance Test"
echo "=========================================="
echo ""

# Test with small sample
export PDF_PATH=data/01-messenger/inbox/quentinbellet_10223556634716316
export PGTABLE=test_backend_speed
export RESET_TABLE=1

echo "Test 1: MLX Backend (batch=32)"
echo "----------------------------------------"
export EMBED_BACKEND=mlx
export EMBED_BATCH=32
time python rag_low_level_m1_16gb_verbose.py 2>&1 | grep -E "(Embedding batches|batch\]|MLX model loaded)"

echo ""
echo ""
echo "Test 2: PyTorch MPS Backend (batch=32)"
echo "----------------------------------------"
export EMBED_BACKEND=huggingface
export EMBED_BATCH=32
time python rag_low_level_m1_16gb_verbose.py 2>&1 | grep -E "(Embedding batches|batch\]|Embedding model loaded)"

echo ""
echo "=========================================="
echo "✅ Test complete! Check which was faster."
echo "=========================================="
