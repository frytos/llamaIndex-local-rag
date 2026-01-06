#!/bin/bash
# Index messenger_clean_small with bge-small-en (MLX) for baseline comparison

cd /Users/frytos/code/llamaIndex-local-rag
source .venv/bin/activate

# Load base config
source optimized_config.sh

# Use bge-small-en with MLX (current model)
export EMBED_MODEL=BAAI/bge-small-en
export EMBED_DIM=384
export EMBED_BACKEND=mlx
export EMBED_BATCH=64

# Set table name for bge-small-en index
export PDF_PATH=data/messenger_clean_small
export PGTABLE=messenger_clean_small_cs700_ov150_bge
export RESET_TABLE=1

echo "==========================================="
echo "INDEXING WITH BGE-SMALL-EN (BASELINE)"
echo "==========================================="
echo "Model: $EMBED_MODEL"
echo "Backend: $EMBED_BACKEND (MLX acceleration)"
echo "Table: $PGTABLE"
echo "Documents: $PDF_PATH"
echo "==========================================="
echo ""
echo "Dataset: 207 conversations, 80 MB"
echo "Expected time: ~2-3 minutes (fast with MLX)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python rag_low_level_m1_16gb_verbose.py --index-only
fi
