#!/bin/bash
# Re-index messenger data with multilingual-e5-small for better French/English support

cd /Users/frytos/code/llamaIndex-local-rag
source .venv/bin/activate

# Load base config
source optimized_config.sh

# Override with multilingual embedding model
export EMBED_MODEL=intfloat/multilingual-e5-small
export EMBED_DIM=384
export EMBED_BACKEND=huggingface  # E5 doesn't have MLX support yet
export EMBED_BATCH=64

# Set table name to distinguish from bge-small-en index
export PDF_PATH=data/messenger_clean_small
export PGTABLE=messenger_clean_small_cs700_ov150_e5
export RESET_TABLE=1

echo "==========================================="
echo "RE-INDEXING WITH MULTILINGUAL-E5-SMALL"
echo "==========================================="
echo "Model: $EMBED_MODEL"
echo "Backend: $EMBED_BACKEND (no MLX yet for E5)"
echo "Table: $PGTABLE"
echo "Documents: $PDF_PATH"
echo "==========================================="
echo ""
echo "This will create a NEW index for comparison"
echo "Your current bge-small-en index will be preserved"
echo ""
echo "Dataset: 207 conversations, 80 MB"
echo "Expected time: ~2-3 minutes (slower without MLX)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python rag_low_level_m1_16gb_verbose.py --index-only
fi
