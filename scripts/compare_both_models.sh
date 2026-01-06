#!/bin/bash
# Compare bge-small-en vs multilingual-e5-small on messenger_clean_small dataset
# This script indexes with BOTH models sequentially, then provides comparison queries

cd /Users/frytos/code/llamaIndex-local-rag
source .venv/bin/activate

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  EMBEDDING MODEL COMPARISON                                          ║"
echo "║  Dataset: messenger_clean_small (207 conversations, 80 MB)           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will create TWO indexes for comparison:"
echo ""
echo "  1. bge-small-en (MLX) → messenger_clean_small_cs700_ov150_bge"
echo "  2. multilingual-e5-small (HF) → messenger_clean_small_cs700_ov150_e5"
echo ""
echo "Expected total time: ~4-6 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/2: Indexing with bge-small-en (MLX)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Load base config
source optimized_config.sh

# Index with bge-small-en
export EMBED_MODEL=BAAI/bge-small-en
export EMBED_DIM=384
export EMBED_BACKEND=mlx
export EMBED_BATCH=64
export PDF_PATH=data/messenger_clean_small
export PGTABLE=messenger_clean_small_cs700_ov150_bge
export RESET_TABLE=1

echo "⚙️  Configuration:"
echo "    EMBED_MODEL: $EMBED_MODEL"
echo "    EMBED_BACKEND: $EMBED_BACKEND"
echo "    EMBED_DIM: $EMBED_DIM"
echo "    EMBED_BATCH: $EMBED_BATCH"
echo "    CHUNK_SIZE: $CHUNK_SIZE"
echo "    CHUNK_OVERLAP: $CHUNK_OVERLAP"
echo "    PDF_PATH: $PDF_PATH"
echo "    PGTABLE: $PGTABLE"
echo "    TOP_K: $TOP_K"
echo ""

START_BGE=$(date +%s)
python rag_low_level_m1_16gb_verbose.py --index-only
BGE_EXIT=$?
END_BGE=$(date +%s)
BGE_TIME=$((END_BGE - START_BGE))

if [ $BGE_EXIT -ne 0 ]; then
    echo ""
    echo "❌ Error during bge-small-en indexing (exit code: $BGE_EXIT)"
    exit 1
fi

echo ""
echo "✅ bge-small-en indexing completed in ${BGE_TIME}s"
echo ""

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/2: Indexing with multilingual-e5-small (HuggingFace)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Index with multilingual-e5-small
export EMBED_MODEL=intfloat/multilingual-e5-small
export EMBED_DIM=384
export EMBED_BACKEND=huggingface
export EMBED_BATCH=64
export PDF_PATH=data/messenger_clean_small
export PGTABLE=messenger_clean_small_cs700_ov150_e5
export RESET_TABLE=1

echo "⚙️  Configuration:"
echo "    EMBED_MODEL: $EMBED_MODEL"
echo "    EMBED_BACKEND: $EMBED_BACKEND"
echo "    EMBED_DIM: $EMBED_DIM"
echo "    EMBED_BATCH: $EMBED_BATCH"
echo "    CHUNK_SIZE: $CHUNK_SIZE"
echo "    CHUNK_OVERLAP: $CHUNK_OVERLAP"
echo "    PDF_PATH: $PDF_PATH"
echo "    PGTABLE: $PGTABLE"
echo "    TOP_K: $TOP_K"
echo ""

START_E5=$(date +%s)
python rag_low_level_m1_16gb_verbose.py --index-only
E5_EXIT=$?
END_E5=$(date +%s)
E5_TIME=$((END_E5 - START_E5))

if [ $E5_EXIT -ne 0 ]; then
    echo ""
    echo "❌ Error during multilingual-e5-small indexing (exit code: $E5_EXIT)"
    exit 1
fi

echo ""
echo "✅ multilingual-e5-small indexing completed in ${E5_TIME}s"
echo ""

# Summary
TOTAL_TIME=$((BGE_TIME + E5_TIME))
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  INDEXING COMPLETE                                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  ✅ bge-small-en:           ${BGE_TIME}s"
echo "  ✅ multilingual-e5-small:  ${E5_TIME}s"
echo "  ⏱️  Total time:             ${TOTAL_TIME}s"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NEXT STEPS: Test Queries"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Test with bge-small-en:"
echo "  PGTABLE=messenger_clean_small_cs700_ov150_bge python rag_low_level_m1_16gb_verbose.py --query-only"
echo ""
echo "Test with multilingual-e5-small:"
echo "  PGTABLE=messenger_clean_small_cs700_ov150_e5 python rag_low_level_m1_16gb_verbose.py --query-only"
echo ""
echo "Suggested test queries:"
echo "  • 'conversations about Paris restaurants'"
echo "  • 'discussions sur les restaurants'"
echo "  • 'weekend plans'"
echo "  • 'vacances et voyages'"
echo ""
