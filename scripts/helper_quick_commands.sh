#!/bin/bash
#
# Quick Commands - Copy/paste ready commands for post-embedding execution
#
# Usage: Source this file or copy commands as needed
#

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Quick Commands Ready${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================================================
# Phase 1: Install MLX
# ============================================================================
install_mlx() {
    echo -e "${GREEN}[1/3] Installing MLX...${NC}"
    pip install mlx mlx-embedding-models rank-bm25

    echo -e "${GREEN}[2/3] Verifying MLX...${NC}"
    python -c "import mlx.core as mx; print(f'✓ MLX {mx.__version__}')"

    echo -e "${GREEN}[3/3] Verifying mlx-embedding-models...${NC}"
    python -c "from mlx_embedding_models.embedding import EmbeddingModel; print('✓ mlx-embedding-models')"

    echo -e "${GREEN}✓ MLX installation complete${NC}"
}

# ============================================================================
# Phase 2: Benchmark
# ============================================================================
benchmark_mlx() {
    echo -e "${GREEN}Running benchmark (5 minutes)...${NC}"
    python scripts/benchmark_embeddings.py \
        --compare \
        --model BAAI/bge-large-en-v1.5 \
        --batch-sizes "16,32,64,128"
}

# ============================================================================
# Phase 3: Apply HNSW
# ============================================================================
apply_hnsw() {
    echo -e "${GREEN}Applying HNSW index (3-5 minutes)...${NC}"
    ./scripts/apply_hnsw.sh inbox_clean
}

# ============================================================================
# Phase 4: Test Query Speed
# ============================================================================
test_query_speed() {
    echo -e "${GREEN}Testing query speed with HNSW...${NC}"
    time python rag_low_level_m1_16gb_verbose.py \
        --query-only \
        --query "What did Elena say about Morocco?"
}

# ============================================================================
# Phase 5: Test MLX on Small Subset
# ============================================================================
test_mlx_small() {
    echo -e "${GREEN}Testing MLX with 50 conversations...${NC}"
    EMBED_BACKEND=mlx \
    EMBED_BATCH=64 \
    CHUNK_SIZE=300 \
    CHUNK_OVERLAP=100 \
    PDF_PATH=data/inbox_small \
    PGTABLE=test_mlx_small \
    RESET_TABLE=1 \
    python rag_low_level_m1_16gb_verbose.py --index-only

    echo -e "${GREEN}Querying test index...${NC}"
    PGTABLE=test_mlx_small \
    ENABLE_FILTERS=1 \
    HYBRID_ALPHA=0.5 \
    python rag_low_level_m1_16gb_verbose.py \
        --query-only \
        --query "participant:EB Morocco"
}

# ============================================================================
# Phase 6: Compare Models
# ============================================================================
compare_models() {
    echo -e "${GREEN}Comparing embedding models...${NC}"
    python scripts/compare_models.py --backend mlx
}

# ============================================================================
# Phase 7: Full Re-index with MLX
# ============================================================================
reindex_mlx_full() {
    echo -e "${YELLOW}⚠️  This will take 5-8 minutes${NC}"
    echo -e "${YELLOW}⚠️  Make sure benchmark results look good first!${NC}"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        return 1
    fi

    echo -e "${GREEN}Creating backup...${NC}"
    PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD not set} psql -h localhost -U fryt -d vector_db -c \
        "CREATE TABLE inbox_clean_backup AS SELECT * FROM inbox_clean;"

    echo -e "${GREEN}Re-indexing with MLX...${NC}"
    EMBED_BACKEND=mlx \
    EMBED_BATCH=64 \
    EXTRACT_CHAT_METADATA=1 \
    CHUNK_SIZE=300 \
    CHUNK_OVERLAP=100 \
    PDF_PATH=data/inbox_clean \
    PGTABLE=inbox_mlx_optimized \
    RESET_TABLE=1 \
    python rag_low_level_m1_16gb_verbose.py --index-only

    echo -e "${GREEN}Applying HNSW to new table...${NC}"
    ./scripts/apply_hnsw.sh inbox_mlx_optimized
}

# ============================================================================
# Phase 8: Test Production Query
# ============================================================================
test_production() {
    echo -e "${GREEN}Testing production query pipeline...${NC}"
    PGTABLE=inbox_mlx_optimized \
    HYBRID_ALPHA=0.5 \
    ENABLE_FILTERS=1 \
    MMR_THRESHOLD=0.7 \
    TOP_K=6 \
    LOG_FULL_CHUNKS=1 \
    COLORIZE_CHUNKS=1 \
    python rag_low_level_m1_16gb_verbose.py \
        --query-only \
        --interactive
}

# ============================================================================
# All-in-one execution
# ============================================================================
run_all() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Running Full Optimization Plan${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    install_mlx
    echo ""

    benchmark_mlx
    echo ""

    apply_hnsw
    echo ""

    test_query_speed
    echo ""

    test_mlx_small
    echo ""

    compare_models
    echo ""

    echo -e "${YELLOW}Ready to re-index full corpus with MLX?${NC}"
    echo -e "${YELLOW}This will take 5-8 minutes (vs 45-50 min baseline)${NC}"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        reindex_mlx_full
        echo ""
        test_production
    fi

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  ✓ Optimization Complete${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# ============================================================================
# Usage instructions
# ============================================================================
show_usage() {
    echo "Available commands:"
    echo ""
    echo "  install_mlx         - Install MLX dependencies"
    echo "  benchmark_mlx       - Benchmark HuggingFace vs MLX"
    echo "  apply_hnsw          - Apply HNSW index (faster queries)"
    echo "  test_query_speed    - Test query speed"
    echo "  test_mlx_small      - Test MLX on small subset"
    echo "  compare_models      - Compare bge-small vs bge-large"
    echo "  reindex_mlx_full    - Re-index full corpus with MLX"
    echo "  test_production     - Test production query pipeline"
    echo "  run_all             - Run everything in sequence"
    echo ""
    echo "Usage:"
    echo "  source QUICK_COMMANDS.sh"
    echo "  install_mlx"
    echo "  benchmark_mlx"
    echo "  ..."
    echo ""
    echo "Or run everything:"
    echo "  source QUICK_COMMANDS.sh && run_all"
}

# Show usage if sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script should be sourced:"
    echo "  source QUICK_COMMANDS.sh"
    echo ""
    show_usage
else
    show_usage
fi
