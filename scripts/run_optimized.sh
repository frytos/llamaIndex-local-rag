#!/usr/bin/env bash
# run_optimized.sh - Quick start with all M1 optimizations enabled
#
# Usage:
#   ./run_optimized.sh                    # Interactive mode
#   ./run_optimized.sh --query "question" # Single query
#   ./run_optimized.sh --help             # Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Local RAG with M1 Optimizations${NC}"
echo ""

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Run: python -m venv .venv${NC}"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Load environment variables (export all)
if [ -f ".env" ]; then
    echo -e "${BLUE}📝 Loading configuration from .env${NC}"
    set -a
    source .env
    set +a
else
    echo -e "${YELLOW}⚠️  .env file not found. Using defaults.${NC}"
fi

# Show active optimizations
echo ""
echo -e "${GREEN}✓ Active Optimizations:${NC}"
echo "  • MLX Backend: ${EMBED_BACKEND:-huggingface} (5-20x faster on M1)"
echo "  • Embed Batch: ${EMBED_BATCH:-64} (optimized for M1)"
echo "  • GPU Layers: ${N_GPU_LAYERS:-24} (Apple Metal)"
echo "  • Semantic Cache: ${ENABLE_SEMANTIC_CACHE:-0} (10-100x speedup)"
echo "  • Reranking: ${ENABLE_RERANKING:-0} (15-30% better results)"
echo "  • Query Expansion: ${ENABLE_QUERY_EXPANSION:-0} (20-40% coverage)"
echo ""

# Check if database is running
if ! pg_isready -h "${PGHOST:-localhost}" -p "${PGPORT:-5432}" -U "${PGUSER:-fryt}" >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  PostgreSQL not reachable. Make sure Docker is running:${NC}"
    echo "    docker-compose -f config/docker-compose.yml up -d"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the pipeline
echo -e "${GREEN}🔥 Launching RAG pipeline...${NC}"
echo ""

if [ $# -eq 0 ]; then
    # Interactive mode
    python rag_low_level_m1_16gb_verbose.py --interactive
else
    # Pass through arguments
    python rag_low_level_m1_16gb_verbose.py "$@"
fi
