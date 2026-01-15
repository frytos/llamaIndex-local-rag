#!/bin/bash
# Optimized RAG Configuration for M1 Mac Mini 16GB
# December 2025 - Performance tuned
#
# Usage:
#   source optimized_config.sh
#   python rag_low_level_m1_16gb_verbose.py --query-only --interactive

# =============================================================================
# LLM Configuration (30% faster generation)
# =============================================================================
export N_GPU_LAYERS=30              # Offload 30/32 layers to Metal GPU
export N_BATCH=256                  # Larger batch for efficiency
export CTX=4096                     # Context window (up from 3072)
export MAX_NEW_TOKENS=256           # Max generation length
export TEMP=0.1                     # Low temperature for factual answers

# =============================================================================
# Embedding Configuration (11x faster with MLX!)
# =============================================================================
export EMBED_BACKEND=mlx            # Use Apple MLX (759 chunks/s vs 67)
export EMBED_MODEL=BAAI/bge-small-en
export EMBED_DIM=384
export EMBED_BATCH=128              # MLX handles larger batches

# =============================================================================
# Chunking Configuration
# =============================================================================
export CHUNK_SIZE=700               # Current chunks (or 500 for re-index)
export CHUNK_OVERLAP=150            # 21% overlap

# =============================================================================
# Retrieval Configuration
# =============================================================================
export TOP_K=4                      # Number of chunks to retrieve
export HYBRID_ALPHA=1.0             # Pure vector search (or 0.5 for hybrid)
export ENABLE_FILTERS=1             # Enable metadata filtering

# =============================================================================
# Database Configuration
# =============================================================================
export PGHOST=localhost
export PGPORT=5432
export PGUSER=fryt
export PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD not set}
export DB_NAME=vector_db
export PGTABLE=data_messages_text_cs700_ov150

# =============================================================================
# Logging Configuration
# =============================================================================
export LOG_LEVEL=INFO               # INFO | DEBUG | WARNING | ERROR
export LOG_FULL_CHUNKS=0            # Set to 1 to see full chunk content
export COLORIZE_CHUNKS=0            # Set to 1 for colored output
export LOG_QUERIES=0                # Set to 1 to save query logs

echo "✅ Optimized configuration loaded!"
echo ""
echo "Configuration Summary:"
echo "  LLM: N_GPU_LAYERS=$N_GPU_LAYERS, N_BATCH=$N_BATCH, CTX=$CTX"
echo "  Embeddings: $EMBED_BACKEND ($EMBED_MODEL)"
echo "  Chunking: $CHUNK_SIZE chars, $CHUNK_OVERLAP overlap"
echo "  Database: $PGTABLE"
echo ""
echo "Ready to run:"
echo "  python rag_low_level_m1_16gb_verbose.py --query-only --interactive"
