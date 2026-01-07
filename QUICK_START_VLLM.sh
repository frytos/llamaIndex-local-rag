#!/bin/bash
# Quick Start with vLLM on Runpod RTX 4090
# One-liner to run optimized GPU queries

cd /workspace/rag-pipeline
source .venv/bin/activate
source runpod_vllm_config.env

echo "🚀 RAG Pipeline with vLLM GPU Backend"
echo "======================================"
echo ""
echo "Configuration:"
echo "  LLM: $VLLM_MODEL"
echo "  Embedding: $EMBED_MODEL (GPU)"
echo "  Table: $PGTABLE"
echo "  Context: ${CTX} tokens"
echo "  Max output: ${MAX_NEW_TOKENS} tokens"
echo ""

# Parse command
case "$1" in
  query|q)
    shift
    QUERY="$*"
    if [ -z "$QUERY" ]; then
      echo "Usage: $0 query <your question>"
      exit 1
    fi
    echo "❓ Query: $QUERY"
    echo ""
    time python3 rag_low_level_m1_16gb_verbose.py --query-only --query "$QUERY"
    ;;

  index|i)
    echo "📊 Indexing data from: $PDF_PATH"
    echo "   This will take ~2-3 minutes on RTX 4090"
    echo ""
    time python3 rag_low_level_m1_16gb_verbose.py
    ;;

  benchmark|b)
    echo "🔬 Running benchmark (12 queries)..."
    echo "   This will take ~1-2 minutes on RTX 4090"
    echo ""
    ./scripts/test_query_quality.sh
    ;;

  gpu|g)
    echo "🎮 GPU Status:"
    nvidia-smi
    ;;

  *)
    echo "Usage: $0 {query|index|benchmark|gpu} [args]"
    echo ""
    echo "Commands:"
    echo "  query <question>    Run a single query"
    echo "  index               Index documents"
    echo "  benchmark           Run full benchmark"
    echo "  gpu                 Show GPU status"
    echo ""
    echo "Examples:"
    echo "  $0 query when did I go to New York"
    echo "  $0 index"
    echo "  $0 benchmark"
    echo "  $0 gpu"
    ;;
esac
