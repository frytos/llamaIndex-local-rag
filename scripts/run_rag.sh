#!/bin/bash
# Convenience script to run RAG pipeline with environment variables loaded

# Activate virtual environment
source .venv/bin/activate

# Load environment variables from .env
set -a
source .env
set +a

# Run the RAG pipeline
python rag_low_level_m1_16gb_verbose.py "$@"
