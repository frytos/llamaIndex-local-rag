# Dockerfile for Railway deployment using pre-built base image
# Base image (frytos/llamaindex-rag-base:latest) contains:
#   - Python 3.13-slim
#   - System dependencies (PostgreSQL client, gcc, g++, etc.)
#   - ALL Python packages from requirements.txt
#   - Pre-created directories (/app/logs, /app/data, /app/query_logs, /app/auth)
#
# This Dockerfile only copies application code = 30-60 second rebuilds!
#
# To update base image (when requirements.txt changes):
#   ./build-base-image.sh

FROM frytos/llamaindex-rag-base:latest

# Copy application code (this is the ONLY layer that changes)
# .dockerignore excludes: data/, logs/, .git/, tests/, .planning/
COPY . /app

# Expose port (Railway assigns $PORT dynamically)
EXPOSE 8080

# Health check (curl is available in base image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Start command - use shell form to allow $PORT expansion
# Railway sets $PORT dynamically, shell wrapper expands it at runtime
CMD streamlit run rag_web.py --server.port=${PORT:-8080} --server.address=0.0.0.0
