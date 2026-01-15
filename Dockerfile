# Multi-stage Dockerfile for Railway deployment (optimized for fast rebuilds)

# =============================================================================
# Stage 1: Base image with system dependencies
# =============================================================================
FROM python:3.13-slim AS base

# Install system dependencies (PostgreSQL client libraries)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libpq5 \
    postgresql-client \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Dependencies layer (HEAVILY CACHED)
# =============================================================================
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy ONLY requirements.txt first (this layer is cached unless requirements.txt changes)
COPY requirements.txt .

# Install Python dependencies
# This step takes 10-15 minutes but is CACHED on subsequent builds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Application layer (rebuilds quickly when code changes)
# =============================================================================
FROM dependencies AS app

# Copy application code
# This layer rebuilds when you change code, but dependencies are already installed!
COPY . /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/query_logs

# Expose port (Railway assigns $PORT dynamically)
EXPOSE 8080

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Set environment variables defaults
ENV PYTHONUNBUFFERED=1

# Start command - use shell form to allow $PORT expansion
# Railway sets $PORT dynamically, shell wrapper expands it at runtime
CMD streamlit run rag_web.py --server.port=${PORT:-8080} --server.address=0.0.0.0
