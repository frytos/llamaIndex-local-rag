#!/bin/bash
set -e

echo "🚀 Launching RAG Pipeline (Memory-Optimized Mode)"
echo "================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "1️⃣  Starting PostgreSQL (minimal footprint)..."
cd config
docker-compose -f docker-compose.minimal.yml up -d
echo ""

echo "2️⃣  Waiting for database to be ready..."
timeout=30
counter=0
until docker exec rag_postgres pg_isready -U "${PGUSER:-postgres}" > /dev/null 2>&1; do
    counter=$((counter + 1))
    if [ $counter -ge $timeout ]; then
        echo "   ❌ Database failed to start within ${timeout}s"
        exit 1
    fi
    sleep 1
done
echo "   ✓ Database is ready!"
echo ""

echo "3️⃣  Checking Python environment..."
if [ ! -d "../.venv" ]; then
    echo "   ⚠️  Virtual environment not found. Creating..."
    python3 -m venv ../.venv
fi
echo ""

echo "4️⃣  Checking Streamlit installation..."
if ! ../.venv/bin/python -c "import streamlit" 2>/dev/null; then
    echo "   Installing Streamlit..."
    ../.venv/bin/pip install streamlit plotly scikit-learn
fi
echo "   ✓ Streamlit ready"
echo ""

echo "✅ System ready (minimal mode)!"
echo ""
echo "🌐 Launching Streamlit Web UI..."
echo "   Open your browser to: http://localhost:8501"
echo ""
echo "🗄️  Database:"
echo "   PostgreSQL: localhost:5432"
echo "   Database:   ${DB_NAME:-vector_db}"
echo "   User:       ${PGUSER:-postgres}"
echo ""
echo "⚡ Memory Optimizations Active:"
echo "   • Monitoring stack disabled (saves ~400MB)"
echo "   • PostgreSQL memory limited to 512MB"
echo "   • Reduced embedding batch size"
echo ""
echo "⏹️  To stop:"
echo "   cd config && docker-compose -f docker-compose.minimal.yml down"
echo ""
echo "================================"
echo ""

cd ..
exec .venv/bin/streamlit run rag_web.py
