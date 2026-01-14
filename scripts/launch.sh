#!/bin/bash
# Complete launch script for RAG Web UI with Docker PostgreSQL
# Usage: ./launch.sh

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Launching RAG Pipeline Web UI"
echo "================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Creating from example..."
    cp config/.env.example .env
    echo "⚠️  Please edit .env and set your database credentials:"
    echo "   nano .env"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Check if PGPASSWORD is set
if [ -z "$PGPASSWORD" ]; then
    echo "❌ PGPASSWORD not set in .env file!"
    echo "Please edit .env and set PGPASSWORD"
    exit 1
fi

echo ""
echo "1️⃣  Starting Docker services (Database + Monitoring)..."

# Create symlink to .env if it doesn't exist in config/
if [ ! -f config/.env ] && [ ! -L config/.env ]; then
    echo "   Creating symlink to .env in config directory..."
    ln -s ../.env config/.env
fi

# Create monitoring symlink if it doesn't exist
if [ ! -L monitoring ]; then
    echo "   Creating monitoring symlink..."
    ln -s config/monitoring monitoring
fi

cd config
echo "   Starting all Docker services..."
docker-compose up -d

echo ""
echo "2️⃣  Waiting for database to be ready..."
for i in {1..30}; do
    if docker-compose exec -T db pg_isready -U "${PGUSER:-postgres}" -d "${DB_NAME:-vector_db}" > /dev/null 2>&1; then
        echo "   ✓ Database is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ Database failed to start after 30 seconds"
        echo "   Check logs: docker-compose logs db"
        exit 1
    fi
    echo "   Waiting... ($i/30)"
    sleep 1
done

cd ..

echo ""
echo "3️⃣  Checking Python environment..."
if [ ! -d .venv ]; then
    echo "❌ Virtual environment not found!"
    echo "Create it with: python3 -m venv .venv"
    exit 1
fi

source .venv/bin/activate

echo ""
echo "4️⃣  Checking Streamlit installation..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "   Installing Streamlit and dependencies..."
    pip install -q streamlit plotly umap-learn watchdog
    echo "   ✓ Streamlit installed"
else
    echo "   ✓ Streamlit already installed"
    # Check for watchdog and install if missing
    if ! python -c "import watchdog" 2>/dev/null; then
        echo "   Installing watchdog for better performance..."
        pip install -q watchdog
    fi
fi

echo ""
echo "5️⃣  Testing database connection..."
if psql -h "${PGHOST:-localhost}" -U "${PGUSER:-postgres}" -d "${DB_NAME:-vector_db}" -c "SELECT 1;" > /dev/null 2>&1; then
    echo "   ✓ Database connection successful"
else
    echo "   ⚠️  Direct psql connection failed (but Docker container is running)"
    echo "   This is OK if you don't have psql client installed"
fi

echo ""
echo "6️⃣  Starting macOS metrics exporter..."
# Create logs directory if it doesn't exist
mkdir -p logs

# Check if exporter is already running
if pgrep -f "macos_exporter.py" > /dev/null; then
    echo "   ✓ macOS exporter already running"
else
    # Start exporter in background
    nohup python macos_exporter.py --port 9101 > logs/macos_exporter.log 2>&1 &
    echo $! > logs/macos_exporter.pid
    echo "   ✓ macOS exporter started on port 9101"
    echo "   Logs: logs/macos_exporter.log"

    # Give it a moment to start
    sleep 2

    # Verify it started
    if curl -s http://localhost:9101/metrics > /dev/null 2>&1; then
        echo "   ✓ macOS exporter responding"
    else
        echo "   ⚠️  macOS exporter may not have started correctly"
        echo "   Check logs: tail -f logs/macos_exporter.log"
    fi
fi

echo ""
echo "✅ All systems ready!"
echo ""
echo "🌐 Launching Streamlit Web UI..."
echo "   Open your browser to: http://localhost:8501"
echo ""
echo "📊 Monitoring Stack:"
echo "   Grafana:       http://localhost:3000 (admin/admin)"
echo "   Prometheus:    http://localhost:9090"
echo "   cAdvisor:      http://localhost:8080"
echo "   macOS Metrics: http://localhost:9101/metrics"
echo ""
echo "🗄️  Database:"
echo "   PostgreSQL: localhost:5432"
echo "   Database:   ${DB_NAME:-vector_db}"
echo "   User:       ${PGUSER:-postgres}"
echo ""
echo "⏹️  To stop everything:"
echo "   cd config && docker-compose down && pkill -f 'streamlit run' && pkill -f 'macos_exporter.py'"
echo ""
echo "================================"
echo ""

# Launch Streamlit
streamlit run rag_web_enhanced.py
