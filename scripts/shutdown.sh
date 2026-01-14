#!/bin/bash
# Shutdown script for RAG Pipeline
# Usage: ./shutdown.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🛑 Shutting Down RAG Pipeline"
echo "=============================="

echo ""
echo "1️⃣  Stopping Streamlit..."
pkill -f 'streamlit run' && echo "   ✓ Streamlit stopped" || echo "   (Streamlit not running)"

echo ""
echo "2️⃣  Stopping macOS metrics exporter..."
pkill -f 'macos_exporter.py' && echo "   ✓ macOS exporter stopped" || echo "   (macOS exporter not running)"
rm -f logs/macos_exporter.pid

echo ""
echo "3️⃣  Stopping Docker services..."
cd config
docker-compose down
cd ..

echo ""
echo "✅ All services stopped!"
echo ""
echo "🔄 To restart:"
echo "   ./launch.sh"
echo ""
