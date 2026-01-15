#!/bin/bash
# Audit Dashboard Server
# Serves the audit reports on http://localhost:8888

echo "🚀 Starting Audit Dashboard Server..."
echo "📊 Dashboard: http://localhost:8888/index.html"
echo "💡 Tip: Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"

# Try port 8888 first, then 8080, then 9000
for port in 8888 8080 9000; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is in use, trying next..."
    else
        echo "✅ Starting server on port $port"
        python3 -m http.server $port
        exit 0
    fi
done

echo "❌ No available ports found. Please stop other services."
exit 1
