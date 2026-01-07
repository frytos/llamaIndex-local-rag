#!/bin/bash
# vLLM Server Control Script
# Manage vLLM server lifecycle (start, stop, status, restart)

set -e

PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-TheBloke/Mistral-7B-Instruct-v0.2-AWQ}"

case "$1" in
    start)
        echo "🚀 Starting vLLM server..."
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "❌ Server already running on port $PORT"
            echo "   Use: $0 stop"
            exit 1
        fi

        nohup ./scripts/start_vllm_server.sh > /tmp/vllm_server.log 2>&1 &
        PID=$!
        echo "✅ Server starting in background (PID: $PID)"
        echo "   Logs: tail -f /tmp/vllm_server.log"
        echo "   Wait ~60s for warmup..."
        sleep 3
        echo "   Checking status in 60s..."
        sleep 57
        $0 status
        ;;

    stop)
        echo "🛑 Stopping vLLM server..."
        if pkill -f 'vllm serve'; then
            echo "✅ Server stopped"
        else
            echo "⚠️  No server running"
        fi
        ;;

    restart)
        echo "🔄 Restarting vLLM server..."
        $0 stop
        sleep 2
        $0 start
        ;;

    status)
        echo "📊 vLLM Server Status"
        echo "───────────────────────"

        # Check process
        if pgrep -f 'vllm serve' > /dev/null; then
            PID=$(pgrep -f 'vllm serve')
            echo "✅ Server running (PID: $PID)"
        else
            echo "❌ Server not running"
            echo "   Start with: $0 start"
            exit 1
        fi

        # Check port
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "✅ Port $PORT listening"
        else
            echo "⚠️  Port $PORT not listening (still warming up?)"
        fi

        # Check health endpoint
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "✅ Health check passed"
            echo ""
            echo "Server ready! Query with:"
            echo "  python3 rag_low_level_m1_16gb_verbose.py --query-only --query 'test'"
        else
            echo "⚠️  Health check failed (still warming up?)"
            echo "   Wait 30-60s and try: $0 status"
        fi
        ;;

    logs)
        echo "📝 vLLM Server Logs (last 50 lines)"
        echo "───────────────────────────────────"
        tail -50 /tmp/vllm_server.log
        echo ""
        echo "Follow logs: tail -f /tmp/vllm_server.log"
        ;;

    test)
        echo "🧪 Testing vLLM server..."
        if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "❌ Server not running!"
            echo "   Start with: $0 start"
            exit 1
        fi

        echo "Sending test query..."
        RESPONSE=$(curl -s http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$MODEL"'",
                "prompt": "2+2=",
                "max_tokens": 10,
                "temperature": 0.1
            }')

        echo "✅ Response:"
        echo "$RESPONSE" | python3 -m json.tool
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start    Start vLLM server in background"
        echo "  stop     Stop vLLM server"
        echo "  restart  Restart vLLM server"
        echo "  status   Check server status"
        echo "  logs     Show server logs"
        echo "  test     Send test query"
        echo ""
        echo "Examples:"
        echo "  $0 start           # Start server"
        echo "  $0 status          # Check if ready"
        echo "  $0 logs            # View logs"
        echo "  $0 test            # Test query"
        echo "  $0 stop            # Stop server"
        exit 1
        ;;
esac
