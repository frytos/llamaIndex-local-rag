#!/bin/bash
# Quick script to check what's happening on the pod

if [ $# -lt 2 ]; then
    echo "Usage: $0 TCP_HOST TCP_PORT [SSH_KEY]"
    echo "Example: $0 213.173.102.169 26459 ~/.ssh/runpod_key"
    exit 1
fi

TCP_HOST=$1
TCP_PORT=$2
SSH_KEY="${3:-${HOME}/.ssh/runpod_key}"

echo "=========================================================================="
echo "RunPod Status Check"
echo "=========================================================================="
echo ""

echo "Checking running processes..."
ssh -i "$SSH_KEY" -p "$TCP_PORT" root@${TCP_HOST} << 'EOF'
echo "=== Active processes (apt/dpkg/python) ==="
ps aux | grep -E "apt|dpkg|python|bash.*init" | grep -v grep

echo ""
echo "=== Disk activity (showing if downloading) ==="
df -h /tmp /workspace 2>/dev/null || df -h

echo ""
echo "=== Network connections (downloading?) ==="
netstat -tunp 2>/dev/null | grep ESTABLISHED | head -5 || ss -tunp | grep ESTABLISHED | head -5

echo ""
echo "=== Check if PostgreSQL is installed ==="
which psql && psql --version || echo "PostgreSQL not yet installed"

echo ""
echo "=== Check init script log if exists ==="
if [ -f /workspace/rag-pipeline/logs/auto-init.log ]; then
    tail -20 /workspace/rag-pipeline/logs/auto-init.log
fi

echo ""
echo "=== Last 10 lines of apt logs ==="
tail -10 /var/log/apt/term.log 2>/dev/null || echo "No apt logs found"
EOF

echo ""
echo "=========================================================================="
