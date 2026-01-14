#!/bin/bash
# ============================================================================
# RunPod Automatic Startup Script
# ============================================================================
# This script runs automatically when the pod starts.
# It clones the repository from GitHub and initializes all services.
#
# Usage: Configured via docker_args when creating the pod
#
# Environment Variables (set during pod creation):
#   GITHUB_REPO - GitHub repository URL (e.g., https://github.com/user/repo.git)
#   GITHUB_BRANCH - Branch to clone (default: main)
# ============================================================================

set -e

# Configuration
WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/rag-pipeline"
LOG_FILE="$WORKSPACE/startup.log"

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================================="
echo "RunPod Automatic Startup - $(date)"
echo "=========================================================================="
echo ""

# Get GitHub repo from environment
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"

echo "Configuration:"
echo "  GitHub Repo: ${GITHUB_REPO:-Not set}"
echo "  Branch: $GITHUB_BRANCH"
echo "  Project Dir: $PROJECT_DIR"
echo ""

# ============================================================================
# Step 1: Clone Repository
# ============================================================================

echo "[1/3] Cloning repository from GitHub..."

if [ -z "$GITHUB_REPO" ]; then
    echo "⚠️  GITHUB_REPO not set. Skipping clone."
    echo "   Set GITHUB_REPO environment variable when creating pod"
    echo "   Example: GITHUB_REPO=https://github.com/username/llamaIndex-local-rag.git"
    echo ""
    echo "Creating empty project structure..."
    mkdir -p "$PROJECT_DIR"/{scripts,config,logs,data}
else
    echo "Cloning from: $GITHUB_REPO"

    if [ -d "$PROJECT_DIR/.git" ]; then
        echo "Repository already exists, pulling latest..."
        cd "$PROJECT_DIR"
        git pull origin "$GITHUB_BRANCH"
    else
        echo "Cloning fresh..."
        git clone --branch "$GITHUB_BRANCH" "$GITHUB_REPO" "$PROJECT_DIR"
    fi

    echo "✅ Repository cloned/updated"
fi

cd "$PROJECT_DIR"
echo ""

# ============================================================================
# Step 2: Run Initialization Script
# ============================================================================

echo "[2/3] Running service initialization..."

if [ -f "$PROJECT_DIR/scripts/init_runpod_services.sh" ]; then
    echo "Found init script, running..."
    bash "$PROJECT_DIR/scripts/init_runpod_services.sh"

    if [ $? -eq 0 ]; then
        echo "✅ Services initialized successfully"
    else
        echo "❌ Initialization failed (check logs above)"
    fi
else
    echo "⚠️  Init script not found at: $PROJECT_DIR/scripts/init_runpod_services.sh"
    echo "   Skipping service initialization"
    echo ""
    echo "To initialize manually:"
    echo "  1. Upload your init script"
    echo "  2. SSH in and run: bash /workspace/rag-pipeline/scripts/init_runpod_services.sh"
fi

echo ""

# ============================================================================
# Step 3: Start Background Services
# ============================================================================

echo "[3/3] Starting background services..."

# Start vLLM if available
if [ -f "$PROJECT_DIR/start_vllm.sh" ]; then
    echo "Starting vLLM server..."
    bash "$PROJECT_DIR/start_vllm.sh"
    echo "✅ vLLM started (loading model in background)"
else
    echo "⚠️  vLLM startup script not found"
fi

echo ""

# ============================================================================
# Complete
# ============================================================================

echo "=========================================================================="
echo "STARTUP COMPLETE - $(date)"
echo "=========================================================================="
echo ""
echo "Services status:"
echo "  PostgreSQL: $(service postgresql status 2>&1 | grep -q active && echo '✅ Running' || echo '❌ Not running')"
echo "  vLLM: $(curl -s http://localhost:8000/health 2>/dev/null | grep -q ok && echo '✅ Ready' || echo '⏳ Loading (wait 60s)')"
echo ""
echo "Logs:"
echo "  Startup: $LOG_FILE"
echo "  vLLM: $PROJECT_DIR/logs/vllm.log"
echo ""
echo "Connection info:"
echo "  SSH: ssh <pod-host>@ssh.runpod.io"
echo "  vLLM: http://localhost:8000"
echo "  PostgreSQL: localhost:5432 (user: fryt, password: frytos)"
echo ""
echo "=========================================================================="
echo "Pod is ready for use!"
echo "=========================================================================="

# Keep container running
exec sleep infinity
