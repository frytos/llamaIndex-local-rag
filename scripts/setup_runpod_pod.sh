#!/bin/bash
# ============================================================================
# Automated RunPod Pod Setup Script
# ============================================================================
# This script automates the entire pod setup process after creation.
#
# Usage:
#   bash scripts/setup_runpod_pod.sh POD_HOST
#
# Example:
#   bash scripts/setup_runpod_pod.sh vlog7cun0nk47z-64411ede
#
# What it does:
#   1. Uploads init script and requirements to pod
#   2. SSHs into pod and runs initialization
#   3. Waits for services to start
#   4. Creates SSH tunnel
#   5. Tests services
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 POD_HOST [SSH_KEY]${NC}"
    echo ""
    echo "Example:"
    echo "  $0 vlog7cun0nk47z-64411ede"
    echo "  $0 vlog7cun0nk47z-64411ede ~/.ssh/id_ed25519"
    echo ""
    echo "Get POD_HOST from Streamlit UI or pod creation output"
    exit 1
fi

POD_HOST=$1
SSH_KEY="${2:-${HOME}/.ssh/runpod_key}"  # Use 2nd arg if provided, else default

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    echo ""
    echo "Available keys:"
    ls -la ~/.ssh/id_* 2>/dev/null | grep -v ".pub" || echo "  No SSH keys found"
    echo ""
    echo "To use a different key:"
    echo "  $0 $POD_HOST ~/.ssh/id_ed25519"
    exit 1
fi

echo -e "${GREEN}Using SSH key:${NC} $SSH_KEY"

# Check SSH key permissions
KEY_PERMS=$(stat -f "%OLp" "$SSH_KEY" 2>/dev/null || stat -c "%a" "$SSH_KEY" 2>/dev/null)
if [ "$KEY_PERMS" != "600" ] && [ "$KEY_PERMS" != "400" ]; then
    echo -e "${YELLOW}⚠️  Warning: SSH key has permissions $KEY_PERMS (should be 600 or 400)${NC}"
    echo "   Fixing permissions..."
    chmod 600 "$SSH_KEY"
    echo -e "${GREEN}✅ Permissions fixed${NC}"
fi

echo ""
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}RunPod Automated Setup${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo -e "Pod Host: ${GREEN}${POD_HOST}${NC}"
echo ""

# Step 1: Wait for SSH to be ready
echo -e "${YELLOW}Step 1/6:${NC} Waiting for pod SSH to be ready..."
echo -e "${BLUE}Debug Info:${NC}"
echo "  SSH Host: ${POD_HOST}@ssh.runpod.io"
echo "  SSH Key: $SSH_KEY"
echo ""

MAX_RETRIES=30
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    RETRY=$((RETRY + 1))
    echo -e "${BLUE}Attempt $RETRY/$MAX_RETRIES:${NC}"

    # Try SSH with verbose output
    echo "  Testing SSH connection..."
    SSH_OUTPUT=$(ssh -i "$SSH_KEY" \
        -T \
        -o ConnectTimeout=5 \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        "${POD_HOST}@ssh.runpod.io" "echo 'SSH Ready'" 2>&1)

    SSH_EXIT_CODE=$?

    if echo "$SSH_OUTPUT" | grep -q "SSH Ready"; then
        echo -e "  ${GREEN}✅ SSH is ready!${NC}"
        break
    else
        echo -e "  ${YELLOW}⏳ Not ready yet${NC}"
        echo "  Exit code: $SSH_EXIT_CODE"

        # Show first line of error for debugging
        if [ -n "$SSH_OUTPUT" ]; then
            ERROR_LINE=$(echo "$SSH_OUTPUT" | head -n 1)
            echo "  Error: $ERROR_LINE"
        fi

        if [ $RETRY -lt $MAX_RETRIES ]; then
            echo "  Waiting 10 seconds..."
            sleep 10
        fi
    fi
    echo ""
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Failed to connect via SSH after $MAX_RETRIES attempts${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo ""
    echo "1. Verify pod is running:"
    echo "   Check Streamlit UI → RunPod Deployment → Existing Pods"
    echo ""
    echo "2. Test SSH manually with verbose output:"
    echo "   ssh -v -i $SSH_KEY ${POD_HOST}@ssh.runpod.io"
    echo ""
    echo "3. Check your SSH key is registered on RunPod:"
    echo "   https://runpod.io/console/user/settings (SSH Keys tab)"
    echo ""
    echo "4. Verify the public key matches:"
    echo "   cat ${SSH_KEY}.pub"
    echo ""
    exit 1
fi

echo ""

# Step 2: Create directories on pod
echo -e "${YELLOW}Step 2/6:${NC} Creating directories on pod..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${POD_HOST}@ssh.runpod.io" << 'REMOTE'
mkdir -p /workspace/rag-pipeline/{scripts,config,logs,data}
echo "Directories created"
REMOTE

echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Step 3: Upload files
echo -e "${YELLOW}Step 3/6:${NC} Uploading init script and requirements..."

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    scripts/init_runpod_services.sh \
    "${POD_HOST}@ssh.runpod.io:/workspace/rag-pipeline/scripts/"

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    requirements.txt \
    "${POD_HOST}@ssh.runpod.io:/workspace/rag-pipeline/"

if [ -f config/requirements_vllm.txt ]; then
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        config/requirements_vllm.txt \
        "${POD_HOST}@ssh.runpod.io:/workspace/rag-pipeline/config/"
fi

echo -e "${GREEN}✅ Files uploaded${NC}"
echo ""

# Step 4: Run initialization script
echo -e "${YELLOW}Step 4/6:${NC} Running initialization script (this takes 5-10 minutes)..."
echo -e "   ${BLUE}You can watch progress in real-time...${NC}"
echo ""

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${POD_HOST}@ssh.runpod.io" << 'REMOTE'
cd /workspace/rag-pipeline
bash scripts/init_runpod_services.sh
REMOTE

echo ""
echo -e "${GREEN}✅ Initialization complete!${NC}"
echo ""

# Step 5: Wait for services
echo -e "${YELLOW}Step 5/6:${NC} Waiting for services to start..."
echo "   Waiting 60 seconds for vLLM model to load..."
sleep 60

echo -e "${GREEN}✅ Services should be ready${NC}"
echo ""

# Step 6: Create SSH tunnel
echo -e "${YELLOW}Step 6/6:${NC} Creating SSH tunnel..."
echo -e "   ${BLUE}Starting tunnel in background...${NC}"

# Kill any existing tunnels
pkill -f "ssh.*${POD_HOST}.*8000" 2>/dev/null || true

# Create tunnel in background
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    -f -N \
    -L 8000:localhost:8000 \
    -L 5432:localhost:5432 \
    "${POD_HOST}@ssh.runpod.io"

echo -e "${GREEN}✅ SSH tunnel created (background)${NC}"
echo ""

# Test services
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}Testing Services${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

echo -e "${YELLOW}Testing vLLM...${NC}"
if curl -s http://localhost:8000/health | grep -q "ok"; then
    echo -e "${GREEN}✅ vLLM is responding!${NC}"
else
    echo -e "${YELLOW}⚠️  vLLM not responding yet (may need more time for model loading)${NC}"
fi

echo ""
echo -e "${YELLOW}Testing PostgreSQL...${NC}"
if psql -h localhost -U fryt -d vector_db -c "SELECT 1" 2>/dev/null | grep -q "1 row"; then
    echo -e "${GREEN}✅ PostgreSQL is responding!${NC}"
else
    echo -e "${YELLOW}⚠️  PostgreSQL connection failed${NC}"
    echo "   Password: frytos"
fi

echo ""
echo -e "${BLUE}========================================================================${NC}"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""
echo -e "Pod: ${GREEN}${POD_HOST}@ssh.runpod.io${NC}"
echo -e "SSH: ${BLUE}ssh -i ~/.ssh/runpod_key ${POD_HOST}@ssh.runpod.io${NC}"
echo ""
echo "Services available on localhost:"
echo -e "  • vLLM: ${GREEN}http://localhost:8000${NC}"
echo -e "  • PostgreSQL: ${GREEN}localhost:5432${NC}"
echo ""
echo "Test with:"
echo -e "  ${BLUE}curl http://localhost:8000/health${NC}"
echo -e "  ${BLUE}python rag_low_level_m1_16gb_verbose.py --query-only --query \"test\"${NC}"
echo ""
echo -e "${YELLOW}Note: SSH tunnel is running in background${NC}"
echo "To stop it: pkill -f 'ssh.*${POD_HOST}'"
echo ""
