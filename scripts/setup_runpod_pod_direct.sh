#!/bin/bash
# ============================================================================
# Automated RunPod Pod Setup Script (Direct TCP Connection)
# ============================================================================
# Uses direct TCP connection instead of SSH proxy for better reliability
#
# Usage:
#   bash scripts/setup_runpod_pod_direct.sh TCP_HOST TCP_PORT [SSH_KEY]
#
# Example (get values from RunPod web UI):
#   bash scripts/setup_runpod_pod_direct.sh 213.173.102.169 26459
#   bash scripts/setup_runpod_pod_direct.sh 213.173.102.169 26459 ~/.ssh/id_ed25519
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Usage: $0 TCP_HOST TCP_PORT [SSH_KEY]${NC}"
    echo ""
    echo "Get TCP connection details from RunPod web UI:"
    echo "  Pods → Your Pod → Connect → 'SSH over exposed TCP'"
    echo ""
    echo "Example:"
    echo "  $0 213.173.102.169 26459"
    echo "  $0 213.173.102.169 26459 ~/.ssh/id_ed25519"
    exit 1
fi

TCP_HOST=$1
TCP_PORT=$2
SSH_KEY="${3:-${HOME}/.ssh/id_ed25519}"  # Default to id_ed25519

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    echo ""
    echo "Available keys:"
    ls -la ~/.ssh/id_* 2>/dev/null | grep -v ".pub" || echo "  No SSH keys found"
    exit 1
fi

echo -e "${GREEN}Using SSH key:${NC} $SSH_KEY"
echo -e "${GREEN}TCP Connection:${NC} root@${TCP_HOST}:${TCP_PORT}"
echo ""

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}RunPod Automated Setup (Direct TCP)${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

# Step 1: Test connection
echo -e "${YELLOW}Step 1/6:${NC} Testing SSH connection..."

if ssh -i "$SSH_KEY" -p "$TCP_PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
    "root@${TCP_HOST}" "echo 'Connected'" 2>&1 | grep -q "Connected"; then
    echo -e "${GREEN}✅ SSH connection successful!${NC}"
else
    echo -e "${RED}❌ Failed to connect${NC}"
    echo ""
    echo "Try connecting manually:"
    echo "  ssh -i $SSH_KEY -p $TCP_PORT root@${TCP_HOST}"
    exit 1
fi
echo ""

# Step 2: Create directories
echo -e "${YELLOW}Step 2/6:${NC} Creating directories on pod..."
ssh -i "$SSH_KEY" -p "$TCP_PORT" -o StrictHostKeyChecking=no "root@${TCP_HOST}" \
    "mkdir -p /workspace/rag-pipeline/{scripts,config,logs,data} && echo 'Directories created'"
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Step 3: Upload files
echo -e "${YELLOW}Step 3/6:${NC} Uploading files..."

scp -i "$SSH_KEY" -P "$TCP_PORT" -o StrictHostKeyChecking=no \
    scripts/init_runpod_services.sh \
    "root@${TCP_HOST}:/workspace/rag-pipeline/scripts/"

scp -i "$SSH_KEY" -P "$TCP_PORT" -o StrictHostKeyChecking=no \
    requirements.txt \
    "root@${TCP_HOST}:/workspace/rag-pipeline/"

if [ -f config/requirements_vllm.txt ]; then
    scp -i "$SSH_KEY" -P "$TCP_PORT" -o StrictHostKeyChecking=no \
        config/requirements_vllm.txt \
        "root@${TCP_HOST}:/workspace/rag-pipeline/config/"
fi

echo -e "${GREEN}✅ Files uploaded${NC}"
echo ""

# Step 4: Run initialization
echo -e "${YELLOW}Step 4/6:${NC} Running initialization script..."
echo -e "   ${BLUE}This takes 5-10 minutes. Watch the progress below:${NC}"
echo ""

ssh -i "$SSH_KEY" -p "$TCP_PORT" -o StrictHostKeyChecking=no "root@${TCP_HOST}" \
    "cd /workspace/rag-pipeline && bash scripts/init_runpod_services.sh"

echo ""
echo -e "${GREEN}✅ Initialization complete!${NC}"
echo ""

# Step 5: Wait for services
echo -e "${YELLOW}Step 5/6:${NC} Waiting for services to start..."
echo "   Waiting 60 seconds for vLLM model to load..."
sleep 60
echo -e "${GREEN}✅ Services should be ready${NC}"
echo ""

# Step 6: Test services
echo -e "${YELLOW}Step 6/6:${NC} Testing services..."

echo -e "${BLUE}Testing vLLM...${NC}"
if ssh -i "$SSH_KEY" -p "$TCP_PORT" -o StrictHostKeyChecking=no "root@${TCP_HOST}" \
    "curl -s http://localhost:8000/health" | grep -q "ok"; then
    echo -e "${GREEN}✅ vLLM is responding!${NC}"
else
    echo -e "${YELLOW}⚠️  vLLM not responding yet (may need more time)${NC}"
fi

echo ""
echo -e "${BLUE}Testing PostgreSQL...${NC}"
if ssh -i "$SSH_KEY" -p "$TCP_PORT" -o StrictHostKeyChecking=no "root@${TCP_HOST}" \
    "psql -h localhost -U fryt -d vector_db -c 'SELECT 1'" 2>&1 | grep -q "1 row"; then
    echo -e "${GREEN}✅ PostgreSQL is responding!${NC}"
else
    echo -e "${YELLOW}⚠️  PostgreSQL check failed${NC}"
fi

echo ""
echo -e "${BLUE}========================================================================${NC}"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""
echo -e "Direct SSH access:"
echo -e "  ${BLUE}ssh -i $SSH_KEY -p $TCP_PORT root@${TCP_HOST}${NC}"
echo ""
echo -e "Create SSH tunnel to access services from your Mac:"
echo -e "  ${BLUE}ssh -i $SSH_KEY -p $TCP_PORT -N -L 8000:localhost:8000 -L 5432:localhost:5432 root@${TCP_HOST}${NC}"
echo ""
echo "Then test from your Mac:"
echo -e "  ${BLUE}curl http://localhost:8000/health${NC}"
echo -e "  ${BLUE}python rag_low_level_m1_16gb_verbose.py --query-only --query \"test\"${NC}"
echo ""
