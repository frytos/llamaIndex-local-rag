#!/bin/bash
# ============================================================================
# Quick Deploy to RunPod
# ============================================================================
# One-command deployment script for RAG pipeline to RunPod.
#
# Usage:
#   export RUNPOD_API_KEY=your_api_key_here
#   bash scripts/quick_deploy_runpod.sh
#
# Or:
#   bash scripts/quick_deploy_runpod.sh YOUR_API_KEY
#
# This script will:
#   1. Create RunPod pod with RTX 4090
#   2. Wait for pod to be ready
#   3. Provide SSH connection details
#   4. Show next steps
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================================================="
echo "Quick Deploy to RunPod - RAG Pipeline"
echo "=========================================================================="
echo ""

# Check for API key
API_KEY="${1:-$RUNPOD_API_KEY}"

if [ -z "$API_KEY" ]; then
    echo -e "${RED}❌ Error: No API key provided${NC}"
    echo ""
    echo "Usage:"
    echo "  1. Pass as argument:"
    echo "     bash scripts/quick_deploy_runpod.sh YOUR_API_KEY"
    echo ""
    echo "  2. Set environment variable:"
    echo "     export RUNPOD_API_KEY=your_api_key"
    echo "     bash scripts/quick_deploy_runpod.sh"
    echo ""
    echo "Get your API key from: https://runpod.io/settings"
    exit 1
fi

# Check if Python script exists
if [ ! -f "scripts/deploy_to_runpod.py" ]; then
    echo -e "${RED}❌ Error: deploy_to_runpod.py not found${NC}"
    echo "Make sure you're in the project root directory"
    exit 1
fi

# Generate unique pod name
POD_NAME="rag-pipeline-$(date +%Y%m%d-%H%M%S)"

echo -e "${GREEN}📝 Configuration:${NC}"
echo "   Pod Name: $POD_NAME"
echo "   GPU: NVIDIA RTX 4090"
echo "   Volume: 100GB"
echo ""

# Deploy
echo -e "${GREEN}🚀 Starting deployment...${NC}"
echo ""

python3 scripts/deploy_to_runpod.py \
    --api-key "$API_KEY" \
    --name "$POD_NAME" \
    --gpu "NVIDIA RTX 4090" \
    --volume 100

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================================================="
    echo -e "${GREEN}✅ DEPLOYMENT SUCCESSFUL${NC}"
    echo "=========================================================================="
    echo ""
    echo "📋 Next Steps:"
    echo ""
    echo "1. Save your pod name:"
    echo "   POD_NAME=$POD_NAME"
    echo ""
    echo "2. Get pod ID and SSH host:"
    echo "   python3 scripts/runpod_cli.py list"
    echo ""
    echo "3. Create SSH tunnel:"
    echo "   python3 scripts/runpod_cli.py tunnel POD_ID"
    echo ""
    echo "4. Initialize services (from within pod):"
    echo "   SSH into pod and run:"
    echo "   bash /workspace/rag-pipeline/scripts/init_runpod_services.sh"
    echo ""
    echo "5. Test vLLM (after tunnel and init):"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "6. Run query:"
    echo "   python rag_low_level_m1_16gb_verbose.py --query-only --query 'test'"
    echo ""
    echo "=========================================================================="
else
    echo ""
    echo "=========================================================================="
    echo -e "${RED}❌ DEPLOYMENT FAILED${NC}"
    echo "=========================================================================="
    echo ""
    echo "Check the error messages above for details."
    echo ""
    exit $EXIT_CODE
fi
