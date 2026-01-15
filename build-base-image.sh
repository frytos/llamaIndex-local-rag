#!/bin/bash
# Build and push base Docker image for Railway deployment
#
# Usage:
#   ./build-base-image.sh
#
# Prerequisites:
#   1. Docker installed and running
#   2. Docker Hub account created at https://hub.docker.com
#   3. Logged in: docker login
#
# What this does:
#   - Builds base image with all Python dependencies pre-installed
#   - Tags as: frytos/llamaindex-rag-base:latest
#   - Pushes to Docker Hub (public or private)
#   - Takes 10-15 minutes (but only needs to run when requirements.txt changes)

set -e  # Exit on error

# Configuration
IMAGE_NAME="frytos/llamaindex-rag-base"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Building Base Image for llamaIndex-local-rag                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Image: ${FULL_IMAGE}"
echo "Build time: ~10-15 minutes (large dependencies: torch, transformers)"
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "   Start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Build the base image
echo "══════════════════════════════════════════════════════════════════"
echo "Step 1/3: Building base image..."
echo "══════════════════════════════════════════════════════════════════"
echo ""

docker build \
    -f Dockerfile.base \
    -t "${FULL_IMAGE}" \
    --progress=plain \
    .

echo ""
echo "✅ Base image built successfully!"
echo ""

# Show image size
IMAGE_SIZE=$(docker images "${FULL_IMAGE}" --format "{{.Size}}")
echo "📦 Image size: ${IMAGE_SIZE}"
echo ""

# Ask if user wants to push
read -p "Push to Docker Hub? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "══════════════════════════════════════════════════════════════════"
    echo "Step 2/3: Checking Docker Hub authentication..."
    echo "══════════════════════════════════════════════════════════════════"
    echo ""

    # Check if logged in
    if ! docker info | grep -q "Username"; then
        echo "Not logged into Docker Hub. Logging in..."
        docker login
    else
        echo "✓ Already logged into Docker Hub"
    fi

    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "Step 3/3: Pushing to Docker Hub..."
    echo "══════════════════════════════════════════════════════════════════"
    echo ""

    docker push "${FULL_IMAGE}"

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║ ✅ Base Image Published Successfully!                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Image: ${FULL_IMAGE}"
    echo "Size: ${IMAGE_SIZE}"
    echo ""
    echo "📝 Next Steps:"
    echo "  1. Update Dockerfile to use this base image:"
    echo "     FROM ${FULL_IMAGE}"
    echo ""
    echo "  2. Commit and push Dockerfile change"
    echo ""
    echo "  3. Railway will rebuild (10-15 min first time)"
    echo ""
    echo "  4. Future code changes = 30-60 second rebuilds! 🚀"
    echo ""
    echo "📌 Maintenance:"
    echo "  - Rebuild this base image when you update requirements.txt"
    echo "  - Run: ./build-base-image.sh"
    echo ""
else
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║ Base Image Built (Not Pushed)                                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "To push later:"
    echo "  docker push ${FULL_IMAGE}"
    echo ""
    echo "To test locally:"
    echo "  docker run --rm -it ${FULL_IMAGE} python --version"
    echo ""
fi
