#!/bin/bash
# Build and push multi-platform base Docker image for Railway deployment
#
# Usage:
#   ./build-base-image.sh
#
# Prerequisites:
#   1. Docker installed and running (with buildx support)
#   2. Docker Hub account created at https://hub.docker.com
#   3. Logged in: docker login
#
# What this does:
#   - Builds base image for BOTH linux/amd64 (Railway) and linux/arm64 (Mac M1/M2/M3)
#   - Tags as: frytos/llamaindex-rag-base:latest
#   - Automatically pushes to Docker Hub (multi-platform builds require push)
#   - Takes 20-30 minutes (building for two platforms with large dependencies)

set -e  # Exit on error

# Configuration
IMAGE_NAME="frytos/llamaindex-rag-base"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
PLATFORMS="linux/amd64,linux/arm64"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Building Multi-Platform Base Image                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Image:     ${FULL_IMAGE}"
echo "Platforms: ${PLATFORMS}"
echo "           - linux/amd64 (Railway, Intel/AMD servers)"
echo "           - linux/arm64 (Mac M1/M2/M3, ARM servers)"
echo ""
echo "Build time: ~20-30 minutes (2 platforms + large dependencies)"
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "   Start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check if buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo "❌ Error: docker buildx is not available"
    echo "   Update to Docker Desktop 19.03+ which includes buildx"
    exit 1
fi

echo "✓ Docker buildx is available"
echo ""

# Create/use buildx builder
echo "══════════════════════════════════════════════════════════════════"
echo "Step 1/4: Setting up buildx builder..."
echo "══════════════════════════════════════════════════════════════════"
echo ""

BUILDER_NAME="llamaindex-multiplatform"

# Check if builder already exists
if docker buildx inspect "${BUILDER_NAME}" > /dev/null 2>&1; then
    echo "✓ Using existing builder: ${BUILDER_NAME}"
else
    echo "Creating new builder: ${BUILDER_NAME}"
    docker buildx create --name "${BUILDER_NAME}" --use
fi

# Use the builder
docker buildx use "${BUILDER_NAME}"

# Bootstrap the builder (ensures it's running)
docker buildx inspect --bootstrap

echo ""
echo "✓ Builder ready"
echo ""

# Check Docker Hub authentication
echo "══════════════════════════════════════════════════════════════════"
echo "Step 2/4: Checking Docker Hub authentication..."
echo "══════════════════════════════════════════════════════════════════"
echo ""

if ! docker info | grep -q "Username"; then
    echo "Not logged into Docker Hub. Logging in..."
    docker login
else
    echo "✓ Already logged into Docker Hub"
fi

echo ""

# Confirm before building (multi-platform builds auto-push)
echo "══════════════════════════════════════════════════════════════════"
echo "⚠️  IMPORTANT: Multi-platform builds automatically push to Docker Hub"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "This will:"
echo "  1. Build for linux/amd64 (Railway)"
echo "  2. Build for linux/arm64 (Your Mac)"
echo "  3. Push both to Docker Hub as: ${FULL_IMAGE}"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# Build and push multi-platform image
echo "══════════════════════════════════════════════════════════════════"
echo "Step 3/4: Building multi-platform image..."
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "⏳ This will take 20-30 minutes..."
echo ""

docker buildx build \
    --platform "${PLATFORMS}" \
    -f Dockerfile.base \
    -t "${FULL_IMAGE}" \
    --push \
    --progress=plain \
    .

echo ""
echo "✅ Multi-platform image built and pushed successfully!"
echo ""

# Show image info from Docker Hub (can't get size locally for multi-platform)
echo "══════════════════════════════════════════════════════════════════"
echo "Step 4/4: Verifying pushed image..."
echo "══════════════════════════════════════════════════════════════════"
echo ""

echo "🔍 Checking image manifest..."
docker buildx imagetools inspect "${FULL_IMAGE}"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ ✅ Multi-Platform Base Image Published Successfully!           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Image: ${FULL_IMAGE}"
echo "Platforms: linux/amd64, linux/arm64"
echo ""
echo "📝 Next Steps:"
echo "  1. Update Dockerfile to use this base image:"
echo "     FROM ${FULL_IMAGE}"
echo ""
echo "  2. Commit and push Dockerfile change:"
echo "     git add Dockerfile"
echo "     git commit -m 'feat: switch to multi-platform base image'"
echo "     git push origin main"
echo ""
echo "  3. Railway will rebuild quickly (~60 seconds) 🚀"
echo ""
echo "📌 Maintenance:"
echo "  - Rebuild when you update requirements.txt"
echo "  - Run: ./build-base-image.sh"
echo ""
echo "💡 Local testing:"
echo "  docker run --rm -it ${FULL_IMAGE} python --version"
echo ""
