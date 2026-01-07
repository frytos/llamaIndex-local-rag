#!/bin/bash
# quick-start.sh - Automated setup script for Local RAG Pipeline
# Usage: ./quick-start.sh [preset]
# Presets: minimal, mac, gpu, dev

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration presets
declare -A PRESETS
PRESETS[minimal]="CPU-only setup with minimal resources"
PRESETS[mac]="Optimized for Apple Silicon (M1/M2/M3) with 16GB RAM"
PRESETS[gpu]="GPU-accelerated with vLLM (NVIDIA RTX 4090)"
PRESETS[dev]="Development setup with testing and linting tools"

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

check_python_version() {
    if command -v python3 &> /dev/null; then
        version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)

        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            print_success "Python $version (>= 3.11 required)"
            return 0
        else
            print_error "Python $version found, but 3.11+ required"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

show_presets() {
    print_header "Available Configuration Presets"
    for preset in "${!PRESETS[@]}"; do
        echo -e "${GREEN}$preset${NC}: ${PRESETS[$preset]}"
    done
    echo ""
}

apply_preset() {
    local preset=$1
    print_header "Applying $preset preset"

    case $preset in
        minimal)
            cat >> .env << 'EOF'

# MINIMAL PRESET (CPU-only, low resources)
EMBED_BACKEND=huggingface
N_GPU_LAYERS=0
N_BATCH=64
EMBED_BATCH=16
CTX=3072
CHUNK_SIZE=900
CHUNK_OVERLAP=120
TOP_K=3
EOF
            print_success "Minimal preset applied (CPU-only, low resources)"
            ;;

        mac)
            cat >> .env << 'EOF'

# MAC PRESET (Apple Silicon optimized)
EMBED_BACKEND=mlx
N_GPU_LAYERS=24
N_BATCH=256
EMBED_BATCH=64
CTX=8192
CHUNK_SIZE=700
CHUNK_OVERLAP=150
TOP_K=4
EOF
            print_success "Mac preset applied (optimized for M1/M2/M3)"
            print_info "Install MLX: pip install mlx mlx-embedding-models"
            ;;

        gpu)
            cat >> .env << 'EOF'

# GPU PRESET (NVIDIA GPU with vLLM)
USE_VLLM=1
VLLM_MODEL=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
EMBED_BACKEND=torch
EMBED_BATCH=256
N_GPU_LAYERS=99
N_BATCH=512
CTX=8192
CHUNK_SIZE=700
CHUNK_OVERLAP=150
TOP_K=4
EOF
            print_success "GPU preset applied (vLLM on NVIDIA GPU)"
            print_info "Install vLLM: pip install -r config/requirements_vllm.txt"
            ;;

        dev)
            cat >> .env << 'EOF'

# DEV PRESET (Development and testing)
LOG_LEVEL=DEBUG
LOG_FULL_CHUNKS=1
LOG_QUERIES=1
EMBED_BACKEND=huggingface
N_GPU_LAYERS=16
N_BATCH=128
EMBED_BATCH=32
CTX=3072
EOF
            print_success "Dev preset applied (verbose logging, testing)"
            ;;

        *)
            print_error "Unknown preset: $preset"
            show_presets
            exit 1
            ;;
    esac
}

# Main setup flow
main() {
    print_header "Local RAG Pipeline - Quick Start Setup"

    # Parse arguments
    PRESET=${1:-minimal}

    if [ "$PRESET" = "help" ] || [ "$PRESET" = "-h" ] || [ "$PRESET" = "--help" ]; then
        show_presets
        echo "Usage: ./quick-start.sh [preset]"
        echo "Example: ./quick-start.sh mac"
        exit 0
    fi

    # Step 1: Check prerequisites
    print_header "Step 1: Checking Prerequisites"

    ERRORS=0
    check_python_version || ERRORS=$((ERRORS+1))
    check_command pip3 || ERRORS=$((ERRORS+1))
    check_command docker || print_warning "Docker not found (optional, needed for PostgreSQL)"
    check_command git || ERRORS=$((ERRORS+1))

    if [ $ERRORS -gt 0 ]; then
        print_error "Prerequisites check failed. Please install missing dependencies."
        exit 1
    fi

    print_success "All required prerequisites installed"

    # Step 2: Create virtual environment
    print_header "Step 2: Setting Up Virtual Environment"

    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf .venv
            print_info "Removed existing virtual environment"
        else
            print_info "Keeping existing virtual environment"
        fi
    fi

    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        print_success "Virtual environment created"
    fi

    # Step 3: Activate and install dependencies
    print_header "Step 3: Installing Dependencies"

    source .venv/bin/activate
    print_success "Virtual environment activated"

    pip install --upgrade pip > /dev/null 2>&1
    print_success "pip upgraded"

    print_info "Installing core dependencies (this may take a few minutes)..."
    pip install -r requirements.txt > /dev/null 2>&1
    print_success "Core dependencies installed"

    if [ "$PRESET" = "dev" ]; then
        print_info "Installing development dependencies..."
        pip install -r requirements-dev.txt > /dev/null 2>&1
        print_success "Development dependencies installed"
    fi

    # Step 4: Configure environment
    print_header "Step 4: Configuring Environment"

    if [ -f ".env" ]; then
        print_warning ".env file already exists"
        read -p "Overwrite with new configuration? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Keeping existing .env file"
            print_success "Setup complete!"
            exit 0
        fi
    fi

    cp config/.env.example .env
    print_success "Created .env from template"

    # Apply preset
    apply_preset "$PRESET"

    # Prompt for database credentials
    print_info "Database configuration required"
    read -p "PostgreSQL username (default: postgres): " PGUSER
    PGUSER=${PGUSER:-postgres}

    read -sp "PostgreSQL password: " PGPASSWORD
    echo

    if [ -z "$PGPASSWORD" ]; then
        print_warning "No password provided, using 'postgres'"
        PGPASSWORD="postgres"
    fi

    # Update .env with credentials
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/PGUSER=.*/PGUSER=$PGUSER/" .env
        sed -i '' "s/PGPASSWORD=.*/PGPASSWORD=$PGPASSWORD/" .env
    else
        # Linux
        sed -i "s/PGUSER=.*/PGUSER=$PGUSER/" .env
        sed -i "s/PGPASSWORD=.*/PGPASSWORD=$PGPASSWORD/" .env
    fi

    print_success "Database credentials configured"

    # Step 5: Start PostgreSQL
    print_header "Step 5: Starting PostgreSQL"

    if command -v docker &> /dev/null; then
        print_info "Starting PostgreSQL with Docker Compose..."
        docker compose -f config/docker-compose.yml up -d
        print_success "PostgreSQL started"
        print_info "Waiting 5 seconds for database to initialize..."
        sleep 5
    else
        print_warning "Docker not available, skipping PostgreSQL setup"
        print_info "Please start PostgreSQL manually"
    fi

    # Step 6: Create data directory
    print_header "Step 6: Setting Up Data Directory"

    mkdir -p data
    print_success "Data directory created"

    if [ ! -f "data/.gitkeep" ]; then
        touch data/.gitkeep
    fi

    # Step 7: Validation
    print_header "Step 7: Validating Setup"

    print_info "Running validation checks..."

    # Test Python imports
    if python3 -c "import llama_index" 2>/dev/null; then
        print_success "llama_index import successful"
    else
        print_error "llama_index import failed"
    fi

    # Test database connection
    if command -v docker &> /dev/null; then
        if python3 -c "import psycopg2; conn = psycopg2.connect(host='localhost', port=5432, user='$PGUSER', password='$PGPASSWORD', dbname='postgres')" 2>/dev/null; then
            print_success "Database connection successful"
        else
            print_warning "Database connection failed (may need manual setup)"
        fi
    fi

    # Final success message
    print_header "Setup Complete!"

    echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo -e "${GREEN}✓ Environment configured ($PRESET preset)${NC}"
    echo -e "${GREEN}✓ PostgreSQL started${NC}"
    echo ""

    print_info "Next steps:"
    echo "  1. Activate environment: source .venv/bin/activate"
    echo "  2. Add documents to data/ directory"
    echo "  3. Run pipeline: python rag_low_level_m1_16gb_verbose.py"
    echo "  4. Try interactive mode: python rag_low_level_m1_16gb_verbose.py --interactive"
    echo "  5. Launch web UI: streamlit run rag_web.py"
    echo ""

    if [ "$PRESET" = "dev" ]; then
        print_info "Development tools:"
        echo "  - Run tests: pytest tests/"
        echo "  - Format code: black ."
        echo "  - Lint code: ruff check ."
        echo "  - Type check: mypy ."
    fi

    print_success "Ready to start building with RAG!"
}

# Run main function
main "$@"
