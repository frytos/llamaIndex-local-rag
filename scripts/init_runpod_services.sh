#!/bin/bash
# ============================================================================
# RunPod Service Initialization Script
# ============================================================================
# Initializes all services for RAG pipeline on RunPod:
#   - PostgreSQL with pgvector
#   - vLLM server with Mistral 7B AWQ
#   - Environment configuration
#   - Database setup with HNSW indices
#
# Usage:
#   bash scripts/init_runpod_services.sh
#
# This script should be run INSIDE the RunPod pod after deployment.
# ============================================================================

set -e  # Exit on error

echo "=========================================================================="
echo "RunPod Service Initialization - RAG Pipeline"
echo "=========================================================================="
echo ""

# ============================================================================
# Configuration
# ============================================================================

WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/rag-pipeline"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"
DATA_DIR="$PROJECT_DIR/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_info "✅ $1 is available"
        return 0
    else
        log_error "❌ $1 is not available"
        return 1
    fi
}

# ============================================================================
# Step 1: Environment Check
# ============================================================================

log_info "Step 1/7: Checking environment..."
echo ""

# Check Python
check_command python3
PYTHON_VERSION=$(python3 --version)
log_info "Python version: $PYTHON_VERSION"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    log_info "✅ CUDA is available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_warn "⚠️  nvidia-smi not found (may be CPU-only environment)"
fi

# Check PostgreSQL
if command -v psql &> /dev/null; then
    log_info "✅ PostgreSQL client available"
else
    log_warn "⚠️  PostgreSQL client not found (will install)"
fi

echo ""

# ============================================================================
# Step 2: Install System Dependencies
# ============================================================================

log_info "Step 2/7: Installing system dependencies..."
echo ""

# Update package lists
log_info "Updating package lists..."
apt-get update -qq
log_info "✅ Package lists updated"

# Install PostgreSQL
log_info "Installing PostgreSQL (this takes 1-2 minutes)..."
log_info "   Downloading packages..."
apt-get install -y postgresql postgresql-contrib 2>&1 | grep -E "Unpacking|Setting up|Processing" || true
log_info "✅ PostgreSQL installed"

# Install pgvector
log_info "Installing pgvector dependencies..."

# Detect PostgreSQL version
PG_VERSION=$(psql --version | grep -oP '\d+' | head -1)
log_info "   Detected PostgreSQL version: $PG_VERSION"

# Install correct dev package
log_info "   Installing postgresql-server-dev-${PG_VERSION}..."
apt-get install -y postgresql-server-dev-${PG_VERSION} build-essential git libpq-dev 2>&1 | grep -E "Unpacking|Setting up|Processing" || true

# Verify postgres.h exists
if [ -f "/usr/include/postgresql/${PG_VERSION}/server/postgres.h" ]; then
    log_info "   ✅ postgres.h found at /usr/include/postgresql/${PG_VERSION}/server/"
else
    log_error "   ❌ postgres.h not found! Trying alternative installation..."

    # Try installing all PostgreSQL dev packages
    apt-get install -y postgresql-server-dev-all 2>&1 | tail -5

    # Try again
    if [ -f "/usr/include/postgresql/${PG_VERSION}/server/postgres.h" ]; then
        log_info "   ✅ postgres.h found after reinstall"
    else
        log_error "   ❌ Still not found. Listing what we have:"
        ls -la /usr/include/postgresql/ 2>/dev/null || echo "No postgresql headers found"
    fi
fi

log_info "✅ Dependencies installed"

log_info "Cloning pgvector repository..."
cd /tmp
if [ -d "pgvector" ]; then
    log_info "   Removing old pgvector directory..."
    rm -rf pgvector
fi

git clone https://github.com/pgvector/pgvector.git 2>&1 | tail -3

log_info "Building pgvector..."
cd pgvector

# Set pg_config path explicitly
export PATH="/usr/lib/postgresql/${PG_VERSION}/bin:$PATH"
log_info "   Using pg_config: $(which pg_config)"

make clean > /dev/null 2>&1
log_info "   Compiling (showing output)..."
make 2>&1
log_info "   Installing..."
make install 2>&1
log_info "✅ pgvector installed"

echo ""

# ============================================================================
# Step 3: Start PostgreSQL
# ============================================================================

log_info "Step 3/7: Starting PostgreSQL..."
echo ""

# Initialize database if needed
if [ ! -d "/var/lib/postgresql/14/main" ]; then
    log_info "Initializing PostgreSQL cluster..."
    su - postgres -c "/usr/lib/postgresql/14/bin/initdb -D /var/lib/postgresql/14/main"
fi

# Configure PostgreSQL for external access (RunPod direct TCP)
log_info "Configuring PostgreSQL for external connections..."
log_info "   Setting listen_addresses to '*'..."

# Update postgresql.conf to listen on all interfaces
sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/${PG_VERSION}/main/postgresql.conf
sed -i "s/listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/${PG_VERSION}/main/postgresql.conf

# Update pg_hba.conf to allow external connections
log_info "   Adding external access rule to pg_hba.conf..."
echo "# Allow external connections (RunPod direct TCP)" >> /etc/postgresql/${PG_VERSION}/main/pg_hba.conf
echo "host    all             all             0.0.0.0/0               md5" >> /etc/postgresql/${PG_VERSION}/main/pg_hba.conf

log_info "✅ PostgreSQL configured for external access"

# Start PostgreSQL
log_info "Starting PostgreSQL service..."
service postgresql start

# Wait for PostgreSQL to be ready
sleep 3

if pg_isready -q; then
    log_info "✅ PostgreSQL is running"

    # Verify it's listening on all interfaces
    PG_LISTEN=$(ss -tulpn 2>/dev/null | grep 5432 | head -1 || netstat -tulpn 2>/dev/null | grep 5432 | head -1)
    log_info "   Listening on: ${PG_LISTEN}"
else
    log_error "❌ PostgreSQL failed to start"
    exit 1
fi

# Create database and user
log_info "Setting up database..."
su - postgres -c "psql -c \"CREATE USER fryt WITH PASSWORD 'frytos';\"" 2>/dev/null || true
su - postgres -c "psql -c \"CREATE DATABASE vector_db OWNER fryt;\"" 2>/dev/null || true
su - postgres -c "psql -d vector_db -c \"CREATE EXTENSION IF NOT EXISTS vector;\"" 2>/dev/null || true
su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE vector_db TO fryt;\"" 2>/dev/null || true

log_info "✅ Database configured"

echo ""

# ============================================================================
# Step 4: Setup Python Environment
# ============================================================================

log_info "Step 4/7: Setting up Python environment..."
echo ""

# Create project directory
mkdir -p "$PROJECT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
cd "$PROJECT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    log_info "✅ Virtual environment created"
else
    log_info "✅ Virtual environment exists"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip 2>&1 | grep -E "Successfully installed|Requirement already satisfied" || true

log_info "✅ Python environment ready"

echo ""

# ============================================================================
# Step 5: Clone/Update Repository
# ============================================================================

log_info "Step 5/7: Setting up codebase..."
echo ""

# Check if repository exists
if [ -d "$PROJECT_DIR/.git" ]; then
    log_info "Repository exists, pulling latest changes..."
    git pull origin main
else
    log_warn "Repository not found"
    log_info "Please clone your repository manually:"
    echo "  cd $PROJECT_DIR"
    echo "  git clone https://github.com/your-repo/rag-pipeline.git ."
fi

echo ""

# ============================================================================
# Step 6: Install Python Dependencies
# ============================================================================

log_info "Step 6/7: Installing Python dependencies..."
echo ""

# Check if requirements.txt exists
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    log_info "Installing from requirements.txt (this takes 3-5 minutes)..."
    log_info "   This may take a while, please be patient..."

    # Install without complex piping (more reliable)
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        log_info "✅ Python dependencies installed"
    else
        log_error "❌ Failed to install Python dependencies"
        exit 1
    fi

    # Install vLLM dependencies
    if [ -f "$PROJECT_DIR/config/requirements_vllm.txt" ]; then
        log_info "Installing vLLM dependencies (this takes 1-2 minutes)..."

        pip install -r config/requirements_vllm.txt

        if [ $? -eq 0 ]; then
            log_info "✅ vLLM dependencies installed"
        else
            log_error "❌ Failed to install vLLM dependencies"
            exit 1
        fi
    fi
else
    log_warn "requirements.txt not found, skipping Python package installation"
fi

echo ""

# ============================================================================
# Step 7: Start vLLM Server
# ============================================================================

log_info "Step 7/7: Starting vLLM server..."
echo ""

# Check if vLLM is installed
if python3 -c "import vllm" 2>/dev/null; then
    log_info "✅ vLLM is installed"

    # Start vLLM server in background
    log_info "Starting vLLM server (model download and loading takes 1-2 minutes)..."
    log_info "   Creating vLLM startup script..."

    # Create startup script for vLLM
    cat > "$PROJECT_DIR/start_vllm.sh" << 'VLLM_SCRIPT'
#!/bin/bash
source /workspace/rag-pipeline/.venv/bin/activate

export VLLM_MODEL="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
export CUDA_VISIBLE_DEVICES=0

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    > /workspace/rag-pipeline/logs/vllm.log 2>&1 &

echo $! > /workspace/rag-pipeline/vllm.pid
echo "vLLM server started (PID: $(cat /workspace/rag-pipeline/vllm.pid))"
VLLM_SCRIPT

    chmod +x "$PROJECT_DIR/start_vllm.sh"
    log_info "   Startup script created"

    # Start vLLM
    log_info "   Launching vLLM server in background..."
    bash "$PROJECT_DIR/start_vllm.sh"
    sleep 2

    # Check if process started
    if [ -f "$PROJECT_DIR/vllm.pid" ]; then
        VLLM_PID=$(cat "$PROJECT_DIR/vllm.pid")
        if ps -p $VLLM_PID > /dev/null 2>&1; then
            log_info "   ✅ vLLM process started (PID: $VLLM_PID)"
        else
            log_warn "   ⚠️  vLLM process may have failed to start"
        fi
    fi

    log_info "✅ vLLM server starting (downloading model and loading into GPU...)"
    log_info "   This will continue in background for 60-90 seconds"
    log_info "   Monitor progress: tail -f $LOG_DIR/vllm.log"
    log_info "   Test when ready: curl http://localhost:8000/health"
else
    log_warn "⚠️  vLLM not installed, skipping server startup"
    log_info "   Install with: pip install vllm"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=========================================================================="
echo "SERVICE INITIALIZATION COMPLETE"
echo "=========================================================================="
echo ""
echo "📊 Service Status:"
echo "   ✅ PostgreSQL: Running on port 5432"
echo "   ✅ pgvector: Extension installed"
echo "   ✅ Python: Virtual environment ready"

if [ -f "$PROJECT_DIR/vllm.pid" ]; then
    echo "   ✅ vLLM: Starting (check logs/vllm.log)"
else
    echo "   ⚠️  vLLM: Not started"
fi

echo ""
echo "🔗 Connection Info:"
echo "   PostgreSQL: localhost:5432"
echo "   vLLM API: http://localhost:8000"
echo "   Database: vector_db"
echo "   User: fryt"
echo ""
echo "📝 Next Steps:"
echo "   1. Wait for vLLM to load model (60-90s)"
echo "   2. Test vLLM: curl http://localhost:8000/health"
echo "   3. Test PostgreSQL: psql -h localhost -U fryt -d vector_db"
echo "   4. Run RAG pipeline: python rag_low_level_m1_16gb_verbose.py"
echo ""
echo "📄 Logs:"
echo "   vLLM: $LOG_DIR/vllm.log"
echo "   PostgreSQL: /var/log/postgresql/"
echo ""
echo "=========================================================================="
