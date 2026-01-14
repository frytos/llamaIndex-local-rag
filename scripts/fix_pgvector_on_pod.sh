#!/bin/bash
# Quick fix for pgvector compilation issue on RunPod

if [ $# -lt 2 ]; then
    echo "Usage: $0 TCP_HOST TCP_PORT [SSH_KEY]"
    exit 1
fi

TCP_HOST=$1
TCP_PORT=$2
SSH_KEY="${3:-${HOME}/.ssh/runpod_key}"

echo "Fixing pgvector installation on pod..."

ssh -i "$SSH_KEY" -p "$TCP_PORT" root@${TCP_HOST} << 'EOF'
set -e

echo "Detecting PostgreSQL version..."
PG_VERSION=$(psql --version | grep -oP '\d+' | head -1)
echo "PostgreSQL version: $PG_VERSION"

echo "Installing correct development headers..."
apt-get update -qq
apt-get install -y postgresql-server-dev-${PG_VERSION} libpq-dev

echo "Verifying postgres.h exists..."
if [ -f "/usr/include/postgresql/${PG_VERSION}/server/postgres.h" ]; then
    echo "✅ postgres.h found!"
else
    echo "❌ Still not found, trying alternative..."
    apt-get install -y postgresql-server-dev-all
fi

echo "Building pgvector..."
cd /tmp
rm -rf pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector

export PATH="/usr/lib/postgresql/${PG_VERSION}/bin:$PATH"
make clean
make
make install

echo "✅ pgvector installed successfully!"

echo "Testing pgvector..."
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;" template1 || echo "Will create extension later in vector_db"

echo "Done! Continue with initialization..."
EOF

echo ""
echo "✅ pgvector fixed!"
echo "Now re-run the setup script or SSH in and run init script manually"
