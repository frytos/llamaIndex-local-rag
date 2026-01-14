#!/bin/bash
# Start PostgreSQL database in Docker
# Usage: ./start_db.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🗄️  Starting PostgreSQL Database"
echo "================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Create it with: cp config/.env.example .env"
    exit 1
fi

# Load environment
source .env

# Check PGPASSWORD
if [ -z "$PGPASSWORD" ]; then
    echo "❌ PGPASSWORD not set in .env!"
    exit 1
fi

# Create symlink if needed
if [ ! -f config/.env ] && [ ! -L config/.env ]; then
    echo "Creating .env symlink in config/..."
    ln -s ../.env config/.env
fi

# Start database
cd config
echo "Starting PostgreSQL container..."
docker-compose up -d db

echo ""
echo "Waiting for database to be ready..."
for i in {1..30}; do
    if docker-compose exec -T db pg_isready -U "${PGUSER:-postgres}" -d "${DB_NAME:-vector_db}" > /dev/null 2>&1; then
        echo "✅ Database is ready!"
        echo ""
        echo "Connection info:"
        echo "  Host: ${PGHOST:-localhost}"
        echo "  Port: ${PGPORT:-5432}"
        echo "  Database: ${DB_NAME:-vector_db}"
        echo "  User: ${PGUSER:-postgres}"
        echo ""
        echo "Test connection:"
        echo "  psql -h ${PGHOST:-localhost} -U ${PGUSER:-postgres} -d ${DB_NAME:-vector_db}"
        echo ""
        echo "Stop database:"
        echo "  cd config && docker-compose down"
        exit 0
    fi
    sleep 1
done

echo "❌ Database failed to start after 30 seconds"
echo "Check logs: cd config && docker-compose logs db"
exit 1
