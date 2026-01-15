#!/bin/bash
#
# Apply HNSW index to PostgreSQL vector table for 2-5x faster queries.
#
# Usage:
#   ./scripts/apply_hnsw.sh [table_name]
#   ./scripts/apply_hnsw.sh inbox_clean
#
# HNSW (Hierarchical Navigable Small World) provides fast approximate
# nearest neighbor search. Much faster than default IVFFlat for queries.
#

set -e  # Exit on error

PGTABLE=${1:-inbox_clean}
PGHOST=${PGHOST:-localhost}
PGPORT=${PGPORT:-5432}
PGUSER=${PGUSER:-fryt}
PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD must be set in .env}}
DB_NAME=${DB_NAME:-vector_db}

echo "========================================================================"
echo "  Applying HNSW Index"
echo "========================================================================"
echo "Table:    $PGTABLE"
echo "Database: $DB_NAME"
echo "Host:     $PGHOST:$PGPORT"
echo ""
echo "This will:"
echo "  1. Drop existing index (if any)"
echo "  2. Create HNSW index for fast vector search"
echo "  3. Update table statistics"
echo ""
echo "Expected time: 2-5 minutes for 47k rows"
echo "Expected speedup: 2-5x faster queries"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Starting index creation..."
echo ""

# Execute SQL
PGPASSWORD=$PGPASSWORD psql -h $PGHOST -p $PGPORT -U $PGUSER -d $DB_NAME <<EOF

-- Drop old index if exists
DO \$\$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = '$PGTABLE'
        AND indexname = '${PGTABLE}_embedding_idx'
    ) THEN
        EXECUTE 'DROP INDEX ${PGTABLE}_embedding_idx';
        RAISE NOTICE 'Dropped existing index';
    END IF;
END \$\$;

-- Create HNSW index
-- Parameters:
--   m = 16: Number of connections per layer (default, good balance)
--   ef_construction = 64: Construction time/quality tradeoff
\timing on
CREATE INDEX ${PGTABLE}_embedding_idx ON $PGTABLE
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Update statistics for query planner
VACUUM ANALYZE $PGTABLE;

-- Verify index was created
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
FROM pg_indexes
WHERE tablename = '$PGTABLE'
AND indexname LIKE '%embedding%';

EOF

echo ""
echo "========================================================================"
echo "  ✓ HNSW Index Applied Successfully"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Test query speed improvement:"
echo "     python rag_low_level_m1_16gb_verbose.py --query-only \\"
echo "       --query \"test query\""
echo ""
echo "  2. Expected improvements:"
echo "     • 2-5x faster retrieval"
echo "     • More accurate approximate search"
echo "     • Better performance with large datasets"
echo ""
