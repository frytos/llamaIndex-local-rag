#!/bin/bash
# Quick test script to verify RAG queries work
# Usage: ./test_query.sh

cd "$(dirname "$0")"

echo "🧪 Testing RAG Query Pipeline"
echo "=============================="
echo ""

# Check database is running
echo "1️⃣  Checking database connection..."
if PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD not set} psql -h localhost -U fryt -d vector_db -c "SELECT 1;" > /dev/null 2>&1; then
    echo "   ✅ Database connected"
else
    echo "   ❌ Database not running!"
    echo "   Start it with: ./start_db.sh"
    exit 1
fi

# Check tables
echo ""
echo "2️⃣  Checking available indexes..."
PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD not set} psql -h localhost -U fryt -d vector_db -t -c "
SELECT
    table_name || ': ' || COUNT(*) || ' chunks'
FROM information_schema.tables t,
     LATERAL (SELECT COUNT(*) FROM (SELECT table_name as tn) x) c
WHERE table_schema='public'
  AND EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_name = t.table_name AND column_name = 'embedding'
  )
GROUP BY table_name
ORDER BY table_name;
" 2>&1 | grep -v "^$" | while read line; do
    echo "   $line"
done

# Test query
echo ""
echo "3️⃣  Testing query on data_messages-text-slim_fast_1883_260108..."

count=$(PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD not set} psql -h localhost -U fryt -d vector_db -t -A -c "
SELECT COUNT(*) FROM \"data_messages-text-slim_fast_1883_260108\" WHERE LOWER(text) LIKE '%agathe%';
")

echo "   Found $count chunks containing 'agathe'"

if [ "$count" -gt 0 ]; then
    echo "   ✅ Data exists in table"
else
    echo "   ⚠️  No chunks found with 'agathe'"
fi

# Test vector search
echo ""
echo "4️⃣  Testing vector similarity search..."

result=$(PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD not set} psql -h localhost -U fryt -d vector_db -t -A -c "
WITH sample AS (
    SELECT embedding FROM \"data_messages-text-slim_fast_1883_260108\"
    WHERE LOWER(text) LIKE '%agathe%' LIMIT 1
)
SELECT COUNT(*)
FROM \"data_messages-text-slim_fast_1883_260108\" d, sample s
WHERE (1 - (d.embedding <=> s.embedding)) > 0.5;
" 2>&1)

echo "   Found $result similar chunks (threshold > 0.5)"

if [ "$result" -gt 0 ]; then
    echo "   ✅ Vector search works!"
else
    echo "   ⚠️  Vector search returned 0 results"
fi

echo ""
echo "=============================="
echo "✅ Database and table are ready!"
echo ""
echo "🌐 Launch Streamlit:"
echo "   streamlit run rag_web_enhanced.py"
echo ""
echo "💡 Tips:"
echo "   - Use Query page to search"
echo "   - Check Settings page for configuration"
echo "   - Ensure Hybrid Search = 1.0"
echo "   - Disable all Advanced Features first"
