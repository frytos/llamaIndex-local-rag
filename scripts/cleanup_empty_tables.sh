#!/bin/bash
# Cleanup script to remove all empty vector tables
# Usage: ./cleanup_empty_tables.sh

PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD not set} psql -h localhost -U fryt -d vector_db << 'EOF'
DO $$
DECLARE
    tbl TEXT;
    cnt BIGINT;
    deleted INT := 0;
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '🧹 Cleaning Empty Vector Tables';
    RAISE NOTICE '================================';
    RAISE NOTICE '';

    FOR tbl IN
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND EXISTS (
              SELECT 1 FROM information_schema.columns
              WHERE table_name = tables.table_name
              AND column_name = 'embedding'
          )
        ORDER BY table_name
    LOOP
        EXECUTE format('SELECT COUNT(*) FROM %I', tbl) INTO cnt;

        IF cnt = 0 THEN
            RAISE NOTICE '❌ Deleting: % (0 chunks)', tbl;
            EXECUTE format('DROP TABLE %I CASCADE', tbl);
            deleted := deleted + 1;
        ELSE
            RAISE NOTICE '✅ Keeping:  % (% chunks)', tbl, cnt;
        END IF;
    END LOOP;

    RAISE NOTICE '';
    RAISE NOTICE '================================';
    IF deleted > 0 THEN
        RAISE NOTICE '🧹 Deleted % empty table(s)', deleted;
    ELSE
        RAISE NOTICE '✅ No empty tables found';
    END IF;
    RAISE NOTICE '';
END $$;
EOF

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining tables:"
PGPASSWORD=${PGPASSWORD:?Error: PGPASSWORD not set} psql -h localhost -U fryt -d vector_db -t -c "
SELECT table_name || ': ' || (SELECT COUNT(*) FROM pg_class WHERE relname = table_name) || ' exists'
FROM information_schema.tables
WHERE table_schema = 'public'
  AND EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_name = tables.table_name
      AND column_name = 'embedding'
  )
ORDER BY table_name;
"
