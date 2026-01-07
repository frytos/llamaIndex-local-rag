# Runbook: Database Failure

**Severity**: P0 (Critical)
**Estimated Resolution Time**: 5-15 minutes
**Last Updated**: 2026-01-07

## Overview

This runbook covers PostgreSQL database connection failures, corruption, and recovery scenarios for the Local RAG Pipeline.

## Symptoms

- `psycopg2.OperationalError: could not connect to server`
- `FATAL: database "vector_db" does not exist`
- `ERROR: relation "table_name" does not exist`
- Pipeline hangs during database operations
- Slow query performance (>5s for simple queries)
- Transaction aborted errors

## Quick Diagnosis

```bash
# 1. Check PostgreSQL container status
docker ps | grep postgres
# Expected: Container running on port 5432

# 2. Test connection
psql -h localhost -U $PGUSER -d vector_db -c "SELECT version();"

# 3. Check database size
docker exec -it postgres psql -U $PGUSER -c "\l+"

# 4. Check active connections
docker exec -it postgres psql -U $PGUSER -d vector_db -c "SELECT count(*) FROM pg_stat_activity;"
```

## Common Issues & Solutions

### Issue 1: Container Not Running

**Symptoms**: `could not connect to server: Connection refused`

**Solution**:
```bash
# Start PostgreSQL container
cd /Users/frytos/code/llamaIndex-local-rag
docker compose -f config/docker-compose.yml up -d

# Wait for startup
sleep 10

# Verify
docker ps | grep postgres
```

**Verification**:
```bash
psql -h localhost -U $PGUSER -d postgres -c "SELECT 1;"
# Expected: Returns 1
```

### Issue 2: Database Does Not Exist

**Symptoms**: `FATAL: database "vector_db" does not exist`

**Solution**:
```bash
# Connect to default database
psql -h localhost -U $PGUSER -d postgres

# Create database
CREATE DATABASE vector_db;

# Enable pgvector extension
\c vector_db
CREATE EXTENSION IF NOT EXISTS vector;

# Exit
\q
```

**Automated Fix**:
```bash
# Run pipeline with RESET_DB=0 (safe mode)
python rag_low_level_m1_16gb_verbose.py
# Pipeline auto-creates database if missing
```

### Issue 3: pgvector Extension Missing

**Symptoms**: `ERROR: type "vector" does not exist`

**Solution**:
```bash
# Connect to database
psql -h localhost -U $PGUSER -d vector_db

# Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

# Verify
\dx
# Should show 'vector' extension

# Exit
\q
```

### Issue 4: Table Does Not Exist

**Symptoms**: `ERROR: relation "table_name" does not exist`

**Solution**:
```bash
# List all tables
psql -h localhost -U $PGUSER -d vector_db -c "\dt"

# If table missing, re-index
PGTABLE=my_table RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
```

### Issue 5: Transaction Aborted Errors

**Symptoms**: `InFailedSqlTransaction: current transaction is aborted`

**Root Cause**: Connection not using autocommit mode

**Solution**:
```python
# Fix in code: Ensure autocommit is enabled
import psycopg2

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    dbname=os.getenv("DB_NAME")
)
conn.autocommit = True  # Critical!
```

**Workaround**: Restart PostgreSQL
```bash
docker compose -f config/docker-compose.yml restart
```

### Issue 6: Connection Pool Exhausted

**Symptoms**: `FATAL: sorry, too many clients already`

**Solution**:
```bash
# 1. Check active connections
psql -h localhost -U $PGUSER -d vector_db -c \
  "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

# 2. Kill idle connections
psql -h localhost -U $PGUSER -d vector_db -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
   WHERE state = 'idle' AND pid <> pg_backend_pid();"

# 3. Increase max_connections (if needed)
# Edit docker-compose.yml and add:
#   command: postgres -c max_connections=200
docker compose -f config/docker-compose.yml restart
```

### Issue 7: Slow Query Performance

**Symptoms**: Queries taking >5s, high CPU usage

**Diagnosis**:
```bash
# Check query performance
psql -h localhost -U $PGUSER -d vector_db -c \
  "SELECT query, mean_exec_time, calls FROM pg_stat_statements
   ORDER BY mean_exec_time DESC LIMIT 10;"

# Check table statistics
psql -h localhost -U $PGUSER -d vector_db -c \
  "SELECT schemaname, tablename, n_live_tup, n_dead_tup
   FROM pg_stat_user_tables;"
```

**Solution**: Add HNSW index for vector similarity
```bash
# Run index optimization script
./scripts/database_apply_hnsw.sh

# Or manually:
psql -h localhost -U $PGUSER -d vector_db << EOF
CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
  ON llama2_paper USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
EOF
```

**Expected Improvement**: 10-100x faster similarity search

## Complete Database Reset (Nuclear Option)

**WARNING**: This destroys all data! Only use if database is corrupted beyond repair.

```bash
# 1. Stop container
docker compose -f config/docker-compose.yml down

# 2. Remove volumes (destroys data)
docker volume rm llamaindex-local-rag_postgres_data

# 3. Restart fresh
docker compose -f config/docker-compose.yml up -d

# 4. Wait for startup
sleep 10

# 5. Verify
psql -h localhost -U $PGUSER -d postgres -c "SELECT version();"

# 6. Re-index all documents
RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
```

## Database Backup & Restore

### Backup

```bash
# Backup single table
pg_dump -h localhost -U $PGUSER -d vector_db -t llama2_paper > backup_table.sql

# Backup entire database
pg_dump -h localhost -U $PGUSER -d vector_db > backup_full.sql

# Backup with compression
pg_dump -h localhost -U $PGUSER -d vector_db | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore

```bash
# Restore table
psql -h localhost -U $PGUSER -d vector_db < backup_table.sql

# Restore full database
psql -h localhost -U $PGUSER -d vector_db < backup_full.sql

# Restore compressed backup
gunzip -c backup_20260107.sql.gz | psql -h localhost -U $PGUSER -d vector_db
```

## Health Checks

### Automated Health Check Script

```bash
#!/bin/bash
# database-health-check.sh

echo "=== PostgreSQL Health Check ==="

# 1. Container status
echo "1. Container Status:"
docker ps --filter "name=postgres" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. Connection test
echo -e "\n2. Connection Test:"
if psql -h localhost -U $PGUSER -d vector_db -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✓ Connection successful"
else
    echo "✗ Connection failed"
fi

# 3. Database size
echo -e "\n3. Database Size:"
psql -h localhost -U $PGUSER -d vector_db -c \
  "SELECT pg_size_pretty(pg_database_size('vector_db'));"

# 4. Active connections
echo -e "\n4. Active Connections:"
psql -h localhost -U $PGUSER -d vector_db -c \
  "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 5. Tables and row counts
echo -e "\n5. Tables:"
psql -h localhost -U $PGUSER -d vector_db -c \
  "SELECT schemaname, tablename, n_live_tup as rows
   FROM pg_stat_user_tables
   ORDER BY n_live_tup DESC;"

echo -e "\n✓ Health check complete"
```

Save as `database-health-check.sh` and run:
```bash
chmod +x database-health-check.sh
./database-health-check.sh
```

## Performance Monitoring

### Key Metrics to Monitor

1. **Connection count**: Should be < max_connections
2. **Query latency**: Should be <1s for similarity search
3. **Database size**: Monitor growth rate
4. **Cache hit ratio**: Should be >95%
5. **Index usage**: Ensure indexes are used

### Monitoring Query

```sql
-- Save as monitoring-query.sql
SELECT
  'Connections' as metric,
  count(*) as value,
  '< ' || current_setting('max_connections') as threshold
FROM pg_stat_activity
UNION ALL
SELECT
  'Cache Hit Ratio',
  round(sum(blks_hit) / (sum(blks_hit) + sum(blks_read)) * 100, 2),
  '> 95%'
FROM pg_stat_database
WHERE datname = 'vector_db'
UNION ALL
SELECT
  'Database Size',
  pg_size_pretty(pg_database_size('vector_db')),
  'Monitor growth'
UNION ALL
SELECT
  'Active Queries',
  count(*),
  '< 50'
FROM pg_stat_activity
WHERE state = 'active';
```

Run periodically:
```bash
psql -h localhost -U $PGUSER -d vector_db -f monitoring-query.sql
```

## Escalation Path

If issue persists after following this runbook:

1. **Check logs**:
   ```bash
   docker logs postgres --tail 100
   ```

2. **Check disk space**:
   ```bash
   df -h
   # Ensure >10GB free
   ```

3. **Check system resources**:
   ```bash
   docker stats postgres
   # CPU should be <80%, memory <80%
   ```

4. **Open GitHub issue** with:
   - Error messages
   - PostgreSQL logs
   - Output of health check script
   - System specs (RAM, disk space)

## Prevention

### Best Practices

1. **Regular backups**: Backup database daily
2. **Monitor disk space**: Keep >20% free
3. **Use connection pooling**: Avoid connection leaks
4. **Enable autocommit**: Prevent transaction errors
5. **Add HNSW indexes**: Optimize vector search
6. **Vacuum regularly**: Clean dead tuples

### Automated Maintenance

Add to cron (run weekly):
```bash
# Vacuum and analyze
psql -h localhost -U $PGUSER -d vector_db -c "VACUUM ANALYZE;"

# Reindex
psql -h localhost -U $PGUSER -d vector_db -c "REINDEX DATABASE vector_db;"
```

## Related Runbooks

- [vLLM Server Crash](vllm-crash.md)
- [Out of Memory](out-of-memory.md)

## Change Log

- 2026-01-07: Initial version
