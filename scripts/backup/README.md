# Backup System Documentation

Automated backup and recovery system for PostgreSQL + pgvector database.

## Quick Start

```bash
# Run backup immediately
./backup_postgres.sh

# Setup automated daily backups at 2 AM
./setup_cron.sh

# Verify backup
./verify_backup.sh
```

## Features

- Full PostgreSQL database backups with pgvector extension
- Gzip compression to save disk space
- Configurable retention policy (default: 7 days)
- Backup verification and integrity checks
- Test restore capability
- Prometheus metrics export
- Automated scheduling via cron
- Docker-based backup service option

## Files

| File | Purpose |
|------|---------|
| `backup_postgres.sh` | Main backup script |
| `verify_backup.sh` | Backup verification and testing |
| `setup_cron.sh` | Cron job configuration |
| `README.md` | This file |

## Usage

### Manual Backup

**Basic backup**:
```bash
./backup_postgres.sh
```

**Custom retention** (14 days):
```bash
./backup_postgres.sh --retention 14
```

**List backups**:
```bash
./backup_postgres.sh --list
```

**Verify backup**:
```bash
./backup_postgres.sh --verify
```

### Automated Backups

**Option 1: Cron Job** (recommended for local/VM)

```bash
# Setup daily backup at 2 AM
./setup_cron.sh

# Custom schedule (every 6 hours)
./setup_cron.sh --time "0 */6 * * *"

# Check status
./setup_cron.sh --status

# Remove cron job
./setup_cron.sh --remove
```

**Option 2: Docker Service** (recommended for containerized)

```bash
# Start backup service
cd ../../config
docker-compose --profile backup up -d

# Check logs
docker-compose logs -f backup
```

### Backup Verification

**Quick verification** (integrity check only):
```bash
./verify_backup.sh
```

**Specific backup**:
```bash
./verify_backup.sh backups/postgres/vector_db_20260107_120000.sql.gz
```

**Test restore** (creates temporary database):
```bash
./verify_backup.sh --test-restore
```

**Verify all backups**:
```bash
./verify_backup.sh --all
```

### Restore from Backup

**Interactive restore**:
```bash
./backup_postgres.sh --restore backups/postgres/vector_db_20260107_120000.sql.gz
```

**Manual restore**:
```bash
# Drop existing database
psql -h localhost -U fryt -d postgres -c "DROP DATABASE vector_db;"
psql -h localhost -U fryt -d postgres -c "CREATE DATABASE vector_db;"

# Restore backup
zcat backups/postgres/vector_db_20260107_120000.sql.gz | \
  psql -h localhost -U fryt -d vector_db

# Enable pgvector
psql -h localhost -U fryt -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Configuration

### Environment Variables

Set in `.env` file or export:

```bash
# Database connection
PGHOST=localhost
PGPORT=5432
PGUSER=fryt
PGPASSWORD=frytos
DB_NAME=vector_db

# Backup settings
BACKUP_DIR=backups/postgres        # Backup directory
BACKUP_RETENTION_DAYS=7            # Days to keep backups
METRICS_DIR=metrics                # Prometheus metrics directory
```

### Backup Location

**Default**: `<project_root>/backups/postgres/`

**Format**: `vector_db_YYYYMMDD_HHMMSS.sql.gz`

**Example**: `vector_db_20260107_143022.sql.gz`

### Retention Policy

**Default**: 7 days

Backups older than retention period are automatically deleted during backup runs.

**Change retention**:
```bash
# Environment variable
export BACKUP_RETENTION_DAYS=14
./backup_postgres.sh

# Command line flag
./backup_postgres.sh --retention 14
```

## Cron Schedules

Common cron schedule formats:

| Schedule | Cron Expression | Description |
|----------|----------------|-------------|
| Daily at 2 AM | `0 2 * * *` | Default, low-traffic time |
| Every 6 hours | `0 */6 * * *` | High-change rate |
| Every 12 hours | `0 */12 * * *` | Moderate frequency |
| Weekly (Sunday 3 AM) | `0 3 * * 0` | Low-change rate |
| Monthly (1st at 4 AM) | `0 4 1 * *` | Archive backups |

**Cron format**:
```
* * * * *
│ │ │ │ │
│ │ │ │ └─── Day of week (0-7, Sunday=0 or 7)
│ │ │ └───── Month (1-12)
│ │ └─────── Day of month (1-31)
│ └───────── Hour (0-23)
└─────────── Minute (0-59)
```

## Backup Process

1. **Connection Check**: Verify database connectivity
2. **Backup Execution**: Run `pg_dump` with compression
3. **Verification**: Test gzip integrity and SQL format
4. **Metadata**: Write backup metadata JSON
5. **Cleanup**: Remove old backups beyond retention
6. **Metrics**: Export Prometheus metrics
7. **Logging**: Record all operations

## Metrics

Backup metrics are exported to `metrics/backup.prom` for Prometheus:

| Metric | Type | Description |
|--------|------|-------------|
| `rag_backup_success` | gauge | Last backup status (1=success, 0=fail) |
| `rag_backup_last_success_timestamp` | gauge | Timestamp of last successful backup |
| `rag_backup_last_failure_timestamp` | gauge | Timestamp of last failed backup |
| `rag_backup_duration_seconds` | gauge | Duration of last backup |
| `rag_backup_size_bytes` | gauge | Size of last backup |
| `rag_backup_verification_success` | gauge | Verification status |
| `rag_backup_total_count` | gauge | Total backups retained |

**Query backup age** (Prometheus):
```promql
(time() - rag_backup_last_success_timestamp) / 3600
```

## Monitoring

### Alerts

Configured in `config/monitoring/alerts.yml`:

- **BackupFailed**: Critical alert if backup fails
- **BackupStale**: Warning if no backup in 48 hours
- **BackupStorageLow**: Warning if <20% storage available

### Grafana Dashboard

Backup metrics included in **RAG Pipeline Overview** dashboard:
- Backup status indicator
- Last backup timestamp
- Backup size trend
- Success/failure rate

## Troubleshooting

### Backup Fails

**Check logs**:
```bash
tail -f logs/backup_*.log
```

**Common issues**:

1. **Connection failure**:
   ```bash
   # Test connection
   psql -h $PGHOST -U $PGUSER -d $DB_NAME -c "SELECT 1"

   # Check credentials in .env
   cat .env | grep PG
   ```

2. **Disk space**:
   ```bash
   # Check available space
   df -h backups/

   # Clean old backups manually
   rm backups/postgres/vector_db_202501*.sql.gz
   ```

3. **Permissions**:
   ```bash
   # Make script executable
   chmod +x backup_postgres.sh

   # Check directory permissions
   ls -la backups/
   ```

### Verification Fails

**Gzip corruption**:
```bash
# Test gzip integrity
gzip -t backups/postgres/vector_db_20260107_120000.sql.gz

# Re-run backup
./backup_postgres.sh
```

**Invalid SQL**:
```bash
# Check backup contents
zcat backups/postgres/vector_db_20260107_120000.sql.gz | head -n 20

# Should see PostgreSQL dump header
```

### Restore Issues

**Extension missing**:
```bash
# Enable pgvector after restore
psql -h localhost -U fryt -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Permission errors**:
```bash
# Grant permissions
psql -h localhost -U fryt -d vector_db -c "
GRANT ALL ON DATABASE vector_db TO fryt;
GRANT ALL ON ALL TABLES IN SCHEMA public TO fryt;
"
```

## Best Practices

### Backup Strategy

1. **Frequency**: Daily backups minimum
2. **Retention**: 7 days for rolling, monthly for archives
3. **Verification**: Weekly test restores
4. **Monitoring**: Alert on failed backups
5. **Storage**: Separate disk/volume from database

### Disaster Recovery

1. **Off-site Backups**: Copy to remote storage
2. **Encryption**: Encrypt sensitive backups
3. **Documentation**: Maintain recovery procedures
4. **Testing**: Regular DR drills

### Storage Management

```bash
# Monitor backup storage
du -sh backups/

# Archive old backups to external storage
tar -czf backups_archive_$(date +%Y%m).tar.gz backups/postgres/

# Clean up after archiving
find backups/ -name "*.sql.gz" -mtime +30 -delete
```

## Advanced Usage

### Backup to Remote Storage

**S3 (AWS)**:
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Backup and upload
./backup_postgres.sh
aws s3 cp backups/postgres/vector_db_*.sql.gz s3://your-bucket/backups/
```

**rsync to remote server**:
```bash
# Add to backup_postgres.sh or run separately
rsync -avz --progress backups/postgres/ user@remote-server:/path/to/backups/
```

### Encrypted Backups

```bash
# Backup with GPG encryption
./backup_postgres.sh
gpg --encrypt --recipient your@email.com \
  backups/postgres/vector_db_20260107_120000.sql.gz

# Decrypt for restore
gpg --decrypt backups/postgres/vector_db_20260107_120000.sql.gz.gpg | \
  gunzip | psql -h localhost -U fryt -d vector_db
```

### Incremental Backups

For large databases, consider incremental backups:

```bash
# Full backup weekly
0 2 * * 0 /path/to/backup_postgres.sh

# WAL archiving (continuous)
# Configure in postgresql.conf:
# archive_mode = on
# archive_command = 'cp %p /path/to/archive/%f'
```

## Testing

### Test Backup Workflow

```bash
# 1. Create test data
psql -h localhost -U fryt -d vector_db -c "
CREATE TABLE test_backup (id serial PRIMARY KEY, data text);
INSERT INTO test_backup (data) VALUES ('test_' || generate_series(1,100));
"

# 2. Run backup
./backup_postgres.sh

# 3. Drop test table
psql -h localhost -U fryt -d vector_db -c "DROP TABLE test_backup;"

# 4. Restore backup
./backup_postgres.sh --restore <backup-file>

# 5. Verify test data
psql -h localhost -U fryt -d vector_db -c "SELECT count(*) FROM test_backup;"
# Should return 100
```

## Support

For issues or questions:
1. Check logs: `tail -f logs/backup_*.log`
2. Run health check: `python utils/health_check.py`
3. Review [OPERATIONS.md](../../docs/OPERATIONS.md)
4. Check Grafana alerts: http://localhost:3000

## License

Part of the Local RAG Pipeline project.
