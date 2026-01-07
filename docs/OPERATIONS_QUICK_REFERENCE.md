# Operations Quick Reference

One-page cheat sheet for RAG pipeline operations.

## Start/Stop Services

```bash
# Start monitoring stack
./scripts/start_monitoring.sh

# Start with application
./scripts/start_monitoring.sh --full

# Start with backups
./scripts/start_monitoring.sh --backup

# Stop all services
./scripts/start_monitoring.sh --stop

# Check status
./scripts/start_monitoring.sh --status
```

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| Alertmanager | http://localhost:9093 | - |
| PostgreSQL | localhost:5432 | See .env |
| cAdvisor | http://localhost:8080 | - |

## Health Checks

```bash
# Full health check
python utils/health_check.py

# Readiness check
python utils/health_check.py readiness

# Liveness check
python utils/health_check.py liveness

# Quick database check
psql -h localhost -U fryt -d vector_db -c "SELECT 1"
```

## Backup Operations

```bash
# Run backup now
./scripts/backup/backup_postgres.sh

# Setup automated backups (2 AM daily)
./scripts/backup/setup_cron.sh

# Verify backup
./scripts/backup/verify_backup.sh

# List backups
./scripts/backup/backup_postgres.sh --list

# Restore backup
./scripts/backup/backup_postgres.sh --restore <backup-file>
```

## Logs

```bash
# Application logs
tail -f logs/*.log

# Docker logs
docker-compose logs -f <service-name>

# Backup logs
tail -f logs/backup_*.log

# All logs
docker-compose logs -f
```

## Common Issues

### Database Down

```bash
# Restart database
docker-compose restart db

# Check logs
docker-compose logs db

# Verify connection
psql -h localhost -U fryt -d vector_db -c "SELECT 1"
```

### High Latency

```bash
# Check active queries
psql -h localhost -U fryt -d vector_db -c "
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state='active' AND now() - query_start > interval '5 minutes';
"

# Check system resources
docker stats
```

### Backup Failed

```bash
# Check logs
tail -f logs/backup_*.log

# Test connection
psql -h localhost -U fryt -d vector_db -c "SELECT 1"

# Run manual backup
./scripts/backup/backup_postgres.sh
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Clean old backups
./scripts/backup/backup_postgres.sh --retention 3
```

## Monitoring Queries

```promql
# Query success rate
(sum(rate(rag_query_success_total[5m])) / sum(rate(rag_query_total[5m]))) * 100

# P95 latency
histogram_quantile(0.95, rate(rag_query_duration_seconds_bucket[5m]))

# Error rate
rate(rag_query_errors_total[5m])

# Database connections
pg_stat_database_numbackends{datname="vector_db"}

# Memory usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Disk usage
(1 - (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"})) * 100
```

## Alert Actions

| Alert | Severity | Action |
|-------|----------|--------|
| PostgreSQLDown | Critical | Restart: `docker-compose restart db` |
| BackupFailed | Critical | Check logs, run manual backup |
| HighMemoryUsage | Critical | Free memory, check processes |
| DiskSpaceLow | Critical | Clean logs/backups, add storage |
| RAGHighLatency | Warning | Check DB load, review queries |
| RAGHighErrorRate | Warning | Check application logs |

## Maintenance Schedule

**Daily**:
- Check Grafana dashboards
- Review error logs
- Verify backup completion

**Weekly**:
- Test backup restore
- Review query performance
- Check disk space

**Monthly**:
- Update dependencies
- Review alert history
- Disaster recovery drill

## Emergency Contacts

```
On-call: [Your contact]
Database: [DB admin contact]
Infrastructure: [Infra contact]
```

## Quick Commands

```bash
# Restart everything
docker-compose restart

# View resource usage
docker stats

# Check container health
docker-compose ps

# Cleanup old data
docker system prune -a

# Database vacuum
psql -h localhost -U fryt -d vector_db -c "VACUUM ANALYZE;"

# Export metrics
python utils/metrics.py
```

## Environment Variables

```bash
# Database
PGHOST=localhost
PGPORT=5432
PGUSER=fryt
PGPASSWORD=frytos
DB_NAME=vector_db

# Backup
BACKUP_RETENTION_DAYS=7
METRICS_DIR=metrics
```

## Testing

```bash
# Test operations setup
./scripts/test_operations_setup.sh

# Test backup workflow
./scripts/backup/verify_backup.sh --test-restore

# Test health checks
python utils/health_check.py
```

## Documentation

- [OPERATIONS.md](./OPERATIONS.md) - Full operations guide
- [Backup README](../scripts/backup/README.md) - Backup documentation
- [ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md) - Configuration
- [PERFORMANCE_QUICK_START.md](./PERFORMANCE_QUICK_START.md) - Performance tuning

## Support

1. Check Grafana dashboards
2. Review logs
3. Run health checks
4. Check documentation
5. Contact on-call engineer
