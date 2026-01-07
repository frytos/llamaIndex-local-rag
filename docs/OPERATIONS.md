# Operations Guide - Local RAG Pipeline

**Version**: 1.0.0 | **Last Updated**: January 2026

## Overview

This guide covers operational procedures for running, monitoring, and maintaining the Local RAG Pipeline in production.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Monitoring](#monitoring)
3. [Backup and Recovery](#backup-and-recovery)
4. [Health Checks](#health-checks)
5. [Alerting](#alerting)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)
8. [Incident Response](#incident-response)

---

## Quick Start

### Starting the Stack

```bash
# Start core services (database + monitoring)
cd config
docker-compose up -d

# Verify services
docker-compose ps

# Check logs
docker-compose logs -f
```

### Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics database |
| Alertmanager | http://localhost:9093 | Alert management |
| PostgreSQL | localhost:5432 | Vector database |
| cAdvisor | http://localhost:8080 | Container metrics |

### Starting with Backup Service

```bash
# Start with automated backups
docker-compose --profile backup up -d
```

### Starting with Application

```bash
# Start full stack including RAG app
docker-compose --profile app up -d
```

---

## Monitoring

### Grafana Dashboards

**Access**: http://localhost:3000 (default: admin/admin)

**Available Dashboards**:
- **RAG Pipeline Overview**: Query metrics, latency, success rates
- **Database Performance**: Connection pool, query time, size
- **System Resources**: CPU, memory, disk usage
- **Container Health**: Docker metrics, resource limits

**Key Metrics**:
- Query success rate (target: >99%)
- P95 query latency (target: <10s)
- Database connections (warning: >80%)
- Backup status (critical if failed)

### Prometheus Queries

**Access**: http://localhost:9090

Useful queries:
```promql
# Query success rate
(sum(rate(rag_query_success_total[5m])) / sum(rate(rag_query_total[5m]))) * 100

# P95 latency
histogram_quantile(0.95, rate(rag_query_duration_seconds_bucket[5m]))

# Error rate
rate(rag_query_errors_total[5m])

# Cache hit rate
(sum(rate(rag_cache_hits_total[5m])) / sum(rate(rag_cache_requests_total[5m]))) * 100

# Database size growth
rate(pg_database_size_bytes{datname="vector_db"}[1h])
```

### Logs

```bash
# Application logs
tail -f logs/*.log

# Docker container logs
docker-compose logs -f db
docker-compose logs -f grafana
docker-compose logs -f prometheus

# Backup logs
tail -f logs/backup_*.log
```

---

## Backup and Recovery

### Automated Backups

**Schedule**: Daily at 2 AM (configurable)

**Setup Cron Job**:
```bash
# Setup daily backup at 2 AM
./scripts/backup/setup_cron.sh

# Custom schedule (every 6 hours)
./scripts/backup/setup_cron.sh --time "0 */6 * * *"

# Check status
./scripts/backup/setup_cron.sh --status

# Remove cron job
./scripts/backup/setup_cron.sh --remove
```

**Docker-based Backup** (alternative):
```bash
# Start backup service
docker-compose --profile backup up -d
```

### Manual Backup

```bash
# Run backup immediately
./scripts/backup/backup_postgres.sh

# Backup with 14-day retention
BACKUP_RETENTION_DAYS=14 ./scripts/backup/backup_postgres.sh

# List backups
./scripts/backup/backup_postgres.sh --list
```

### Verify Backup

```bash
# Verify most recent backup
./scripts/backup/verify_backup.sh

# Verify specific backup
./scripts/backup/verify_backup.sh backups/postgres/vector_db_20260107_120000.sql.gz

# Test restore to temporary database
./scripts/backup/verify_backup.sh --test-restore
```

### Restore from Backup

**WARNING**: This will overwrite the current database!

```bash
# Interactive restore
./scripts/backup/backup_postgres.sh --restore backups/postgres/vector_db_20260107_120000.sql.gz

# Will prompt for confirmation
```

**Manual Restore**:
```bash
# Drop and recreate database
psql -h localhost -U fryt -d postgres -c "DROP DATABASE vector_db;"
psql -h localhost -U fryt -d postgres -c "CREATE DATABASE vector_db;"

# Restore backup
zcat backups/postgres/vector_db_20260107_120000.sql.gz | psql -h localhost -U fryt -d vector_db

# Re-enable pgvector
psql -h localhost -U fryt -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Backup Storage

**Location**: `./backups/postgres/`

**Retention**: 7 days (configurable via `BACKUP_RETENTION_DAYS`)

**Disk Space Monitoring**:
```bash
# Check backup disk usage
du -sh backups/

# List backups by size
ls -lh backups/postgres/
```

---

## Health Checks

### Comprehensive Health Check

```bash
# Run all health checks
python utils/health_check.py

# Check readiness (ready to serve requests)
python utils/health_check.py readiness

# Check liveness (application alive)
python utils/health_check.py liveness
```

**Example Output**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-07T12:00:00",
  "checks": [
    {
      "component": "database",
      "status": "healthy",
      "message": "Database connection successful",
      "latency_ms": 15.2
    },
    {
      "component": "system_resources",
      "status": "healthy",
      "message": "System resources OK"
    }
  ],
  "summary": {
    "healthy": 5,
    "degraded": 0,
    "unhealthy": 0,
    "total": 5
  }
}
```

### Quick Health Checks

```bash
# Database connectivity
psql -h localhost -U fryt -d vector_db -c "SELECT 1"

# Check Prometheus
curl -s http://localhost:9090/-/healthy

# Check Grafana
curl -s http://localhost:3000/api/health

# Check all containers
docker-compose ps
```

### Health Check Integration

**Kubernetes/Docker Compose**:
```yaml
healthcheck:
  test: ["CMD", "python", "utils/health_check.py", "readiness"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

---

## Alerting

### Alert Rules

Configured in `/Users/frytos/code/llamaIndex-local-rag/config/monitoring/alerts.yml`

**Critical Alerts**:
- Database down (fires after 1 minute)
- Backup failed
- High memory usage (>90%)
- Disk space low (<15%)

**Warning Alerts**:
- High latency (P95 >10s)
- High error rate (>10%)
- Database connections high (>80%)
- Backup stale (>48 hours)

### Viewing Alerts

**Prometheus**: http://localhost:9090/alerts

**Alertmanager**: http://localhost:9093

### Silence Alerts

```bash
# Silence specific alert for 1 hour
amtool silence add alertname=PostgreSQLDown --duration=1h --comment="Maintenance"

# List active silences
amtool silence query

# Expire silence
amtool silence expire <silence-id>
```

### Configure Notifications

Edit `/Users/frytos/code/llamaIndex-local-rag/config/monitoring/alertmanager.yml`:

**Email**:
```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@example.com'
  smtp_auth_username: 'alerts@example.com'
  smtp_auth_password: 'your-app-password'
```

**Slack**:
```yaml
receivers:
  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
```

**Reload configuration**:
```bash
docker-compose restart alertmanager
```

---

## Troubleshooting

### Database Issues

**Database won't start**:
```bash
# Check logs
docker-compose logs db

# Verify disk space
df -h

# Reset database (WARNING: data loss)
docker-compose down -v
docker-compose up -d db
```

**High connection usage**:
```bash
# List active connections
psql -h localhost -U fryt -d vector_db -c "
SELECT pid, usename, application_name, client_addr, state, query_start
FROM pg_stat_activity
WHERE datname='vector_db' AND state='active';
"

# Terminate specific connection
psql -h localhost -U fryt -d vector_db -c "SELECT pg_terminate_backend(<pid>);"
```

**Slow queries**:
```bash
# Check long-running queries
psql -h localhost -U fryt -d vector_db -c "
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state='active' AND now() - query_start > interval '5 minutes';
"
```

### High Latency

1. **Check database load**:
   ```bash
   psql -h localhost -U fryt -d vector_db -c "SELECT * FROM pg_stat_activity;"
   ```

2. **Check system resources**:
   ```bash
   # CPU usage
   top -n 1

   # Memory usage
   free -h

   # Disk I/O
   iostat -x 1 5
   ```

3. **Review chunk configuration**:
   - Reduce `CHUNK_SIZE` or `TOP_K`
   - Check index configuration

### High Error Rate

1. **Check application logs**:
   ```bash
   tail -f logs/*.log | grep ERROR
   ```

2. **Check database errors**:
   ```bash
   docker-compose logs db | grep ERROR
   ```

3. **Verify models loaded**:
   ```bash
   python utils/health_check.py
   ```

### Backup Failures

1. **Check backup logs**:
   ```bash
   tail -f logs/backup_*.log
   ```

2. **Verify disk space**:
   ```bash
   df -h backups/
   ```

3. **Test database connection**:
   ```bash
   psql -h localhost -U fryt -d vector_db -c "SELECT 1"
   ```

4. **Run manual backup**:
   ```bash
   ./scripts/backup/backup_postgres.sh
   ```

### Container Issues

```bash
# Restart specific service
docker-compose restart db

# Restart all services
docker-compose restart

# View resource usage
docker stats

# Check container health
docker-compose ps
docker inspect rag_postgres | grep Health -A 10
```

---

## Maintenance

### Daily Tasks

- [x] Check Grafana dashboards for anomalies
- [x] Verify backup completion
- [x] Review error logs

### Weekly Tasks

- [x] Review alert history
- [x] Check disk space usage
- [x] Verify backup restoration (test restore)
- [x] Review query performance trends

### Monthly Tasks

- [x] Update dependencies
- [x] Review and optimize chunk configurations
- [x] Clean old logs and backups
- [x] Review system resource trends
- [x] Disaster recovery drill

### Update Docker Images

```bash
# Pull latest images
docker-compose pull

# Restart with new images
docker-compose up -d

# Remove old images
docker image prune -a
```

### Clean Up Logs

```bash
# Remove logs older than 30 days
find logs/ -name "*.log" -mtime +30 -delete

# Rotate large log files
for log in logs/*.log; do
  if [ $(stat -f%z "$log" 2>/dev/null || stat -c%s "$log") -gt 104857600 ]; then
    gzip "$log"
    mv "$log.gz" "logs/archive/"
  fi
done
```

### Optimize Database

```bash
# Vacuum and analyze
psql -h localhost -U fryt -d vector_db -c "VACUUM ANALYZE;"

# Reindex (if performance degrades)
psql -h localhost -U fryt -d vector_db -c "REINDEX DATABASE vector_db;"

# Update statistics
psql -h localhost -U fryt -d vector_db -c "ANALYZE;"
```

---

## Incident Response

### Severity Levels

**P0 - Critical**: Complete service outage
- Database down
- All queries failing

**P1 - High**: Significant degradation
- High error rate (>25%)
- Extreme latency (>30s)

**P2 - Medium**: Partial degradation
- Elevated error rate (10-25%)
- High latency (10-30s)

**P3 - Low**: Minor issues
- Backup failures
- Resource warnings

### Response Procedures

**P0 - Critical Incident**:

1. **Acknowledge**: Silence non-critical alerts
2. **Assess**: Check Grafana dashboards and logs
3. **Mitigate**:
   ```bash
   # Restart services
   docker-compose restart

   # If database corrupted, restore from backup
   ./scripts/backup/backup_postgres.sh --restore <backup-file>
   ```
4. **Communicate**: Notify stakeholders
5. **Document**: Record timeline and actions

**P1 - High Severity**:

1. Check system resources
2. Review recent changes
3. Scale resources if needed
4. Investigate root cause

**Post-Incident**:

1. Write incident report
2. Conduct blameless postmortem
3. Implement preventive measures
4. Update runbooks

### Emergency Contacts

Document your team contacts:

```
On-call Engineer: [Name] <email@example.com>
Database Admin: [Name] <email@example.com>
Infrastructure Lead: [Name] <email@example.com>
```

### Rollback Procedures

```bash
# Restore previous database state
./scripts/backup/backup_postgres.sh --restore <previous-backup>

# Revert Docker images
docker-compose down
git checkout <previous-version>
docker-compose up -d
```

---

## Metrics Reference

### Query Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rag_query_total` | counter | Total queries |
| `rag_query_success_total` | counter | Successful queries |
| `rag_query_errors_total` | counter | Failed queries |
| `rag_query_duration_seconds` | histogram | Query latency |

### Retrieval Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rag_retrieval_score_avg` | gauge | Average similarity score |
| `rag_retrieval_documents_total` | counter | Documents retrieved |

### Database Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `pg_up` | gauge | Database status (1=up) |
| `pg_database_size_bytes` | gauge | Database size |
| `pg_stat_database_numbackends` | gauge | Active connections |

### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `node_cpu_seconds_total` | counter | CPU usage |
| `node_memory_MemAvailable_bytes` | gauge | Available memory |
| `node_filesystem_avail_bytes` | gauge | Available disk space |

---

## Additional Resources

- [ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md) - Configuration reference
- [PERFORMANCE_QUICK_START.md](./PERFORMANCE_QUICK_START.md) - Performance tuning
- [VLLM_SERVER_GUIDE.md](./VLLM_SERVER_GUIDE.md) - vLLM server setup
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

## Appendix: Service Dependencies

```
RAG Application
    ↓
PostgreSQL (pgvector)
    ↓
Postgres Exporter → Prometheus → Grafana
                        ↓
                   Alertmanager
```

**Start Order**:
1. Database (PostgreSQL)
2. Monitoring (Prometheus, Node Exporter)
3. Visualization (Grafana)
4. Alerting (Alertmanager)
5. Application (RAG)
6. Backup (scheduled)
