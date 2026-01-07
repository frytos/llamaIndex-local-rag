# Runbooks

Operational runbooks for troubleshooting and recovering from common failure scenarios in the Local RAG Pipeline.

## Available Runbooks

### [Database Failure](database-failure.md)
**Severity**: P0 (Critical)
**Time to Resolve**: 5-15 minutes

Covers PostgreSQL connection failures, corruption, and recovery:
- Container not running
- Database doesn't exist
- pgvector extension missing
- Transaction aborted errors
- Connection pool exhausted
- Slow query performance
- Complete database reset
- Backup and restore procedures

**When to Use**: Cannot connect to database, queries failing, data corruption

---

### [vLLM Server Crash](vllm-crash.md)
**Severity**: P1 (High - Fallback available)
**Time to Resolve**: 5-10 minutes

Covers vLLM server crashes and GPU issues:
- Server not running
- GPU out of memory
- Server hung/not responding
- Model download failures
- Slow generation speed
- Port conflicts
- CUDA driver version mismatch
- Fallback to llama.cpp

**When to Use**: vLLM crashes, GPU errors, slow inference, connection failures

---

### [Out of Memory](out-of-memory.md)
**Severity**: P0-P1 (Critical to High)
**Time to Resolve**: 2-10 minutes

Covers memory exhaustion across all components:
- Embedding model OOM
- LLM inference OOM
- vLLM GPU OOM
- Database OOM
- Document loading OOM
- Disk space exhaustion
- Memory budget guidelines
- Emergency memory clearing

**When to Use**: System crashes, swap at 100%, GPU OOM, disk full

---

## Quick Reference

### Most Common Issues

| Symptom | Runbook | Quick Fix |
|---------|---------|-----------|
| Cannot connect to database | [Database Failure](database-failure.md#issue-1-container-not-running) | `docker compose up -d` |
| CUDA out of memory | [Out of Memory](out-of-memory.md#3-vllm-gpu-oom) | Reduce `--gpu-memory-utilization` |
| vLLM not responding | [vLLM Crash](vllm-crash.md#issue-3-vllm-hung--not-responding) | `pkill -9 -f vllm && ./scripts/start_vllm_server.sh` |
| System RAM exhausted | [Out of Memory](out-of-memory.md#1-embedding-model-oom) | Reduce `EMBED_BATCH` and `N_BATCH` |
| Disk space full | [Out of Memory](out-of-memory.md#6-disk-space-exhaustion) | Clean model cache and logs |
| Query very slow | [Database Failure](database-failure.md#issue-7-slow-query-performance) | Add HNSW index |

### Emergency Procedures

**System Completely Unresponsive:**
```bash
# Kill all RAG processes
pkill -f python
pkill -f vllm
pkill -f streamlit

# Clear caches
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Restart database
docker compose restart
```

**GPU Locked Up:**
```bash
# Reset GPU
nvidia-smi --gpu-reset

# Or restart GPU driver
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

**Database Corrupted:**
```bash
# Nuclear option - destroys all data
docker compose down
docker volume rm llamaindex-local-rag_postgres_data
docker compose up -d
RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
```

## Health Check Commands

Run these to quickly diagnose system state:

```bash
# Database health
psql -h localhost -U $PGUSER -d vector_db -c "SELECT 1;"

# vLLM health
curl http://localhost:8000/health

# System resources
free -h && df -h && nvidia-smi

# Process status
ps aux | grep -E "(python|vllm|postgres)" | grep -v grep
```

## Monitoring Setup

### Recommended Monitoring

1. **System Resources** (every 5 minutes):
   ```bash
   watch -n 300 'free -h; df -h'
   ```

2. **GPU Utilization** (every 5 minutes):
   ```bash
   watch -n 300 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv'
   ```

3. **Database Connections** (every 10 minutes):
   ```bash
   watch -n 600 'psql -h localhost -U $PGUSER -d vector_db -c \
       "SELECT count(*) FROM pg_stat_activity;"'
   ```

### Automated Alerting

Set up cron jobs to alert on issues:

```bash
# Add to crontab: crontab -e
*/15 * * * * /path/to/health-check.sh
0 2 * * * /path/to/cleanup-old-logs.sh
0 0 * * 0 /path/to/backup-database.sh
```

## Escalation Matrix

| Severity | Resolution Time | Escalation After | Action |
|----------|----------------|------------------|--------|
| P0 (Critical) | 15 minutes | 30 minutes | Page on-call, open critical issue |
| P1 (High) | 30 minutes | 1 hour | Notify team, open high-priority issue |
| P2 (Medium) | 2 hours | 4 hours | Create issue, investigate in next sprint |
| P3 (Low) | Best effort | N/A | Create issue for backlog |

## Contributing

When creating new runbooks:

1. **Follow the template**:
   - Severity and time to resolve
   - Overview and symptoms
   - Quick diagnosis commands
   - Issue-specific solutions
   - Prevention and monitoring

2. **Include working commands**: Test all commands before documenting

3. **Add to this README**: Update the quick reference table

4. **Cross-reference**: Link to related runbooks

## Feedback

Found an issue or have a suggestion? Please:
1. Test the runbook and document what didn't work
2. Open a GitHub issue with details
3. Submit a PR with improvements

## Related Documentation

- [Troubleshooting Guide](../START_HERE.md#troubleshooting)
- [Performance Tuning](../RAG_OPTIMIZATION_GUIDE.md)
- [Environment Variables](../ENVIRONMENT_VARIABLES.md)
- [Architecture Overview](../ARCHITECTURE.md)

## Change Log

- 2026-01-07: Initial runbooks created (database-failure, vllm-crash, out-of-memory)
