# Quick Start Fixes - 30 Minutes to 4x Performance

## The 30-Minute Win

### 1. Enable vLLM (30min) → 4x Faster Queries

```bash
# Terminal 1 (leave running):
pip install vllm
./scripts/start_vllm_server.sh

# Terminal 2:
export USE_VLLM=1
python rag_interactive.py
```

**Result:** 12s → 3s per query

---

## The 4-Hour Critical Security Fix

### 2. Fix Hardcoded Passwords (3h)

```bash
# Update docker-compose.yml
POSTGRES_USER: ${PGUSER}
POSTGRES_PASSWORD: ${PGPASSWORD}

# Create .env
echo "PGUSER=fryt" > .env
echo "PGPASSWORD=$(openssl rand -base64 32)" >> .env
chmod 600 .env

# Restart
docker-compose --env-file .env up -d
```

### 3. Add Automated Backups (1h)

```bash
cat > backup.sh << 'EOF'
#!/bin/bash
pg_dump -U fryt vector_db > backup_$(date +%Y%m%d).sql
find . -name "backup_*.sql" -mtime +7 -delete
EOF

chmod +x backup.sh
(crontab -l; echo "0 2 * * * /path/to/backup.sh") | crontab -
```

---

## The 8-Hour Full Security + Performance Fix

**Add:**
4. Fix SQL Injection (4h) - Use `psycopg2.sql.Identifier()` everywhere
5. Optimize Settings (5min) - `EMBED_BATCH=128 N_GPU_LAYERS=24`

**Result:**
- Security: 66 → 85 (+19 points)
- Performance: 67 → 82 (+15 points)
- Queries: 12s → 2s (6x faster)

---

## The 40-Hour Complete P0 Fix

**Everything above plus:**
6. Web UI auth (4h)
7. Monitoring (8h) - Prometheus + Grafana
8. Alerts (4h)
9. Runbooks (12h)

**Result:**
- Overall health: 62 → 74 (+12 points)
- Production-capable for personal use

---

## Command Cheatsheet

```bash
# Performance boost (instant)
USE_VLLM=1 EMBED_BATCH=128 N_GPU_LAYERS=24 python rag_interactive.py

# Check security
pip-audit  # Find vulnerabilities
grep -r "POSTGRES_PASSWORD" .  # Find hardcoded credentials
grep -r "execute(f" --include="*.py"  # Find SQL injection

# Monitor system
docker-compose ps  # Check services
tail -f query_logs/*.json  # Watch queries
htop  # Monitor resources

# Backup/restore
pg_dump -U fryt vector_db > backup.sql  # Backup
pg_restore -d vector_db backup.sql  # Restore

# Test
pytest tests/ -v  # Run all tests
pytest --cov=. --cov-report=html  # Coverage report
```

---

## Priority Order

1. **Now (30min):** Enable vLLM → Immediate 4x speedup
2. **Today (4h):** Fix credentials + backups → Critical security
3. **This Week (8h):** Full security fixes → Production-safe
4. **This Month (40h):** Complete P0 → Ready for sharing

Start with #1 right now. Takes 30 minutes, makes everything 4x faster.
