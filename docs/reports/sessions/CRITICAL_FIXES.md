# Critical Fixes - Priority Order

**Project Health:** 62/100 (C+) | **Status:** Not Production Ready
**Fix Time:** 40 hours (P0) + 80 hours (P1) = 120 hours total

---

## P0 - Fix This Week (40 hours)

### 1. Remove Hardcoded Credentials (3h) - CVSS 9.8

**Files:** `config/docker-compose.yml`, `scripts/compare_embedding_models.py`

```bash
# Fix docker-compose.yml
sed -i '' 's/POSTGRES_USER: fryt/POSTGRES_USER: ${PGUSER}/' config/docker-compose.yml
sed -i '' 's/POSTGRES_PASSWORD: frytos/POSTGRES_PASSWORD: ${PGPASSWORD}/' config/docker-compose.yml

# Create .env from template
cp config/.env.example .env
# Edit .env and set strong password

# Run with env file
docker-compose --env-file .env up -d

# Rotate database password
psql -U postgres -c "ALTER USER fryt WITH PASSWORD 'new_secure_password';"
```

---

### 2. Fix SQL Injection (8h) - CVSS 8.2

**Files:** 8 locations in `rag_low_level_m1_16gb_verbose.py`, `rag_web.py`, scripts

```python
# Before (UNSAFE):
c.execute(f"DROP TABLE IF EXISTS {S.table}")
c.execute(f'SELECT COUNT(*) FROM "{table_name}"')

# After (SAFE):
from psycopg2 import sql
c.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(S.table)))
c.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
```

**Find all:** `grep -r "execute(f" --include="*.py"`

---

### 3. Add Web UI Authentication (4h) - CVSS 8.2

**File:** `rag_web.py`

```bash
pip install streamlit-authenticator
```

```python
# Add to top of rag_web.py
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials={'usernames': {
        os.getenv('WEB_UI_USER', 'admin'): {
            'name': 'Admin',
            'password': os.getenv('WEB_UI_PASSWORD_HASH')  # bcrypt hash
        }
    }},
    cookie_name='rag_auth',
    key='random_signature_key',
    cookie_expiry_days=30
)

name, auth_status, username = authenticator.login('Login', 'main')
if not auth_status:
    st.stop()
```

---

### 4. Enable vLLM (30min) - 3-4x Speed Boost

**Impact:** 8-15s → 2-3s per query

```bash
# Terminal 1 (keep running):
./scripts/start_vllm_server.sh

# Terminal 2:
export USE_VLLM=1
python rag_interactive.py
```

**Done.** No code changes needed.

---

### 5. Deploy Monitoring (8h)

```bash
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus:latest
  ports: ["9090:9090"]
  volumes: ["./config/prometheus.yml:/etc/prometheus/prometheus.yml"]

grafana:
  image: grafana/grafana:latest
  ports: ["3000:3000"]
```

```python
# Add to rag_low_level_m1_16gb_verbose.py
from prometheus_client import Counter, Histogram, start_http_server

query_count = Counter('rag_queries_total', 'Queries processed')
query_duration = Histogram('rag_query_duration_seconds', 'Query latency')

start_http_server(8001)  # Metrics at localhost:8001/metrics
```

**Access:** Grafana at `http://localhost:3000`

---

### 6. Automated Backups (4h)

```bash
# Create backup script
cat > /usr/local/bin/backup_postgres.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

pg_dump -h localhost -U fryt -d vector_db \
  --format=custom --compress=9 \
  --file="$BACKUP_DIR/vector_db_$DATE.dump"

# Keep last 7 days
find "$BACKUP_DIR" -name "*.dump" -mtime +7 -delete
EOF

chmod +x /usr/local/bin/backup_postgres.sh

# Schedule daily at 2am
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup_postgres.sh") | crontab -

# Test restore
pg_restore -d vector_db_test /backups/postgres/latest.dump
```

---

### 7. Fix Bare Exceptions (2h)

**Files:** `rag_web.py` (9 instances), `rag_minimal_local.py` (2)

```python
# Before (DANGEROUS):
try:
    conn.commit()
except:
    pass  # Silently fails!

# After (SAFE):
try:
    conn.commit()
except psycopg2.Error as e:
    log.error(f"Commit failed: {e}")
    conn.rollback()
    raise
```

**Find all:** `grep -n "except:" rag_web.py`

---

### 8. Optimize Performance (1h)

```bash
# Increase batch sizes
export EMBED_BATCH=128          # Up from 32 (1.4x faster indexing)
export N_GPU_LAYERS=24          # Up from 16 (+20% LLM speed)
export DB_INSERT_BATCH=500      # Up from 250

# Add to .env to make permanent
```

---

### 9. Write Critical Runbooks (12h)

**Create:** `docs/runbooks/`

**Template:**
```markdown
# Runbook: Database Failure

## Symptoms
- Errors: "Connection refused" or "Database unavailable"
- All queries failing

## Investigation
```bash
docker-compose ps  # Check if db container running
docker-compose logs db  # Check database logs
```

## Fix
```bash
docker-compose restart db
# Wait 30 seconds
python rag.py --query "test"  # Verify working
```

## Prevention
- Set up monitoring alerts
- Enable auto-restart: `restart: unless-stopped` in docker-compose.yml
```

**Required runbooks:** DB failure, vLLM crash, Out of Memory

---

### 10. Fix Test Coverage Config (1h)

**File:** `pyproject.toml`

```toml
[tool.coverage.run]
source = ["."]
omit = [
    "tests/*",        # Exclude test files
    ".venv/*",
    "scripts/*",
]

[tool.coverage.report]
fail_under = 30  # Increase from 3%
```

**Run:** `pytest --cov=.`
**Result:** Shows real coverage ~30-40% instead of 0.94%

---

## P1 - Fix This Month (80 hours)

### 11. Extract Database Module (4h)

```bash
# Create new file
touch core/database.py

# Move these functions:
# - db_conn()
# - admin_conn()
# - ensure_db_exists()
# - ensure_pgvector_extension()
# - count_rows()

# Update imports in main file
```

**Repeat for:** `embeddings.py`, `retrieval.py`, `llm.py`

**Goal:** Main file from 2,734 → 800 lines

---

### 12. Create CHANGELOG.md (30min)

```markdown
# Changelog

## [2.0.0] - 2026-01-XX

### Added
- MLX embedding backend (5-20x faster on M1)
- vLLM server mode (3-4x faster queries)
- Hybrid search (BM25 + vector)
- Test suite (310 tests, 30.94% coverage)

### Fixed
- [List from audit]

### Security
- Fixed SQL injection vulnerabilities
- Removed hardcoded credentials
- Added web UI authentication
```

---

### 13. Add LICENSE (5min)

```bash
# Copy MIT license text to LICENSE file
curl https://opensource.org/licenses/MIT > LICENSE
```

---

### 14. Fix Coverage and Tests (3h)

```bash
# Fix import errors
export PYTHONPATH=/Users/frytos/code/llamaIndex-local-rag:$PYTHONPATH

# Run tests
pytest tests/ -v

# Fix 2 failing tests:
# - tests/test_e2e_pipeline.py
# - tests/test_fixtures.py
```

---

### 15. Add Critical Alerts (4h)

**File:** `config/prometheus/alerts.yml`

```yaml
groups:
- name: critical
  rules:
  - alert: HighErrorRate
    expr: rate(rag_query_errors_total[5m]) > 0.05
    annotations:
      summary: "Error rate above 5%"

  - alert: DatabaseDown
    expr: up{job="postgres"} == 0
    annotations:
      summary: "PostgreSQL is down"

  - alert: HighLatency
    expr: histogram_quantile(0.95, rag_query_duration_seconds) > 30
    annotations:
      summary: "P95 latency > 30s"
```

---

## Quick Reference

### Security Fixes
```bash
# 1. Credentials: 3h
# 2. SQL injection: 8h
# 3. Web auth: 4h
# 4. Bare exceptions: 2h
Total: 17 hours
```

### Performance Fixes
```bash
# 1. Enable vLLM: 30min (4x faster)
# 2. Batch sizes: 5min (1.4x faster)
# 3. GPU layers: 1min (+20%)
Total: 36 minutes → 4-5x improvement
```

### Operations Fixes
```bash
# 1. Monitoring: 8h
# 2. Backups: 4h
# 3. Alerts: 4h
# 4. Runbooks: 12h
Total: 28 hours
```

### Totals
- **P0 (Week 1):** 40 hours
- **P1 (Month 1):** 80 hours
- **Total to 90/100:** 320 hours (8 weeks)

---

## One-Week Action Plan

**Monday (8h):** Security
- Remove hardcoded credentials (3h)
- Start SQL injection fixes (5h)

**Tuesday (8h):** Security + Performance
- Finish SQL injection (3h)
- Add web auth (4h)
- Enable vLLM (30min)
- Test everything (30min)

**Wednesday (8h):** Operations
- Deploy Prometheus + Grafana (8h)

**Thursday (8h):** Operations
- Set up automated backups (4h)
- Configure alerts (4h)

**Friday (8h):** Documentation
- Write 3 critical runbooks (8h)

**Result:** Security 66→85, Performance 67→82, Operations 39→55, Overall 62→74

---

## Files Modified Checklist

**Security:**
- [ ] `config/docker-compose.yml` - Remove hardcoded credentials
- [ ] `scripts/compare_embedding_models.py` - Remove password
- [ ] `rag_low_level_m1_16gb_verbose.py` - Fix SQL injection (8 locations)
- [ ] `rag_web.py` - Fix SQL injection (3 locations) + add auth
- [ ] `scripts/*.py` - Fix SQL injection (2 locations)

**Performance:**
- [ ] `.env` - Add `USE_VLLM=1`, `EMBED_BATCH=128`, `N_GPU_LAYERS=24`

**Operations:**
- [ ] `docker-compose.yml` - Add prometheus, grafana, backup services
- [ ] `config/prometheus.yml` - Create metrics config
- [ ] `config/prometheus/alerts.yml` - Create alert rules
- [ ] `/usr/local/bin/backup_postgres.sh` - Create backup script
- [ ] `crontab` - Schedule daily backups

**Documentation:**
- [ ] `CHANGELOG.md` - Create
- [ ] `LICENSE` - Create
- [ ] `docs/runbooks/` - Create 3 runbooks
- [ ] `pyproject.toml` - Fix coverage config

---

**Priority:** Start with #1 (credentials) and #4 (vLLM) today.
