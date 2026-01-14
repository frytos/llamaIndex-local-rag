# Risk Priority Matrix - Comprehensive Audit

**Project**: LlamaIndex Local RAG Pipeline
**Date**: January 9, 2026
**Total Issues Identified**: 47 items (10 P0, 15 P1, 14 P2, 8 P3)

---

## Summary

This document provides a **prioritized action plan** for addressing all findings from the comprehensive 360° audit. Issues are categorized by priority (P0-P3) with clear ownership, effort estimates, and expected impact.

**Critical Path**: P0 issues must be resolved before production deployment

---

## P0 - Critical (Fix Immediately) - 10 Items

**Total Effort**: 126 hours (3.2 weeks)
**Risk Reduction**: 85%
**Production Blocker**: YES

### Security (3 items - 7 hours)

| ID | Issue | Location | Impact | Effort | Owner | Deadline |
|----|-------|----------|--------|--------|-------|----------|
| **S1** | No authentication/authorization | `rag_web.py`, `rag_web_enhanced.py` | Data breach, unauthorized access | 4h | Security Team | Week 1 |
| **S2** | Hardcoded database password | `cleanup_empty_tables.sh:5` | Credential exposure | 15min | Dev Team | **TODAY** |
| **S3** | Docker secrets in plaintext | `docker-compose.yml` | Security violation | 2h | DevOps | Week 1 |

**Fix Instructions**:

**S1 - Add Authentication**:
```python
# Install required packages
pip install python-jose[cryptography] passlib[bcrypt]

# Implement JWT authentication
# See SECURITY_QUICK_FIXES.md for full code
```

**S2 - Remove Hardcoded Password** (**DO THIS NOW**):
```bash
# 1. Edit cleanup_empty_tables.sh
# Remove: PGPASSWORD=frytos
# Add: export PGPASSWORD="${PGPASSWORD:-}"

# 2. Rotate password immediately
psql -U frytos -c "ALTER USER frytos WITH PASSWORD 'new_secure_password';"

# 3. Add to .env
echo "PGPASSWORD=new_secure_password" >> .env
```

**S3 - Use Docker Secrets**:
```yaml
# docker-compose.yml
secrets:
  db_password:
    file: ./secrets/db_password.txt
services:
  postgres:
    secrets:
      - db_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
```

---

### Architecture (1 item - 96 hours)

| ID | Issue | Location | Impact | Effort | Owner | Deadline |
|----|-------|----------|--------|--------|-------|----------|
| **A1** | Monolithic core blocks velocity | `rag_low_level_m1_16gb_verbose.py` (3,092 lines) | 40% velocity decline, hard to test | 96h (12 days) | Tech Lead | Month 2 |

**Refactoring Plan**:
1. **Phase 1** (40h): Extract services
   - `document_loader.py` (500 lines)
   - `chunking_service.py` (300 lines)
   - `embedding_service.py` (400 lines)
   - `vector_store_manager.py` (600 lines)

2. **Phase 2** (32h): Extract retrieval & generation
   - `retriever.py` (500 lines)
   - `generator.py` (400 lines)

3. **Phase 3** (24h): Create orchestrator
   - `pipeline_orchestrator.py` (800 lines)
   - Dependency injection setup

**Benefits**:
- Restore velocity (+40%)
- Enable parallel development
- Reduce onboarding time (3 weeks → 5 days)
- Increase testability (30% → 65% coverage)

---

### Operations (4 items - 84 hours)

| ID | Issue | Location | Impact | Effort | Owner | Deadline |
|----|-------|----------|--------|--------|-------|----------|
| **O1** | No SLI/SLO/SLA definitions | N/A | Cannot measure reliability | 16h | SRE Team | Week 2 |
| **O2** | No automated deployment | N/A | Manual deployment risk, no rollback | 24h | DevOps | Week 3 |
| **O3** | No Infrastructure as Code | N/A | Cannot reproduce environments | 40h | DevOps | Month 2 |
| **O4** | Application metrics not exported | `rag_web.py`, core modules | Blind to app health | 4h | Dev Team | Week 2 |

**Fix Instructions**:

**O1 - Define SLOs**:
```yaml
# slo_definitions.yml
slos:
  availability:
    target: 99.5%  # 3.6 hours downtime/month
    measurement: uptime_percentage

  latency:
    target: 95% of requests < 3 seconds
    measurement: response_time_p95

  error_rate:
    target: < 0.5% of requests fail
    measurement: error_rate_percentage

error_budget:
  monthly_downtime_allowed: 216 minutes
  current_burn_rate: track_hourly
```

**O2 - Set Up CI/CD**:
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    tags:
      - 'v*'
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy via SSH
        run: |
          ssh deploy@production "cd /app && git pull && docker-compose up -d"
      - name: Health Check
        run: curl -f https://production.example.com/health || exit 1
```

**O4 - Export Metrics**:
```python
# Add to rag_web.py
from prometheus_client import Counter, Histogram, generate_latest

query_count = Counter('rag_queries_total', 'Total queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

### Project Risk (2 items - Ongoing)

| ID | Risk | Impact | Mitigation | Effort | Owner | Deadline |
|----|------|--------|------------|--------|-------|----------|
| **R1** | Bus factor of 1 | Project halt if developer unavailable | Hire backup dev + knowledge transfer | 160h + Hiring | PM | Month 2 |
| **R2** | Test coverage 30% (target: 80%) | High bug risk, dangerous refactoring | Add 200+ tests | 40h | QA Team | Month 2 |

**R1 - Bus Factor Mitigation**:
1. **Week 1**: Create architecture walkthrough video (4h)
2. **Week 2**: Document critical procedures (8h)
3. **Week 3**: Create ADRs for key decisions (4h)
4. **Month 2**: Hire and onboard backup developer (160h)
5. **Month 3**: Establish pair programming (ongoing)

**R2 - Increase Test Coverage**:
```python
# Priority test additions (40 hours total):
# 1. Core pipeline tests (12h) - 60 tests
#    - Document loading edge cases
#    - Chunking boundary conditions
#    - Embedding error handling

# 2. Database integration tests (10h) - 40 tests
#    - Connection failure scenarios
#    - Query error handling
#    - Migration testing

# 3. Web UI tests (8h) - 30 tests
#    - User authentication flows
#    - File upload validation
#    - Query submission

# 4. End-to-end tests (10h) - 20 tests
#    - Full pipeline workflows
#    - Multi-user scenarios
#    - Performance regression
```

**Target Progress**:
- Month 1: 30% → 50% (+20%)
- Month 2: 50% → 70% (+20%)
- Month 3: 70% → 80% (+10%)

---

## P1 - High Priority (Fix This Sprint) - 15 Items

**Total Effort**: 98 hours (2.5 weeks)
**Risk Reduction**: Additional 10%
**Production Impact**: Medium

### Security (4 items - 18 hours)

| ID | Issue | Severity | Effort | Deadline |
|----|-------|----------|--------|----------|
| **S4** | Vulnerable dependencies (10 CVEs) | High | 2h | Week 2 |
| **S5** | No HTTPS/TLS encryption | High | 4h | Week 3 |
| **S6** | No rate limiting (DoS risk) | Medium | 4h | Week 3 |
| **S7** | Missing security logging | Medium | 8h | Week 4 |

**Fix Commands**:

**S4 - Update Dependencies**:
```bash
# Update vulnerable packages
pip install --upgrade aiohttp urllib3 marshmallow

# Run security scan
pip-audit

# Update requirements.txt
pip freeze > requirements.txt
```

**S5 - Enable HTTPS**:
```python
# Use Caddy as reverse proxy
# Caddyfile
rag.example.com {
    reverse_proxy localhost:8501
    tls internal  # Auto HTTPS
}
```

**S6 - Add Rate Limiting**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/query")
@limiter.limit("10/minute")  # 10 requests per minute
def query_endpoint(question: str):
    # Your code here
```

---

### Architecture (3 items - 16 hours)

| ID | Issue | Location | Effort | Deadline |
|----|-------|----------|--------|----------|
| **A2** | Code duplication (3-4 web UI files) | `rag_web*.py` (40-60% overlap) | 8h | Week 3 |
| **A3** | Database connection duplication | 3 files, ~150 lines | 4h | Week 2 |
| **A4** | Documentation sprawl (72 MD files) | Root directory | 4h | Week 2 |

**Fix Plan**:

**A2 - Consolidate Web UIs**:
```bash
# Decision: Keep rag_web_enhanced.py as canonical
# It has the most features (2,975 lines)

# Archive others
mkdir -p archive/web_ui_variants
mv rag_web.py archive/web_ui_variants/
mv rag_web_enhanced_new.py archive/web_ui_variants/
mv rag_web_backend.py archive/web_ui_variants/

# Update documentation
echo "Canonical web UI: rag_web_enhanced.py" >> README.md
```

**A3 - Create Connection Manager**:
```python
# utils/db_connection.py
class DatabaseConnectionManager:
    """Centralized database connection management"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.conn = psycopg2.connect(
            host=os.getenv("PGHOST"),
            port=os.getenv("PGPORT"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            database=os.getenv("DB_NAME")
        )
        self.conn.autocommit = True

    def get_connection(self):
        return self.conn

# Usage everywhere:
from utils.db_connection import DatabaseConnectionManager
conn = DatabaseConnectionManager().get_connection()
```

**A4 - Consolidate Documentation**:
```bash
# Create docs archive
mkdir -p docs/archive

# Move session notes and old audits
mv *_AUDIT.md docs/archive/
mv *_ANALYSIS.md docs/archive/
mv *_REVIEW.md docs/archive/

# Keep only core docs (10 files):
# - README.md
# - CLAUDE.md
# - docs/START_HERE.md
# - docs/QUICKSTART.md
# - docs/ARCHITECTURE.md
# - docs/SECURITY.md
# - docs/DEPLOYMENT.md
# - docs/TROUBLESHOOTING.md
# - docs/API_REFERENCE.md
# - CHANGELOG.md

# Create index
cat > docs/README.md <<EOF
# Documentation Index
- [Start Here](START_HERE.md) - New users
- [Quick Start](QUICKSTART.md) - 15-minute setup
- [Architecture](ARCHITECTURE.md) - System design
- [Security](SECURITY.md) - Security best practices
- [Deployment](DEPLOYMENT.md) - Production deployment
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [API Reference](API_REFERENCE.md) - API documentation

Archived documentation: [archive/](archive/)
EOF
```

---

### Operations (4 items - 36 hours)

| ID | Issue | Impact | Effort | Deadline |
|----|-------|--------|--------|----------|
| **O5** | No error tracking (Sentry) | Slow incident detection | 2h | Week 2 |
| **O6** | No log aggregation (Loki) | Difficult debugging | 6h | Week 3 |
| **O7** | Untested disaster recovery | Unknown RTO/RPO | 4h | Week 2 |
| **O8** | Missing runbooks (5 scenarios) | Slow incident response | 24h | Month 2 |

**Fix Instructions**:

**O5 - Integrate Sentry**:
```python
# Install Sentry SDK
pip install sentry-sdk

# Initialize in main files
import sentry_sdk

sentry_sdk.init(
    dsn="https://your-dsn@sentry.io/project-id",
    environment="production",
    traces_sample_rate=1.0,
)

# Errors automatically captured
```

**O6 - Deploy Loki**:
```yaml
# docker-compose.yml - Add Loki service
loki:
  image: grafana/loki:latest
  ports:
    - "3100:3100"
  volumes:
    - ./config/loki-config.yml:/etc/loki/local-config.yaml
  command: -config.file=/etc/loki/local-config.yaml

promtail:
  image: grafana/promtail:latest
  volumes:
    - ./logs:/var/log
    - ./config/promtail-config.yml:/etc/promtail/config.yml
  command: -config.file=/etc/promtail/config.yml
```

**O7 - Test Disaster Recovery**:
```bash
# Test backup restore procedure (DO THIS NOW)
# 1. Create test database
psql -U postgres -c "CREATE DATABASE vector_db_test;"

# 2. Restore latest backup
gunzip < backups/latest.sql.gz | psql -U postgres -d vector_db_test

# 3. Verify data integrity
psql -U postgres -d vector_db_test -c "SELECT COUNT(*) FROM vector_table;"

# 4. Measure restore time (should be < 30 minutes)

# 5. Document results
echo "Backup restore test: SUCCESS" >> docs/DR_TEST_LOG.md
echo "Restore time: 15 minutes" >> docs/DR_TEST_LOG.md
echo "Data integrity: 100% verified" >> docs/DR_TEST_LOG.md
```

**O8 - Create Missing Runbooks** (24 hours total):
1. **Slow Query Response** (4h)
2. **High Memory Usage** (4h)
3. **Authentication Failures** (4h)
4. **Embedding Service Down** (4h)
5. **Database Connection Pool Exhausted** (4h)
6. **Disk Space Alert** (4h)

Template for each runbook:
```markdown
# Runbook: [Issue Name]

## Symptoms
- Alert: [Alert name]
- Metrics: [Specific metrics]
- User Impact: [What users experience]

## Diagnosis
1. Check [metric/log]
2. Run [command]
3. Verify [condition]

## Resolution
### Quick Fix (< 5 min)
[Immediate mitigation steps]

### Root Cause Fix (< 1 hour)
[Permanent resolution steps]

### Escalation
If issue persists after 30 minutes:
- Contact: [Person/Team]
- Slack: [Channel]
- On-call: [Rotation schedule]

## Prevention
[Steps to prevent recurrence]

## Related Alerts
- [Related alert 1]
- [Related alert 2]
```

---

### Quality (4 items - 28 hours)

| ID | Issue | Location | Effort | Deadline |
|----|-------|----------|--------|----------|
| **Q1** | Complex functions (>10 cyclomatic complexity) | 12 functions | 12h | Month 2 |
| **Q2** | Missing error handling (12 async functions) | Various files | 6h | Week 4 |
| **Q3** | 23 TODO comments without tickets | Various files | 2h | Week 2 |
| **Q4** | 7 files >500 lines | Various files | 8h | Month 2 |

**Fix Plan**:

**Q1 - Refactor Complex Functions**:
```python
# Example: load_documents() has complexity 26
# Before (185 lines, complexity 26):
def load_documents(path):
    # Complex branching logic
    # Multiple responsibilities
    # Hard to test

# After: Extract to smaller functions
def load_documents(path):
    """Load documents from path (simplified)"""
    if os.path.isfile(path):
        return _load_single_file(path)
    elif os.path.isdir(path):
        return _load_directory(path)
    else:
        raise ValueError(f"Invalid path: {path}")

def _load_single_file(file_path):
    """Load single file based on extension"""
    loader = _get_loader_for_file(file_path)
    return loader.load()

def _get_loader_for_file(file_path):
    """Get appropriate loader for file type"""
    ext = os.path.splitext(file_path)[1]
    if ext == '.pdf':
        return PDFReader()
    elif ext == '.html':
        return HTMLReader()
    # ... etc

def _load_directory(dir_path):
    """Load all files from directory"""
    files = glob.glob(f"{dir_path}/**/*", recursive=True)
    return [_load_single_file(f) for f in files if os.path.isfile(f)]
```

**Q2 - Add Error Handling**:
```python
# Pattern to apply everywhere:
import logging
logger = logging.getLogger(__name__)

async def some_async_function():
    try:
        result = await risky_operation()
        return result
    except SpecificException as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        # Graceful fallback
        return default_value
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise  # Re-raise unexpected errors
```

**Q3 - Convert TODOs to Issues**:
```bash
# Script to extract TODOs and create GitHub issues
grep -r "TODO\|FIXME\|HACK\|XXX" --include="*.py" . > todos.txt

# For each TODO:
# 1. Create GitHub issue
# 2. Add issue number to code comment
# 3. Remove or update TODO

# Example:
# Before: # TODO: Add validation
# After:  # See issue #123: Add input validation
```

---

## P2 - Medium Priority (Fix Next Sprint) - 14 Items

**Total Effort**: 64 hours (1.5 weeks)
**Impact**: Quality of life improvements

### Security (2 items - 12 hours)

| ID | Issue | Effort | Benefit |
|----|-------|--------|---------|
| **S8** | Missing input validation (8 endpoints) | 8h | Prevent injection attacks |
| **S9** | No Content Security Policy headers | 4h | XSS protection |

### Performance (3 items - 16 hours)

| ID | Issue | Effort | Benefit |
|----|-------|--------|---------|
| **P1** | No connection pooling | 4h | Handle more concurrent users |
| **P2** | No horizontal scaling support | 8h | Scale to 10x traffic |
| **P3** | Cache hit rate only 42% (target: >60%) | 4h | Reduce latency 20% |

### Quality (5 items - 20 hours)

| ID | Issue | Effort | Benefit |
|----|-------|--------|---------|
| **Q5** | No integration tests | 12h | Catch multi-component bugs |
| **Q6** | Magic numbers in code (15 occurrences) | 2h | Improve maintainability |
| **Q7** | Inconsistent error patterns | 4h | Better error handling |
| **Q8** | Missing type hints (30% of functions) | 2h | Better IDE support |

### Operations (4 items - 16 hours)

| ID | Issue | Effort | Benefit |
|----|-------|--------|---------|
| **O9** | No distributed tracing | 8h | Trace requests across services |
| **O10** | No capacity planning process | 4h | Prevent outages |
| **O11** | No load testing | 4h | Know system limits |

---

## P3 - Low Priority (Backlog) - 8 Items

**Total Effort**: 24 hours
**Impact**: Nice-to-have improvements

| ID | Issue | Category | Effort |
|----|-------|----------|--------|
| **L1** | Dead code (8 unused components) | Quality | 2h |
| **L2** | Outdated comments | Quality | 2h |
| **L3** | Inconsistent logging format | Ops | 4h |
| **L4** | No API documentation | Docs | 8h |
| **L5** | No changelog | Docs | 2h |
| **L6** | No contributing guide | Docs | 4h |
| **L7** | No code of conduct | Docs | 1h |
| **L8** | No security policy | Docs | 1h |

---

## Implementation Timeline

### Week 1 - Critical Security (40 hours)

**Monday-Tuesday** (16h):
- [ ] S2: Remove hardcoded password (**DO FIRST**)
- [ ] S1: Implement authentication (OAuth2/JWT)
- [ ] S3: Set up Docker secrets

**Wednesday-Thursday** (16h):
- [ ] O4: Export application metrics
- [ ] O7: Test disaster recovery
- [ ] Q3: Convert TODOs to issues

**Friday** (8h):
- [ ] S4: Update vulnerable dependencies
- [ ] Enable Dependabot
- [ ] Set up GitHub Issues

**Deliverable**: Eliminate 3 P0 security blockers

---

### Week 2 - Operations Foundations (40 hours)

**Monday-Tuesday** (16h):
- [ ] O1: Define SLIs/SLOs
- [ ] O5: Integrate Sentry

**Wednesday-Friday** (24h):
- [ ] O2: Create CI/CD pipeline
- [ ] A3: Create database connection manager
- [ ] A4: Consolidate documentation

**Deliverable**: Operational visibility + automation

---

### Week 3 - Quick Wins (24 hours)

**Monday-Wednesday** (24h):
- [ ] A2: Consolidate web UI files
- [ ] S5: Enable HTTPS/TLS
- [ ] S6: Add rate limiting
- [ ] O6: Deploy log aggregation

**Deliverable**: Code quality improvements

---

### Month 2 - Architecture Refactoring (120 hours)

**Weeks 4-7** (120h):
- [ ] A1: Break monolith into 8 modules (96h)
- [ ] R2: Increase test coverage to 70% (24h)

**Deliverable**: Modular architecture + high test coverage

---

### Month 3 - Team & Production Hardening (150 hours)

**Weeks 8-11** (150h):
- [ ] R1: Hire/train backup developer (80h)
- [ ] O3: Implement Infrastructure as Code (40h)
- [ ] O8: Create 5 missing runbooks (20h)
- [ ] P2 issues: Performance improvements (10h)

**Deliverable**: Production-ready system with backup team

---

## Success Criteria

### By End of Month 1 (Security + Ops)

- ✅ Zero P0 security issues
- ✅ Application metrics exported to Prometheus
- ✅ SLOs defined and tracked
- ✅ CI/CD pipeline operational
- ✅ Disaster recovery tested
- ✅ Code duplication reduced 50%

**Risk Level**: HIGH → MEDIUM

---

### By End of Month 2 (Architecture)

- ✅ Monolithic core refactored into 8 modules
- ✅ Test coverage increased to 70%
- ✅ Velocity restored (+40%)
- ✅ Onboarding time reduced (3 weeks → 5 days)

**Risk Level**: MEDIUM → LOW

---

### By End of Month 3 (Team + Hardening)

- ✅ Bus factor increased to 2 (backup developer)
- ✅ Infrastructure as Code implemented
- ✅ All critical runbooks created
- ✅ Load testing completed
- ✅ SRE maturity: 2.5/5 → 3.5/5

**Risk Level**: LOW → VERY LOW
**Production Status**: ✅ **READY**

---

## Tracking & Accountability

### GitHub Project Board Structure

**Columns**:
1. **Backlog** (P2, P3 items)
2. **To Do** (P0, P1 items not yet started)
3. **In Progress** (Limit: 3 items max)
4. **In Review** (Awaiting approval)
5. **Done** (Completed & verified)

**Labels**:
- `priority/p0-critical` - Must fix before production
- `priority/p1-high` - Fix this sprint
- `priority/p2-medium` - Fix next sprint
- `priority/p3-low` - Backlog
- `security` - Security-related
- `architecture` - Architecture changes
- `operations` - Ops/SRE work
- `quality` - Testing/quality
- `quick-win` - High impact, low effort

### Weekly Progress Review

**Every Friday**:
1. Review completed items
2. Update risk assessment
3. Adjust priorities if needed
4. Plan next week's work

**Metrics to Track**:
- Security issues remaining
- Test coverage %
- Technical debt days
- Velocity (story points/week)
- MTTR (mean time to resolve)

---

## Contact & Escalation

### Issue Ownership

| Category | Owner | Backup | Escalation |
|----------|-------|--------|------------|
| **Security** | Security Team Lead | Dev Team Lead | CTO |
| **Architecture** | Tech Lead | Senior Engineer | CTO |
| **Operations** | DevOps Lead | SRE Team | CTO |
| **Quality** | QA Lead | Dev Team Lead | Engineering Manager |
| **Product** | Product Manager | CEO | Board |

### Emergency Contacts

**P0 Security Issues**:
- Primary: Security Team Lead
- Escalate after: 4 hours
- Contact: [Email/Slack/Phone]

**Production Outages**:
- Primary: On-Call Engineer (see PagerDuty)
- Escalate after: 30 minutes
- War room: #incident-response Slack channel

---

## Audit Follow-Up

**Next Comprehensive Audit**: April 9, 2026 (3 months)

**Expected Improvements by Next Audit**:
- Overall Health Score: 67/100 → 80/100
- Security Score: 58/100 → 95/100
- Test Coverage: 30% → 80%
- SRE Maturity: 2.5/5 → 3.5/5
- Risk Level: HIGH → LOW
- Production Status: NOT READY → READY ✅

---

**This Risk Priority Matrix provides the roadmap from current state to production-ready. Focus on P0 items first - they are blocking production deployment.**

**Start with S2 (hardcoded password) - it takes 15 minutes and is a critical security vulnerability.**
