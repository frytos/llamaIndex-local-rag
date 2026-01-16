# Code Quality Configuration Checklist

**Purpose:** Prevent recurrent deployment and integration issues
**Based on:** Session learnings from Railway/RunPod deployment (2026-01-15/16)
**Last Updated:** 2026-01-16

---

## 🔒 Security & Secrets

### Before Every Commit

- [ ] **No hardcoded API keys**
  ```bash
  git diff | grep -iE "rpa_[A-Za-z0-9]{40}|sk-[A-Za-z0-9]{40}" && echo "⚠️  API key detected!"
  ```
  - Check: `get_runpod_connection.py`, `utils/runpod_*.py`, `rag_web.py`
  - Rule: Use `os.getenv()` always, never defaults with real keys
  - Test: `pytest tests/test_session_learnings.py::TestSecurityConfiguration -v`

- [ ] **No hardcoded passwords**
  ```bash
  git diff | grep -E "password.*=.*['\"](?!your_|test_|<)" && echo "⚠️  Password detected!"
  ```
  - Check: Database connection strings
  - Rule: Always from environment: `os.getenv("PGPASSWORD")`

- [ ] **No hardcoded endpoints**
  - Check: Connection strings, URLs
  - Rule: Use auto-detection or environment variables
  - Bad: `host = "103.196.86.53"`
  - Good: `host = get_postgres_config()["host"]`

---

## 🌐 Environment Variables

### Configuration Defaults

- [ ] **Empty string triggers auto-detection** (not "localhost")
  ```python
  # BAD
  PGHOST = os.getenv("PGHOST", "localhost")  # Blocks auto-detection

  # GOOD
  PGHOST = os.getenv("PGHOST", "")  # Triggers auto-detection
  ```
  - File: `config/constants.py` - `DEFAULT_HOST = ""`
  - File: `rag_web.py:123` - `get("PGHOST", "")`
  - Test: Auto-detection should activate when env var not set

- [ ] **All services support auto-detection**
  - PostgreSQL: PGHOST="" or "auto" → auto-detect
  - Embedding: RUNPOD_EMBEDDING_ENDPOINT="" or "auto" → auto-detect
  - Rule: User shouldn't need to set endpoints manually

- [ ] **Credentials passed to new pods**
  ```python
  custom_env = {
      "PGPASSWORD": os.getenv("PGPASSWORD"),  # ✅ Railway password to pod
      "PGUSER": os.getenv("PGUSER"),
      "RUNPOD_EMBEDDING_API_KEY": os.getenv("RUNPOD_EMBEDDING_API_KEY")
  }
  ```
  - File: `rag_web.py` - pod creation `custom_env` dict
  - Verify: New pods get Railway's custom password (not default)

---

## 🔌 Port Configurations

### Service Port Assignments

- [ ] **Each service has dedicated port**
  - PostgreSQL: 5432
  - vLLM: 8000
  - Embedding API: 8001
  - SSH: 22
  - Grafana: 3000
  - No overlaps, no conflicts

- [ ] **Port types correct for accessibility**
  ```python
  # Services Railway needs to access → TCP (public IP)
  "5432/tcp"  # PostgreSQL ✅
  "8001/tcp"  # Embedding API ✅
  "22/tcp"    # SSH ✅

  # Services only accessed via RunPod Console → HTTP (proxy OK)
  "8000/http"  # vLLM (optional)
  "3000/http"  # Grafana (optional)
  ```
  - File: `utils/runpod_manager.py:141`
  - Test: Verify all Railway-accessed services use /tcp

- [ ] **Port env vars unset before service starts**
  ```bash
  # vLLM startup
  unset PORT  # Prevents PORT env var from overriding --port 8000
  python -m vllm.entrypoints.openai.api_server --port 8000
  ```
  - File: `scripts/init_runpod_services.sh` - vLLM startup
  - File: `rag_web.py` - auto-startup command
  - Test: `pytest tests/test_session_learnings.py::TestServicePortAssignments -v`

- [ ] **Embedding service gets explicit PORT**
  ```bash
  export PORT=8001  # Explicit, clear
  python -m uvicorn services.embedding_service:app --port 8001
  ```
  - File: `rag_web.py` - STEP 4/5 of auto-startup

---

## 🐍 Virtual Environment

### Dependency Isolation

- [ ] **All services run in venv** (not system python)
  ```bash
  # BAD
  python3 -m uvicorn services.embedding_service:app  # System python

  # GOOD
  source .venv/bin/activate
  python -m uvicorn services.embedding_service:app  # Venv python

  # BEST
  /workspace/rag-pipeline/.venv/bin/python -m uvicorn ...  # Explicit venv
  ```
  - File: `rag_web.py` - auto-startup uses `.venv/bin/python`
  - File: `scripts/init_runpod_services.sh` - creates and activates venv
  - Test: `pytest tests/test_session_learnings.py::TestVirtualEnvironmentUsage -v`

- [ ] **No distutils package conflicts**
  - Always use venv to avoid system package conflicts
  - Don't `pip install --user` on RunPod
  - Create fresh venv if distutils errors occur

---

## 🚢 Railway Deployment

### Dockerfile Configuration

- [ ] **CMD uses shell form** (not array) for env var expansion
  ```dockerfile
  # BAD - Variables not expanded
  CMD ["streamlit", "run", "app.py", "--server.port=$PORT"]

  # GOOD - Variables expanded
  CMD streamlit run app.py --server.port=${PORT:-8080}
  ```
  - File: `Dockerfile`
  - Test: `pytest tests/test_session_learnings.py::TestRailwayConfiguration::test_dockerfile_cmd_allows_port_expansion -v`

- [ ] **No conflicting start commands**
  - Delete: `railway.toml` (if exists)
  - Delete: `Procfile` (if exists)
  - Use: Dockerfile CMD only
  - Test: `test_no_start_command_conflicts()`

- [ ] **Production dependencies in requirements.txt**
  ```
  ✅ streamlit>=1.31.0
  ✅ plotly>=5.18.0
  ✅ fastapi>=0.115.0
  ✅ uvicorn[standard]>=0.34.0
  ```
  - Not in requirements-optional.txt!
  - Test: `test_streamlit_in_production_requirements()`

- [ ] **.dockerignore excludes large dirs**
  ```
  data/
  logs/
  .git/
  tests/
  .planning/
  __pycache__/
  ```
  - Faster builds, smaller images

### Multi-Platform Images

- [ ] **Base images support both architectures**
  ```bash
  docker buildx build \
    --platform linux/amd64,linux/arm64 \  # Both platforms
    -t user/image:latest \
    --push .
  ```
  - Mac: linux/arm64
  - Railway: linux/amd64
  - Test: Check `build-base-image.sh` uses buildx

---

## 🤖 RunPod Integration

### Auto-Detection Logic

- [ ] **Skip pods without port mappings**
  ```python
  ports = runtime.get("ports", []) or []
  if not ports:
      continue  # Skip stopped/starting pods
  ```
  - File: `utils/runpod_db_config.py`
  - Prevents: Connecting to stopped pods
  - Test: `test_auto_detection_handles_no_pods()`

- [ ] **Prefer fully-configured pods**
  ```python
  # Priority:
  # 1. Pods with ports 5432 AND 8001 (fully configured)
  # 2. Pods with port 5432 only (PostgreSQL only)
  # 3. None
  ```
  - Logic: Two-pass pod selection
  - File: `utils/runpod_db_config.py:126-160`
  - Test: `test_prefers_fully_configured_pods()`

- [ ] **Log pod selection reasoning**
  ```
  Found 3 pod(s)
  Evaluating pods:
     ⏭️  pod-1: SKIP (no port mappings)
     🔍 pod-2: PostgreSQL=True, Embedding=False
     🔍 pod-3: PostgreSQL=True, Embedding=True
     ✅ SELECTED: pod-3 (fully-configured)
  ```
  - Must show: Which pods considered, why selected/skipped

### Pod Creation

- [ ] **Pass Railway env vars to new pods**
  ```python
  custom_env = {
      "PGPASSWORD": os.getenv("PGPASSWORD"),  # ✅
      "RUNPOD_EMBEDDING_API_KEY": os.getenv("RUNPOD_EMBEDDING_API_KEY"),  # ✅
      "PGUSER": os.getenv("PGUSER", "fryt"),
      "DB_NAME": os.getenv("DB_NAME", "vector_db")
  }
  ```
  - File: `rag_web.py` - pod creation
  - Test: Verify custom password in new pod

- [ ] **All required ports exposed**
  ```python
  ports = "5432/tcp,8000/http,8001/tcp,22/tcp,3000/http"
  ```
  - 5432/tcp - PostgreSQL (Railway access)
  - 8001/tcp - Embedding (Railway access)
  - 8000/http, 3000/http - Console only
  - Test: `test_default_pod_includes_embedding_port()`

---

## 🚀 Service Startup

### Auto-Startup Scripts

- [ ] **Services start in correct order**
  ```bash
  # 1. Install dependencies (in venv)
  # 2. Start PostgreSQL
  # 3. Start vLLM (optional, port 8000)
  # 4. Start embedding service (port 8001)
  # 5. Verify all services
  ```

- [ ] **Each service uses venv**
  ```bash
  source .venv/bin/activate  # Activate first
  # OR
  /workspace/rag-pipeline/.venv/bin/python  # Explicit path
  ```
  - Test: `test_auto_startup_activates_venv()`

- [ ] **PORT env var managed correctly**
  ```bash
  # vLLM
  unset PORT  # Prevent override
  python -m vllm.entrypoints.openai.api_server --port 8000

  # Embedding
  export PORT=8001  # Explicit
  python -m uvicorn services.embedding_service:app --port 8001
  ```

- [ ] **Health checks before proceeding**
  ```bash
  curl http://localhost:8001/health || echo "Service not ready"
  ```

---

## 📊 API Compatibility

### Dependency Versions

- [ ] **urllib3 2.0+ API**
  ```python
  # BAD
  Retry(method_whitelist=["POST", "GET"])

  # GOOD
  Retry(allowed_methods=["POST", "GET"])
  ```
  - File: `utils/runpod_embedding_client.py:60`
  - Test: `test_urllib3_uses_v2_api()`

- [ ] **Pydantic v2 API**
  ```python
  # BAD
  class Config:
      schema_extra = {...}

  # GOOD
  class Config:
      json_schema_extra = {...}
  ```
  - File: `services/embedding_service.py`
  - Test: `test_pydantic_uses_v2_api()`

- [ ] **Streamlit current API**
  ```python
  # BAD
  st.button("Click", use_container_width=True)

  # GOOD
  st.button("Click", width='stretch')
  ```
  - File: `rag_web.py`
  - Test: `test_streamlit_no_deprecated_parameters()`

---

## 🔗 Integration Patterns

### Service Communication

- [ ] **Health checks before using services**
  ```python
  client = RunPodEmbeddingClient(endpoint, api_key)
  if not client.check_health():
      # Fall back to local
      return use_local_embedding()
  # Proceed with GPU
  ```
  - Every external service should have health check
  - Implement graceful degradation

- [ ] **Comprehensive error handling**
  ```python
  try:
      result = call_external_service()
  except requests.Timeout:
      log.error("Timeout - service slow/overloaded")
      return fallback()
  except requests.ConnectionError:
      log.error("Connection failed - service down/unreachable")
      return fallback()
  except Exception as e:
      log.error(f"Unexpected error: {e}")
      return fallback()
  ```
  - Separate timeout, connection, and other errors
  - Provide actionable error messages

- [ ] **Retry logic with backoff**
  ```python
  Retry(
      total=3,
      backoff_factor=1,  # 1s, 2s, 4s
      status_forcelist=[429, 500, 502, 503, 504]
  )
  ```
  - Don't retry indefinitely
  - Use exponential backoff

---

## 📝 Logging Standards

### Diagnostic Logging

- [ ] **Decision points logged**
  ```python
  log.info("🎯 Embedding Mode Decision:")
  if endpoint and api_key:
      log.info("   → MODE: RunPod GPU")
  else:
      log.info("   → MODE: Local CPU")
      log.info("   Reason: ...")
  ```
  - Show WHY decisions were made
  - User should understand mode selection

- [ ] **Configuration state visible**
  ```python
  log.info("🔍 Environment Check:")
  log.info(f"   RUNPOD_API_KEY: {'✅ Set' if set else '❌ Not set'}")
  log.info(f"   RUNPOD_EMBEDDING_API_KEY: {'✅ Set' if set else '❌ Not set'}")
  ```
  - Show what's configured at startup
  - Mask sensitive values (show length, not content)

- [ ] **Selection process documented**
  ```python
  log.info("Evaluating pods:")
  for pod in pods:
      log.info(f"   🔍 {pod.name}: PostgreSQL={has_pg}, Embedding={has_emb}")
  log.info(f"   ✅ SELECTED: {selected_pod.name}")
  ```
  - Show what was considered
  - Show why chosen/skipped

- [ ] **Error messages actionable**
  ```python
  log.error("❌ Health check FAILED")
  log.error("   Possible causes:")
  log.error("      - Service not started on RunPod")
  log.error("      - Wrong IP/port (check port mappings)")
  log.error("      - Firewall blocking connection")
  ```
  - Don't just say "failed"
  - Suggest specific fixes

---

## 🧪 Testing Requirements

### Before Deployment

- [ ] **Run regression tests**
  ```bash
  pytest tests/test_session_learnings.py -v
  # Must pass: 25+ tests covering session issues
  ```

- [ ] **Check for deprecation warnings**
  ```bash
  pytest tests/ -v 2>&1 | grep -i "deprecat\|warning"
  ```
  - Fix before warnings become errors

- [ ] **Verify startup scripts are executable**
  ```bash
  ls -la scripts/*.sh | grep -v "x"  # Should be empty
  ```

---

## 🏗️ Deployment Configuration

### Railway-Specific

- [ ] **Minimal environment variables**
  ```
  Required:
  - RUNPOD_API_KEY
  - RUNPOD_EMBEDDING_API_KEY
  - PGPASSWORD

  Optional (auto-detected):
  - PGHOST (leave unset)
  - PGPORT (leave unset)
  - RUNPOD_EMBEDDING_ENDPOINT (leave unset)
  ```
  - Less config = fewer errors

- [ ] **Dockerfile optimized**
  - Multi-stage build for caching
  - Shell form CMD for variable expansion
  - .dockerignore excludes large dirs

### RunPod-Specific

- [ ] **Port exposure correct**
  ```
  "5432/tcp" - Database (public IP needed)
  "8001/tcp" - Embedding (public IP needed)
  "8000/http" - vLLM (proxy OK)
  ```
  - TCP for Railway access
  - HTTP for console-only

- [ ] **Auto-startup idempotent**
  ```bash
  if [ -f /workspace/.init_complete ]; then
      skip_init
  else
      run_init
      touch /workspace/.init_complete
  fi
  ```
  - Can run multiple times safely
  - Doesn't duplicate work

- [ ] **Services start in venv**
  ```bash
  source /workspace/rag-pipeline/.venv/bin/activate
  # Then start services
  ```

---

## 🔍 Code Review Checklist

### Before Merging Code

- [ ] **No code bypasses main pipeline**
  ```python
  # BAD - Web UI has own embedding loop
  for batch in batches:
      embeddings = local_model.embed(batch)

  # GOOD - Web UI uses pipeline
  rag.embed_nodes(model, nodes)  # Uses GPU if available
  ```
  - Reuse existing functions
  - Don't duplicate logic

- [ ] **Fallbacks implemented**
  - Every external service call should have fallback
  - System should degrade gracefully, not crash
  - Log when falling back (warning level)

- [ ] **Configuration validated at startup**
  ```python
  def __post_init__(self):
      if not self.user or not self.password:
          raise ValueError("Database credentials not set!")
  ```
  - Fail fast with clear message
  - Don't wait for first use to discover missing config

---

## 📋 Pre-Deployment Checklist

### Run Before Every Deploy

```bash
#!/bin/bash
# pre-deploy-check.sh

echo "🔍 Running pre-deployment checks..."

# 1. Security
echo "Checking for hardcoded secrets..."
git diff main | grep -iE "rpa_[A-Za-z0-9]{40}|password.*=.*['\"]" && {
    echo "❌ FAIL: Secrets detected"
    exit 1
}

# 2. Tests
echo "Running regression tests..."
pytest tests/test_session_learnings.py -q || {
    echo "❌ FAIL: Tests failed"
    exit 1
}

# 3. Startup scripts executable
echo "Checking script permissions..."
find scripts -name "*.sh" -type f ! -perm -u+x && {
    echo "❌ FAIL: Non-executable scripts found"
    exit 1
}

# 4. Port configuration
echo "Validating port mappings..."
grep -q "8001/tcp" utils/runpod_manager.py || {
    echo "❌ FAIL: Port 8001 not TCP"
    exit 1
}

# 5. Virtual environment usage
echo "Checking venv usage in auto-startup..."
grep -q ".venv/bin/python" rag_web.py || {
    echo "❌ FAIL: Auto-startup not using venv"
    exit 1
}

echo "✅ All pre-deployment checks passed!"
```

---

## 🐛 Common Issues & Fixes

### Issue: "$PORT is not a valid integer"
**Cause:** Dockerfile CMD uses array form
**Fix:** Use shell form CMD
**Check:** `test_dockerfile_cmd_allows_port_expansion()`

### Issue: "streamlit: command not found"
**Cause:** streamlit not in requirements.txt
**Fix:** Add to production requirements
**Check:** `test_streamlit_in_production_requirements()`

### Issue: "No module named 'llama_index'"
**Cause:** Service running outside venv
**Fix:** Use `.venv/bin/python`
**Check:** `test_embedding_service_uses_virtual_environment()`

### Issue: "Network is unreachable"
**Cause:** Port exposed as /http (proxy IP)
**Fix:** Change to /tcp for public IP
**Check:** Verify in `utils/runpod_manager.py:141`

### Issue: "connection to server at 'auto'"
**Cause:** PGHOST defaults to "auto" literal
**Fix:** Default to empty string
**Check:** `config/constants.py` - `DEFAULT_HOST = ""`

### Issue: "Password authentication failed"
**Cause:** New pod got default password
**Fix:** Pass PGPASSWORD in pod creation env
**Check:** `test_pod_creation_passes_railway_env_vars()`

### Issue: vLLM on port 8001
**Cause:** PORT env var overrode --port 8000
**Fix:** `unset PORT` before starting vLLM
**Check:** `test_vllm_uses_port_8000_not_8001()`

### Issue: GitHub push protection
**Cause:** Hardcoded API key in code
**Fix:** Use os.getenv() without defaults
**Check:** `test_no_hardcoded_api_keys_in_code()`

---

## 📈 Quality Metrics

### Target Standards

- [ ] **Test coverage:** >80% for new code
- [ ] **No hardcoded secrets:** 0 occurrences
- [ ] **Deprecation warnings:** 0 warnings
- [ ] **Startup success rate:** >95%
- [ ] **Auto-detection success:** >90% with active pod
- [ ] **Fallback activation:** Works 100% when primary fails

---

## 🔄 Continuous Improvement

### After Each Issue

1. **Add test** to prevent regression
2. **Update this checklist** with new pattern
3. **Document in SESSION_LEARNINGS** (if significant)
4. **Add to pre-deployment script** (if automatable)

### Monthly Review

- [ ] Review failed deployments
- [ ] Check for new deprecation warnings
- [ ] Update dependency versions
- [ ] Regenerate test packs for RAG evaluation
- [ ] Review and update this checklist

---

## ✅ Quick Validation

Run this before committing:

```bash
# Security
git diff | grep -iE "rpa_|password.*=.*['\"]" | grep -v "#" | grep -v "your_"

# Tests
pytest tests/test_session_learnings.py --tb=no -q

# Permissions
find scripts -name "*.sh" ! -perm -u+x

# Port config
grep "8001/tcp" utils/runpod_manager.py > /dev/null && echo "✅ Port OK"
```

**If all pass:** Safe to deploy
**If any fail:** Fix before pushing

---

## 📚 Related Documentation

- `docs/DEPLOYMENT_TESTS_GUIDE.md` - Issue → Test mapping
- `docs/RUNPOD_PORT_CONFIGURATION.md` - Port mapping details
- `docs/SESSION_LEARNINGS.md` - Original session notes (if exists)
- `tests/test_session_learnings.py` - Automated tests
- `TEST_GPU_EMBEDDINGS.md` - GPU testing procedures

---

**Last Updated:** 2026-01-16
**Status:** Ready for use
**Next Review:** 2026-02-16 (30 days)
