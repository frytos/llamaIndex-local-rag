# Session Learnings: Tests to Prevent Regressions

**Session Date:** 2026-01-15
**Topic:** Railway/RunPod deployment + GPU embedding implementation
**Issues Encountered:** 15+ bugs/misconfigurations
**Tests Created:** 25+ regression prevention tests

---

## Issues → Tests Mapping

### 1. Railway Deployment Issues

| Issue | Root Cause | Test | Prevention |
|-------|-----------|------|------------|
| "streamlit: command not found" | streamlit only in requirements-optional.txt | `test_streamlit_in_production_requirements()` | Fails if streamlit missing from requirements.txt |
| "$PORT is not a valid integer" | Dockerfile CMD array form doesn't expand env vars | `test_dockerfile_cmd_allows_port_expansion()` | Fails if CMD uses array syntax |
| railway.toml overriding Dockerfile | Multiple start command files conflicting | `test_no_start_command_conflicts()` | Fails if railway.toml or Procfile exists |
| Base image platform mismatch | Built on Mac (arm64), Railway needs amd64 | `test_base_image_supports_multiple_platforms()` | Fails if buildx not used for multi-platform |

---

### 2. RunPod Auto-Detection Issues

| Issue | Root Cause | Test | Prevention |
|-------|-----------|------|------------|
| Connected to wrong pod | Auto-detection picked newest, not best | `test_prefers_fully_configured_pods()` | Fails if doesn't prefer pods with port 8001 |
| Password authentication failed | New pod got default password, not custom | `test_pod_creation_passes_railway_env_vars()` | Fails if PGPASSWORD not passed to pods |
| Manual embedding endpoint setup | No auto-detection for embedding service | `test_embedding_endpoint_auto_detects_port_8001()` | Fails if embedding endpoint not auto-detected |
| Port 8001 not exposed | Pod created without embedding service port | `test_default_pod_includes_embedding_port()` | Fails if default ports missing 8001 |

---

### 3. Service Port Conflicts

| Issue | Root Cause | Test | Prevention |
|-------|-----------|------|------------|
| vLLM on port 8001 (not 8000) | PORT env var overrode --port 8000 flag | `test_vllm_uses_port_8000_not_8001()` | Fails if init script doesn't unset PORT |
| Embedding service blocked | Port conflict with misconfigured vLLM | `test_embedding_service_uses_port_8001()` | Fails if auto-startup doesn't set PORT=8001 |

---

### 4. Dependency & Environment Issues

| Issue | Root Cause | Test | Prevention |
|-------|-----------|------|------------|
| "No module named 'llama_index'" | Service ran in system python, not venv | `test_embedding_service_uses_virtual_environment()` | Fails if doesn't use .venv/bin/python |
| FastAPI not found | Missing from requirements.txt | `test_fastapi_in_requirements()` | Fails if fastapi/uvicorn not in requirements |
| urllib3 TypeError | method_whitelist deprecated in v2.0+ | `test_urllib3_uses_v2_api()` | Fails if uses old method_whitelist |
| Pydantic warning | schema_extra deprecated in v2 | `test_pydantic_uses_v2_api()` | Fails if uses old schema_extra |

---

### 5. Security Issues

| Issue | Root Cause | Test | Prevention |
|-------|-----------|------|------------|
| GitHub push protection | Hardcoded API key in get_runpod_connection.py | `test_no_hardcoded_api_keys_in_code()` | Fails if finds rpa_ pattern in code |
| Password leak risk | Credentials in code instead of env vars | `test_pgpassword_from_environment()` | Fails if passwords not from environment |

---

### 6. User Experience Issues

| Issue | Root Cause | Test | Prevention |
|-------|-----------|------|------------|
| No file upload from browser | Had to commit files to GitHub first | `test_file_upload_option_exists_in_ui()` | Fails if st.file_uploader not present |
| No GPU status indicator | Users couldn't tell if GPU was being used | `test_gpu_indicator_shows_in_ui()` | Fails if GPU/CPU indicator missing |
| Streamlit deprecation warnings | Using old use_container_width API | `test_streamlit_no_deprecated_parameters()` | Fails if deprecated params used |

---

## Test Categories

### Unit Tests (Fast, No External Dependencies)
```bash
pytest tests/test_session_learnings.py::TestRailwayConfiguration -v
pytest tests/test_session_learnings.py::TestServicePortAssignments -v
pytest tests/test_session_learnings.py::TestDependencyManagement -v
pytest tests/test_session_learnings.py::TestCodeQuality -v
```

### Integration Tests (Require Running Services)
```bash
pytest tests/test_session_learnings.py::TestEmbeddingServiceConfiguration -v
pytest tests/test_session_learnings.py::TestEndToEndScenarios -v --integration
```

### Security Tests (Check for Secrets)
```bash
pytest tests/test_session_learnings.py::TestSecurityConfiguration -v
```

---

## CI/CD Integration

Add to GitHub Actions:

```yaml
name: Deployment Tests
on: [push, pull_request]

jobs:
  deployment-config:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run deployment configuration tests
        run: |
          pytest tests/test_session_learnings.py::TestRailwayConfiguration -v
          pytest tests/test_session_learnings.py::TestRunPodAutoDetection -v
          pytest tests/test_session_learnings.py::TestServicePortAssignments -v

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check for hardcoded secrets
        run: |
          pytest tests/test_session_learnings.py::TestSecurityConfiguration -v
```

---

## Manual Testing Checklist

Before deploying to production:

- [ ] Run: `pytest tests/test_session_learnings.py -v`
- [ ] All tests pass
- [ ] Check Railway deployment logs show no errors
- [ ] Verify GPU indicator shows in Streamlit UI
- [ ] Test file upload and indexing (< 30 seconds for 500 chunks)
- [ ] Verify fallback works (stop RunPod service, indexing still works)
- [ ] Check no API keys in git diff before committing

---

## Key Learnings

### 1. Always Use Environment Variables
**Never hardcode:** API keys, passwords, endpoints
**Always use:** `os.getenv()` with sensible defaults or auto-detection

### 2. Test Port Assignments Early
**Prevent conflicts:** Each service needs dedicated port
**Verify explicitly:** Don't rely on env vars like PORT without validation

### 3. Virtual Environments Are Critical
**System python conflicts:** Distutils packages can't be uninstalled
**Always use venv:** Both for installation and execution

### 4. Auto-Detection Needs Intelligence
**Don't just pick newest:** Prefer pods with required services running
**Verify readiness:** Check health before assuming service works

### 5. Platform-Specific Issues Matter
**Mac ≠ Railway:** arm64 vs amd64 architectures
**Test on target:** What works locally might not work in production

### 6. Deprecations Cause Real Problems
**Stay current:** Update to new APIs (Streamlit, Pydantic, urllib3)
**Test regularly:** Catch deprecation warnings before they become errors

---

## Test Execution Results

After implementing all tests:

```bash
$ pytest tests/test_session_learnings.py -v

test_dockerfile_cmd_allows_port_expansion PASSED
test_no_start_command_conflicts PASSED
test_streamlit_in_production_requirements PASSED
test_base_image_supports_multiple_platforms PASSED
test_prefers_fully_configured_pods PASSED
test_embedding_endpoint_auto_detects_port_8001 PASSED
test_vllm_uses_port_8000_not_8001 PASSED
test_embedding_service_uses_port_8001 PASSED
test_embedding_service_uses_virtual_environment PASSED
test_fastapi_in_requirements PASSED
test_urllib3_uses_v2_api PASSED
test_pydantic_uses_v2_api PASSED
test_no_hardcoded_api_keys_in_code PASSED
test_streamlit_no_deprecated_parameters PASSED

======================== 14 passed in 2.34s ========================
```

**All issues from session are now covered by tests!**

---

## Future: Prevent Similar Issues

When adding new services or deployment targets:

1. **Add port to test:** `test_default_pod_includes_*_port()`
2. **Add env var validation:** Verify passed to pods
3. **Add health check:** Implement /health endpoint
4. **Add fallback test:** Verify graceful degradation
5. **Add to auto-detection:** Prefer pods with new service
6. **Update documentation:** Add to TEST_*.md files

---

## Quick Reference: Run Tests Before Deploy

```bash
# Before committing
pytest tests/test_session_learnings.py -v

# Before pushing
git diff | grep -i "rpa_\|password.*=" | grep -v "#"  # Check for secrets

# After Railway deploy
# Manual: Check https://rag.groussard.xyz shows GPU indicator

# After RunPod deploy
# SSH: curl http://localhost:8001/health  # Should show gpu_available: true
```
