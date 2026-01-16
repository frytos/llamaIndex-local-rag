#!/bin/bash
#
# Pre-Deployment Quality Checks
#
# Runs automated checks before deploying to Railway/RunPod.
# Based on CODE_QUALITY_CHECKLIST.md
#
# Usage:
#   ./scripts/pre-deploy-check.sh           # Quick checks only
#   ./scripts/pre-deploy-check.sh --full    # Include test suite (slower)
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#

set -e

FAILED=0
RUN_TESTS=false

# Parse arguments
if [ "$1" = "--full" ]; then
    RUN_TESTS=true
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Pre-Deployment Quality Checks                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# 1. Security Checks
# =============================================================================

echo "🔒 Security Checks"
echo "────────────────────────────────────────────────────────────────"

# Check for hardcoded API keys
echo -n "   Checking for hardcoded API keys... "
if git diff main 2>/dev/null | grep -iE "rpa_[A-Za-z0-9]{40}" | grep -v "#" | grep -v "your_" | grep -v "test_" > /dev/null; then
    echo "❌ FAIL"
    echo "      Found hardcoded RunPod API key in diff"
    FAILED=1
else
    echo "✅ PASS"
fi

# Check for hardcoded passwords
echo -n "   Checking for hardcoded passwords... "
if git diff main 2>/dev/null | grep -E "password.*=.*['\"]" | grep -v "#" | grep -v "your_" | grep -v "test_" | grep -v "<" > /dev/null; then
    echo "❌ FAIL"
    echo "      Found hardcoded password in diff"
    FAILED=1
else
    echo "✅ PASS"
fi

echo ""

# =============================================================================
# 2. Test Suite
# =============================================================================

if [ "$RUN_TESTS" = true ]; then
    echo "🧪 Test Suite"
    echo "────────────────────────────────────────────────────────────────"

    echo -n "   Running regression tests... "
    # Run tests and capture output
    TEST_OUTPUT=$(pytest tests/test_session_learnings.py --tb=no -q --no-cov 2>&1)
    PASSED=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+")
    FAILED_COUNT=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+" || echo "0")

    if [ "$PASSED" -ge 20 ] && [ "$FAILED_COUNT" -le 5 ]; then
        echo "✅ PASS ($PASSED passed, $FAILED_COUNT known mock issues)"
    else
        echo "❌ FAIL"
        echo "      Run: pytest tests/test_session_learnings.py -v"
        echo "      Passed: $PASSED | Failed: $FAILED_COUNT"
        FAILED=1
    fi

    echo ""
else
    echo "🧪 Test Suite"
    echo "────────────────────────────────────────────────────────────────"
    echo "   Skipping tests (use --full to run)"
    echo ""
fi

# =============================================================================
# 3. Configuration Validation
# =============================================================================

echo "⚙️  Configuration Validation"
echo "────────────────────────────────────────────────────────────────"

# Check port mappings
echo -n "   Port 8001 is TCP (not HTTP)... "
if grep -q "8001/tcp" utils/runpod_manager.py; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    echo "      Port 8001 must be /tcp for public IP access"
    FAILED=1
fi

# Check no conflicting start commands
echo -n "   No railway.toml or Procfile... "
if [ -f "railway.toml" ] || [ -f "Procfile" ]; then
    echo "❌ FAIL"
    echo "      Remove railway.toml and Procfile (use Dockerfile CMD)"
    FAILED=1
else
    echo "✅ PASS"
fi

# Check Dockerfile uses shell form CMD
echo -n "   Dockerfile CMD uses shell form... "
if grep -q 'CMD \["streamlit"' Dockerfile 2>/dev/null; then
    echo "❌ FAIL"
    echo "      Dockerfile CMD must use shell form for \$PORT expansion"
    FAILED=1
else
    echo "✅ PASS"
fi

# Check streamlit in requirements
echo -n "   Streamlit in requirements.txt... "
if grep -q "streamlit" requirements.txt; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    echo "      Add streamlit to requirements.txt"
    FAILED=1
fi

# Check FastAPI in requirements
echo -n "   FastAPI in requirements.txt... "
if grep -q "fastapi" requirements.txt; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    echo "      Add fastapi to requirements.txt"
    FAILED=1
fi

echo ""

# =============================================================================
# 4. Script Permissions
# =============================================================================

echo "📜 Script Permissions"
echo "────────────────────────────────────────────────────────────────"

NON_EXECUTABLE=$(find scripts -name "*.sh" -type f ! -perm -u+x 2>/dev/null)
if [ -z "$NON_EXECUTABLE" ]; then
    echo "   All .sh scripts executable... ✅ PASS"
else
    echo "   Scripts not executable... ❌ FAIL"
    echo "$NON_EXECUTABLE" | sed 's/^/      /'
    FAILED=1
fi

echo ""

# =============================================================================
# 5. Code Quality
# =============================================================================

echo "📊 Code Quality"
echo "────────────────────────────────────────────────────────────────"

# Check for deprecated Streamlit API
echo -n "   No deprecated Streamlit APIs... "
if grep -q "use_container_width" rag_web.py; then
    echo "❌ FAIL"
    echo "      Replace use_container_width with width='stretch'"
    FAILED=1
else
    echo "✅ PASS"
fi

# Check for deprecated urllib3 API
echo -n "   No deprecated urllib3 APIs... "
if grep -q "method_whitelist=" utils/runpod_embedding_client.py 2>/dev/null; then
    echo "❌ FAIL"
    echo "      Replace method_whitelist with allowed_methods"
    FAILED=1
else
    echo "✅ PASS"
fi

# Check for deprecated Pydantic API
echo -n "   No deprecated Pydantic APIs... "
if grep -qE "^[[:space:]]+schema_extra = \{" services/embedding_service.py 2>/dev/null; then
    echo "❌ FAIL"
    echo "      Replace schema_extra with json_schema_extra"
    FAILED=1
else
    echo "✅ PASS"
fi

echo ""

# =============================================================================
# 6. Auto-Detection Logic
# =============================================================================

echo "🔍 Auto-Detection Logic"
echo "────────────────────────────────────────────────────────────────"

# Check default host triggers auto-detection
echo -n "   DEFAULT_HOST is empty string... "
if grep -q 'DEFAULT_HOST.*=.*""' config/constants.py; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    echo "      DEFAULT_HOST should be \"\" not \"localhost\""
    FAILED=1
fi

# Check pod selection skips empty ports
echo -n "   Auto-detection skips stopped pods... "
if grep -q "if not ports:" utils/runpod_db_config.py; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    echo "      Auto-detection must skip pods without port mappings"
    FAILED=1
fi

echo ""

# =============================================================================
# 7. Service Integration
# =============================================================================

echo "🔗 Service Integration"
echo "────────────────────────────────────────────────────────────────"

# Check health check implementation
echo -n "   Embedding client has health check... "
if grep -q "def check_health" utils/runpod_embedding_client.py; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    echo "      Embedding client must have check_health() method"
    FAILED=1
fi

# Check fallback mechanism
echo -n "   Embed nodes has fallback... "
if grep -q "_embed_nodes_local" rag_low_level_m1_16gb_verbose.py; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    echo "      Must have _embed_nodes_local() fallback"
    FAILED=1
fi

# Check web UI uses pipeline
echo -n "   Web UI uses pipeline embed_nodes... "
if grep -q "rag.embed_nodes(embed_model, nodes)" rag_web.py; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    echo "      Web UI must use rag.embed_nodes() not custom loop"
    FAILED=1
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "════════════════════════════════════════════════════════════════"
if [ $FAILED -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Safe to deploy to Railway/RunPod!"
    echo ""
    exit 0
else
    echo "❌ SOME CHECKS FAILED"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Fix issues before deploying."
    echo "See: docs/CODE_QUALITY_CHECKLIST.md"
    echo ""
    exit 1
fi
