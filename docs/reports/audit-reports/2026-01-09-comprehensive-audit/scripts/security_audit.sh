#!/bin/bash
# Automated Security Audit Script
# Scans for common security vulnerabilities in the RAG pipeline
# Usage: ./scripts/security_audit.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=0

echo -e "${BLUE}================================================"
echo "Security Audit - RAG Pipeline"
echo "================================================${NC}"
echo ""
echo "Starting security scan at $(date)"
echo ""

# 1. Check for hardcoded credentials
echo -e "${BLUE}[1/10] Checking for hardcoded credentials...${NC}"
if grep -rn "password\|secret\|api_key\|token" . \
    --include="*.sh" --include="*.py" --include="*.yml" \
    --exclude-dir=".venv" --exclude-dir="node_modules" --exclude-dir=".git" \
    | grep -v ".example\|.md\|def password\|validate_password\|type=\"password\"\|PGPASSWORD}" \
    | grep -E "password=|secret=|key=|token=" > /tmp/hardcoded_creds.txt 2>/dev/null; then
    echo -e "${RED}✗ CRITICAL: Hardcoded credentials found:${NC}"
    cat /tmp/hardcoded_creds.txt | head -10
    CRITICAL=$((CRITICAL + 1))
else
    echo -e "${GREEN}✓ No hardcoded credentials detected${NC}"
fi
echo ""

# 2. Check for authentication implementation
echo -e "${BLUE}[2/10] Checking for authentication in web interfaces...${NC}"
if ! grep -q "authenticate\|check_auth\|login" rag_web.py 2>/dev/null && \
   ! grep -q "authenticate\|check_auth\|login" rag_web_enhanced.py 2>/dev/null; then
    echo -e "${RED}✗ CRITICAL: No authentication found in web interfaces${NC}"
    echo "  Files: rag_web.py, rag_web_enhanced.py"
    CRITICAL=$((CRITICAL + 1))
else
    echo -e "${GREEN}✓ Authentication implementation detected${NC}"
fi
echo ""

# 3. Check for SQL injection protection
echo -e "${BLUE}[3/10] Checking SQL query safety...${NC}"
if grep -rn "execute.*format\|execute.*%" . \
    --include="*.py" --exclude-dir=".venv" \
    | grep -v "sql.SQL\|sql.Identifier" > /tmp/sql_unsafe.txt 2>/dev/null; then
    echo -e "${YELLOW}⚠ HIGH: Potential SQL injection vulnerabilities:${NC}"
    cat /tmp/sql_unsafe.txt | head -5
    HIGH=$((HIGH + 1))
else
    echo -e "${GREEN}✓ SQL queries appear to use parameterized statements${NC}"
fi
echo ""

# 4. Check for path traversal protection
echo -e "${BLUE}[4/10] Checking path traversal protection...${NC}"
if grep -rn "Path(.*input\|open(.*input" . \
    --include="*.py" --exclude-dir=".venv" \
    | grep -v "validate.*path" > /tmp/path_traversal.txt 2>/dev/null; then
    echo -e "${YELLOW}⚠ HIGH: Potential path traversal vulnerabilities:${NC}"
    cat /tmp/path_traversal.txt | head -5
    HIGH=$((HIGH + 1))
else
    echo -e "${GREEN}✓ Path operations appear validated${NC}"
fi
echo ""

# 5. Check for rate limiting
echo -e "${BLUE}[5/10] Checking for rate limiting...${NC}"
if ! grep -q "rate_limit\|RateLimiter" rag_web.py 2>/dev/null && \
   ! grep -q "rate_limit\|RateLimiter" rag_web_enhanced.py 2>/dev/null; then
    echo -e "${YELLOW}⚠ HIGH: No rate limiting detected in web interfaces${NC}"
    HIGH=$((HIGH + 1))
else
    echo -e "${GREEN}✓ Rate limiting implementation found${NC}"
fi
echo ""

# 6. Check for security logging
echo -e "${BLUE}[6/10] Checking security logging...${NC}"
if ! grep -rq "security.*log\|SecurityLogger" . --include="*.py" --exclude-dir=".venv"; then
    echo -e "${YELLOW}⚠ MEDIUM: No security logging implementation found${NC}"
    MEDIUM=$((MEDIUM + 1))
else
    echo -e "${GREEN}✓ Security logging detected${NC}"
fi
echo ""

# 7. Check for HTTPS/TLS configuration
echo -e "${BLUE}[7/10] Checking HTTPS/TLS configuration...${NC}"
if [ ! -f "nginx.conf" ] && [ ! -f "config/nginx.conf" ]; then
    echo -e "${YELLOW}⚠ MEDIUM: No HTTPS/TLS configuration found${NC}"
    echo "  Recommendation: Add nginx reverse proxy with SSL"
    MEDIUM=$((MEDIUM + 1))
else
    echo -e "${GREEN}✓ Reverse proxy configuration exists${NC}"
fi
echo ""

# 8. Check Docker security
echo -e "${BLUE}[8/10] Checking Docker security configuration...${NC}"
if [ -f "config/docker-compose.yml" ]; then
    # Check for secrets
    if ! grep -q "secrets:" config/docker-compose.yml; then
        echo -e "${RED}✗ CRITICAL: Docker secrets not configured${NC}"
        CRITICAL=$((CRITICAL + 1))
    else
        echo -e "${GREEN}✓ Docker secrets configured${NC}"
    fi

    # Check for root user
    if ! grep -q "user:" config/docker-compose.yml; then
        echo -e "${YELLOW}⚠ HIGH: Containers may run as root${NC}"
        HIGH=$((HIGH + 1))
    else
        echo -e "${GREEN}✓ Non-root user configured${NC}"
    fi

    # Check for SSL mode
    if grep -q "sslmode=disable" config/docker-compose.yml; then
        echo -e "${YELLOW}⚠ HIGH: PostgreSQL SSL disabled${NC}"
        HIGH=$((HIGH + 1))
    else
        echo -e "${GREEN}✓ PostgreSQL SSL enabled or not explicitly disabled${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Docker Compose file not found${NC}"
fi
echo ""

# 9. Check .env file security
echo -e "${BLUE}[9/10] Checking environment variable security...${NC}"
if [ -f ".env" ]; then
    # Check permissions
    PERMS=$(stat -f "%A" .env 2>/dev/null || stat -c "%a" .env 2>/dev/null)
    if [ "$PERMS" != "600" ] && [ "$PERMS" != "400" ]; then
        echo -e "${YELLOW}⚠ MEDIUM: .env file has weak permissions: $PERMS${NC}"
        echo "  Recommendation: chmod 600 .env"
        MEDIUM=$((MEDIUM + 1))
    else
        echo -e "${GREEN}✓ .env file has correct permissions${NC}"
    fi
else
    echo -e "${YELLOW}⚠ .env file not found (may be intentional)${NC}"
fi

# Check if .env is in gitignore
if [ -f ".gitignore" ]; then
    if grep -q "^\.env$" .gitignore; then
        echo -e "${GREEN}✓ .env is properly gitignored${NC}"
    else
        echo -e "${RED}✗ CRITICAL: .env not in .gitignore!${NC}"
        CRITICAL=$((CRITICAL + 1))
    fi
fi
echo ""

# 10. Check dependency vulnerabilities
echo -e "${BLUE}[10/10] Checking dependency vulnerabilities...${NC}"
if command -v pip-audit &> /dev/null; then
    echo "Running pip-audit..."
    if pip-audit -r requirements.txt --desc 2>&1 | grep -q "Found.*vulnerabilities"; then
        echo -e "${YELLOW}⚠ MEDIUM: Vulnerable dependencies found${NC}"
        pip-audit -r requirements.txt --desc 2>&1 | tail -20
        MEDIUM=$((MEDIUM + 1))
    else
        echo -e "${GREEN}✓ No known vulnerabilities in dependencies${NC}"
    fi
elif command -v safety &> /dev/null; then
    echo "Running safety check..."
    if safety check -r requirements.txt --json 2>&1 | grep -q "vulnerabilities_found"; then
        echo -e "${YELLOW}⚠ MEDIUM: Vulnerable dependencies found${NC}"
        safety check -r requirements.txt 2>&1 | tail -20
        MEDIUM=$((MEDIUM + 1))
    else
        echo -e "${GREEN}✓ No known vulnerabilities in dependencies${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Dependency scanning tools not installed${NC}"
    echo "  Install with: pip install pip-audit safety"
    LOW=$((LOW + 1))
fi
echo ""

# Summary
echo -e "${BLUE}================================================"
echo "Security Audit Summary"
echo "================================================${NC}"
echo ""
echo -e "${RED}Critical Issues:  $CRITICAL${NC}"
echo -e "${YELLOW}High Priority:    $HIGH${NC}"
echo -e "${YELLOW}Medium Priority:  $MEDIUM${NC}"
echo -e "${GREEN}Low Priority:     $LOW${NC}"
echo ""

TOTAL=$((CRITICAL + HIGH + MEDIUM + LOW))
if [ $CRITICAL -gt 0 ]; then
    echo -e "${RED}✗ AUDIT FAILED - Critical vulnerabilities found!${NC}"
    echo "Review SECURITY_AUDIT_REPORT.md for detailed remediation steps"
    exit 1
elif [ $HIGH -gt 0 ]; then
    echo -e "${YELLOW}⚠ AUDIT WARNING - High priority issues found${NC}"
    echo "Address high priority issues before production deployment"
    exit 1
elif [ $TOTAL -gt 0 ]; then
    echo -e "${YELLOW}⚠ AUDIT PASSED with warnings${NC}"
    echo "Consider addressing medium/low priority issues"
    exit 0
else
    echo -e "${GREEN}✓ AUDIT PASSED - No security issues detected${NC}"
    exit 0
fi
