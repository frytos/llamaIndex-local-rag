#!/bin/bash
#
# Security Scan Script
# Runs comprehensive security checks on the RAG pipeline codebase
#

set -e  # Exit on error

echo "======================================================================"
echo "Security Scan - Local RAG Pipeline"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if in correct directory
if [ ! -f "rag_low_level_m1_16gb_verbose.py" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Create reports directory
mkdir -p security_reports
REPORT_DIR="security_reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"

echo "Reports will be saved to: $REPORT_DIR"
echo ""

# Check 1: Environment file security
echo "======================================================================"
echo "Check 1: Environment File Security"
echo "======================================================================"

if [ -f ".env" ]; then
    echo -e "${GREEN}✓ .env file exists${NC}"

    # Check if .env is in gitignore
    if grep -q "^\.env$" .gitignore 2>/dev/null; then
        echo -e "${GREEN}✓ .env is in .gitignore${NC}"
    else
        echo -e "${RED}✗ WARNING: .env is NOT in .gitignore!${NC}"
        echo "  Add it immediately: echo '.env' >> .gitignore"
    fi

    # Check for weak passwords (basic check)
    if grep -E "PASSWORD=.{1,8}$" .env >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠ WARNING: Potentially weak password detected (< 8 chars)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ .env file not found${NC}"
    echo "  Copy from template: cp config/.env.example .env"
fi

echo ""

# Check 2: Hardcoded credentials
echo "======================================================================"
echo "Check 2: Hardcoded Credentials"
echo "======================================================================"

echo "Scanning for hardcoded passwords..."
HARDCODED=$(grep -r -n --include="*.py" --include="*.yml" --include="*.yaml" \
    -E "(password|passwd|pwd|secret|token|api_key)\s*=\s*['\"][^'\"]{3,}" \
    . 2>/dev/null | grep -v ".venv" | grep -v "node_modules" || echo "")

if [ -z "$HARDCODED" ]; then
    echo -e "${GREEN}✓ No obvious hardcoded credentials found${NC}"
else
    echo -e "${RED}✗ Potential hardcoded credentials found:${NC}"
    echo "$HARDCODED"
fi

echo ""

# Check 3: Python security with Bandit
echo "======================================================================"
echo "Check 3: Static Analysis (Bandit)"
echo "======================================================================"

if command -v bandit &> /dev/null; then
    echo "Running Bandit security scan..."
    bandit -r . \
        --exclude .venv,node_modules,data,benchmarks,logs \
        -f json -o "$REPORT_DIR/bandit-report.json" \
        -ll  # Low and above

    # Generate human-readable report
    bandit -r . \
        --exclude .venv,node_modules,data,benchmarks,logs \
        -f txt -o "$REPORT_DIR/bandit-report.txt" \
        -ll

    # Count issues
    HIGH=$(grep -c '"issue_severity": "HIGH"' "$REPORT_DIR/bandit-report.json" 2>/dev/null || echo "0")
    MEDIUM=$(grep -c '"issue_severity": "MEDIUM"' "$REPORT_DIR/bandit-report.json" 2>/dev/null || echo "0")

    echo ""
    if [ "$HIGH" -eq "0" ] && [ "$MEDIUM" -eq "0" ]; then
        echo -e "${GREEN}✓ No high or medium severity issues found${NC}"
    else
        echo -e "${YELLOW}⚠ Found $HIGH high and $MEDIUM medium severity issues${NC}"
        echo "  Review: $REPORT_DIR/bandit-report.txt"
    fi
else
    echo -e "${YELLOW}⚠ Bandit not installed${NC}"
    echo "  Install: pip install bandit"
fi

echo ""

# Check 4: Dependency vulnerabilities
echo "======================================================================"
echo "Check 4: Dependency Vulnerabilities (pip-audit)"
echo "======================================================================"

if command -v pip-audit &> /dev/null; then
    echo "Running pip-audit..."
    pip-audit --format json --output "$REPORT_DIR/pip-audit.json" || true
    pip-audit --format markdown --output "$REPORT_DIR/pip-audit.md" || true

    VULN_COUNT=$(jq '.vulnerabilities | length' "$REPORT_DIR/pip-audit.json" 2>/dev/null || echo "0")

    echo ""
    if [ "$VULN_COUNT" -eq "0" ]; then
        echo -e "${GREEN}✓ No known vulnerabilities in dependencies${NC}"
    else
        echo -e "${RED}✗ Found $VULN_COUNT vulnerable dependencies${NC}"
        echo "  Review: $REPORT_DIR/pip-audit.md"
    fi
else
    echo -e "${YELLOW}⚠ pip-audit not installed${NC}"
    echo "  Install: pip install pip-audit"
fi

echo ""

# Check 5: Safety check
echo "======================================================================"
echo "Check 5: Known Vulnerabilities (Safety)"
echo "======================================================================"

if command -v safety &> /dev/null; then
    echo "Running Safety check..."
    safety check --json --output "$REPORT_DIR/safety-report.json" || true
    safety check --output "$REPORT_DIR/safety-report.txt" || true

    if [ -s "$REPORT_DIR/safety-report.json" ]; then
        SAFETY_ISSUES=$(jq 'length' "$REPORT_DIR/safety-report.json" 2>/dev/null || echo "0")
        if [ "$SAFETY_ISSUES" -eq "0" ]; then
            echo -e "${GREEN}✓ No known vulnerabilities found${NC}"
        else
            echo -e "${RED}✗ Found $SAFETY_ISSUES vulnerabilities${NC}"
            echo "  Review: $REPORT_DIR/safety-report.txt"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Safety not installed${NC}"
    echo "  Install: pip install safety"
fi

echo ""

# Check 6: SQL injection patterns
echo "======================================================================"
echo "Check 6: SQL Injection Patterns"
echo "======================================================================"

echo "Scanning for potential SQL injection vulnerabilities..."

# Look for f-string SQL queries
SQL_FSTRING=$(grep -r -n --include="*.py" \
    -E "cur\.execute\(f['\"]|cursor\.execute\(f['\"]|execute\(f['\"].*SELECT|INSERT|UPDATE|DELETE" \
    . 2>/dev/null | grep -v ".venv" || echo "")

if [ -z "$SQL_FSTRING" ]; then
    echo -e "${GREEN}✓ No f-string SQL queries found${NC}"
else
    echo -e "${RED}✗ Potential SQL injection vulnerabilities:${NC}"
    echo "$SQL_FSTRING" | tee "$REPORT_DIR/sql-injection-suspects.txt"
fi

echo ""

# Check 7: eval/exec usage
echo "======================================================================"
echo "Check 7: Dangerous Code Execution (eval/exec)"
echo "======================================================================"

echo "Scanning for eval() and exec() usage..."

EVAL_EXEC=$(grep -r -n --include="*.py" \
    -E "\beval\(|\bexec\(" \
    . 2>/dev/null | grep -v ".venv" | grep -v "# Safe:" || echo "")

if [ -z "$EVAL_EXEC" ]; then
    echo -e "${GREEN}✓ No eval() or exec() usage found${NC}"
else
    echo -e "${YELLOW}⚠ Found eval()/exec() usage:${NC}"
    echo "$EVAL_EXEC"
    echo "  Ensure these are safe or replace with ast.literal_eval()"
fi

echo ""

# Check 8: Bare exception handlers
echo "======================================================================"
echo "Check 8: Bare Exception Handlers"
echo "======================================================================"

echo "Scanning for bare except clauses..."

BARE_EXCEPT=$(grep -r -n --include="*.py" \
    -E "^\s+except\s*:\s*$" \
    . 2>/dev/null | grep -v ".venv" || echo "")

if [ -z "$BARE_EXCEPT" ]; then
    echo -e "${GREEN}✓ No bare except clauses found${NC}"
else
    echo -e "${YELLOW}⚠ Found bare except clauses:${NC}"
    EXCEPT_COUNT=$(echo "$BARE_EXCEPT" | wc -l | tr -d ' ')
    echo "  Count: $EXCEPT_COUNT"
    echo "$BARE_EXCEPT" | head -10
    echo ""
    echo "  Full list: saved to $REPORT_DIR/bare-except.txt"
    echo "$BARE_EXCEPT" > "$REPORT_DIR/bare-except.txt"
fi

echo ""

# Check 9: Outdated dependencies
echo "======================================================================"
echo "Check 9: Outdated Dependencies"
echo "======================================================================"

echo "Checking for outdated packages..."
pip list --outdated --format=json > "$REPORT_DIR/outdated-packages.json" 2>/dev/null || true

OUTDATED_COUNT=$(jq 'length' "$REPORT_DIR/outdated-packages.json" 2>/dev/null || echo "0")

if [ "$OUTDATED_COUNT" -eq "0" ]; then
    echo -e "${GREEN}✓ All packages are up to date${NC}"
else
    echo -e "${YELLOW}⚠ $OUTDATED_COUNT packages are outdated${NC}"
    echo "  Review: $REPORT_DIR/outdated-packages.json"
    echo ""
    echo "  Update with: pip install --upgrade -r requirements.txt"
fi

echo ""

# Generate summary report
echo "======================================================================"
echo "Summary Report"
echo "======================================================================"

cat > "$REPORT_DIR/SUMMARY.md" << EOF
# Security Scan Summary

**Date**: $(date)
**Project**: Local RAG Pipeline

## Results

### Critical Issues
$([ -n "$SQL_FSTRING" ] && echo "- ❌ SQL injection vulnerabilities found" || echo "- ✅ No SQL injection vulnerabilities")
$([ -n "$EVAL_EXEC" ] && echo "- ⚠️  eval()/exec() usage found" || echo "- ✅ No eval()/exec() usage")
$([ -n "$HARDCODED" ] && echo "- ❌ Hardcoded credentials found" || echo "- ✅ No hardcoded credentials")

### Security Scans
- Bandit: $HIGH high, $MEDIUM medium severity issues
- pip-audit: $VULN_COUNT vulnerable dependencies
- Safety: $SAFETY_ISSUES known vulnerabilities

### Code Quality
- Bare except clauses: $EXCEPT_COUNT instances
- Outdated packages: $OUTDATED_COUNT packages

## Files Generated
- \`bandit-report.json\` - Static analysis results
- \`pip-audit.json\` - Dependency vulnerability scan
- \`safety-report.json\` - Known vulnerability database
- \`sql-injection-suspects.txt\` - Potential SQL injection issues
- \`bare-except.txt\` - Bare exception handlers
- \`outdated-packages.json\` - Packages needing updates

## Recommendations

1. Review all critical issues immediately
2. Update vulnerable dependencies
3. Fix SQL injection vulnerabilities
4. Replace bare except clauses with specific exceptions
5. Implement Web UI authentication
6. Enable database SSL/TLS connections
7. Set up automated security scanning (CI/CD)

## Next Steps

\`\`\`bash
# Fix critical issues
python scripts/fix_sql_injection.py

# Update dependencies
pip install --upgrade -r requirements.txt

# Re-run scan
./scripts/security_scan.sh
\`\`\`

## Resources

- [Security Guide](../docs/SECURITY_GUIDE.md)
- [Security Fixes Applied](../SECURITY_FIXES_APPLIED.md)
EOF

cat "$REPORT_DIR/SUMMARY.md"

echo ""
echo "======================================================================"
echo "Scan Complete"
echo "======================================================================"
echo ""
echo "Full report saved to: $REPORT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Review $REPORT_DIR/SUMMARY.md"
echo "  2. Fix critical issues"
echo "  3. Update vulnerable dependencies"
echo "  4. Re-run this scan to verify"
echo ""
