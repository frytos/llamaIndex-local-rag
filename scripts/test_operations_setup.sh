#!/usr/bin/env bash
# test_operations_setup.sh - Verify operations infrastructure setup
#
# Tests:
#   - Backup scripts exist and are executable
#   - Monitoring configuration files exist
#   - Health check module works
#   - Metrics module works
#   - Docker compose configuration is valid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

pass() {
    echo -e "${GREEN}✓${NC} $*"
    ((TESTS_PASSED++))
}

fail() {
    echo -e "${RED}✗${NC} $*"
    ((TESTS_FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $*"
}

section() {
    echo ""
    echo "======================================"
    echo "$*"
    echo "======================================"
}

# Test 1: Backup scripts
test_backup_scripts() {
    section "Testing Backup Scripts"

    # Check backup script exists
    if [ -f "$PROJECT_ROOT/scripts/backup/backup_postgres.sh" ]; then
        pass "backup_postgres.sh exists"
    else
        fail "backup_postgres.sh not found"
    fi

    # Check executable
    if [ -x "$PROJECT_ROOT/scripts/backup/backup_postgres.sh" ]; then
        pass "backup_postgres.sh is executable"
    else
        fail "backup_postgres.sh is not executable"
    fi

    # Check verify script
    if [ -f "$PROJECT_ROOT/scripts/backup/verify_backup.sh" ]; then
        pass "verify_backup.sh exists"
    else
        fail "verify_backup.sh not found"
    fi

    # Check setup cron script
    if [ -f "$PROJECT_ROOT/scripts/backup/setup_cron.sh" ]; then
        pass "setup_cron.sh exists"
    else
        fail "setup_cron.sh not found"
    fi

    # Check README
    if [ -f "$PROJECT_ROOT/scripts/backup/README.md" ]; then
        pass "Backup README.md exists"
    else
        fail "Backup README.md not found"
    fi
}

# Test 2: Monitoring configuration
test_monitoring_config() {
    section "Testing Monitoring Configuration"

    # Check docker-compose.yml
    if [ -f "$PROJECT_ROOT/config/docker-compose.yml" ]; then
        pass "docker-compose.yml exists"

        # Validate docker-compose syntax
        if cd "$PROJECT_ROOT/config" && docker-compose config > /dev/null 2>&1; then
            pass "docker-compose.yml is valid"
        else
            fail "docker-compose.yml has syntax errors"
        fi
    else
        fail "docker-compose.yml not found"
    fi

    # Check prometheus config
    if [ -f "$PROJECT_ROOT/config/monitoring/prometheus.yml" ]; then
        pass "prometheus.yml exists"
    else
        fail "prometheus.yml not found"
    fi

    # Check alerts config
    if [ -f "$PROJECT_ROOT/config/monitoring/alerts.yml" ]; then
        pass "alerts.yml exists"
    else
        fail "alerts.yml not found"
    fi

    # Check alertmanager config
    if [ -f "$PROJECT_ROOT/config/monitoring/alertmanager.yml" ]; then
        pass "alertmanager.yml exists"
    else
        fail "alertmanager.yml not found"
    fi
}

# Test 3: Grafana configuration
test_grafana_config() {
    section "Testing Grafana Configuration"

    # Check provisioning directories
    if [ -d "$PROJECT_ROOT/config/grafana/provisioning" ]; then
        pass "Grafana provisioning directory exists"
    else
        fail "Grafana provisioning directory not found"
    fi

    # Check datasource provisioning
    if [ -f "$PROJECT_ROOT/config/grafana/provisioning/datasources/prometheus.yml" ]; then
        pass "Datasource provisioning exists"
    else
        fail "Datasource provisioning not found"
    fi

    # Check dashboard provisioning
    if [ -f "$PROJECT_ROOT/config/grafana/provisioning/dashboards/default.yml" ]; then
        pass "Dashboard provisioning exists"
    else
        fail "Dashboard provisioning not found"
    fi

    # Check dashboard JSON
    if [ -f "$PROJECT_ROOT/config/grafana/dashboards/rag_overview.json" ]; then
        pass "RAG overview dashboard exists"

        # Validate JSON
        if python3 -m json.tool "$PROJECT_ROOT/config/grafana/dashboards/rag_overview.json" > /dev/null 2>&1; then
            pass "Dashboard JSON is valid"
        else
            fail "Dashboard JSON is invalid"
        fi
    else
        fail "RAG overview dashboard not found"
    fi
}

# Test 4: Health check module
test_health_check() {
    section "Testing Health Check Module"

    if [ -f "$PROJECT_ROOT/utils/health_check.py" ]; then
        pass "health_check.py exists"

        # Check syntax
        if python3 -m py_compile "$PROJECT_ROOT/utils/health_check.py" 2>/dev/null; then
            pass "health_check.py syntax is valid"
        else
            fail "health_check.py has syntax errors"
        fi
    else
        fail "health_check.py not found"
    fi
}

# Test 5: Metrics module
test_metrics_module() {
    section "Testing Metrics Module"

    if [ -f "$PROJECT_ROOT/utils/metrics.py" ]; then
        pass "metrics.py exists"

        # Check syntax
        if python3 -m py_compile "$PROJECT_ROOT/utils/metrics.py" 2>/dev/null; then
            pass "metrics.py syntax is valid"
        else
            fail "metrics.py has syntax errors"
        fi
    else
        fail "metrics.py not found"
    fi
}

# Test 6: Documentation
test_documentation() {
    section "Testing Documentation"

    if [ -f "$PROJECT_ROOT/docs/OPERATIONS.md" ]; then
        pass "OPERATIONS.md exists"
    else
        fail "OPERATIONS.md not found"
    fi

    if [ -f "$PROJECT_ROOT/scripts/backup/README.md" ]; then
        pass "Backup README.md exists"
    else
        fail "Backup README.md not found"
    fi
}

# Test 7: Helper scripts
test_helper_scripts() {
    section "Testing Helper Scripts"

    if [ -f "$PROJECT_ROOT/scripts/start_monitoring.sh" ]; then
        pass "start_monitoring.sh exists"

        if [ -x "$PROJECT_ROOT/scripts/start_monitoring.sh" ]; then
            pass "start_monitoring.sh is executable"
        else
            fail "start_monitoring.sh is not executable"
        fi
    else
        fail "start_monitoring.sh not found"
    fi
}

# Test 8: Directory structure
test_directories() {
    section "Testing Directory Structure"

    local dirs=(
        "config/monitoring"
        "config/grafana/provisioning/datasources"
        "config/grafana/provisioning/dashboards"
        "config/grafana/dashboards"
        "scripts/backup"
        "utils"
        "docs"
    )

    for dir in "${dirs[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            pass "Directory exists: $dir"
        else
            fail "Directory missing: $dir"
        fi
    done
}

# Test 9: Docker services configuration
test_docker_services() {
    section "Testing Docker Services Configuration"

    cd "$PROJECT_ROOT/config"

    # Check if docker-compose defines required services
    local services=(
        "db"
        "prometheus"
        "grafana"
        "alertmanager"
        "node_exporter"
        "postgres_exporter"
        "cadvisor"
        "backup"
    )

    for service in "${services[@]}"; do
        if docker-compose config | grep -q "^  $service:"; then
            pass "Service defined: $service"
        else
            fail "Service missing: $service"
        fi
    done
}

# Test 10: Alert rules validation
test_alert_rules() {
    section "Testing Alert Rules"

    local alerts_file="$PROJECT_ROOT/config/monitoring/alerts.yml"

    if [ -f "$alerts_file" ]; then
        # Check for critical alert definitions
        local critical_alerts=(
            "PostgreSQLDown"
            "BackupFailed"
            "HighMemoryUsage"
            "DiskSpaceLow"
        )

        for alert in "${critical_alerts[@]}"; do
            if grep -q "alert: $alert" "$alerts_file"; then
                pass "Alert defined: $alert"
            else
                fail "Alert missing: $alert"
            fi
        done
    else
        fail "alerts.yml not found"
    fi
}

# Summary
show_summary() {
    section "Test Summary"

    echo "Tests Passed: $TESTS_PASSED"
    echo "Tests Failed: $TESTS_FAILED"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Start monitoring stack: ./scripts/start_monitoring.sh"
        echo "  2. Setup automated backups: ./scripts/backup/setup_cron.sh"
        echo "  3. Open Grafana: http://localhost:3000 (admin/admin)"
        echo ""
        return 0
    else
        echo -e "${RED}Some tests failed. Please review the output above.${NC}"
        return 1
    fi
}

# Main
main() {
    echo "======================================"
    echo "Operations Infrastructure Test Suite"
    echo "======================================"
    echo ""

    test_backup_scripts
    test_monitoring_config
    test_grafana_config
    test_health_check
    test_metrics_module
    test_documentation
    test_helper_scripts
    test_directories
    test_docker_services
    test_alert_rules

    echo ""
    show_summary
}

main
