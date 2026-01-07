#!/usr/bin/env bash
# verify_backup.sh - Comprehensive backup verification and testing
#
# Usage:
#   ./verify_backup.sh [backup_file]      # Verify specific backup
#   ./verify_backup.sh --all              # Verify all backups
#   ./verify_backup.sh --test-restore     # Test restore in temp database

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups/postgres}"

# Load environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-fryt}"
PGPASSWORD="${PGPASSWORD:-frytos}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

verify_backup_integrity() {
    local backup_file="$1"
    local checks_passed=0
    local checks_total=5

    log "Verifying backup: $(basename "$backup_file")"

    # Check 1: File exists and readable
    if [ -f "$backup_file" ] && [ -r "$backup_file" ]; then
        log "  [OK] File exists and readable"
        ((checks_passed++))
    else
        error "  [FAIL] File not found or not readable"
        return 1
    fi

    # Check 2: File size > 1KB
    local size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null || echo 0)
    if [ "$size" -gt 1024 ]; then
        log "  [OK] File size: $(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "$size bytes")"
        ((checks_passed++))
    else
        error "  [FAIL] File too small: $size bytes"
        return 1
    fi

    # Check 3: Gzip integrity
    if gzip -t "$backup_file" 2>/dev/null; then
        log "  [OK] Gzip compression valid"
        ((checks_passed++))
    else
        error "  [FAIL] Gzip corruption detected"
        return 1
    fi

    # Check 4: SQL header
    local header=$(zcat "$backup_file" | head -n 10)
    if echo "$header" | grep -q "PostgreSQL"; then
        log "  [OK] PostgreSQL dump format detected"
        ((checks_passed++))
    else
        error "  [FAIL] Not a valid PostgreSQL dump"
        return 1
    fi

    # Check 5: pgvector extension present
    if zcat "$backup_file" | grep -q "CREATE EXTENSION.*vector"; then
        log "  [OK] pgvector extension found"
        ((checks_passed++))
    else
        log "  [WARN] pgvector extension not found in backup"
    fi

    # Summary
    log "  Verification: $checks_passed/$checks_total checks passed"

    if [ "$checks_passed" -ge 4 ]; then
        log "  [RESULT] Backup is VALID"
        return 0
    else
        error "  [RESULT] Backup is INVALID"
        return 1
    fi
}

test_restore() {
    local backup_file="$1"
    local test_db="vector_db_test_$(date +%s)"

    log "Testing restore to temporary database: $test_db"

    # Create test database
    PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -c "CREATE DATABASE $test_db;" 2>/dev/null || {
        error "Failed to create test database"
        return 1
    }

    # Restore backup
    log "  Restoring backup..."
    if zcat "$backup_file" | PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$test_db" > /dev/null 2>&1; then
        log "  [OK] Restore successful"

        # Verify data
        local table_count=$(PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$test_db" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null | tr -d ' ')
        log "  [OK] Tables found: $table_count"

        # Check for data_* tables (vector storage)
        local vector_tables=$(PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$test_db" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE 'data_%';" 2>/dev/null | tr -d ' ')
        log "  [OK] Vector tables: $vector_tables"

        # Cleanup
        PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -c "DROP DATABASE $test_db;" 2>/dev/null

        log "  [RESULT] Test restore SUCCESSFUL"
        return 0
    else
        error "  [FAIL] Restore failed"
        PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -c "DROP DATABASE IF EXISTS $test_db;" 2>/dev/null
        return 1
    fi
}

verify_all_backups() {
    log "Verifying all backups in $BACKUP_DIR"

    local total=0
    local passed=0
    local failed=0

    while IFS= read -r backup_file; do
        ((total++))
        if verify_backup_integrity "$backup_file"; then
            ((passed++))
        else
            ((failed++))
        fi
        echo ""
    done < <(find "$BACKUP_DIR" -name "vector_db_*.sql.gz" -type f 2>/dev/null | sort -r)

    log "===== Verification Summary ====="
    log "Total backups: $total"
    log "Passed: $passed"
    log "Failed: $failed"
}

show_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [BACKUP_FILE]

Verify backup integrity and test restore functionality.

OPTIONS:
    --all               Verify all backups in backup directory
    --test-restore      Test restore in temporary database
    --help              Show this help message

EXAMPLES:
    # Verify specific backup
    $(basename "$0") backups/postgres/vector_db_20260107_120000.sql.gz

    # Verify most recent backup
    $(basename "$0")

    # Verify all backups
    $(basename "$0") --all

    # Test restore
    $(basename "$0") --test-restore backups/postgres/vector_db_20260107_120000.sql.gz

EOF
}

main() {
    local mode="single"
    local backup_file=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --all)
                mode="all"
                shift
                ;;
            --test-restore)
                mode="test-restore"
                backup_file="${2:-}"
                shift
                if [ -n "$backup_file" ]; then shift; fi
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                backup_file="$1"
                shift
                ;;
        esac
    done

    case "$mode" in
        all)
            verify_all_backups
            ;;
        test-restore)
            if [ -z "$backup_file" ]; then
                backup_file=$(find "$BACKUP_DIR" -name "vector_db_*.sql.gz" -type f 2>/dev/null | sort -r | head -n1)
            fi
            verify_backup_integrity "$backup_file" && test_restore "$backup_file"
            ;;
        single)
            if [ -z "$backup_file" ]; then
                backup_file=$(find "$BACKUP_DIR" -name "vector_db_*.sql.gz" -type f 2>/dev/null | sort -r | head -n1)
            fi
            verify_backup_integrity "$backup_file"
            ;;
    esac
}

main "$@"
