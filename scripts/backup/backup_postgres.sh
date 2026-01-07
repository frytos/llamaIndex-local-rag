#!/usr/bin/env bash
# backup_postgres.sh - Automated PostgreSQL + pgvector backup with retention and verification
#
# Usage:
#   ./backup_postgres.sh                    # Full backup with default settings
#   ./backup_postgres.sh --retention 14     # Keep backups for 14 days
#   ./backup_postgres.sh --verify           # Verify last backup
#
# Features:
#   - Full database backup with pg_dump
#   - Compression (gzip) to save disk space
#   - Retention policy (default 7 days)
#   - Backup verification
#   - Metrics export for Prometheus
#   - Error handling and logging

set -euo pipefail

# ==================== Configuration ====================

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups/postgres}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
METRICS_DIR="${METRICS_DIR:-$PROJECT_ROOT/metrics}"

# Retention (days)
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"

# Database connection (load from .env if available)
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-fryt}"
PGPASSWORD="${PGPASSWORD:-frytos}"
DB_NAME="${DB_NAME:-vector_db}"

# Backup filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="vector_db_${TIMESTAMP}.sql.gz"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILE"

# Logging
LOG_FILE="$LOG_DIR/backup_$(date +"%Y%m%d").log"

# ==================== Functions ====================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

export_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local metric_type="${3:-gauge}"
    local help_text="${4:-}"

    mkdir -p "$METRICS_DIR"
    local metric_file="$METRICS_DIR/backup.prom"

    {
        if [ -n "$help_text" ]; then
            echo "# HELP $metric_name $help_text"
        fi
        echo "# TYPE $metric_name $metric_type"
        echo "$metric_name $metric_value"
    } >> "$metric_file.tmp"
}

finalize_metrics() {
    if [ -f "$METRICS_DIR/backup.prom.tmp" ]; then
        mv "$METRICS_DIR/backup.prom.tmp" "$METRICS_DIR/backup.prom"
    fi
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check pg_dump
    if ! command -v pg_dump &> /dev/null; then
        error "pg_dump not found. Install PostgreSQL client tools."
        exit 1
    fi

    # Create directories
    mkdir -p "$BACKUP_DIR" "$LOG_DIR" "$METRICS_DIR"

    # Test database connection
    if ! PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$DB_NAME" -c "SELECT 1" &>/dev/null; then
        error "Cannot connect to database: $DB_NAME@$PGHOST:$PGPORT"
        export_metric "rag_backup_success" 0
        export_metric "rag_backup_last_failure_timestamp" "$(date +%s)"
        finalize_metrics
        exit 1
    fi

    log "Prerequisites OK"
}

perform_backup() {
    log "Starting backup: $BACKUP_FILE"

    local start_time=$(date +%s)
    local backup_success=0
    local backup_size=0

    # Perform backup with progress
    if PGPASSWORD="$PGPASSWORD" pg_dump \
        -h "$PGHOST" \
        -p "$PGPORT" \
        -U "$PGUSER" \
        -d "$DB_NAME" \
        --format=plain \
        --no-owner \
        --no-acl \
        --verbose 2>&1 | gzip > "$BACKUP_PATH"; then

        backup_success=1
        backup_size=$(stat -f%z "$BACKUP_PATH" 2>/dev/null || stat -c%s "$BACKUP_PATH" 2>/dev/null || echo 0)
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        log "Backup completed successfully"
        log "  File: $BACKUP_PATH"
        log "  Size: $(numfmt --to=iec-i --suffix=B $backup_size 2>/dev/null || echo "$backup_size bytes")"
        log "  Duration: ${duration}s"

        # Export metrics
        export_metric "rag_backup_success" 1 "gauge" "Last backup success status (1=success, 0=failure)"
        export_metric "rag_backup_last_success_timestamp" "$end_time" "gauge" "Timestamp of last successful backup"
        export_metric "rag_backup_duration_seconds" "$duration" "gauge" "Duration of last backup in seconds"
        export_metric "rag_backup_size_bytes" "$backup_size" "gauge" "Size of last backup in bytes"

        # Write backup metadata
        cat > "${BACKUP_PATH}.meta" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "database": "$DB_NAME",
  "host": "$PGHOST",
  "port": $PGPORT,
  "size_bytes": $backup_size,
  "duration_seconds": $duration,
  "backup_file": "$BACKUP_FILE",
  "success": true
}
EOF
    else
        error "Backup failed"
        export_metric "rag_backup_success" 0
        export_metric "rag_backup_last_failure_timestamp" "$(date +%s)"
        finalize_metrics
        exit 1
    fi
}

verify_backup() {
    local backup_to_verify="${1:-$BACKUP_PATH}"

    if [ ! -f "$backup_to_verify" ]; then
        error "Backup file not found: $backup_to_verify"
        return 1
    fi

    log "Verifying backup: $(basename "$backup_to_verify")"

    # Test gzip integrity
    if ! gzip -t "$backup_to_verify" 2>/dev/null; then
        error "Backup file is corrupted (gzip test failed)"
        export_metric "rag_backup_verification_success" 0
        return 1
    fi

    # Check SQL content
    local sql_check=$(zcat "$backup_to_verify" | head -n 20)
    if ! echo "$sql_check" | grep -q "PostgreSQL"; then
        error "Backup file does not appear to be a valid PostgreSQL dump"
        export_metric "rag_backup_verification_success" 0
        return 1
    fi

    log "Backup verification: OK"
    export_metric "rag_backup_verification_success" 1 "gauge" "Last backup verification status"
    return 0
}

cleanup_old_backups() {
    log "Cleaning up backups older than ${RETENTION_DAYS} days..."

    local deleted_count=0
    local kept_count=0

    while IFS= read -r -d '' backup_file; do
        if [ -f "$backup_file" ]; then
            local age_days=$(( ($(date +%s) - $(stat -f%m "$backup_file" 2>/dev/null || stat -c%Y "$backup_file")) / 86400 ))

            if [ "$age_days" -gt "$RETENTION_DAYS" ]; then
                log "  Deleting old backup: $(basename "$backup_file") (${age_days} days old)"
                rm -f "$backup_file" "${backup_file}.meta"
                ((deleted_count++))
            else
                ((kept_count++))
            fi
        fi
    done < <(find "$BACKUP_DIR" -name "vector_db_*.sql.gz" -print0 2>/dev/null || true)

    log "Cleanup complete: deleted=$deleted_count, kept=$kept_count"
    export_metric "rag_backup_total_count" "$kept_count" "gauge" "Total number of backups retained"
}

list_backups() {
    log "Available backups in $BACKUP_DIR:"

    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]; then
        log "  No backups found"
        return
    fi

    local total_size=0
    while IFS= read -r backup_file; do
        local size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null || echo 0)
        local date=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$backup_file" 2>/dev/null || stat -c "%y" "$backup_file" 2>/dev/null | cut -d. -f1)
        local human_size=$(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "$size bytes")

        log "  $(basename "$backup_file"): $human_size ($date)"
        total_size=$((total_size + size))
    done < <(find "$BACKUP_DIR" -name "vector_db_*.sql.gz" -type f 2>/dev/null | sort -r)

    local human_total=$(numfmt --to=iec-i --suffix=B $total_size 2>/dev/null || echo "$total_size bytes")
    log "  Total backup size: $human_total"
}

restore_backup() {
    local backup_to_restore="$1"

    if [ ! -f "$backup_to_restore" ]; then
        error "Backup file not found: $backup_to_restore"
        exit 1
    fi

    log "WARNING: This will OVERWRITE the current database: $DB_NAME"
    read -p "Are you sure? Type 'yes' to continue: " confirmation

    if [ "$confirmation" != "yes" ]; then
        log "Restore cancelled"
        exit 0
    fi

    log "Restoring backup: $(basename "$backup_to_restore")"

    # Drop existing database and recreate
    PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres <<EOF
DROP DATABASE IF EXISTS ${DB_NAME};
CREATE DATABASE ${DB_NAME};
EOF

    # Restore backup
    if zcat "$backup_to_restore" | PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$DB_NAME" > /dev/null; then
        log "Restore completed successfully"

        # Re-enable pgvector extension
        PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"

        export_metric "rag_backup_last_restore_timestamp" "$(date +%s)"
    else
        error "Restore failed"
        exit 1
    fi
}

show_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Automated PostgreSQL + pgvector backup with verification and retention.

OPTIONS:
    --retention DAYS    Set retention period in days (default: 7)
    --verify [FILE]     Verify backup integrity
    --list              List all available backups
    --restore FILE      Restore from backup file
    --help              Show this help message

EXAMPLES:
    # Perform backup
    $(basename "$0")

    # Set 14-day retention
    $(basename "$0") --retention 14

    # Verify last backup
    $(basename "$0") --verify

    # List all backups
    $(basename "$0") --list

    # Restore from specific backup
    $(basename "$0") --restore backups/postgres/vector_db_20260107_120000.sql.gz

ENVIRONMENT VARIABLES:
    PGHOST              Database host (default: localhost)
    PGPORT              Database port (default: 5432)
    PGUSER              Database user (default: fryt)
    PGPASSWORD          Database password (default: frytos)
    DB_NAME             Database name (default: vector_db)
    BACKUP_DIR          Backup directory (default: ./backups/postgres)
    BACKUP_RETENTION_DAYS  Retention period (default: 7)

EOF
}

# ==================== Main ====================

main() {
    local mode="backup"
    local verify_file=""
    local restore_file=""

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --verify)
                mode="verify"
                verify_file="${2:-}"
                shift
                if [ -n "$verify_file" ] && [ "${verify_file:0:1}" != "-" ]; then
                    shift
                fi
                ;;
            --list)
                mode="list"
                shift
                ;;
            --restore)
                mode="restore"
                restore_file="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Execute mode
    case "$mode" in
        backup)
            log "===== PostgreSQL Backup Started ====="
            check_prerequisites
            perform_backup
            verify_backup
            cleanup_old_backups
            list_backups
            finalize_metrics
            log "===== Backup Complete ====="
            ;;
        verify)
            if [ -z "$verify_file" ]; then
                # Find most recent backup
                verify_file=$(find "$BACKUP_DIR" -name "vector_db_*.sql.gz" -type f 2>/dev/null | sort -r | head -n1)
            fi
            verify_backup "$verify_file"
            finalize_metrics
            ;;
        list)
            list_backups
            ;;
        restore)
            restore_backup "$restore_file"
            ;;
    esac
}

main "$@"
