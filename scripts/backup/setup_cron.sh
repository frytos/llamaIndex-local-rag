#!/usr/bin/env bash
# setup_cron.sh - Configure automated backup cron job
#
# Usage:
#   ./setup_cron.sh                    # Setup daily backup at 2 AM
#   ./setup_cron.sh --time "0 3 * * *" # Custom schedule
#   ./setup_cron.sh --remove           # Remove cron job

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default cron schedule: 2 AM daily
DEFAULT_SCHEDULE="0 2 * * *"
CRON_SCHEDULE="${DEFAULT_SCHEDULE}"
BACKUP_SCRIPT="${SCRIPT_DIR}/backup_postgres.sh"
CRON_JOB_MARKER="# RAG Pipeline Automated Backup"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

check_prerequisites() {
    # Check if backup script exists
    if [ ! -f "$BACKUP_SCRIPT" ]; then
        error "Backup script not found: $BACKUP_SCRIPT"
        exit 1
    fi

    # Make backup script executable
    chmod +x "$BACKUP_SCRIPT"

    log "Prerequisites OK"
}

add_cron_job() {
    log "Setting up cron job: $CRON_SCHEDULE"

    # Get current crontab (ignore errors if empty)
    current_crontab=$(crontab -l 2>/dev/null || echo "")

    # Check if job already exists
    if echo "$current_crontab" | grep -q "$CRON_JOB_MARKER"; then
        log "Cron job already exists. Removing old entry..."
        current_crontab=$(echo "$current_crontab" | grep -v "$CRON_JOB_MARKER" | grep -v "$BACKUP_SCRIPT" || echo "")
    fi

    # Add new cron job
    new_crontab="$current_crontab
$CRON_JOB_MARKER
$CRON_SCHEDULE $BACKUP_SCRIPT >> $PROJECT_ROOT/logs/backup_cron.log 2>&1
"

    # Install new crontab
    echo "$new_crontab" | crontab -

    log "Cron job installed successfully"
    log "Backup will run: $CRON_SCHEDULE"
    log "Logs: $PROJECT_ROOT/logs/backup_cron.log"

    # Create log directory
    mkdir -p "$PROJECT_ROOT/logs"
}

remove_cron_job() {
    log "Removing cron job..."

    # Get current crontab
    current_crontab=$(crontab -l 2>/dev/null || echo "")

    # Remove job
    if echo "$current_crontab" | grep -q "$CRON_JOB_MARKER"; then
        new_crontab=$(echo "$current_crontab" | grep -v "$CRON_JOB_MARKER" | grep -v "$BACKUP_SCRIPT" || echo "")
        echo "$new_crontab" | crontab -
        log "Cron job removed successfully"
    else
        log "No cron job found"
    fi
}

show_status() {
    log "Checking cron job status..."

    current_crontab=$(crontab -l 2>/dev/null || echo "")

    if echo "$current_crontab" | grep -q "$CRON_JOB_MARKER"; then
        log "Cron job is ACTIVE:"
        echo "$current_crontab" | grep -A1 "$CRON_JOB_MARKER"

        # Show last run (if log exists)
        log_file="$PROJECT_ROOT/logs/backup_cron.log"
        if [ -f "$log_file" ]; then
            log "Last backup run:"
            tail -n 20 "$log_file"
        fi
    else
        log "No cron job configured"
    fi
}

show_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Setup automated PostgreSQL backup cron job.

OPTIONS:
    --time SCHEDULE     Cron schedule (default: "0 2 * * *" - 2 AM daily)
    --remove            Remove cron job
    --status            Show cron job status
    --help              Show this help message

EXAMPLES:
    # Setup daily backup at 2 AM
    $(basename "$0")

    # Custom schedule (every 6 hours)
    $(basename "$0") --time "0 */6 * * *"

    # Setup weekly backup (Sundays at 3 AM)
    $(basename "$0") --time "0 3 * * 0"

    # Check status
    $(basename "$0") --status

    # Remove cron job
    $(basename "$0") --remove

CRON SCHEDULE FORMAT:
    * * * * *
    │ │ │ │ │
    │ │ │ │ └─── Day of week (0-7, Sunday=0 or 7)
    │ │ │ └───── Month (1-12)
    │ │ └─────── Day of month (1-31)
    │ └───────── Hour (0-23)
    └─────────── Minute (0-59)

COMMON SCHEDULES:
    0 2 * * *       Daily at 2 AM
    0 */6 * * *     Every 6 hours
    0 3 * * 0       Weekly on Sunday at 3 AM
    0 4 1 * *       Monthly on 1st at 4 AM

EOF
}

main() {
    local action="add"

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --time)
                CRON_SCHEDULE="$2"
                shift 2
                ;;
            --remove)
                action="remove"
                shift
                ;;
            --status)
                action="status"
                shift
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

    # Execute action
    case "$action" in
        add)
            check_prerequisites
            add_cron_job
            ;;
        remove)
            remove_cron_job
            ;;
        status)
            show_status
            ;;
    esac
}

main "$@"
