#!/usr/bin/env bash
# start_monitoring.sh - Quick start script for monitoring stack
#
# Usage:
#   ./start_monitoring.sh              # Start monitoring stack
#   ./start_monitoring.sh --full       # Start with application
#   ./start_monitoring.sh --backup     # Include backup service
#   ./start_monitoring.sh --stop       # Stop all services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $*"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $*" >&2
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Please install Docker."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi

    # Check if .env exists
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        warn ".env file not found. Creating from example..."
        if [ -f "$PROJECT_ROOT/config/.env.example" ]; then
            cp "$PROJECT_ROOT/config/.env.example" "$PROJECT_ROOT/.env"
            log "Created .env file. Please review and update with your credentials."
        else
            error ".env.example not found. Cannot create .env file."
            exit 1
        fi
    fi

    log "Prerequisites OK"
}

start_monitoring() {
    local profiles="$1"

    log "Starting monitoring stack..."

    cd "$CONFIG_DIR"

    # Pull latest images
    log "Pulling Docker images..."
    if [ -n "$profiles" ]; then
        docker-compose --profile "$profiles" pull
    else
        docker-compose pull
    fi

    # Start services
    log "Starting services..."
    if [ -n "$profiles" ]; then
        docker-compose --profile "$profiles" up -d
    else
        docker-compose up -d
    fi

    # Wait for services to be healthy
    log "Waiting for services to be ready..."
    sleep 5

    # Check status
    docker-compose ps

    log ""
    log "======================================"
    log "Monitoring Stack Started Successfully!"
    log "======================================"
    log ""
    log "Service URLs:"
    log "  Grafana:       http://localhost:3000 (admin/admin)"
    log "  Prometheus:    http://localhost:9090"
    log "  Alertmanager:  http://localhost:9093"
    log "  PostgreSQL:    localhost:5432"
    log "  cAdvisor:      http://localhost:8080"
    log ""
    log "View logs:"
    log "  docker-compose logs -f"
    log ""
    log "Stop services:"
    log "  docker-compose down"
    log ""
}

stop_monitoring() {
    log "Stopping monitoring stack..."

    cd "$CONFIG_DIR"
    docker-compose --profile backup --profile app down

    log "Monitoring stack stopped"
}

show_status() {
    cd "$CONFIG_DIR"

    log "Service Status:"
    docker-compose ps

    log ""
    log "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
        rag_postgres rag_grafana rag_prometheus rag_alertmanager 2>/dev/null || log "Some services not running"
}

setup_backup_cron() {
    log "Setting up automated backups..."

    if [ -f "$PROJECT_ROOT/scripts/backup/setup_cron.sh" ]; then
        "$PROJECT_ROOT/scripts/backup/setup_cron.sh"
        log "Backup cron job configured"
    else
        error "Backup setup script not found"
        exit 1
    fi
}

run_health_check() {
    log "Running health checks..."

    if [ -f "$PROJECT_ROOT/utils/health_check.py" ]; then
        python "$PROJECT_ROOT/utils/health_check.py"
    else
        error "Health check script not found"
        exit 1
    fi
}

show_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Quick start script for RAG pipeline monitoring stack.

OPTIONS:
    --full              Start with RAG application
    --backup            Include automated backup service
    --stop              Stop all services
    --status            Show service status
    --health            Run health checks
    --setup-backup      Setup automated backup cron job
    --help              Show this help message

EXAMPLES:
    # Start monitoring stack
    $(basename "$0")

    # Start with application
    $(basename "$0") --full

    # Start with backups
    $(basename "$0") --backup

    # Check status
    $(basename "$0") --status

    # Stop all services
    $(basename "$0") --stop

FIRST TIME SETUP:
    1. Run this script to start monitoring
    2. Open Grafana: http://localhost:3000 (admin/admin)
    3. Navigate to RAG Pipeline Overview dashboard
    4. Setup backup cron: $(basename "$0") --setup-backup

EOF
}

main() {
    local action="start"
    local profiles=""

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --full)
                profiles="app"
                shift
                ;;
            --backup)
                profiles="backup"
                shift
                ;;
            --stop)
                action="stop"
                shift
                ;;
            --status)
                action="status"
                shift
                ;;
            --health)
                action="health"
                shift
                ;;
            --setup-backup)
                action="setup-backup"
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
        start)
            check_prerequisites
            start_monitoring "$profiles"
            ;;
        stop)
            stop_monitoring
            ;;
        status)
            show_status
            ;;
        health)
            run_health_check
            ;;
        setup-backup)
            setup_backup_cron
            ;;
    esac
}

main "$@"
