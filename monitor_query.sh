#!/bin/bash

# RAG Query Performance Monitor for M1 Mac
# Usage: ./monitor_query.sh [query_text]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QUERY="${1:-Where does jealousy come from?}"
LOG_DIR="performance_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MEMORY_LOG="${LOG_DIR}/memory_${TIMESTAMP}.log"
POWER_LOG="${LOG_DIR}/power_${TIMESTAMP}.log"
SUMMARY_LOG="${LOG_DIR}/summary_${TIMESTAMP}.txt"

# Create log directory
mkdir -p "${LOG_DIR}"

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   RAG Query Performance Monitor${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "${GREEN}Query:${NC} ${QUERY}"
echo -e "${GREEN}Timestamp:${NC} ${TIMESTAMP}"
echo -e "${GREEN}Logs will be saved to:${NC} ${LOG_DIR}/"
echo ""

# Check if running as root for powermetrics
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Start memory monitoring in background
echo -e "${YELLOW}Starting memory monitoring...${NC}"
vm_stat 2 > "${MEMORY_LOG}" 2>&1 &
VM_PID=$!

# Start powermetrics monitoring in background
echo -e "${YELLOW}Starting temperature/power monitoring...${NC}"
echo "(You may need to enter your password for sudo)"
$SUDO powermetrics --samplers cpu_power,gpu_power --show-initial-usage -i 2000 > "${POWER_LOG}" 2>&1 &
PM_PID=$!

# Give monitors time to start
sleep 2

echo -e "${GREEN}Monitors started successfully${NC}"
echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   Running Query (this may take 60+ seconds)${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the query with optimized settings
CHUNK_SIZE=5000 \
CHUNK_OVERLAP=1000 \
N_GPU_LAYERS=32 \
N_BATCH=512 \
TOP_K=3 \
PGTABLE=ethical-slut_paper \
LOG_QUERIES=1 \
.venv/bin/python3 rag_low_level_m1_16gb_verbose.py \
--query-only --query "${QUERY}"

QUERY_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Give monitors time to catch final readings
sleep 2

# Stop monitoring processes
echo ""
echo -e "${YELLOW}Stopping monitors...${NC}"
kill $VM_PID 2>/dev/null || true
$SUDO kill $PM_PID 2>/dev/null || true

echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   Performance Analysis${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Create summary report
{
    echo "================================================="
    echo "   RAG Query Performance Report"
    echo "================================================="
    echo ""
    echo "Query: ${QUERY}"
    echo "Timestamp: ${TIMESTAMP}"
    echo "Total Elapsed Time: ${ELAPSED} seconds"
    echo "Exit Code: ${QUERY_EXIT_CODE}"
    echo ""
    echo "================================================="
    echo "   Memory Analysis"
    echo "================================================="
    echo ""

    # Memory stats
    echo "Initial Memory State:"
    head -10 "${MEMORY_LOG}" | grep -E "Pages free|Pages active|Pages inactive|Pages speculative|Pages wired down|Pages purgeable|File-backed pages|Anonymous pages|Pages stored in compressor|Pages occupied by compressor|Pageins|Pageouts|Swapins|Swapouts" || echo "No data"

    echo ""
    echo "Final Memory State:"
    tail -10 "${MEMORY_LOG}" | grep -E "Pages free|Pages active|Pages inactive|Pages speculative|Pages wired down|Pages purgeable|File-backed pages|Anonymous pages|Pages stored in compressor|Pages occupied by compressor|Pageins|Pageouts|Swapins|Swapouts" || echo "No data"

    echo ""
    echo "Swap Usage:"
    sysctl vm.swapusage

    echo ""
    echo "================================================="
    echo "   Temperature & Power Analysis"
    echo "================================================="
    echo ""

    # Temperature and power stats
    if [ -f "${POWER_LOG}" ]; then
        echo "CPU Temperature Readings:"
        grep "CPU die temperature" "${POWER_LOG}" | head -5
        echo "..."
        grep "CPU die temperature" "${POWER_LOG}" | tail -5

        echo ""
        echo "GPU Temperature Readings:"
        grep "GPU die temperature" "${POWER_LOG}" | head -5 || echo "No GPU temp data"
        echo "..."
        grep "GPU die temperature" "${POWER_LOG}" | tail -5 || echo "No GPU temp data"

        echo ""
        echo "Package Power Readings:"
        grep "Package Power" "${POWER_LOG}" | head -5
        echo "..."
        grep "Package Power" "${POWER_LOG}" | tail -5

        echo ""
        echo "CPU Power by Core (sample):"
        grep -A 10 "CPU Power" "${POWER_LOG}" | head -15 || echo "No CPU power data"
    else
        echo "No power metrics data available"
    fi

    echo ""
    echo "================================================="
    echo "   Performance Issues Detected"
    echo "================================================="
    echo ""

    # Check for issues
    ISSUES=0

    # Check swap
    SWAP_USED=$(sysctl vm.swapusage | grep -oE 'used = [0-9.]+' | awk '{print $3}')
    if [ ! -z "$SWAP_USED" ]; then
        SWAP_MB=$(echo "$SWAP_USED" | sed 's/M//')
        if (( $(echo "$SWAP_MB > 500" | bc -l 2>/dev/null || echo 0) )); then
            echo "⚠️  HIGH SWAP USAGE: ${SWAP_USED}M (recommend < 500M)"
            ISSUES=$((ISSUES + 1))
        else
            echo "✓ Swap usage OK: ${SWAP_USED}M"
        fi
    fi

    # Check pageouts (indicates swapping activity)
    PAGEOUTS_START=$(head -50 "${MEMORY_LOG}" | grep "Pageouts" | head -1 | awk '{print $2}' | sed 's/\.//')
    PAGEOUTS_END=$(tail -50 "${MEMORY_LOG}" | grep "Pageouts" | tail -1 | awk '{print $2}' | sed 's/\.//')
    if [ ! -z "$PAGEOUTS_START" ] && [ ! -z "$PAGEOUTS_END" ]; then
        PAGEOUTS_DELTA=$((PAGEOUTS_END - PAGEOUTS_START))
        if [ $PAGEOUTS_DELTA -gt 1000 ]; then
            echo "⚠️  ACTIVE SWAPPING DETECTED: $PAGEOUTS_DELTA pageouts during query"
            echo "   This indicates memory pressure - consider reducing N_BATCH or N_GPU_LAYERS"
            ISSUES=$((ISSUES + 1))
        else
            echo "✓ Minimal pageouts: $PAGEOUTS_DELTA"
        fi
    fi

    # Check temperature
    if [ -f "${POWER_LOG}" ]; then
        MAX_TEMP=$(grep "CPU die temperature" "${POWER_LOG}" | grep -oE '[0-9]+\.[0-9]+' | sort -n | tail -1)
        if [ ! -z "$MAX_TEMP" ]; then
            if (( $(echo "$MAX_TEMP > 95" | bc -l 2>/dev/null || echo 0) )); then
                echo "⚠️  HIGH CPU TEMPERATURE: ${MAX_TEMP}°C (may cause throttling)"
                ISSUES=$((ISSUES + 1))
            elif (( $(echo "$MAX_TEMP > 85" | bc -l 2>/dev/null || echo 0) )); then
                echo "⚠️  ELEVATED CPU TEMPERATURE: ${MAX_TEMP}°C"
                ISSUES=$((ISSUES + 1))
            else
                echo "✓ CPU temperature OK: ${MAX_TEMP}°C"
            fi
        fi
    fi

    echo ""
    if [ $ISSUES -eq 0 ]; then
        echo "✓ No major performance issues detected"
    else
        echo "Found $ISSUES potential issue(s)"
    fi

    echo ""
    echo "================================================="
    echo "   Recommendations"
    echo "================================================="
    echo ""

    if [ $ISSUES -gt 0 ]; then
        echo "Based on the detected issues, try these settings:"
        echo ""

        if [ ! -z "$PAGEOUTS_DELTA" ] && [ $PAGEOUTS_DELTA -gt 1000 ]; then
            echo "For memory pressure:"
            echo "  export N_BATCH=256        # Reduce from 512"
            echo "  export N_GPU_LAYERS=28    # Reduce from 32"
            echo ""
        fi

        if [ ! -z "$MAX_TEMP" ] && (( $(echo "$MAX_TEMP > 85" | bc -l 2>/dev/null || echo 0) )); then
            echo "For thermal issues:"
            echo "  export N_GPU_LAYERS=24    # Reduce GPU load"
            echo "  # Ensure good ventilation"
            echo "  # Close other applications"
            echo ""
        fi
    else
        echo "System is handling the current settings well."
        echo "Current settings appear optimal for your hardware."
    fi

    echo ""
    echo "================================================="
    echo "   Log Files"
    echo "================================================="
    echo ""
    echo "Detailed logs saved to:"
    echo "  Memory: ${MEMORY_LOG}"
    echo "  Power:  ${POWER_LOG}"
    echo "  Summary: ${SUMMARY_LOG}"
    echo ""
    echo "View latest query log:"
    echo "  ls -lt query_logs/ethical-slut_paper/*.json | head -1"

} | tee "${SUMMARY_LOG}"

echo ""
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}   Monitoring Complete${NC}"
echo -e "${GREEN}=================================================${NC}"
echo ""
echo -e "Summary saved to: ${BLUE}${SUMMARY_LOG}${NC}"
echo ""
