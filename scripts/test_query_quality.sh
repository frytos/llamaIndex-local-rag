#!/bin/bash
# Test query quality comparison between bge-small-en and multilingual-e5-small
# Focus: French/English queries about Paris restaurants and New York travel dates
#
# Configuration:
#   - Logging: Full chunks, colorized output, query logging enabled
#   - Retrieval: TOP_K=5, HYBRID_ALPHA=0.5 (balanced BM25+vector)
#   - LLM: CTX=6144, MAX_TOKENS=512 (detailed answers)
#   - Monitoring: CPU/RAM snapshots, swap tracking enabled
date
cd /Users/frytos/code/llamaIndex-local-rag
source .venv/bin/activate
source optimized_config.sh

# Resource monitoring configuration
ENABLE_RESOURCE_MONITORING=1    # Set to 0 to disable
MONITOR_INTERVAL=2              # Seconds between samples (if using continuous monitoring)

# Override config for quality testing
echo "🔧 Applying test-optimized configuration..."

# Logging - Enable full visibility
export LOG_FULL_CHUNKS=1
export COLORIZE_CHUNKS=1
export LOG_QUERIES=1

# Retrieval - Optimized for bilingual quality
export TOP_K=8
export HYBRID_ALPHA=0.8
export ENABLE_FILTERS=1
export MMR_THRESHOLD=0.5

# LLM - Balanced for detailed answers
export TEMP=0.1
export MAX_NEW_TOKENS=128
export CTX=50000
export N_GPU_LAYERS=24
export N_BATCH=256
date
echo "✅ Configuration applied:"
echo "  Logging: FULL_CHUNKS=$LOG_FULL_CHUNKS COLORIZE=$COLORIZE_CHUNKS LOG_QUERIES=$LOG_QUERIES"
echo "  Retrieval: TOP_K=$TOP_K HYBRID_ALPHA=$HYBRID_ALPHA MMR=$MMR_THRESHOLD"
echo "  LLM: TEMP=$TEMP MAX_TOKENS=$MAX_NEW_TOKENS CTX=$CTX GPU_LAYERS=$N_GPU_LAYERS"
echo ""

# Test queries
declare -a QUERIES=(
    "conversations about restaurants in Paris"
    "discussions sur les restaurants à Paris"
    "restaurants parisiens"
    "when did I go to New York"
    "quand suis-je allé à New York"
    "date de mon voyage à New York"
	"resume les difficultes a Centrale Lyon"
	"sum up the challenges at Centrale Lyon"
	"quels sont les principaux sports pratiques ?"
	"what are the main sports practiced?"
	"avis sur les soiréss electroniques a Londres"
	"feedbacks on electronic music noghts out in London"
)

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to capture resource snapshot
capture_resources() {
    local output_file=$1
    local label=$2
    local timestamp=$(date '+%H:%M:%S')

    {
        echo "=== $label @ $timestamp ==="

        # Memory snapshot (vm_stat)
        echo "Memory Status:"
        vm_stat 1 2 | tail -1 | awk '{
            printf "  Pages free: %s\n", $3
            printf "  Pages active: %s\n", $5
            printf "  Pages inactive: %s\n", $7
            printf "  Pages wired: %s\n", $11
            printf "  Pageouts: %s\n", $23
        }'

        # RAM summary (sysctl)
        echo ""
        echo "RAM Summary:"
        sysctl vm.swapusage | awk '{
            printf "  Swap: %s\n", $0
        }'

        # CPU & Memory for Python process
        echo ""
        echo "Python Process:"
        ps aux | head -1
        ps aux | grep -E "python.*rag" | grep -v grep | head -1

        echo ""
    } >> "$output_file"
}
date
# Function to calculate memory delta
calculate_memory_delta() {
    local resources_file=$1

    # Extract pageouts from baseline and peak
    local baseline_pageouts=$(grep -A 10 "BASELINE" "$resources_file" | grep "Pageouts:" | awk '{print $2}' | sed 's/\.//')
    local peak_pageouts=$(grep -A 10 "PEAK" "$resources_file" | grep "Pageouts:" | awk '{print $2}' | sed 's/\.//')

    if [ -n "$baseline_pageouts" ] && [ -n "$peak_pageouts" ]; then
        local delta=$((peak_pageouts - baseline_pageouts))
        echo "$delta"
    else
        echo "0"
    fi
}
date
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  QUERY QUALITY COMPARISON TEST                                       ║"
echo "║  bge-small-en (MLX) vs multilingual-e5-small (HuggingFace)           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Testing ${#QUERIES[@]} queries against both indexes..."
echo ""

# Create output directory
OUTPUT_DIR="query_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Results will be saved to: $OUTPUT_DIR"
echo ""
date
# Function to run query and capture output
run_query() {
    local table=$1
    local query=$2
    local model_name=$3
    local query_num=$4

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Query $query_num with $model_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}\"$query\"${NC}"
    echo ""

    # Output file
    local output_file="$OUTPUT_DIR/q${query_num}_${model_name}.txt"
    local timing_file="$OUTPUT_DIR/q${query_num}_${model_name}_timing.txt"

    # Start timing
    local start_time=$(date +%s.%N)
    echo "⏱️  Start time: $(date '+%H:%M:%S')"

    # Capture resource baseline
    local resources_file="$OUTPUT_DIR/q${query_num}_${model_name}_resources.log"

    if [ $ENABLE_RESOURCE_MONITORING -eq 1 ]; then
        capture_resources "$resources_file" "BASELINE"
    fi

    # Run query and capture output
    PGTABLE=$table python rag_low_level_m1_16gb_verbose.py \
        --query-only \
        --query "$query" \
        2>&1 | tee "$output_file"

    local exit_code=$?

    # End timing
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    # Capture resource peak
    if [ $ENABLE_RESOURCE_MONITORING -eq 1 ]; then
        capture_resources "$resources_file" "PEAK"
    fi

    echo ""
    echo -e "⏱️  End time: $(date '+%H:%M:%S')"
    echo -e "${YELLOW}⏱️  Total query time: ${duration}s${NC}"

    # Extract timing details from output
    local retrieval_time=$(grep "Retrieval took" "$output_file" | sed -E 's/.*took ([0-9.]+)s.*/\1/' | head -n 1)
    local generation_time=$(grep "Generation took" "$output_file" | sed -E 's/.*took ([0-9.]+)s.*/\1/' | head -n 1)

    # Save timing information
    cat > "$timing_file" << EOF
Total: ${duration}s
Retrieval: ${retrieval_time}s
Generation: ${generation_time}s
Model: ${model_name}
Query: ${query_num}
EOF

    # Display breakdown
    if [ -n "$retrieval_time" ] && [ -n "$generation_time" ]; then
        echo -e "  └─ Retrieval: ${retrieval_time}s"
        echo -e "  └─ Generation: ${generation_time}s"
        echo -e "  └─ Overhead: $(echo "$duration - $retrieval_time - $generation_time" | bc)s"
    fi

    # Display memory pressure
    if [ $ENABLE_RESOURCE_MONITORING -eq 1 ]; then
        local pageout_delta=$(calculate_memory_delta "$resources_file")

        echo ""
        echo -e "💾 Memory Pressure:"
        if [ "$pageout_delta" -gt 1000 ]; then
            echo -e "  ${RED}⚠️  High swap activity: ${pageout_delta} pageouts${NC}"
        elif [ "$pageout_delta" -gt 100 ]; then
            echo -e "  ${YELLOW}⚠️  Moderate swap: ${pageout_delta} pageouts${NC}"
        else
            echo -e "  ${GREEN}✓ Normal: ${pageout_delta} pageouts${NC}"
        fi
    fi

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Query completed${NC}"
    else
        echo -e "${RED}✗ Query failed (exit code: $exit_code)${NC}"
    fi

    echo ""
    echo "Results saved to: $output_file"
    echo "Timing saved to: $timing_file"
    if [ $ENABLE_RESOURCE_MONITORING -eq 1 ]; then
        echo "Resources saved to: $resources_file"
    fi
    echo ""

    return $exit_code
}

# Function to extract key metrics from output
extract_metrics() {
    local file=$1

    # Extract average similarity score
    local avg_score=$(grep "Average similarity:" "$file" | sed 's/.*: //')

    # Extract answer snippet (first 200 chars)
    local answer=$(grep -A 20 "ANSWER:" "$file" | head -n 5 | tail -n 4 | tr '\n' ' ' | cut -c1-200)

    echo "Avg Score: $avg_score"
    echo "Answer: $answer..."
}
date
# Main test loop
query_num=1
for query in "${QUERIES[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "TEST $query_num/${#QUERIES[@]}: \"$query\""
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""

    # Test with bge-small-en
    echo -e "${GREEN}[1/2] Testing with bge-small-en (MLX)${NC}"
    run_query "messenger_clean_small_cs700_ov150_bge" "$query" "bge" "$query_num"

    sleep 2

    # Test with multilingual-e5-small
    echo -e "${GREEN}[2/2] Testing with multilingual-e5-small (HuggingFace)${NC}"
    run_query "messenger_clean_small_cs700_ov150_e5" "$query" "e5" "$query_num"

    sleep 2

    ((query_num++))
done
date
# Generate summary report
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  GENERATING COMPARISON REPORT                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

REPORT_FILE="$OUTPUT_DIR/COMPARISON_REPORT.md"

cat > "$REPORT_FILE" << 'REPORT_HEADER'
# Query Quality Comparison Report

**Date:** $(date)
**Models Tested:**
- bge-small-en (MLX) → messenger_clean_small_cs700_ov150_bge
- multilingual-e5-small (HuggingFace) → messenger_clean_small_cs700_ov150_e5

## Test Queries

REPORT_HEADER

# Add test results to report
query_num=1
for query in "${QUERIES[@]}"; do
    echo "" >> "$REPORT_FILE"
    echo "### Query $query_num: \"$query\"" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    # BGE results
    echo "#### bge-small-en (MLX)" >> "$REPORT_FILE"

    # Add timing if available
    if [ -f "$OUTPUT_DIR/q${query_num}_bge_timing.txt" ]; then
        echo "**⏱️ Timing:**" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        cat "$OUTPUT_DIR/q${query_num}_bge_timing.txt" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi

    echo '```' >> "$REPORT_FILE"
    if [ -f "$OUTPUT_DIR/q${query_num}_bge.txt" ]; then
        grep -A 5 "Retrieved chunks:" "$OUTPUT_DIR/q${query_num}_bge.txt" | head -n 6 >> "$REPORT_FILE" || echo "No chunks found" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        grep -A 10 "ANSWER:" "$OUTPUT_DIR/q${query_num}_bge.txt" | head -n 11 >> "$REPORT_FILE" || echo "No answer found" >> "$REPORT_FILE"
    fi
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    # E5 results
    echo "#### multilingual-e5-small (HuggingFace)" >> "$REPORT_FILE"

    # Add timing if available
    if [ -f "$OUTPUT_DIR/q${query_num}_e5_timing.txt" ]; then
        echo "**⏱️ Timing:**" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        cat "$OUTPUT_DIR/q${query_num}_e5_timing.txt" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi

    echo '```' >> "$REPORT_FILE"
    if [ -f "$OUTPUT_DIR/q${query_num}_e5.txt" ]; then
        grep -A 5 "Retrieved chunks:" "$OUTPUT_DIR/q${query_num}_e5.txt" | head -n 6 >> "$REPORT_FILE" || echo "No chunks found" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        grep -A 10 "ANSWER:" "$OUTPUT_DIR/q${query_num}_e5.txt" | head -n 11 >> "$REPORT_FILE" || echo "No answer found" >> "$REPORT_FILE"
    fi
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    ((query_num++))
done

# Calculate and add timing summary
cat >> "$REPORT_FILE" << 'TIMING_HEADER'

## Performance Comparison

### Query Response Times

| Query # | Query | bge (MLX) | e5 (HF) | Speedup |
|---------|-------|-----------|---------|---------|
TIMING_HEADER

# Add timing data for each query
query_num=1
for query in "${QUERIES[@]}"; do
    # Extract timing from files
    bge_total=""
    e5_total=""
    if [ -f "$OUTPUT_DIR/q${query_num}_bge_timing.txt" ]; then
        bge_total=$(grep "Total:" "$OUTPUT_DIR/q${query_num}_bge_timing.txt" | awk '{print $2}')
    fi
    if [ -f "$OUTPUT_DIR/q${query_num}_e5_timing.txt" ]; then
        e5_total=$(grep "Total:" "$OUTPUT_DIR/q${query_num}_e5_timing.txt" | awk '{print $2}')
    fi

    # Calculate speedup if both times available
    speedup=""
    if [ -n "$bge_total" ] && [ -n "$e5_total" ]; then
        bge_num=$(echo "$bge_total" | sed 's/s//')
        e5_num=$(echo "$e5_total" | sed 's/s//')
        if [ -n "$bge_num" ] && [ -n "$e5_num" ]; then
            speedup=$(echo "scale=2; $e5_num / $bge_num" | bc)
            speedup="${speedup}x"
        fi
    fi

    # Truncate query for table
    query_short=$(echo "$query" | cut -c1-30)
    if [ ${#query} -gt 30 ]; then
        query_short="${query_short}..."
    fi

    echo "| $query_num | $query_short | ${bge_total:-N/A} | ${e5_total:-N/A} | ${speedup:-N/A} |" >> "$REPORT_FILE"
    ((query_num++))
done

# Add resource monitoring summary
if [ $ENABLE_RESOURCE_MONITORING -eq 1 ]; then
    cat >> "$REPORT_FILE" << 'RESOURCE_HEADER'

### Resource Utilization

| Query # | Model | Pageouts | Memory Pressure | Status |
|---------|-------|----------|-----------------|--------|
RESOURCE_HEADER

    # Add resource data for each query
    query_num=1
    for query in "${QUERIES[@]}"; do
        # BGE resources
        if [ -f "$OUTPUT_DIR/q${query_num}_bge_resources.log" ]; then
            bge_pageouts=$(calculate_memory_delta "$OUTPUT_DIR/q${query_num}_bge_resources.log")
            bge_status="✓ Normal"
            if [ "$bge_pageouts" -gt 1000 ]; then
                bge_status="⚠️ High"
            elif [ "$bge_pageouts" -gt 100 ]; then
                bge_status="⚠️ Moderate"
            fi
            echo "| $query_num | bge | $bge_pageouts | ${bge_status} | - |" >> "$REPORT_FILE"
        fi

        # E5 resources
        if [ -f "$OUTPUT_DIR/q${query_num}_e5_resources.log" ]; then
            e5_pageouts=$(calculate_memory_delta "$OUTPUT_DIR/q${query_num}_e5_resources.log")
            e5_status="✓ Normal"
            if [ "$e5_pageouts" -gt 1000 ]; then
                e5_status="⚠️ High"
            elif [ "$e5_pageouts" -gt 100 ]; then
                e5_status="⚠️ Moderate"
            fi
            echo "| $query_num | e5 | $e5_pageouts | ${e5_status} | - |" >> "$REPORT_FILE"
        fi

        ((query_num++))
    done

    cat >> "$REPORT_FILE" << 'RESOURCE_FOOTER'

**Notes:**
- Pageouts = memory pages written to swap (disk)
- >1000 pageouts = High memory pressure (performance impact)
- >100 pageouts = Moderate (monitor closely)
- <100 pageouts = Normal operation

RESOURCE_FOOTER
fi

# Add comparison table
cat >> "$REPORT_FILE" << 'REPORT_FOOTER'

**Notes:**
- Speedup = e5_time / bge_time (higher = bge is faster)
- Times include retrieval + generation + overhead

---

## Test Configuration

**Environment Variables:**
- **Logging**: LOG_FULL_CHUNKS=1, COLORIZE_CHUNKS=1, LOG_QUERIES=1
- **Retrieval**: TOP_K=5, HYBRID_ALPHA=0.5 (hybrid), MMR_THRESHOLD=0.5 (diversity)
- **LLM**: TEMP=0.1, MAX_NEW_TOKENS=512, CTX=6144, N_GPU_LAYERS=24

**Why These Values:**
- `HYBRID_ALPHA=0.5`: Balanced keyword+semantic search (better for "Paris restaurants")
- `TOP_K=5`: +25% context vs default (4 chunks)
- `CTX=6144`: Headroom for 5 larger chunks (avoid overflow)
- `MMR_THRESHOLD=0.5`: Diversity to avoid repetitive chunks

## Quality Summary

| Query | Language | bge-small-en Quality | multilingual-e5 Quality | Winner |
|-------|----------|---------------------|------------------------|--------|
| Restaurants Paris (EN) | EN | ? | ? | ? |
| Restaurants Paris (FR) | FR | ? | ? | ? |
| Restaurants parisiens | FR | ? | ? | ? |
| New York trip (EN) | EN | ? | ? | ? |
| New York trip (FR) | FR | ? | ? | ? |
| Date voyage NY (FR) | FR | ? | ? | ? |

## Recommendations

Based on the test results:

1. **For English queries**: [To be filled after review]
2. **For French queries**: [To be filled after review]
3. **For bilingual use case**: [To be filled after review]

REPORT_FOOTER

echo "✅ Comparison report generated: $REPORT_FILE"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "TESTING COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "📁 All results saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Review the report: cat $REPORT_FILE"
echo "  2. Compare individual query results in $OUTPUT_DIR/"
echo "  3. Look for:"
echo "     - Which model found relevant conversations?"
echo "     - Which model has higher similarity scores?"
echo "     - Which answers are more accurate?"
echo ""
echo "Quick comparison:"
echo "  diff $OUTPUT_DIR/q1_bge.txt $OUTPUT_DIR/q1_e5.txt"
echo ""
date
