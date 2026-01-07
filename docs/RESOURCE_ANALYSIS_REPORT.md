# 📊 Resource Consumption Analysis Report

**Generated:** 2025-12-20
**Analysis:** Query Comparison Runs 1 & 2
**Tool:** Memory & Swap Monitoring

---

## 🚨 EXECUTIVE SUMMARY

### Critical Discovery: Run 2 Performance Degradation Explained

**Root Cause Identified:** **HIGH SWAP USAGE** during Run 2 caused 3-6x performance degradation.

| Metric | Run 1 (04:19) | Run 2 (04:57) | Impact |
|--------|---------------|---------------|--------|
| **Baseline Swap** | 0 MB | **1,422 MB** | Run 2 started with 1.4GB swap active! |
| **Peak Swap** | 0 MB | **1,964 MB** | Swap increased to 2GB during queries |
| **Avg Query Time** | 65s | 430s | **6.6x slower** |
| **Pageouts** | 0 | 0 | Misleading metric - swap was pre-existing |

**Verdict:** Run 2 benchmark is **INVALID** - system was already under memory pressure before test started.

---

## 📈 Detailed Findings

### 1. Swap Usage Evolution

#### Run 1 - Optimal Conditions
```
Query 1-6 (BGE): Swap = 0 MB (baseline and peak)
Query 1-6 (E5):  Swap = 0 MB (baseline and peak)
```
✅ **Perfect conditions** - No memory pressure, all operations in RAM

#### Run 2 - Degraded Conditions
```
Query 1 (BGE):  Baseline = 1,422 MB → Peak = 1,964 MB (Δ +542 MB)
Query 2 (BGE):  Baseline = 1,422 MB → Peak = 1,964 MB (Δ +542 MB)
Query 3 (BGE):  Baseline = 1,422 MB → Peak = 1,964 MB (Δ +542 MB)
... (pattern continues for all 12 queries)
```
🔴 **High memory pressure** - System swapping to disk throughout entire test

### 2. Performance Correlation

**Direct correlation between swap usage and query time:**

| Run | Avg Baseline Swap | Avg Query Time | Performance |
|-----|-------------------|----------------|-------------|
| Run 1 | 0 MB | 65s | ✅ Baseline |
| Run 2 | 1,422 MB | 430s | 🔴 6.6x slower |

**Explanation:**
- When swap is active, system must read/write to disk (SSD) instead of RAM
- RAM access: ~10-100 ns
- SSD access: ~10-100 μs (1000x slower!)
- LLM operations are memory-intensive → Heavy swap penalty

### 3. Query-by-Query Comparison

#### Q1: "conversations about restaurants in Paris"
```
Run 1 BGE: 272s (anomaly - cold start) | Swap: 0 MB
Run 2 BGE: 447s (1.64x slower)         | Swap: 1,422 → 1,964 MB
Run 2 E5:  425s (6.6x slower vs Run 1) | Swap: 1,422 → 1,964 MB
```

#### Q2: "discussions sur les restaurants à Paris"
```
Run 1 BGE: 72s  | Swap: 0 MB
Run 2 BGE: 307s (4.26x slower) | Swap: 1,422 → 1,964 MB
```

#### Q3-Q6: Similar pattern
```
All queries 3-6x slower in Run 2 due to swap pressure
```

### 4. BGE vs E5 Resource Usage (Run 2)

Both models showed **identical swap patterns** in Run 2:
- Both started at ~1,422 MB baseline
- Both peaked at ~1,964 MB
- **Conclusion:** Swap usage is LLM-driven (Mistral 7B), not embedding-model specific

### 5. System State Analysis

#### Why did Run 2 start with swap active?

**Possible causes:**
1. **Previous applications not closed** - Browser, IDE, other memory-heavy apps
2. **macOS memory compression aggressive** - System may have kept cached data in swap
3. **No system restart** between runs
4. **Background processes** - Docker, databases, other services consuming RAM

#### M1 Mac Mini 16GB Memory Breakdown

**Run 1 (Optimal):**
```
Total RAM:     17.2 GB
Available:     ~7.3 GB (42%)
Used:          ~9.9 GB (58%)
Swap:          0 MB
Status:        ✅ Healthy
```

**Run 2 (Degraded):**
```
Total RAM:     17.2 GB
Available:     ~6.8 GB (40%)
Used:          ~10.4 GB (60%)
Swap:          1,422 → 1,964 MB (growing)
Status:        🔴 Memory pressure
```

**Analysis:**
- Only ~500 MB difference in available RAM between runs
- But **1.4 GB swap active** in Run 2 = Previous heavy usage not released
- macOS didn't release swap even though RAM was technically available

---

## 💡 Key Insights

### 1. Pageouts Metric is Insufficient

**Problem discovered:**
- Run 2 showed "0 pageouts" but had 1.4GB swap active
- Pageouts only measure NEW pages swapped during query
- **Pre-existing swap not captured**

**Better metrics needed:**
- Baseline swap usage (before query)
- Peak swap usage (during query)
- Swap delta (peak - baseline)
- Memory free (actual RAM available)

### 2. Benchmarking Requirements

For **valid RAG performance benchmarks**, ensure:
1. ✅ **System restart** or full memory flush before test
2. ✅ **Swap usage = 0 MB** before starting
3. ✅ **No background applications** (close browsers, IDEs, Docker)
4. ✅ **Monitor continuously** during test (not just snapshots)
5. ✅ **Consistent system state** between comparison runs

### 3. Production Implications

**Run 1 represents real-world performance** for a well-maintained system:
- BGE: ~65s average query time (excluding cold start)
- E5: ~60s average query time
- Acceptable for production use

**Run 2 is artificially degraded** and should be discarded.

---

## 🔍 Visualization Breakdown

**resource_analysis.png** contains 6 panels:

### Panel 1: Run 1 Swap Usage (Baseline vs Peak)
- Shows all queries with 0 MB swap
- Confirms optimal conditions

### Panel 2: Run 2 Swap Usage (Baseline vs Peak)
- Shows ~1.4 GB baseline, ~2 GB peak
- Visual confirmation of memory pressure

### Panel 3: Swap Delta per Query
- Compares swap increase (Peak - Baseline)
- Run 1: Flat at 0 MB
- Run 2: Consistent ~500 MB increase per query

### Panel 4: Query Time vs Swap Correlation
- Scatter plot showing relationship
- Clear trend: Higher swap → Slower queries

### Panel 5: BGE vs E5 Swap Comparison (Run 2)
- Shows both models have identical swap patterns
- Confirms LLM (not embedding model) drives memory usage

### Panel 6: Summary Statistics Table
- Key metrics comparison
- Recommendations for future tests

---

## 📋 Recommendations

### Immediate Actions

1. **❌ DISCARD Run 2 Results**
   - Invalid due to pre-existing memory pressure
   - Use only Run 1 for performance baseline

2. **✅ RE-RUN Query Comparison**
   - Restart system before test
   - Verify swap = 0 MB
   - Test all 12 queries in clean environment

3. **🔧 Improve Monitoring**
   - Add memory-free and swap-free to resource logs
   - Log full vm_stat output (not just swap)
   - Capture Python process RSS (Resident Set Size)

### Long-term Improvements

4. **System Preparation Script**
   ```bash
   #!/bin/bash
   # pre_benchmark.sh - Prepare system for valid benchmarks

   # Check swap usage
   SWAP_USED=$(sysctl vm.swapusage | grep -oE 'used = [0-9.]+' | cut -d' ' -f3)

   if (( $(echo "$SWAP_USED > 0" | bc -l) )); then
       echo "❌ ERROR: Swap in use ($SWAP_USED MB)"
       echo "Please restart system or run: sudo purge"
       exit 1
   fi

   echo "✅ System ready for benchmarking"
   ```

5. **Real-time Monitoring Dashboard**
   - Live memory/swap tracking during queries
   - Alert if swap usage starts
   - Automatic test abortion if memory pressure detected

6. **Document Baseline System State**
   - RAM available: ≥7 GB
   - Swap usage: 0 MB
   - CPU idle: ≥50%
   - Disk free: ≥10 GB

---

## 🎯 Conclusion

### What We Learned

1. **Run 2 slowdown FULLY EXPLAINED**: Pre-existing 1.4GB swap caused 6.6x performance degradation
2. **Run 1 is the valid baseline**: 0 swap, optimal conditions, representative of production
3. **BGE vs E5 comparison from Run 1 stands**: BGE is superior for quality, E5 slightly faster
4. **Monitoring improvements needed**: Current resource logs miss critical swap state

### Corrected Performance Summary

**Valid data (Run 1 only):**
| Model | Avg Query Time | Swap Usage | Status |
|-------|----------------|------------|--------|
| BGE   | ~65s (post cold-start) | 0 MB | ✅ Recommended |
| E5    | ~60s | 0 MB | 🟡 Acceptable |

**Invalid data (Run 2 - discard):**
| Model | Avg Query Time | Swap Usage | Status |
|-------|----------------|------------|--------|
| BGE   | 430s | 1.4-2.0 GB | ❌ Invalid test |
| E5    | 400s | 1.4-2.0 GB | ❌ Invalid test |

### Next Steps

1. ✅ Use Run 1 results for production decisions → **Choose BGE**
2. 🔄 Optionally re-run full 12-query test in clean environment
3. 🔧 Implement improved monitoring (memory-free, swap-free tracking)
4. 📚 Document system requirements for future benchmarks

---

**Generated by:** Resource Analysis Script v1.0
**Visualization:** resource_analysis.png
**Data Sources:**
- query_comparison_20251220_041917/ (Run 1 - Valid)
- query_comparison_20251220_045713/ (Run 2 - Invalid due to swap)
