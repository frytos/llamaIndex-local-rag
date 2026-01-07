# Performance Tracking Guide

Complete guide to performance tracking, regression testing, and baseline management in the RAG pipeline.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Multi-Platform Baselines](#multi-platform-baselines)
- [CI/CD Integration](#cicd-integration)
- [Dashboard & Visualization](#dashboard--visualization)
- [Baseline Management](#baseline-management)
- [Tracked Metrics](#tracked-metrics)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Overview

The RAG pipeline includes automated performance tracking to:

- **Detect regressions** before they reach production
- **Track performance trends** over time
- **Compare across platforms** (M1 Mac, GPU servers, CI environments)
- **Maintain baselines** automatically
- **Visualize performance** with interactive dashboards

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Performance Tests                         │
│              (tests/test_performance_regression.py)          │
└──────────────────────┬──────────────────────────────────────┘
                       │ Records metrics
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Performance History Database                       │
│        (SQLite: benchmarks/history/performance.db)          │
│                                                              │
│  - Time-series metrics                                       │
│  - Multi-platform support                                    │
│  - Git metadata tracking                                     │
└──────┬────────────────┬────────────────┬─────────────────────┘
       │                │                │
       ▼                ▼                ▼
   Reports         Dashboard        Baseline Updates
 (Markdown/HTML)  (Plotly charts)  (Semi-automated)
```

---

## Quick Start

### Run Performance Tests Locally

```bash
# Run performance tests (no recording)
pytest tests/test_performance_regression.py -v

# Run with recording to history database
ENABLE_PERFORMANCE_RECORDING=1 pytest tests/test_performance_regression.py -v

# Quick mode (skip slow tests)
pytest tests/test_performance_regression.py -v -m "not slow"
```

### Generate Performance Report

```bash
# Markdown report (for PRs)
python scripts/generate_performance_report.py --format markdown --output report.md

# HTML report (for viewing)
python scripts/generate_performance_report.py --format html --output report.html

# JSON report (for programmatic access)
python scripts/generate_performance_report.py --format json --output metrics.json
```

### View Performance Dashboard

```bash
# Generate dashboard (last 30 days)
python scripts/generate_performance_dashboard.py --days 30

# Open in browser
open benchmarks/dashboard.html

# Custom platforms and date range
python scripts/generate_performance_dashboard.py \
  --days 90 \
  --platforms M1_Mac_16GB GitHub_Actions_macOS \
  --output dashboard-90d.html
```

### Check Current Baselines

```bash
# View baselines file
cat tests/performance_baselines.json

# Or pretty-print
python -c "import json; print(json.dumps(json.load(open('tests/performance_baselines.json')), indent=2))"

# Check database stats
python -c "
from utils.performance_history import PerformanceHistory
history = PerformanceHistory()
stats = history.get_stats()
print(f'Total runs: {stats[\"total_runs\"]}')
print(f'Platforms: {stats[\"num_platforms\"]}')
print(f'Date range: {stats[\"first_run\"]} to {stats[\"last_run\"]}')
"
```

---

## Multi-Platform Baselines

### Supported Platforms

The system auto-detects your platform and loads appropriate baselines:

| Platform Identifier | Description | Example Use Case |
|-------------------|-------------|------------------|
| `M1_Mac_16GB` | M1 Mac Mini with 16GB RAM | Local development |
| `M2_Mac_32GB` | M2 Mac with 32GB RAM | Local development |
| `GitHub_Actions_macOS` | GitHub Actions CI runner | CI/CD automation |
| `RTX_4090_24GB` | RTX 4090 GPU server | Production |
| `A100_80GB` | NVIDIA A100 GPU | Cloud/RunPod |
| `RunPod_A100` | RunPod with A100 | Cloud deployment |

### Platform Detection

Platform is automatically detected based on:
- CPU type (M1, M2, x86_64)
- System memory
- GPU type (via nvidia-smi or torch)
- Environment variables (CI, GITHUB_ACTIONS, RUNPOD_POD_ID)

**Override detection:**
```bash
PERFORMANCE_PLATFORM=M1_Mac_16GB pytest tests/test_performance_regression.py
```

### Baseline File Format

The baselines file (`tests/performance_baselines.json`) uses a multi-platform format:

```json
{
  "M1_Mac_16GB": {
    "embedding_throughput": 67.0,
    "vector_search_latency": 0.011,
    "query_latency_no_vllm": 8.0,
    "db_insertion_throughput": 1250.0,
    "peak_memory_gb": 12.5,
    "cache_hit_rate": 0.42,
    "tokens_per_second": 15.0,
    "avg_mrr": 0.85,
    "metadata": {
      "last_updated": "2025-01-07T10:00:00Z",
      "git_commit": "abc123",
      "python_version": "3.11.9"
    }
  },
  "GitHub_Actions_macOS": {
    "embedding_throughput": 50.0,
    "vector_search_latency": 0.015,
    "metadata": {
      "last_updated": "2025-01-07T10:00:00Z",
      "notes": "CI environment ~25% slower than local M1"
    }
  }
}
```

---

## CI/CD Integration

### Automated Checks on Pull Requests

When you create a PR to `main`, the `performance-regression` job automatically:

1. ✅ Runs performance tests with recording enabled
2. ✅ Generates performance report
3. ✅ Posts report as PR comment
4. ✅ Uploads artifacts (90-day retention)
5. ⚠️ **Blocks PR if >20% regression detected**

**Example PR Comment:**
```markdown
## 📊 Performance Report
**Platform**: `GitHub_Actions_macOS`

### 📈 Metrics Comparison
| Metric | Current | Baseline | Change | Status |
|--------|---------|----------|--------|--------|
| query_latency | 12.5s | 12.0s | +4.2% | ✅ |
| embedding_throughput | 48 c/s | 50 c/s | -4.0% | ✅ |

### ✅ No Regressions Detected
```

### Nightly Comprehensive Benchmarks

Every night at 2 AM UTC, the `nightly-benchmark` workflow:

1. 🔄 Runs comprehensive benchmark suite
2. 📊 Generates HTML, Markdown, and JSON reports
3. 📁 Uploads artifacts (365-day retention)
4. 🔔 Creates GitHub issue if regression detected
5. 🎯 Auto-updates baselines on sustained improvements

**View nightly results:**
- Go to Actions tab → Nightly Performance Benchmark
- Download artifacts from latest run
- Open `dashboard.html` to see trends

### Manual Workflow Trigger

Trigger benchmarks manually:

1. Go to Actions tab
2. Select "Nightly Performance Benchmark"
3. Click "Run workflow"
4. Wait for completion (~5-10 minutes)
5. Download artifacts

---

## Dashboard & Visualization

### Interactive Plotly Dashboard

The dashboard shows 8 key metrics over time with:
- Line charts with markers
- Baseline reference lines (dashed)
- Regression thresholds (dotted, red)
- Multi-platform comparison
- Hover details
- Zoom and pan

**Metrics visualized:**
1. Query Latency (s)
2. Embedding Throughput (chunks/s)
3. Vector Search Latency (s)
4. Memory Usage (GB)
5. Cache Hit Rate (%)
6. DB Insertion Throughput (nodes/s)
7. RAG Quality (MRR)
8. Tokens per Second

**Usage:**
```bash
# Generate dashboard
python scripts/generate_performance_dashboard.py

# Custom date range
python scripts/generate_performance_dashboard.py --days 90

# Specific platforms
python scripts/generate_performance_dashboard.py \
  --platforms M1_Mac_16GB RTX_4090_24GB

# Custom output
python scripts/generate_performance_dashboard.py --output custom-dashboard.html
```

**Tips:**
- The dashboard is a standalone HTML file (4-5MB)
- No server required, just open in browser
- Interactive: hover, zoom, pan
- Share via email or upload to S3

---

## Baseline Management

### When to Update Baselines

Update baselines when:
- ✅ Code optimization improves performance **consistently** (5+ runs showing improvement)
- ✅ Hardware upgrade (e.g., new GPU, more memory)
- ✅ Python version upgrade
- ✅ Intentional architectural change that affects performance

**DO NOT update for:**
- ❌ Temporary spikes or one-off improvements
- ❌ Performance regressions
- ❌ Measurement noise

### How to Update Baselines

#### Interactive Mode (Recommended)

```bash
# Run update script
python scripts/update_baselines.py

# Output:
# ======================================================================
# Baseline Update Proposals for M1_Mac_16GB
# ======================================================================
# Metric                         Current      Proposed     Change
# ----------------------------------------------------------------------
# query_latency_no_vllm          8.000        7.500             6.2%
# embedding_throughput           67.0         70.5              5.2%
#
# Apply these updates? (yes/no):
```

Type `yes` to apply, `no` to cancel.

#### Dry-Run Mode (Preview)

```bash
# See proposed changes without applying
python scripts/update_baselines.py --dry-run

# Output shows proposals, but no changes made
```

#### Auto-Approve Mode (CI)

```bash
# Auto-approve improvements (for CI/nightly jobs)
python scripts/update_baselines.py --auto-approve-improvements --min-runs 5
```

#### Specific Platform

```bash
# Update baselines for specific platform
python scripts/update_baselines.py --platform M1_Mac_16GB
```

### Update Process

1. **Analysis**: Script analyzes last 5+ runs
2. **Median Calculation**: Calculates median for each metric
3. **Improvement Detection**: Only proposes if >5% improvement
4. **Approval**: Interactive or automatic
5. **File Update**: Updates `tests/performance_baselines.json`
6. **Git Commit**: You commit the changes

**After update:**
```bash
# Review changes
git diff tests/performance_baselines.json

# Commit
git add tests/performance_baselines.json
git commit -m "perf: update baselines for M1_Mac_16GB"
git push
```

### Safeguards

The update script includes multiple safeguards:

- ✅ **Minimum runs required**: Needs 5+ consecutive runs (configurable)
- ✅ **Improvement threshold**: Only updates if >5% better
- ✅ **No regressions**: Never updates on worse performance
- ✅ **Interactive approval**: Default requires explicit `yes`
- ✅ **Git metadata**: Tracks commit and branch in baseline file
- ✅ **Dry-run mode**: Preview changes safely

---

## Tracked Metrics

### Core Metrics

| Metric | Unit | Direction | Description |
|--------|------|-----------|-------------|
| `embedding_throughput` | chunks/sec | ↑ Higher is better | Document embedding speed |
| `vector_search_latency` | seconds | ↓ Lower is better | pgvector search time |
| `query_latency_no_vllm` | seconds | ↓ Lower is better | End-to-end query (llama.cpp) |
| `query_latency_vllm` | seconds | ↓ Lower is better | End-to-end query (vLLM) |
| `db_insertion_throughput` | nodes/sec | ↑ Higher is better | Database write speed |
| `peak_memory_gb` | GB | ↓ Lower is better | Maximum memory usage |
| `avg_memory_gb` | GB | ↓ Lower is better | Average memory usage |
| `cache_hit_rate` | 0-1 | ↑ Higher is better | Query cache effectiveness |
| `tokens_per_second` | tokens/sec | ↑ Higher is better | LLM generation speed |
| `avg_mrr` | 0-1 | ↑ Higher is better | Mean Reciprocal Rank (quality) |
| `avg_ndcg` | 0-1 | ↑ Higher is better | Normalized DCG (quality) |

### Regression Thresholds

- **Default threshold**: 20% deviation from baseline
- **Configurable** via `REGRESSION_THRESHOLD` environment variable
- **Direction-aware**:
  - Latency: >20% slower = regression
  - Throughput: >20% slower = regression

### Data Collection

Metrics are collected from:
- **Unit tests**: `tests/test_performance_regression.py`
- **Benchmark suite**: `utils/rag_benchmark.py`
- **Performance monitor**: `utils/performance_optimizations.py`

---

## Troubleshooting

### "No baseline for platform"

**Cause**: No baseline exists for your platform in `tests/performance_baselines.json`

**Solution**:
```bash
# Run tests to collect initial data
ENABLE_PERFORMANCE_RECORDING=1 pytest tests/test_performance_regression.py -v

# Check if data was recorded
python -c "
from utils.performance_history import PerformanceHistory
history = PerformanceHistory()
print(f'Total runs: {history.get_stats()[\"total_runs\"]}')
"

# Update baselines with initial data
python scripts/update_baselines.py --dry-run
python scripts/update_baselines.py
```

### False Positive Regressions

**Cause**: System load, background processes, thermal throttling

**Solutions**:
1. **Close unnecessary applications**
2. **Run multiple times** to verify consistency
3. **Check system load**: `top` or Activity Monitor
4. **Temporarily increase threshold**:
   ```bash
   REGRESSION_THRESHOLD=0.30 pytest tests/test_performance_regression.py
   ```
5. **Use dedicated test machine** for consistent results

### "Database locked" Error

**Cause**: Concurrent writes to SQLite database

**Solution**:
- SQLite handles this automatically with retry logic
- If persistent, check for zombie processes:
  ```bash
  lsof benchmarks/history/performance.db
  ```

### Dashboard Shows No Data

**Cause**: No performance runs recorded to database

**Solution**:
```bash
# Check database
python -c "
from utils.performance_history import PerformanceHistory
history = PerformanceHistory()
stats = history.get_stats()
print(f'Total runs: {stats[\"total_runs\"]}')
print(f'Platforms: {history.get_available_platforms()}')
"

# If empty, run tests with recording
ENABLE_PERFORMANCE_RECORDING=1 pytest tests/test_performance_regression.py

# Try dashboard again
python scripts/generate_performance_dashboard.py
```

### Baseline Update Shows No Improvements

**Possible causes:**
1. Performance hasn't improved (expected)
2. Improvement < 5% threshold
3. Insufficient runs (need 5+)

**Check:**
```bash
# How many recent runs?
python -c "
from utils.performance_history import PerformanceHistory
from utils.platform_detection import detect_platform
history = PerformanceHistory()
platform = detect_platform()
runs = history.get_recent_runs(platform, limit=10)
print(f'Recent runs: {len(runs)}')
"

# View recent performance
python -c "
from utils.performance_history import PerformanceHistory
from utils.platform_detection import detect_platform
history = PerformanceHistory()
platform = detect_platform()
runs = history.get_recent_runs(platform, limit=5)
for i, run in enumerate(runs):
    print(f'Run {i+1}: {run.get(\"query_latency_no_vllm\")}s')
"
```

---

## Advanced Usage

### Query Historical Data

```python
from utils.performance_history import PerformanceHistory
from utils.platform_detection import detect_platform

history = PerformanceHistory()
platform = detect_platform()

# Get recent runs
runs = history.get_recent_runs(platform, limit=10)
for run in runs:
    print(f"{run['timestamp']}: {run.get('query_latency_no_vllm')}s")

# Get trend data
trend = history.get_trend(platform, "query_latency_no_vllm", days=30)
print(f"Trend points: {len(trend)}")

# Get baseline
baseline = history.get_baseline(platform, "embedding_throughput")
print(f"Baseline: {baseline:.2f} chunks/sec")

# Compare current to baseline
current_metrics = {"embedding_throughput": 65.0}
comparison = history.compare_to_baseline(current_metrics, platform)
print(f"Regressions: {len(comparison['regressions'])}")
print(f"Improvements: {len(comparison['improvements'])}")
```

### Custom Benchmark

```python
from utils.rag_benchmark import RAGBenchmark
from utils.performance_history import PerformanceHistory
from utils.platform_detection import detect_platform, get_git_metadata

# Run custom benchmark
benchmark = RAGBenchmark()
results = benchmark.run_end_to_end_benchmark(
    query_engine=my_engine,
    test_queries=my_test_queries
)

# Record to history
history = PerformanceHistory()
history.record_run(
    metrics={
        "avg_mrr": results.retrieval_metrics.avg_mrr,
        "avg_ndcg": results.retrieval_metrics.avg_ndcg,
        "query_latency_no_vllm": results.performance_metrics.total_latency_ms / 1000
    },
    metadata={
        "platform": detect_platform(),
        **get_git_metadata(),
        "run_type": "custom"
    }
)

# Generate report
benchmark.save_report(results, "my_benchmark.html", format="html")
```

### Export Data

```python
from utils.performance_history import PerformanceHistory

history = PerformanceHistory()

# Export to JSON
history.export_to_json("M1_Mac_16GB", "m1_performance.json")

# Or manually query and export
import json
runs = history.get_recent_runs("M1_Mac_16GB", limit=100)
with open("performance_export.json", "w") as f:
    json.dump(runs, f, indent=2)
```

### Database Maintenance

```bash
# Check database size
ls -lh benchmarks/history/performance.db

# Backup database
cp benchmarks/history/performance.db benchmarks/history/performance.db.backup

# Compact database (if needed)
sqlite3 benchmarks/history/performance.db "VACUUM;"
```

---

## Environment Variables

### Core Variables

```bash
# Enable recording to history database
ENABLE_PERFORMANCE_RECORDING=1

# Type of performance run
PERFORMANCE_RUN_TYPE=ci          # ci, nightly, manual

# Override auto-detected platform
PERFORMANCE_PLATFORM=M1_Mac_16GB

# Custom database path
PERFORMANCE_DB_PATH=./benchmarks/history/performance.db

# Regression detection threshold (default 0.20 = 20%)
REGRESSION_THRESHOLD=0.20

# Benchmark mode
BENCHMARK_MODE=quick             # quick, standard, comprehensive
```

### Usage Examples

```bash
# CI environment
ENABLE_PERFORMANCE_RECORDING=1 \
PERFORMANCE_RUN_TYPE=ci \
pytest tests/test_performance_regression.py

# Nightly comprehensive benchmark
ENABLE_PERFORMANCE_RECORDING=1 \
PERFORMANCE_RUN_TYPE=nightly \
BENCHMARK_MODE=comprehensive \
python scripts/run_comprehensive_benchmark.py

# Custom platform and threshold
PERFORMANCE_PLATFORM=Custom_Server \
REGRESSION_THRESHOLD=0.15 \
pytest tests/test_performance_regression.py
```

---

## Best Practices

### For Development

1. **Run performance tests regularly** (weekly)
2. **Check baselines** before major refactoring
3. **Use dry-run** when exploring baseline updates
4. **Keep dashboard updated** for team visibility

### For CI/CD

1. **Let CI catch regressions** (don't disable checks)
2. **Review nightly reports** weekly
3. **Update baselines** on intentional improvements
4. **Investigate issues** created by nightly workflow

### For Baseline Management

1. **Require 5+ runs** before updating (default)
2. **Use interactive mode** for manual updates
3. **Document** why baselines changed (in commit message)
4. **Review trends** in dashboard before updating

### For Multi-Platform

1. **Maintain separate baselines** per platform
2. **Don't compare** across platforms directly
3. **Document platform specs** in baseline metadata
4. **Test on target platform** before deployment

---

## References

- **Source Code**: `utils/performance_history.py`, `utils/platform_detection.py`
- **Tests**: `tests/test_performance_regression.py`
- **Scripts**: `scripts/generate_performance_*.py`, `scripts/update_baselines.py`
- **Workflows**: `.github/workflows/ci.yml`, `.github/workflows/nightly-benchmark.yml`
- **Baselines**: `tests/performance_baselines.json`

---

## Getting Help

- **Issues**: File bug reports or feature requests on GitHub
- **Questions**: Ask in project discussions
- **Logs**: Check `benchmarks/history/performance.db` for recorded data
- **Dashboard**: Visual debugging with `python scripts/generate_performance_dashboard.py`
