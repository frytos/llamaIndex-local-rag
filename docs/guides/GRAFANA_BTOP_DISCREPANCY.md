# Grafana vs btop Data Discrepancy - Root Cause Analysis

**Date**: 2026-01-09
**Issue**: Grafana dashboards don't show the same data as btop

---

## Root Cause Identified

### The Problem

**btop shows**: Real M1 Mac system metrics
- 16GB total RAM
- 8 CPU cores (Apple Silicon)
- Real macOS processes (Arc, iTerm2, python3.11)
- Actual swap usage (7.83 GiB)

**Grafana shows**: Docker VM metrics (not your actual Mac)
- Docker VM internal resources
- Container-only metrics
- Disconnected from host reality

### Why This Happens

Docker on macOS runs inside a **Linux Virtual Machine** (Docker Desktop VM). When you mount `/proc`, `/sys`, `/` into node_exporter, you're mounting the **VM's** filesystem, not the Mac's.

```
┌─────────────────────────────────────┐
│         Your M1 Mac (Host)          │
│  ┌───────────────────────────────┐  │
│  │   Docker Desktop VM (Linux)   │  │ ← node_exporter sees THIS
│  │  ┌─────────────────────────┐  │  │
│  │  │  Containers (Postgres,  │  │  │
│  │  │  Grafana, Prometheus)   │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
       ↑
   btop sees THIS (the actual Mac)
```

---

## Missing Metrics Breakdown

### 1. RAG Application Metrics (Missing)

**Dashboard**: `rag_overview.json`
**Expected Metrics**:
- `rag_query_total`
- `rag_query_success_total`
- `rag_query_duration_seconds_bucket`
- `rag_retrieval_score_avg`
- `rag_cache_hit_rate`

**Status**: ❌ **Not emitted**

**Reason**: Your RAG application (`rag_web.py`, `rag_low_level_m1_16gb_verbose.py`) doesn't use the `utils/metrics.py` instrumentation.

**Evidence**:
```bash
# Metrics module exists
$ ls utils/metrics.py
utils/metrics.py

# But NOT imported in main files
$ grep "from utils.metrics" rag_web.py rag_low_level_m1_16gb_verbose.py
# (no results)
```

### 2. Host System Metrics (Wrong Source)

**Dashboard**: `system_overview.json`
**Expected**: M1 Mac host metrics
**Actual**: Docker VM metrics

**Metrics affected**:
- `node_memory_MemAvailable_bytes` - Shows VM memory, not Mac memory
- `node_cpu_seconds_total` - Shows VM CPU, not Mac CPU cores
- `node_filesystem_avail_bytes` - Shows VM disk, not Mac disk

**Status**: ⚠️ **Partially working** (but wrong data source)

### 3. Container Metrics (Working)

**Source**: cAdvisor
**Metrics**:
- `container_memory_usage_bytes{name="rag_postgres"}`
- `container_cpu_usage_seconds_total{name="rag_postgres"}`

**Status**: ✅ **Working correctly** (these are accurate for containers)

### 4. PostgreSQL Metrics (Working)

**Source**: postgres_exporter
**Metrics**:
- `pg_up`
- `pg_database_size_bytes`
- `pg_stat_database_numbackends`

**Status**: ✅ **Working correctly**

---

## Solutions

### Solution 1: Enable RAG Application Metrics (Quick Fix)

**Integrate metrics into your RAG pipeline**:

#### Step 1: Add to `rag_web.py`

Add metrics tracking to your Streamlit app:

```python
# At top of file (after existing imports)
from utils.metrics import get_metrics

# Initialize metrics
metrics = get_metrics()

# In query handling code (where you process queries):
with metrics.query_timer():
    try:
        response = query_engine.query(user_question)
        metrics.record_query_success()
        metrics.record_retrieval(
            num_documents=len(response.source_nodes),
            scores=[node.score for node in response.source_nodes]
        )
    except Exception as e:
        metrics.record_query_error(type(e).__name__)
        raise

# Export metrics periodically (add to app)
if st.sidebar.button("Export Metrics"):
    output_path = metrics.export()
    st.success(f"Metrics exported to {output_path}")
```

#### Step 2: Configure Prometheus to Read File-Based Metrics

Update `config/monitoring/prometheus.yml`:

```yaml
scrape_configs:
  # Add this job for RAG application metrics
  - job_name: 'rag-app'
    static_configs:
      - targets: ['host.docker.internal:9091']  # If using pushgateway
        labels:
          service: 'rag-application'
    # Or use file-based service discovery
    file_sd_configs:
      - files:
          - '/prometheus/sd/*.json'
        refresh_interval: 30s
```

#### Step 3: Export Metrics Regularly

Add a background thread to export metrics:

```python
# In rag_web.py
import threading
import time

def export_metrics_loop():
    while True:
        metrics.export()
        time.sleep(30)  # Export every 30 seconds

# Start background thread
metrics_thread = threading.Thread(target=export_metrics_loop, daemon=True)
metrics_thread.start()
```

### Solution 2: Get Real Host Metrics (macOS Native)

**Option A: Run node_exporter natively** (Recommended)

```bash
# Install node_exporter with Homebrew
brew install node_exporter

# Run as service
brew services start node_exporter

# Or run manually
node_exporter &

# Update Prometheus config to scrape from host
# In prometheus.yml:
scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['host.docker.internal:9100']  # Use Docker's host gateway
```

**Option B: Create macOS-specific exporter**

Use a simple Python script that exposes macOS system metrics:

```python
# macos_exporter.py
import psutil
from prometheus_client import start_http_server, Gauge
import time

# Define metrics
cpu_percent = Gauge('macos_cpu_percent', 'CPU usage percentage')
memory_available = Gauge('macos_memory_available_bytes', 'Available memory in bytes')
memory_percent = Gauge('macos_memory_percent', 'Memory usage percentage')
swap_percent = Gauge('macos_swap_percent', 'Swap usage percentage')

def collect_metrics():
    while True:
        cpu_percent.set(psutil.cpu_percent(interval=1))
        mem = psutil.virtual_memory()
        memory_available.set(mem.available)
        memory_percent.set(mem.percent)
        swap = psutil.swap_memory()
        swap_percent.set(swap.percent)
        time.sleep(10)

if __name__ == '__main__':
    start_http_server(9101)  # Expose on port 9101
    collect_metrics()
```

Run it:
```bash
pip install psutil prometheus_client
python macos_exporter.py &

# Add to Prometheus:
scrape_configs:
  - job_name: 'macos-host'
    static_configs:
      - targets: ['host.docker.internal:9101']
```

### Solution 3: Simplified Approach - Focus on What Works

**Accept the limitations and optimize for what you need**:

1. **Use Grafana for**:
   - PostgreSQL metrics (working)
   - Container metrics (working)
   - RAG application metrics (after adding instrumentation)

2. **Use btop for**:
   - Real-time host CPU/memory monitoring
   - Process-level insights
   - Swap usage monitoring

3. **Use Streamlit UI for**:
   - RAG query performance
   - Retrieval quality metrics
   - Cache hit rates

---

## Quick Implementation Guide

### Immediate Actions (15 minutes)

1. **Add metrics to rag_web.py**:
   ```bash
   # See "Solution 1" above for code changes
   ```

2. **Install psutil for native metrics**:
   ```bash
   pip install psutil prometheus_client
   ```

3. **Create and run macOS exporter**:
   ```bash
   # Create macos_exporter.py with code from "Solution 2"
   python macos_exporter.py &
   ```

4. **Update Prometheus config**:
   ```yaml
   # Add to config/monitoring/prometheus.yml
   scrape_configs:
     - job_name: 'macos-host'
       static_configs:
         - targets: ['host.docker.internal:9101']
   ```

5. **Restart Prometheus**:
   ```bash
   cd config
   docker-compose restart prometheus
   ```

6. **Create new Grafana dashboard** for macOS metrics:
   - Use `macos_cpu_percent`
   - Use `macos_memory_percent`
   - Use `macos_swap_percent`

---

## Expected Results

### After Implementing Solutions

**Grafana will show**:
- ✅ Real M1 Mac CPU usage (from macOS exporter)
- ✅ Real M1 Mac memory usage (from macOS exporter)
- ✅ Real M1 Mac swap usage (from macOS exporter)
- ✅ RAG query metrics (from instrumented application)
- ✅ PostgreSQL metrics (already working)
- ✅ Container metrics (already working)

**Data consistency**:
- Grafana metrics ≈ btop metrics (within 1-2%)
- Real-time monitoring of actual system state
- Accurate alerting based on host conditions

---

## Alternative: Accept Current State

**If you don't want to implement fixes**:

1. Use Grafana for **Docker/PostgreSQL/container metrics only**
2. Use **btop** for host system monitoring
3. Use **Streamlit UI** for RAG performance metrics
4. Document this split responsibility clearly

**Pros**:
- No code changes needed
- Each tool serves its purpose
- Clear separation of concerns

**Cons**:
- No unified monitoring view
- Manual switching between tools
- Harder to correlate issues

---

## Files to Create/Modify

### New Files
1. `macos_exporter.py` - Native macOS metrics exporter
2. `scripts/start_macos_exporter.sh` - Startup script
3. `config/grafana/dashboards/macos_host.json` - Host metrics dashboard

### Modified Files
1. `rag_web.py` - Add metrics instrumentation
2. `config/monitoring/prometheus.yml` - Add macOS scrape target
3. `launch.sh` - Start macOS exporter automatically

---

## Summary

**Current State**:
- Grafana: Shows Docker VM metrics ❌
- btop: Shows real M1 Mac metrics ✅
- **They're measuring different systems!**

**Root Causes**:
1. node_exporter in Docker sees VM, not Mac
2. RAG application doesn't emit metrics
3. No native macOS metrics collection

**Best Solution**:
1. Add metrics to RAG application (15 min)
2. Run native macOS exporter (10 min)
3. Update Prometheus config (5 min)
4. Create macOS dashboard in Grafana (10 min)

**Total time to fix**: ~40 minutes

---

**Next Steps**: Let me know which solution you'd like to implement, and I'll create the code!
