# Quick Fix: Get Real macOS Metrics in Grafana

**Time required**: 10 minutes
**Fixes**: Grafana now shows actual M1 Mac metrics (matching btop)

---

## Problem Summary

Grafana currently shows Docker VM metrics, not your actual M1 Mac metrics.
This quick fix adds a native macOS metrics exporter.

---

## Installation Steps

### 1. Install Required Package (2 min)

```bash
cd ~/code/llamaIndex-local-rag
source .venv/bin/activate
pip install prometheus-client
```

### 2. Test the Exporter (1 min)

```bash
# Run the exporter
python macos_exporter.py --port 9101

# You should see:
# INFO - Starting metrics server on port 9101
# INFO - Metrics available at http://localhost:9101/metrics

# Test it (in another terminal):
curl http://localhost:9101/metrics | head -20
```

You should see metrics like:
```
# HELP macos_cpu_percent CPU usage percentage
# TYPE macos_cpu_percent gauge
macos_cpu_percent 42.3

# HELP macos_memory_percent Memory usage percentage
# TYPE macos_memory_percent gauge
macos_memory_percent 46.4

# HELP macos_swap_percent Swap usage percentage
# TYPE macos_swap_percent gauge
macos_swap_percent 87.0
```

### 3. Update Prometheus Config (3 min)

Edit `config/monitoring/prometheus.yml` and add this scrape config:

```yaml
scrape_configs:
  # ... existing configs ...

  # macOS Host Metrics (native exporter)
  - job_name: 'macos-host'
    static_configs:
      - targets: ['host.docker.internal:9101']
        labels:
          service: 'macos-host'
          instance: 'm1-mac-mini'
```

**Full example**:
```bash
cd config/monitoring

# Backup current config
cp prometheus.yml prometheus.yml.backup

# Add the new job (append to end of scrape_configs)
cat >> prometheus.yml << 'EOF'

  # macOS Host Metrics (native exporter)
  - job_name: 'macos-host'
    static_configs:
      - targets: ['host.docker.internal:9101']
        labels:
          service: 'macos-host'
          instance: 'm1-mac-mini'
EOF
```

### 4. Restart Prometheus (1 min)

```bash
cd ~/code/llamaIndex-local-rag/config
docker-compose restart prometheus

# Wait for it to restart
sleep 5

# Check it's scraping
docker logs rag_prometheus | grep "macos-host"
```

You should see logs like:
```
level=info msg="Scrape target succeeded" target=macos-host
```

### 5. Verify in Prometheus UI (1 min)

1. Open http://localhost:9090
2. Go to **Status → Targets**
3. Find `macos-host` - should be **UP** (green)
4. Go to **Graph** tab
5. Try queries:
   - `macos_cpu_percent`
   - `macos_memory_percent`
   - `macos_swap_percent`

You should see real-time data matching btop!

### 6. Create Grafana Dashboard (2 min)

**Option A: Import pre-built dashboard** (coming soon)

**Option B: Quick manual dashboard**:

1. Open http://localhost:3000 (admin/admin)
2. Click **+** → **Dashboard** → **Add visualization**
3. Select **Prometheus** data source
4. Add these queries:

**Panel 1: CPU Usage**
```
Query: macos_cpu_percent
Title: M1 Mac CPU Usage
Unit: Percent (0-100)
```

**Panel 2: Memory Usage**
```
Query: macos_memory_percent
Title: M1 Mac Memory Usage
Unit: Percent (0-100)
Threshold: Yellow > 70, Red > 85
```

**Panel 3: Swap Usage**
```
Query: macos_swap_percent
Title: M1 Mac Swap Usage
Unit: Percent (0-100)
Threshold: Yellow > 50, Red > 80
```

**Panel 4: Load Average**
```
Query: macos_load_avg_15m
Title: Load Average (15m)
Unit: Short
Threshold: Yellow > 6, Red > 8
```

5. Save dashboard as **"M1 Mac System Metrics"**

---

## Running the Exporter Automatically

### Option 1: Run in Background (Quick)

```bash
cd ~/code/llamaIndex-local-rag

# Start exporter in background
nohup .venv/bin/python macos_exporter.py --port 9101 > logs/macos_exporter.log 2>&1 &

# Save PID for later
echo $! > logs/macos_exporter.pid

# Check it's running
tail -f logs/macos_exporter.log

# To stop later:
# kill $(cat logs/macos_exporter.pid)
```

### Option 2: Add to Launch Script (Recommended)

Update your `launch.sh` to start the exporter automatically:

```bash
# Add before starting Streamlit:

echo "5️⃣  Starting macOS metrics exporter..."
if ! pgrep -f "macos_exporter.py" > /dev/null; then
    nohup .venv/bin/python macos_exporter.py --port 9101 > logs/macos_exporter.log 2>&1 &
    echo "   ✓ macOS exporter started on port 9101"
else
    echo "   ✓ macOS exporter already running"
fi
```

---

## Verification Checklist

After completing all steps, verify everything works:

- [ ] Exporter running: `curl http://localhost:9101/metrics`
- [ ] Prometheus scraping: http://localhost:9090/targets (macos-host UP)
- [ ] Metrics queryable: `macos_cpu_percent` returns data
- [ ] Grafana shows data: Dashboard displays live metrics
- [ ] **Metrics match btop**: CPU/memory/swap values are similar

---

## Available Metrics

Once configured, you'll have these metrics in Grafana:

### CPU Metrics
- `macos_cpu_percent` - CPU usage (%)
- `macos_cpu_count_physical` - Physical cores (4)
- `macos_cpu_count_logical` - Logical cores (8)
- `macos_load_avg_1m/5m/15m` - Load averages

### Memory Metrics
- `macos_memory_total_bytes` - Total RAM (16 GB)
- `macos_memory_available_bytes` - Available RAM
- `macos_memory_used_bytes` - Used RAM
- `macos_memory_percent` - Memory usage (%)
- `macos_memory_active_bytes` - Active memory
- `macos_memory_wired_bytes` - Wired memory

### Swap Metrics
- `macos_swap_total_bytes` - Total swap (9 GB)
- `macos_swap_used_bytes` - Used swap
- `macos_swap_percent` - Swap usage (%)
- `macos_swap_sin_bytes` - Bytes swapped in
- `macos_swap_sout_bytes` - Bytes swapped out

### Disk Metrics
- `macos_disk_usage_percent{path="/"}` - Disk usage (%)
- `macos_disk_total_bytes{path="/"}` - Total disk space
- `macos_disk_free_bytes{path="/"}` - Free disk space

### Process Metrics
- `macos_process_count` - Total processes
- `macos_process_running_count` - Running processes
- `macos_process_sleeping_count` - Sleeping processes

### Network Metrics
- `macos_network_bytes_sent_total{interface="en0"}` - Bytes sent
- `macos_network_bytes_recv_total{interface="en0"}` - Bytes received

---

## Example Grafana Queries

### CPU Dashboard
```
// CPU Usage
macos_cpu_percent

// CPU Cores Utilized
macos_load_avg_15m / macos_cpu_count_logical * 100

// CPU Load per Core
macos_load_avg_15m
```

### Memory Dashboard
```
// Memory Usage %
macos_memory_percent

// Memory Available (GB)
macos_memory_available_bytes / 1024 / 1024 / 1024

// Memory Breakdown (pie chart)
macos_memory_active_bytes (Active)
macos_memory_wired_bytes (Wired)
macos_memory_free_bytes (Free)
```

### Swap Dashboard
```
// Swap Usage %
macos_swap_percent

// Swap Activity Rate (KB/s)
rate(macos_swap_sin_bytes[1m]) / 1024 (Swap In)
rate(macos_swap_sout_bytes[1m]) / 1024 (Swap Out)
```

### System Overview Dashboard
```
// Combined Memory Pressure
(macos_memory_percent + macos_swap_percent) / 2

// System Stress Score (0-100)
(
  (macos_cpu_percent * 0.3) +
  (macos_memory_percent * 0.4) +
  (macos_swap_percent * 0.3)
)

// Disk Space Alert
100 - macos_disk_usage_percent{path="/"}
```

---

## Troubleshooting

### Exporter Won't Start
```bash
# Check if port is in use
lsof -i :9101

# Try different port
python macos_exporter.py --port 9102
```

### Prometheus Can't Scrape
```bash
# Check if host.docker.internal works
docker run --rm curlimages/curl curl http://host.docker.internal:9101/metrics

# If fails, use your Mac's IP:
ifconfig | grep "inet " | grep -v 127.0.0.1
# Use that IP in prometheus.yml instead of host.docker.internal
```

### Metrics Show Wrong Values
```bash
# Compare with btop
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%'); print(f'Mem: {psutil.virtual_memory().percent}%'); print(f'Swap: {psutil.swap_memory().percent}%')"
```

### Grafana Shows "No Data"
1. Check Prometheus targets: http://localhost:9090/targets
2. Query Prometheus directly: http://localhost:9090/graph
3. Check time range in Grafana (top right)
4. Refresh dashboard (Ctrl+R)

---

## Stop/Restart Exporter

```bash
# Find PID
pgrep -f "macos_exporter.py"

# Stop
pkill -f "macos_exporter.py"

# Start
cd ~/code/llamaIndex-local-rag
.venv/bin/python macos_exporter.py --port 9101 &
```

---

## What's Next?

After this fix, you can:

1. **Compare with btop**: Metrics should now match!
2. **Set up alerts**: Configure alerting in Prometheus/Grafana
3. **Add RAG metrics**: Instrument your application (see GRAFANA_BTOP_DISCREPANCY.md)
4. **Create custom dashboards**: Build visualizations for your specific needs

---

**Total time**: ~10 minutes
**Result**: Grafana now shows real M1 Mac metrics matching btop! 🎉
