# System Resource Optimization Guide

**Target System**: M1 Mac Mini 16GB RAM
**Created**: 2026-01-09
**Issue**: High memory pressure, swap thrashing, load average > 8

---

## Current System Status (from btop)

### Critical Metrics
- **CPU Load**: 3.32 (1m), 6.21 (5m), **8.28 (15m)** ← At capacity
- **RAM Used**: 7.43 GiB / 16.0 GiB (46%)
- **Swap Used**: 7.83 GiB / 9.00 GiB (87%) 🔴 **CRITICAL**
- **Free RAM**: 77.1 MiB (0%) 🔴 **CRITICAL**
- **Disk Used**: 1.51 TiB / 1.81 TiB (84%)

### Memory Breakdown
| Process | RAM Usage | CPU % | Notes |
|---------|-----------|-------|-------|
| Docker VM (postgres+monitoring) | 579M | 0.4% | Can be reduced |
| Arc Browser (total) | ~1.5GB | 2-5% | 10+ processes |
| iTerm2 | 297M | 2.7% | Normal |
| **RAG Pipeline (python3.11)** | **152M** | **2.8%** | ✅ Reasonable |
| Browser helpers (various) | 100-216M each | 0-0.5% | Arc overhead |

---

## Optimization Strategy

### Phase 1: Immediate Actions (Do Now)

#### 1. Free Up Browser Memory (~500-800MB)
**Arc Browser is consuming ~1.5GB across 10+ processes**

```bash
# Close unused Arc tabs/windows
# Restart Arc to consolidate memory
```

**Expected savings**: 500-800MB

#### 2. Stop Heavy Monitoring Stack (~400MB)
Your full monitoring stack (Grafana, Prometheus, Alertmanager, etc.) is overkill for development.

```bash
# Stop current stack
cd config
docker-compose down

# Use minimal config (PostgreSQL only)
docker-compose -f docker-compose.minimal.yml up -d
```

**Expected savings**: 300-400MB
**You keep**: PostgreSQL with pgvector (everything you need for RAG)
**You lose**: Grafana dashboards (rarely used during development)

#### 3. Reduce RAG Pipeline Memory Footprint (~150MB)
**Current settings** are optimized for throughput, not constrained memory:

```bash
# Add to .env file
EMBED_BATCH=32     # Reduce from 64 (saves ~50MB peak)
N_BATCH=128        # Reduce from 256 (saves ~100MB peak)
N_GPU_LAYERS=20    # Reduce from 24 (saves ~50MB VRAM)
```

**Expected savings**: ~150MB peak usage
**Trade-off**: 10-15% slower indexing (still acceptable)

---

### Phase 2: Switch to Memory-Optimized Mode

#### Use the New Minimal Launch Script

```bash
# Stop current full stack
./shutdown.sh

# Launch minimal stack (saves ~400MB)
./launch_minimal.sh
```

**What this does**:
1. Launches PostgreSQL with memory limits (512MB max)
2. Skips Grafana, Prometheus, monitoring agents
3. Uses optimized PostgreSQL settings
4. Starts Streamlit UI

**Memory savings**: ~400-500MB total

---

### Phase 3: Long-term Optimizations

#### 1. Use Native PostgreSQL (No Docker)
If you have PostgreSQL installed locally:

```bash
# Install PostgreSQL with Homebrew
brew install postgresql@16 pgvector

# Configure and start
brew services start postgresql@16
```

**Savings**: ~300MB (no Docker VM overhead)

#### 2. Process Documents in Batches
For large datasets (2239+ files), process incrementally:

```bash
# Index in smaller batches
for batch in data/251218-messenger/batch_*; do
    PDF_PATH="$batch" PGTABLE=messenger_batch python rag_low_level_m1_16gb_verbose.py
    sleep 60  # Cool down between batches
done
```

#### 3. Reduce Active Browser Usage
- Close Arc when not needed
- Use Safari for browsing (lower memory footprint)
- Keep Arc only for development tools

---

## Quick Reference Commands

### Check Memory Pressure
```bash
# macOS memory pressure
sudo memory_pressure

# Detailed breakdown
vm_stat

# Process memory usage
ps aux | sort -nrk 4 | head -10
```

### Monitor System Load
```bash
# Real-time monitoring
btop

# Or simpler
htop

# Load average only
uptime
```

### Docker Memory Usage
```bash
# See container memory
docker stats

# Set memory limits
docker update --memory 512m --memory-swap 1g rag_postgres
```

### Clear Swap (if needed)
```bash
# Disable swap temporarily to force cleanup
sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.dynamic_pager.plist
sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.dynamic_pager.plist
```

---

## Recommended Configuration for M1 16GB

### For Development (Current Use Case)
```bash
# .env settings
EMBED_BATCH=32
N_BATCH=128
N_GPU_LAYERS=20
CHUNK_SIZE=700
TOP_K=3

# Launch
./launch_minimal.sh
```

**Memory footprint**: ~2.5GB total (PostgreSQL 512MB + RAG 400MB + OS overhead)

### For Production Indexing (Batch Processing)
```bash
# .env settings
EMBED_BATCH=64
N_BATCH=256
N_GPU_LAYERS=24

# Launch full stack
./launch.sh
```

**Memory footprint**: ~4GB total
**Requirement**: Close browser, non-critical apps

---

## Troubleshooting

### Issue: Still High Swap Usage
**Cause**: Too many background applications
**Fix**:
```bash
# Find memory hogs
ps aux | sort -nrk 4 | head -20

# Kill non-essential processes
killall "Application Name"
```

### Issue: PostgreSQL OOM (Out of Memory)
**Cause**: Database memory limits too low
**Fix**:
```bash
# Increase PostgreSQL memory in docker-compose.minimal.yml
mem_limit: 1g  # Increase from 512m
```

### Issue: Embedding Process Crashes
**Cause**: EMBED_BATCH too high
**Fix**:
```bash
# Reduce batch size
EMBED_BATCH=16 python rag_low_level_m1_16gb_verbose.py
```

---

## Performance Benchmarks

### Full Stack (Before Optimization)
- **Memory**: 7.43 GiB + 7.83 GiB swap = 15.26 GiB total
- **Load**: 8.28 (15min avg)
- **Index Speed**: ~67 chunks/s
- **Query Speed**: ~0.3s retrieval + 5-15s generation

### Minimal Stack (After Optimization)
- **Expected Memory**: ~4-5 GiB + minimal swap
- **Expected Load**: 3-5 (15min avg)
- **Index Speed**: ~60 chunks/s (10% slower)
- **Query Speed**: Same (no impact)

---

## Summary

### What's Working
✅ RAG pipeline is stable (152M RAM, 2.8% CPU)
✅ Document loading successful (2238/2239 files)
✅ PostgreSQL healthy

### What Needs Fixing
🔴 Swap thrashing (87% usage)
🔴 Memory pressure (77MB free)
⚠️ High load average (8.28)

### Priority Actions
1. **Now**: Close Arc tabs, stop monitoring stack → Free ~1GB
2. **Next**: Use `./launch_minimal.sh` → Save ~400MB
3. **Soon**: Reduce EMBED_BATCH/N_BATCH → Save ~150MB

### Expected Outcome
- **Swap usage**: 87% → <20%
- **Free RAM**: 77MB → 4-6GB
- **Load average**: 8.28 → 3-5
- **Performance**: Significantly faster, more responsive system

---

**Last Updated**: 2026-01-09
**Status**: Ready to implement
