# RAG Pipeline Monitoring Guide

**Last Updated**: January 2026
**Purpose**: Monitor RAG improvements with easy-to-read dashboards and alerts

---

## Table of Contents

1. [Overview](#overview)
2. [Built-in Monitoring (Quick Start)](#built-in-monitoring-quick-start)
3. [Streamlit Dashboard (Recommended)](#streamlit-dashboard-recommended)
4. [Prometheus + Grafana (Production)](#prometheus--grafana-production)
5. [Key Metrics to Track](#key-metrics-to-track)
6. [Alert Configurations](#alert-configurations)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Your RAG pipeline now has **built-in monitoring capabilities** through the modules we created. This guide shows you how to visualize and track performance.

### What to Monitor

**Quality Metrics**:
- Answer relevance, faithfulness, confidence
- Retrieval quality (MRR, nDCG, precision, recall)
- Hallucination rate

**Performance Metrics**:
- Query latency (p50, p95, p99)
- Cache hit rate
- Throughput (queries per second)
- Component-level timing (embedding, retrieval, generation)

**System Metrics**:
- Memory usage
- Database connection pool utilization
- Error rates
- Query type distribution

### Monitoring Approaches

| Approach | Setup Time | Best For | Cost |
|----------|------------|----------|------|
| **Built-in Stats** | 5 min | Quick insights, debugging | Free |
| **Streamlit Dashboard** | 15 min | Development, demos | Free |
| **Prometheus + Grafana** | 1 hour | Production, alerts | Free (self-hosted) |
| **Cloud Monitoring** | Variable | Enterprise, managed | $$ |

---

## Built-in Monitoring (Quick Start)

### 1. Module Statistics

Every module has a `get_stats()` or `stats()` method:

```python
from utils.query_cache import semantic_cache
from utils.query_router import QueryRouter
from utils.answer_validator import AnswerValidator
from utils.conversation_memory import session_manager
from utils.performance_optimizations import PerformanceMonitor

# Get stats from all modules
cache_stats = semantic_cache.stats()
print(f"Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
print(f"Cached Queries: {cache_stats['count']}")

# Query router stats
router = QueryRouter()
# ... after routing some queries
router_stats = router.get_stats()
print(f"Query Types: {router_stats['classifications']}")
print(f"Routing Cache Hit Rate: {router_stats['cache_hit_rate']:.1%}")

# Session manager stats
session_stats = session_manager.stats()
print(f"Active Sessions: {session_stats['active_sessions']}")
print(f"Total Created: {session_stats['total_created']}")

# Performance monitor stats
monitor = PerformanceMonitor()
# ... after tracking operations
perf_stats = monitor.get_stats()
for op, metrics in perf_stats.items():
    print(f"{op}:")
    print(f"  p50: {metrics['p50']:.3f}s")
    print(f"  p95: {metrics['p95']:.3f}s")
```

### 2. Simple Logging Dashboard

Create a quick monitoring script:

```python
#!/usr/bin/env python3
"""
Simple monitoring dashboard - run periodically to check health
"""

from utils.query_cache import semantic_cache, cache
from utils.conversation_memory import session_manager

def print_monitoring_summary():
    print("=" * 70)
    print("RAG Pipeline Monitoring Summary")
    print("=" * 70)

    # Semantic cache
    print("\n📦 Semantic Cache:")
    stats = semantic_cache.stats()
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    print(f"  Total Queries: {stats['hits'] + stats['misses']}")
    print(f"  Cache Size: {stats['count']}")
    print(f"  Disk Usage: {stats['size_mb']:.2f} MB")

    # Embedding cache
    print("\n🔢 Embedding Cache:")
    embed_stats = cache.stats()
    print(f"  Cached Embeddings: {embed_stats['count']}")
    print(f"  Disk Usage: {embed_stats['size_mb']:.2f} MB")

    # Conversations
    print("\n💬 Conversations:")
    conv_stats = session_manager.stats()
    print(f"  Active Sessions: {conv_stats['active_sessions']}")
    print(f"  Total Created: {conv_stats['total_created']}")
    print(f"  Total Expired: {conv_stats['total_expired']}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    print_monitoring_summary()
```

Save as `scripts/monitor_rag.py` and run:
```bash
python scripts/monitor_rag.py
```

---

## Streamlit Dashboard (Recommended)

### Quick Start

Create a real-time monitoring dashboard with Streamlit (already in requirements.txt):

**File**: `monitoring_dashboard.py`

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# Import monitoring components
from utils.query_cache import semantic_cache, cache
from utils.conversation_memory import session_manager
from utils.performance_optimizations import PerformanceMonitor

# Page config
st.set_page_config(
    page_title="RAG Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Pipeline Monitoring Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Auto-refresh
st.sidebar.header("⚙️ Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=True)
if auto_refresh:
    time.sleep(10)
    st.rerun()

# Main metrics in columns
col1, col2, col3, col4 = st.columns(4)

# Semantic cache stats
cache_stats = semantic_cache.stats()
with col1:
    st.metric(
        "Cache Hit Rate",
        f"{cache_stats['hit_rate']:.1%}",
        delta=f"{cache_stats['hits']} hits"
    )

with col2:
    total_queries = cache_stats['hits'] + cache_stats['misses']
    st.metric(
        "Total Queries",
        total_queries,
        delta=f"{cache_stats['count']} cached"
    )

with col3:
    st.metric(
        "Cache Size",
        f"{cache_stats['size_mb']:.1f} MB",
        delta=f"{cache_stats['count']} entries"
    )

# Conversations
conv_stats = session_manager.stats()
with col4:
    st.metric(
        "Active Sessions",
        conv_stats['active_sessions'],
        delta=f"{conv_stats['total_created']} created"
    )

# Detailed sections
st.header("📈 Detailed Metrics")

tab1, tab2, tab3 = st.tabs(["Cache Performance", "Conversations", "System Health"])

with tab1:
    st.subheader("Cache Performance")

    # Cache effectiveness chart
    fig_cache = go.Figure()

    fig_cache.add_trace(go.Bar(
        name='Hits',
        x=['Semantic Cache', 'Embedding Cache'],
        y=[cache_stats['hits'], 0],  # Embedding cache doesn't track hits
        marker_color='green'
    ))

    fig_cache.add_trace(go.Bar(
        name='Misses',
        x=['Semantic Cache', 'Embedding Cache'],
        y=[cache_stats['misses'], 0],
        marker_color='red'
    ))

    fig_cache.update_layout(
        title="Cache Hits vs Misses",
        barmode='group',
        yaxis_title="Count"
    )

    st.plotly_chart(fig_cache, use_container_width=True)

    # Cache details
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Semantic Cache Details**")
        st.json({
            "enabled": cache_stats['enabled'],
            "threshold": cache_stats['threshold'],
            "max_size": cache_stats['max_size'],
            "ttl": f"{cache_stats['ttl']}s",
            "hit_rate": f"{cache_stats['hit_rate']:.2%}",
        })

    with col2:
        embed_stats = cache.stats()
        st.write("**Embedding Cache Details**")
        st.json({
            "count": embed_stats['count'],
            "size_mb": f"{embed_stats['size_mb']:.2f}",
            "cache_dir": embed_stats['cache_dir'],
        })

with tab2:
    st.subheader("Conversation Sessions")

    # Session stats
    st.write(f"**Total Sessions Created:** {conv_stats['total_created']}")
    st.write(f"**Active Sessions:** {conv_stats['active_sessions']}")
    st.write(f"**Expired Sessions:** {conv_stats['total_expired']}")

    # Active sessions list
    active_sessions = session_manager.list_active_sessions()
    if active_sessions:
        st.write(f"**Active Session IDs ({len(active_sessions)}):**")
        for session_id in active_sessions[:10]:  # Show first 10
            memory = session_manager.get(session_id)
            if memory:
                mem_stats = memory.stats()
                st.write(f"- `{session_id}`: {mem_stats['total_turns']} turns, "
                        f"{mem_stats['idle_seconds']:.0f}s idle")
    else:
        st.info("No active sessions")

with tab3:
    st.subheader("System Health")

    # System checks
    checks = []

    # Check cache health
    if cache_stats['hit_rate'] > 0.3:
        checks.append(("✅", "Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}", "Good"))
    elif cache_stats['hit_rate'] > 0.1:
        checks.append(("⚠️", "Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}", "Low"))
    else:
        checks.append(("❌", "Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}", "Very Low"))

    # Check cache size
    if cache_stats['size_mb'] < 500:
        checks.append(("✅", "Cache Size", f"{cache_stats['size_mb']:.1f} MB", "Normal"))
    elif cache_stats['size_mb'] < 1000:
        checks.append(("⚠️", "Cache Size", f"{cache_stats['size_mb']:.1f} MB", "Growing"))
    else:
        checks.append(("❌", "Cache Size", f"{cache_stats['size_mb']:.1f} MB", "High"))

    # Check session count
    if conv_stats['active_sessions'] < 50:
        checks.append(("✅", "Active Sessions", conv_stats['active_sessions'], "Normal"))
    elif conv_stats['active_sessions'] < 100:
        checks.append(("⚠️", "Active Sessions", conv_stats['active_sessions'], "Growing"))
    else:
        checks.append(("❌", "Active Sessions", conv_stats['active_sessions'], "High"))

    # Display checks
    df_health = pd.DataFrame(checks, columns=["Status", "Metric", "Value", "Assessment"])
    st.dataframe(df_health, use_container_width=True, hide_index=True)

# Footer with refresh info
st.sidebar.markdown("---")
st.sidebar.caption(f"Dashboard v1.0 | Last refresh: {datetime.now().strftime('%H:%M:%S')}")
```

**Run the dashboard**:
```bash
streamlit run monitoring_dashboard.py
```

**Features**:
- ✅ Auto-refresh every 10 seconds
- ✅ Real-time cache performance
- ✅ Conversation session tracking
- ✅ System health checks
- ✅ Interactive Plotly charts

---

## Prometheus + Grafana (Production)

### Architecture

```
RAG Pipeline → Prometheus (metrics) → Grafana (visualization)
     ↓
  Metrics exported via /metrics endpoint
```

### Setup (15 minutes)

#### Step 1: Add Prometheus Metrics Exporter

**File**: `utils/prometheus_exporter.py`

```python
"""
Prometheus metrics exporter for RAG pipeline.

Exports metrics in Prometheus format for scraping.
"""

import time
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client import REGISTRY

from utils.query_cache import semantic_cache
from utils.conversation_memory import session_manager

# Define metrics
query_counter = Counter(
    'rag_queries_total',
    'Total number of queries processed',
    ['query_type', 'cache_hit']
)

query_latency = Histogram(
    'rag_query_latency_seconds',
    'Query latency in seconds',
    ['component'],  # embedding, retrieval, generation
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

cache_hit_rate = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate (0-1)'
)

active_sessions = Gauge(
    'rag_active_sessions',
    'Number of active conversation sessions'
)

answer_confidence = Histogram(
    'rag_answer_confidence',
    'Answer confidence score',
    buckets=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)

retrieval_quality = Histogram(
    'rag_retrieval_mrr',
    'Retrieval MRR score',
    buckets=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)

# Info metric for configuration
rag_info = Info('rag_config', 'RAG configuration info')


def update_metrics():
    """Update all Prometheus metrics from module stats"""

    # Update cache metrics
    stats = semantic_cache.stats()
    total_requests = stats['hits'] + stats['misses']
    if total_requests > 0:
        cache_hit_rate.set(stats['hit_rate'])

    # Update session metrics
    conv_stats = session_manager.stats()
    active_sessions.set(conv_stats['active_sessions'])


def export_metrics():
    """Export metrics in Prometheus format"""
    update_metrics()
    return generate_latest(REGISTRY)


# Example: Track a query
def track_query(query_type: str, cache_hit: bool, latency_s: float,
                component: str, confidence: float = None, mrr: float = None):
    """Track query metrics"""
    query_counter.labels(query_type=query_type, cache_hit=str(cache_hit)).inc()
    query_latency.labels(component=component).observe(latency_s)

    if confidence is not None:
        answer_confidence.observe(confidence)

    if mrr is not None:
        retrieval_quality.observe(mrr)
```

#### Step 2: Add Metrics Endpoint (if using FastAPI)

```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI()

# Mount prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

#### Step 3: Create Grafana Dashboard

**File**: `config/grafana/dashboards/rag_monitoring.json`

(See full configuration below in Grafana section)

#### Step 4: Run Monitoring Stack

```bash
# Start Prometheus + Grafana with Docker Compose
docker-compose -f config/docker-compose-monitoring.yml up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

---

## Key Metrics to Track

### 1. Cache Performance

**Primary Metrics**:
```python
cache_stats = semantic_cache.stats()

# Must-track
hit_rate = cache_stats['hit_rate']  # Target: >30%
total_hits = cache_stats['hits']
total_misses = cache_stats['misses']
cache_size = cache_stats['count']  # Monitor growth

# Alerts
if hit_rate < 0.2:
    alert("Low cache hit rate - consider lowering threshold")
if cache_size > 5000:
    alert("Cache size growing large - check TTL settings")
```

**Graph**: Line chart of hit rate over time
```python
# Streamlit
st.line_chart(hit_rate_history)

# Grafana
rate(rag_cache_hits_total[5m]) / rate(rag_cache_total[5m])
```

### 2. Query Performance

**Primary Metrics**:
```python
monitor = PerformanceMonitor()
stats = monitor.get_stats()

# Must-track
p50_latency = stats['query']['p50']  # Target: <1s
p95_latency = stats['query']['p95']  # Target: <3s
p99_latency = stats['query']['p99']  # SLA: <10s

# Component breakdown
embedding_p95 = stats['embedding']['p95']  # Target: <0.2s
retrieval_p95 = stats['retrieval']['p95']  # Target: <0.5s
generation_p95 = stats['generation']['p95']  # Target: <3s

# Alerts
if p95_latency > 5000:  # 5 seconds
    alert("High query latency - investigate bottleneck")
```

**Graph**: Histogram of latency distribution
```python
# Streamlit
fig = go.Figure(data=[go.Histogram(x=latencies)])
fig.update_layout(title="Query Latency Distribution")
st.plotly_chart(fig)

# Grafana
histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))
```

### 3. Answer Quality

**Primary Metrics**:
```python
from utils.answer_validator import AnswerValidator

# Track per query
validation = validator.validate_answer(answer, query, chunks)

# Must-track
avg_confidence = np.mean(confidence_scores)  # Target: >0.7
hallucination_rate = num_hallucinations / total_queries  # Target: <0.1
low_confidence_rate = low_conf_count / total  # Target: <0.2

# Alerts
if avg_confidence < 0.6:
    alert("Low average confidence - check retrieval quality")
if hallucination_rate > 0.15:
    alert("High hallucination rate - lower temperature or increase context")
```

**Graph**: Gauge chart for avg confidence
```python
# Streamlit
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=avg_confidence,
    title={'text': "Average Confidence"},
    gauge={'axis': {'range': [0, 1]},
           'bar': {'color': "darkblue"},
           'threshold': {
               'line': {'color': "red", 'width': 4},
               'thickness': 0.75,
               'value': 0.7
           }}
))
st.plotly_chart(fig)
```

### 4. Query Type Distribution

**Primary Metrics**:
```python
router = QueryRouter()
stats = router.get_stats()

classifications = stats['classifications']
# {'factual': 45, 'conceptual': 30, 'procedural': 15, ...}

# Analysis
total = sum(classifications.values())
distribution = {k: v/total for k, v in classifications.items()}

# Insights
most_common = max(classifications.items(), key=lambda x: x[1])
print(f"Most common query type: {most_common[0]} ({most_common[1]/total:.1%})")
```

**Graph**: Pie chart of query types
```python
# Streamlit
fig = go.Figure(data=[go.Pie(
    labels=list(classifications.keys()),
    values=list(classifications.values())
)])
fig.update_layout(title="Query Type Distribution")
st.plotly_chart(fig)
```

### 5. System Health

**Primary Metrics**:
```python
import psutil

# Memory
memory = psutil.virtual_memory()
mem_percent = memory.percent
mem_available_gb = memory.available / (1024**3)

# Database connections
# (if using connection pool)
from utils.performance_optimizations import DatabaseConnectionPool
pool_stats = pool.get_stats()
pool_utilization = (pool_stats['pool_size'] - pool_stats['pool_free']) / pool_stats['max_size']

# Alerts
if mem_percent > 90:
    alert("High memory usage")
if pool_utilization > 0.8:
    alert("Connection pool nearly exhausted")
```

**Graph**: Multi-metric gauge dashboard
```python
# Streamlit
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Memory Usage", f"{mem_percent:.1f}%")

with col2:
    st.metric("Pool Utilization", f"{pool_utilization:.1%}")

with col3:
    st.metric("Available Memory", f"{mem_available_gb:.1f} GB")
```

---

## Grafana Dashboards (Production-Ready)

### Dashboard 1: RAG Overview

**Metrics to Display**:

**Row 1: Key Performance Indicators (KPIs)**
- Cache Hit Rate (gauge, target >30%)
- Avg Query Latency (stat panel, target <2s)
- Queries per Second (stat panel)
- Answer Confidence (gauge, target >0.7)

**Row 2: Performance Over Time**
- Query Latency (graph, p50/p95/p99)
- Cache Performance (graph, hit rate trend)
- Throughput (graph, QPS over time)

**Row 3: Quality Metrics**
- Answer Confidence Distribution (heatmap)
- Retrieval Quality (graph, MRR over time)
- Hallucination Rate (graph, percentage over time)

**Row 4: System Resources**
- Memory Usage (graph)
- Connection Pool Utilization (graph)
- Active Sessions (graph)

### Dashboard Configuration

**File**: `config/grafana/rag_dashboard.json`

```json
{
  "dashboard": {
    "title": "RAG Pipeline Monitoring",
    "panels": [
      {
        "id": 1,
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "rate(rag_cache_hits_total[5m]) / rate(rag_cache_total[5m])"
          }
        ],
        "thresholds": {
          "mode": "absolute",
          "steps": [
            {"color": "red", "value": 0},
            {"color": "yellow", "value": 0.2},
            {"color": "green", "value": 0.3}
          ]
        }
      },
      {
        "id": 2,
        "title": "Query Latency (p95)",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))"
          }
        ],
        "thresholds": {
          "steps": [
            {"color": "green", "value": 0},
            {"color": "yellow", "value": 2},
            {"color": "red", "value": 5}
          ]
        }
      },
      {
        "id": 3,
        "title": "Query Latency Over Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(rag_query_latency_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(rag_query_latency_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      }
    ]
  }
}
```

### PromQL Queries (for Grafana)

**Cache Performance**:
```promql
# Hit rate (5-minute window)
rate(rag_cache_hits_total[5m]) / rate(rag_cache_total[5m])

# Cache size
rag_cache_size
```

**Query Performance**:
```promql
# p95 latency
histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))

# QPS
rate(rag_queries_total[1m])
```

**Quality Metrics**:
```promql
# Average confidence (5-minute window)
rate(rag_answer_confidence_sum[5m]) / rate(rag_answer_confidence_count[5m])

# Hallucination rate
rate(rag_hallucinations_total[5m]) / rate(rag_queries_total[5m])
```

---

## Alert Configurations

### Prometheus Alerts

**File**: `config/prometheus/alerts.yml`

```yaml
groups:
  - name: rag_alerts
    interval: 30s
    rules:
      # Cache performance
      - alert: LowCacheHitRate
        expr: rate(rag_cache_hits_total[10m]) / rate(rag_cache_total[10m]) < 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate ({{ $value | humanizePercentage }})"
          description: "Cache hit rate below 20% for 5 minutes"

      # Query latency
      - alert: HighQueryLatency
        expr: histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High query latency (p95: {{ $value }}s)"
          description: "95th percentile latency above 5 seconds"

      - alert: CriticalQueryLatency
        expr: histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m])) > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Critical query latency (p95: {{ $value }}s)"
          description: "95th percentile latency above 10 seconds"

      # Answer quality
      - alert: LowAnswerConfidence
        expr: rate(rag_answer_confidence_sum[10m]) / rate(rag_answer_confidence_count[10m]) < 0.6
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low average answer confidence ({{ $value }})"
          description: "Average confidence below 0.6 for 5 minutes"

      - alert: HighHallucinationRate
        expr: rate(rag_hallucinations_total[10m]) / rate(rag_queries_total[10m]) > 0.15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High hallucination rate ({{ $value | humanizePercentage }})"
          description: "Hallucination rate above 15%"

      # System health
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 / 1024 > 14
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage ({{ $value }}GB)"
          description: "Memory usage above 14GB (approaching 16GB limit)"

      - alert: ConnectionPoolExhausted
        expr: rag_db_pool_free_connections == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool exhausted"
          description: "No free connections available"
```

### Alert Channels

**Slack Integration**:
```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'slack-notifications'

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#rag-alerts'
        title: 'RAG Pipeline Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

---

## Quick Monitoring Scripts

### Script 1: Real-Time Monitoring

**File**: `scripts/monitor_live.py`

```python
#!/usr/bin/env python3
"""
Real-time RAG monitoring with auto-refresh.

Usage:
    python scripts/monitor_live.py
"""

import time
import os
from datetime import datetime

from utils.query_cache import semantic_cache
from utils.conversation_memory import session_manager

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_metrics():
    clear_screen()

    print("=" * 70)
    print(f"RAG Pipeline Live Monitoring - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    # Cache metrics
    cache_stats = semantic_cache.stats()
    total_requests = cache_stats['hits'] + cache_stats['misses']

    print("\n📦 CACHE PERFORMANCE")
    print(f"  Hit Rate:      {cache_stats['hit_rate']:>6.1%} ", end="")
    print("✅" if cache_stats['hit_rate'] > 0.3 else "⚠️" if cache_stats['hit_rate'] > 0.1 else "❌")

    print(f"  Total Queries: {total_requests:>6d}")
    print(f"  Cache Hits:    {cache_stats['hits']:>6d}")
    print(f"  Cache Misses:  {cache_stats['misses']:>6d}")
    print(f"  Cache Size:    {cache_stats['count']:>6d} entries")
    print(f"  Disk Usage:    {cache_stats['size_mb']:>6.1f} MB")

    # Session metrics
    conv_stats = session_manager.stats()

    print("\n💬 CONVERSATION SESSIONS")
    print(f"  Active:        {conv_stats['active_sessions']:>6d}")
    print(f"  Total Created: {conv_stats['total_created']:>6d}")
    print(f"  Total Expired: {conv_stats['total_expired']:>6d}")

    # Performance indicators
    print("\n🎯 HEALTH INDICATORS")

    indicators = []

    # Cache health
    if cache_stats['hit_rate'] > 0.3:
        indicators.append(("✅", "Cache Performance", "GOOD"))
    elif cache_stats['hit_rate'] > 0.1:
        indicators.append(("⚠️", "Cache Performance", "FAIR"))
    else:
        indicators.append(("❌", "Cache Performance", "POOR"))

    # Cache size health
    if cache_stats['size_mb'] < 500:
        indicators.append(("✅", "Cache Size", "NORMAL"))
    elif cache_stats['size_mb'] < 1000:
        indicators.append(("⚠️", "Cache Size", "GROWING"))
    else:
        indicators.append(("❌", "Cache Size", "HIGH"))

    # Session health
    if conv_stats['active_sessions'] < 50:
        indicators.append(("✅", "Session Count", "NORMAL"))
    elif conv_stats['active_sessions'] < 100:
        indicators.append(("⚠️", "Session Count", "GROWING"))
    else:
        indicators.append(("❌", "Session Count", "HIGH"))

    for status, metric, assessment in indicators:
        print(f"  {status} {metric:<20s} {assessment}")

    print("\n" + "=" * 70)
    print("Press Ctrl+C to exit | Auto-refresh every 5s")
    print("=" * 70)

def main():
    try:
        while True:
            print_metrics()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()
```

**Run it**:
```bash
python scripts/monitor_live.py
```

### Script 2: Daily Summary Report

**File**: `scripts/daily_report.py`

```python
#!/usr/bin/env python3
"""
Generate daily monitoring report.

Usage:
    python scripts/daily_report.py > reports/daily_$(date +%Y%m%d).txt
"""

from datetime import datetime, timedelta
from utils.query_cache import semantic_cache
from utils.conversation_memory import session_manager

def generate_daily_report():
    print("=" * 70)
    print(f"RAG Pipeline Daily Report - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 70)

    # Cache summary
    cache_stats = semantic_cache.stats()
    print("\n📦 CACHE SUMMARY")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
    print(f"  Total Requests: {cache_stats['hits'] + cache_stats['misses']}")
    print(f"  Speedup from Cache: {cache_stats['hits']} queries (~{cache_stats['hits'] * 10:.0f}s saved)")
    print(f"  Cache Size: {cache_stats['count']} entries ({cache_stats['size_mb']:.1f} MB)")

    # Conversation summary
    conv_stats = session_manager.stats()
    print("\n💬 CONVERSATION SUMMARY")
    print(f"  Active Sessions: {conv_stats['active_sessions']}")
    print(f"  Sessions Created Today: {conv_stats['total_created']}")
    print(f"  Sessions Expired Today: {conv_stats['total_expired']}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS")

    if cache_stats['hit_rate'] < 0.2:
        print("  ⚠️  Consider lowering SEMANTIC_CACHE_THRESHOLD (current queries too diverse)")

    if cache_stats['size_mb'] > 1000:
        print("  ⚠️  Consider reducing SEMANTIC_CACHE_MAX_SIZE or lowering TTL")

    if conv_stats['active_sessions'] > 80:
        print("  ⚠️  Consider running session_manager.cleanup_expired()")

    if cache_stats['hit_rate'] > 0.4:
        print("  ✅ Excellent cache performance!")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    generate_daily_report()
```

**Schedule with cron**:
```bash
# Add to crontab
0 9 * * * cd /path/to/project && python scripts/daily_report.py > reports/daily_$(date +\%Y\%m\%d).txt
```

---

## Visualization Examples

### 1. Cache Performance Trend

```python
import plotly.graph_objects as go

# Collect data over time
timestamps = [...]  # datetime objects
hit_rates = [...]   # hit rate at each timestamp

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=timestamps,
    y=hit_rates,
    mode='lines+markers',
    name='Hit Rate',
    line=dict(color='green', width=2)
))

# Add threshold line
fig.add_hline(y=0.3, line_dash="dash", line_color="red",
              annotation_text="Target: 30%")

fig.update_layout(
    title="Cache Hit Rate Over Time",
    xaxis_title="Time",
    yaxis_title="Hit Rate",
    yaxis=dict(range=[0, 1], tickformat=".0%")
)

fig.show()
# or fig.write_html("cache_performance.html")
```

### 2. Query Latency Distribution

```python
import plotly.graph_objects as go

# Collect latency data
latencies = [...]  # list of latency values in ms

fig = go.Figure()

# Histogram
fig.add_trace(go.Histogram(
    x=latencies,
    nbinsx=50,
    name='Distribution',
    marker_color='blue'
))

# Add percentile lines
import numpy as np
p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

fig.add_vline(x=p50, line_dash="dot", line_color="green",
              annotation_text=f"p50: {p50:.0f}ms")
fig.add_vline(x=p95, line_dash="dash", line_color="orange",
              annotation_text=f"p95: {p95:.0f}ms")
fig.add_vline(x=p99, line_dash="solid", line_color="red",
              annotation_text=f"p99: {p99:.0f}ms")

fig.update_layout(
    title="Query Latency Distribution",
    xaxis_title="Latency (ms)",
    yaxis_title="Count"
)

fig.show()
```

### 3. Query Type Distribution (Pie Chart)

```python
import plotly.graph_objects as go

# Get query type stats
router_stats = router.get_stats()
classifications = router_stats['classifications']

# Filter out zero counts
active_types = {k: v for k, v in classifications.items() if v > 0}

fig = go.Figure(data=[go.Pie(
    labels=list(active_types.keys()),
    values=list(active_types.values()),
    hole=0.3,  # Donut chart
    textinfo='label+percent',
    marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
)])

fig.update_layout(
    title="Query Type Distribution",
    annotations=[dict(text='Query<br>Types', x=0.5, y=0.5, font_size=20, showarrow=False)]
)

fig.show()
```

### 4. Multi-Metric Dashboard

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Cache Hit Rate", "Query Latency (p95)",
                   "Answer Confidence", "Active Sessions"),
    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
           [{'type': 'indicator'}, {'type': 'indicator'}]]
)

# Cache hit rate gauge
fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=cache_stats['hit_rate'] * 100,
    title={'text': "Cache Hit Rate (%)"},
    gauge={'axis': {'range': [0, 100]},
           'threshold': {'line': {'color': "red", 'width': 4}, 'value': 30}}
), row=1, col=1)

# Query latency gauge
fig.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=p95_latency,
    delta={'reference': 2000},
    title={'text': "Query Latency p95 (ms)"},
    gauge={'axis': {'range': [0, 10000]},
           'threshold': {'line': {'color': "red", 'width': 4}, 'value': 5000}}
), row=1, col=2)

# Answer confidence gauge
fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=avg_confidence,
    title={'text': "Avg Confidence"},
    gauge={'axis': {'range': [0, 1]},
           'threshold': {'line': {'color': "red", 'width': 4}, 'value': 0.7}}
), row=2, col=1)

# Active sessions gauge
fig.add_trace(go.Indicator(
    mode="number+delta",
    value=conv_stats['active_sessions'],
    delta={'reference': 50},
    title={'text': "Active Sessions"}
), row=2, col=2)

fig.update_layout(height=600, title_text="RAG Pipeline Health Dashboard")
fig.show()
```

---

## Simple Text-Based Monitoring

### Option 1: Log File Analysis

**Enable query logging**:
```bash
export LOG_QUERIES=1
export LOG_QUERY_DIR=query_logs/
```

**Analyze logs**:
```bash
# Count queries per hour
grep "Query:" query_logs/*.log | awk -F'[: ]' '{print $1":"$2}' | sort | uniq -c

# Average latency
grep "Latency:" query_logs/*.log | awk '{sum+=$NF; count++} END {print sum/count "ms"}'

# Cache hit rate
grep "Cache hit" query_logs/*.log | wc -l
grep "Cache miss" query_logs/*.log | wc -l
```

### Option 2: SQLite Metrics Database

**File**: `utils/metrics_logger.py`

```python
"""
Log metrics to SQLite for analysis.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

class MetricsLogger:
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS query_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query_type TEXT,
                latency_ms REAL,
                cache_hit BOOLEAN,
                confidence REAL,
                mrr REAL,
                hallucinations INTEGER
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                hit_rate REAL,
                cache_size INTEGER,
                size_mb REAL
            )
        """)

        self.conn.commit()

    def log_query(self, query_type, latency_ms, cache_hit,
                  confidence=None, mrr=None, hallucinations=None):
        self.conn.execute("""
            INSERT INTO query_metrics
            (query_type, latency_ms, cache_hit, confidence, mrr, hallucinations)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query_type, latency_ms, cache_hit, confidence, mrr, hallucinations))
        self.conn.commit()

    def log_cache_snapshot(self):
        from utils.query_cache import semantic_cache
        stats = semantic_cache.stats()

        self.conn.execute("""
            INSERT INTO cache_snapshots (hit_rate, cache_size, size_mb)
            VALUES (?, ?, ?)
        """, (stats['hit_rate'], stats['count'], stats['size_mb']))
        self.conn.commit()

    def get_summary(self, hours=24):
        """Get metrics summary for last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)

        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_queries,
                AVG(latency_ms) as avg_latency,
                AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END) as hit_rate,
                AVG(confidence) as avg_confidence,
                SUM(hallucinations) as total_hallucinations
            FROM query_metrics
            WHERE timestamp > ?
        """, (cutoff,))

        return cursor.fetchone()

# Usage in your RAG pipeline
metrics_logger = MetricsLogger()

def query_with_logging(query_text):
    # ... your query logic ...

    # Log metrics
    metrics_logger.log_query(
        query_type=routing_result.query_type.value,
        latency_ms=elapsed_ms,
        cache_hit=was_cached,
        confidence=validation['confidence_score'],
        mrr=retrieval_metrics['mrr']
    )
```

**Query the database**:
```bash
sqlite3 metrics.db "SELECT AVG(latency_ms), AVG(confidence) FROM query_metrics WHERE timestamp > datetime('now', '-1 day')"
```

---

## Recommended Setup

### For Development (Local)

**Best Choice**: **Streamlit Dashboard**

```bash
# Create monitoring_dashboard.py (provided above)
streamlit run monitoring_dashboard.py
```

**Pros**:
- Quick setup (5 minutes)
- Auto-refresh
- Interactive charts
- Easy to customize

### For Production (RunPod/Cloud)

**Best Choice**: **Prometheus + Grafana**

```bash
# 1. Add prometheus_exporter.py to your utils/
# 2. Start monitoring stack
docker-compose -f config/docker-compose-monitoring.yml up -d

# 3. Access Grafana
open http://localhost:3000
```

**Pros**:
- Industry standard
- Historical data retention
- Alerting built-in
- Scales to multiple instances

### For Quick Checks

**Best Choice**: **Built-in stats + scripts/monitor_live.py**

```bash
# Real-time monitoring
python scripts/monitor_live.py

# One-time check
python -c "from utils.query_cache import semantic_cache; print(semantic_cache.stats())"
```

---

## Summary: Recommended Monitoring Stack

### Immediate (Day 1)
1. ✅ Run `scripts/monitor_live.py` to see real-time stats
2. ✅ Check stats after every 100 queries
3. ✅ Use built-in `get_stats()` methods in Python

### Week 1 (Development)
1. ✅ Create Streamlit dashboard (15 min setup)
2. ✅ Monitor cache hit rate, latency, confidence
3. ✅ Adjust thresholds based on actual data

### Production (RunPod)
1. ✅ Deploy Prometheus + Grafana stack
2. ✅ Set up alerts for critical metrics
3. ✅ Create custom dashboards for your use case
4. ✅ Schedule daily reports

---

**Next**: I'll create the actual monitoring scripts and Streamlit dashboard if you'd like! Just say the word.
