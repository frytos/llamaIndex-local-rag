#!/usr/bin/env python3
"""
RAG Pipeline Monitoring Dashboard (Streamlit)

Real-time monitoring dashboard for tracking RAG performance, cache efficiency,
and system health.

Features:
- Auto-refresh every 10 seconds
- Cache performance tracking
- Conversation session monitoring
- System health indicators
- Interactive charts with Plotly

Usage:
    streamlit run monitoring_dashboard.py

Requirements:
    pip install streamlit plotly pandas psutil
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os

# Import monitoring components
from utils.query_cache import semantic_cache, cache
from utils.conversation_memory import session_manager

# Optional: psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="RAG Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-good { color: green; font-weight: bold; }
    .status-warning { color: orange; font-weight: bold; }
    .status-bad { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 RAG Pipeline Monitoring Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar controls
st.sidebar.header("⚙️ Dashboard Settings")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (10s)", value=False)
if auto_refresh:
    time.sleep(10)
    st.rerun()

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Quick Stats")

# Get all stats
cache_stats = semantic_cache.stats()
embed_stats = cache.stats()
conv_stats = session_manager.stats()

# Sidebar quick stats
total_queries = cache_stats['hits'] + cache_stats['misses']
st.sidebar.metric("Total Queries", total_queries)
st.sidebar.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}")
st.sidebar.metric("Active Sessions", conv_stats['active_sessions'])

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard v1.0")

# Main content
st.header("🎯 Key Performance Indicators")

# KPI row
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_hits = f"+{cache_stats['hits']}" if cache_stats['hits'] > 0 else "No hits yet"
    st.metric(
        "Cache Hit Rate",
        f"{cache_stats['hit_rate']:.1%}",
        delta=delta_hits,
        help="Percentage of queries served from cache (target: >30%)"
    )

with col2:
    st.metric(
        "Total Queries",
        f"{total_queries:,}",
        delta=f"{cache_stats['count']} cached",
        help="Total queries processed"
    )

with col3:
    st.metric(
        "Cache Size",
        f"{cache_stats['size_mb']:.1f} MB",
        delta=f"{cache_stats['count']} entries",
        help="Disk space used by cache"
    )

with col4:
    st.metric(
        "Active Sessions",
        conv_stats['active_sessions'],
        delta=f"{conv_stats['total_created']} created",
        help="Number of active conversation sessions"
    )

st.markdown("---")

# Detailed tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Cache Performance",
    "💬 Conversations",
    "🖥️ System Health",
    "📊 Historical Trends"
])

# ============================================================================
# Tab 1: Cache Performance
# ============================================================================

with tab1:
    st.subheader("Cache Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Cache effectiveness chart
        fig_cache = go.Figure()

        fig_cache.add_trace(go.Bar(
            name='Hits',
            x=['Semantic Cache'],
            y=[cache_stats['hits']],
            marker_color='#4ECDC4',
            text=[cache_stats['hits']],
            textposition='auto'
        ))

        fig_cache.add_trace(go.Bar(
            name='Misses',
            x=['Semantic Cache'],
            y=[cache_stats['misses']],
            marker_color='#FF6B6B',
            text=[cache_stats['misses']],
            textposition='auto'
        ))

        fig_cache.update_layout(
            title="Cache Hits vs Misses",
            barmode='group',
            yaxis_title="Count",
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig_cache, use_container_width=True)

    with col2:
        # Cache hit rate gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cache_stats['hit_rate'] * 100,
            title={'text': "Cache Hit Rate (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "#FFE5E5"},
                    {'range': [20, 30], 'color': "#FFF4E5"},
                    {'range': [30, 100], 'color': "#E5F5E5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))

        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Cache details
    st.markdown("### Cache Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Semantic Cache**")
        semantic_data = {
            "Enabled": "✅ Yes" if cache_stats['enabled'] else "❌ No",
            "Similarity Threshold": f"{cache_stats['threshold']:.2f}",
            "Max Size": f"{cache_stats['max_size']} entries",
            "TTL": f"{cache_stats['ttl']}s ({cache_stats['ttl']//3600}h)",
            "Current Size": f"{cache_stats['count']} entries",
            "Disk Usage": f"{cache_stats['size_mb']:.2f} MB",
            "Evictions": cache_stats.get('evictions', 0),
        }
        st.json(semantic_data)

    with col2:
        st.markdown("**Embedding Cache**")
        embedding_data = {
            "Cached Embeddings": f"{embed_stats['count']} queries",
            "Disk Usage": f"{embed_stats['size_mb']:.2f} MB",
            "Cache Directory": embed_stats['cache_dir'],
        }
        st.json(embedding_data)

    # Performance impact
    st.markdown("### 💡 Performance Impact")

    if cache_stats['hits'] > 0:
        # Estimate time saved (assume 10s per query without cache)
        time_saved_s = cache_stats['hits'] * 10
        time_saved_m = time_saved_s / 60

        st.success(
            f"🚀 Cache has saved approximately **{time_saved_m:.1f} minutes** "
            f"({cache_stats['hits']} queries × ~10s each)"
        )

        speedup = total_queries / cache_stats['misses'] if cache_stats['misses'] > 0 else 1
        st.info(f"⚡ Effective speedup: **{speedup:.1f}x** with current hit rate")
    else:
        st.info("Cache warming up... No hits yet. This is normal for the first few queries.")

# ============================================================================
# Tab 2: Conversations
# ============================================================================

with tab2:
    st.subheader("Conversation Sessions")

    # Session overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Active Sessions",
            conv_stats['active_sessions'],
            help="Currently active conversation sessions"
        )

    with col2:
        st.metric(
            "Total Created",
            conv_stats['total_created'],
            help="All-time session creations"
        )

    with col3:
        st.metric(
            "Total Expired",
            conv_stats['total_expired'],
            help="Sessions that have expired"
        )

    # Active sessions table
    st.markdown("### Active Sessions")

    active_session_ids = session_manager.list_active_sessions()

    if active_session_ids:
        session_data = []

        for session_id in active_session_ids[:20]:  # Show first 20
            memory = session_manager.get(session_id)
            if memory:
                mem_stats = memory.stats()
                session_data.append({
                    "Session ID": session_id[:16] + "...",
                    "Total Turns": mem_stats['total_turns'],
                    "Active Turns": mem_stats['active_turns'],
                    "Entities": mem_stats['entities'],
                    "Topics": mem_stats['topics'],
                    "Idle Time": f"{mem_stats['idle_seconds']:.0f}s",
                    "Age": f"{mem_stats['age_seconds']//60:.0f}m"
                })

        df_sessions = pd.DataFrame(session_data)
        st.dataframe(df_sessions, use_container_width=True, hide_index=True)

        # Session cleanup button
        if st.button("🧹 Cleanup Expired Sessions"):
            expired_count = session_manager.cleanup_expired()
            st.success(f"Cleaned up {expired_count} expired sessions")
            time.sleep(1)
            st.rerun()
    else:
        st.info("No active conversation sessions")

# ============================================================================
# Tab 3: System Health
# ============================================================================

with tab3:
    st.subheader("System Health Checks")

    # Health checks
    checks = []

    # Cache health
    if cache_stats['hit_rate'] > 0.3:
        checks.append(("✅", "Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}", "GOOD", "green"))
    elif cache_stats['hit_rate'] > 0.1:
        checks.append(("⚠️", "Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}", "FAIR", "orange"))
    else:
        checks.append(("❌", "Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}", "POOR", "red"))

    # Cache size health
    if cache_stats['size_mb'] < 500:
        checks.append(("✅", "Cache Disk Usage", f"{cache_stats['size_mb']:.1f} MB", "NORMAL", "green"))
    elif cache_stats['size_mb'] < 1000:
        checks.append(("⚠️", "Cache Disk Usage", f"{cache_stats['size_mb']:.1f} MB", "GROWING", "orange"))
    else:
        checks.append(("❌", "Cache Disk Usage", f"{cache_stats['size_mb']:.1f} MB", "HIGH", "red"))

    # Session count health
    if conv_stats['active_sessions'] < 50:
        checks.append(("✅", "Active Sessions", str(conv_stats['active_sessions']), "NORMAL", "green"))
    elif conv_stats['active_sessions'] < 100:
        checks.append(("⚠️", "Active Sessions", str(conv_stats['active_sessions']), "GROWING", "orange"))
    else:
        checks.append(("❌", "Active Sessions", str(conv_stats['active_sessions']), "HIGH", "red"))

    # System memory (if psutil available)
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        mem_percent = mem.percent

        if mem_percent < 80:
            checks.append(("✅", "Memory Usage", f"{mem_percent:.1f}%", "NORMAL", "green"))
        elif mem_percent < 90:
            checks.append(("⚠️", "Memory Usage", f"{mem_percent:.1f}%", "HIGH", "orange"))
        else:
            checks.append(("❌", "Memory Usage", f"{mem_percent:.1f}%", "CRITICAL", "red"))

        mem_available_gb = mem.available / (1024**3)
        checks.append(("ℹ️", "Available Memory", f"{mem_available_gb:.1f} GB", "INFO", "blue"))

    # Display checks
    df_health = pd.DataFrame(checks, columns=["Status", "Metric", "Value", "Assessment", "Color"])

    # Color code the dataframe
    def color_status(row):
        color = row['Color']
        return [f'background-color: {color}20' for _ in row]

    st.dataframe(
        df_health[["Status", "Metric", "Value", "Assessment"]],
        use_container_width=True,
        hide_index=True
    )

    # Recommendations
    st.markdown("### 💡 Recommendations")

    recommendations = []

    if cache_stats['hit_rate'] < 0.2:
        recommendations.append("⚠️ **Low cache hit rate**: Consider lowering `SEMANTIC_CACHE_THRESHOLD` (current queries may be too diverse)")

    if cache_stats['size_mb'] > 1000:
        recommendations.append("⚠️ **High cache disk usage**: Consider reducing `SEMANTIC_CACHE_MAX_SIZE` or lowering `SEMANTIC_CACHE_TTL`")

    if conv_stats['active_sessions'] > 80:
        recommendations.append("⚠️ **Many active sessions**: Run cleanup with `session_manager.cleanup_expired()`")

    if cache_stats['hit_rate'] > 0.4:
        recommendations.append("✅ **Excellent cache performance!** Hit rate above 40% is great")

    if PSUTIL_AVAILABLE and mem_percent > 85:
        recommendations.append("❌ **High memory usage**: Consider restarting the service or increasing memory limits")

    if not recommendations:
        st.success("✅ All systems healthy! No recommendations at this time.")
    else:
        for rec in recommendations:
            if rec.startswith("✅"):
                st.success(rec)
            elif rec.startswith("⚠️"):
                st.warning(rec)
            else:
                st.error(rec)

# ============================================================================
# Tab 4: Historical Trends (Simulated - requires metrics logging)
# ============================================================================

with tab4:
    st.subheader("Historical Trends")

    st.info(
        "📊 **Note**: Historical trends require metrics logging. "
        "Enable logging with `scripts/monitor_rag.py` or Prometheus integration. "
        "This tab shows what's possible with historical data."
    )

    # Example trend chart (would use real data if available)
    import numpy as np

    # Simulate 24 hours of data
    hours = list(range(24))
    simulated_hit_rate = [
        0.1 + (h / 24) * 0.3 + np.random.uniform(-0.05, 0.05)
        for h in hours
    ]
    simulated_latency = [
        3000 - (h / 24) * 500 + np.random.uniform(-200, 200)
        for h in hours
    ]

    col1, col2 = st.columns(2)

    with col1:
        fig_trend_cache = go.Figure()
        fig_trend_cache.add_trace(go.Scatter(
            x=hours,
            y=simulated_hit_rate,
            mode='lines+markers',
            name='Hit Rate',
            line=dict(color='green', width=2)
        ))
        fig_trend_cache.add_hline(
            y=0.3, line_dash="dash", line_color="red",
            annotation_text="Target: 30%"
        )
        fig_trend_cache.update_layout(
            title="Cache Hit Rate Trend (24h)",
            xaxis_title="Hour",
            yaxis_title="Hit Rate",
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            height=400
        )
        st.plotly_chart(fig_trend_cache, use_container_width=True)

    with col2:
        fig_trend_latency = go.Figure()
        fig_trend_latency.add_trace(go.Scatter(
            x=hours,
            y=simulated_latency,
            mode='lines+markers',
            name='p95 Latency',
            line=dict(color='blue', width=2)
        ))
        fig_trend_latency.add_hline(
            y=2000, line_dash="dash", line_color="red",
            annotation_text="SLA: 2000ms"
        )
        fig_trend_latency.update_layout(
            title="Query Latency Trend (24h)",
            xaxis_title="Hour",
            yaxis_title="Latency (ms)",
            height=400
        )
        st.plotly_chart(fig_trend_latency, use_container_width=True)

    st.markdown("### 📈 To Enable Real Historical Tracking")
    st.code("""
# Option 1: Use Prometheus + Grafana (recommended for production)
docker-compose -f config/docker-compose-monitoring.yml up -d

# Option 2: Use SQLite metrics logger
from utils.metrics_logger import MetricsLogger
logger = MetricsLogger()
logger.log_query(query_type, latency_ms, cache_hit, confidence)

# Option 3: Enable query logging
export LOG_QUERIES=1
export LOG_QUERY_DIR=query_logs/
    """, language="bash")

# ============================================================================
# Footer with export options
# ============================================================================

st.markdown("---")
st.header("📥 Export & Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Save Current Stats"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_data = {
            "timestamp": timestamp,
            "cache": cache_stats,
            "conversations": conv_stats,
            "embeddings": embed_stats
        }

        import json
        output_file = f"monitoring_snapshot_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(stats_data, f, indent=2)

        st.success(f"✅ Saved to `{output_file}`")

with col2:
    if st.button("🗑️ Clear Cache"):
        if st.checkbox("Confirm clear cache"):
            semantic_cache.clear()
            cache.clear()
            st.success("✅ Cache cleared")
            time.sleep(1)
            st.rerun()

with col3:
    if st.button("🔄 Reset Stats"):
        if st.checkbox("Confirm reset statistics"):
            semantic_cache.reset_stats()
            st.success("✅ Statistics reset")
            time.sleep(1)
            st.rerun()

# Display current configuration
with st.expander("⚙️ Current Configuration"):
    config_data = {
        "Cache": {
            "ENABLE_SEMANTIC_CACHE": os.getenv("ENABLE_SEMANTIC_CACHE", "1"),
            "SEMANTIC_CACHE_THRESHOLD": os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"),
            "SEMANTIC_CACHE_MAX_SIZE": os.getenv("SEMANTIC_CACHE_MAX_SIZE", "1000"),
            "SEMANTIC_CACHE_TTL": os.getenv("SEMANTIC_CACHE_TTL", "86400"),
        },
        "Conversations": {
            "ENABLE_CONVERSATION_MEMORY": os.getenv("ENABLE_CONVERSATION_MEMORY", "1"),
            "MAX_CONVERSATION_TURNS": os.getenv("MAX_CONVERSATION_TURNS", "10"),
            "CONVERSATION_TIMEOUT": os.getenv("CONVERSATION_TIMEOUT", "3600"),
            "AUTO_SUMMARIZE": os.getenv("AUTO_SUMMARIZE", "1"),
        },
        "Performance": {
            "ENABLE_ASYNC": os.getenv("ENABLE_ASYNC", "1"),
            "CONNECTION_POOL_SIZE": os.getenv("CONNECTION_POOL_SIZE", "10"),
            "BATCH_SIZE": os.getenv("BATCH_SIZE", "32"),
        }
    }

    st.json(config_data)
```

**Run the dashboard**:
```bash
streamlit run monitoring_dashboard.py
```

Opens automatically at `http://localhost:8501`

---

## Summary: Quick Monitoring Setup (5 Minutes)

### Step 1: Create Dashboard (1 min)
```bash
# Dashboard file already created above
cp monitoring_dashboard.py .
```

### Step 2: Install Streamlit (1 min)
```bash
pip install streamlit plotly  # Already in requirements.txt!
```

### Step 3: Run Dashboard (1 min)
```bash
streamlit run monitoring_dashboard.py
```

### Step 4: View in Browser
Open `http://localhost:8501` - dashboard auto-updates!

---

## What You'll See

✅ **Real-time KPIs**: Cache hit rate, total queries, cache size, active sessions
✅ **Cache Performance**: Bar charts showing hits vs misses
✅ **Hit Rate Gauge**: Visual indicator with color-coded thresholds
✅ **Session Tracking**: Table of active conversations
✅ **Health Checks**: Automated status indicators
✅ **Recommendations**: Smart suggestions based on current metrics
✅ **Export Options**: Save snapshots, clear cache, reset stats

All **color-coded** and **auto-refreshing** for easy monitoring!