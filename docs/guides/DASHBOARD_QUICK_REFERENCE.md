# Grafana Dashboards Quick Reference

**Quick access guide for RAG system monitoring**

## Dashboard URLs

Access dashboards at: `http://localhost:3000/d/`

1. **Pipeline Internals**: `http://localhost:3000/d/rag-pipeline-internals`
2. **System Resources**: `http://localhost:3000/d/rag-system-resources`
3. **Quality & Performance**: `http://localhost:3000/d/rag-quality-performance`

## Quick Start (5 Minutes)

### 1. Start Monitoring Stack
```bash
# Start macOS exporter (for system metrics)
python macos_exporter.py --port 9101 &

# Start your RAG application with metrics
python rag_low_level_m1_16gb_verbose.py

# Access Grafana
open http://localhost:3000
# Login: admin / admin (default)
```

### 2. Navigate to Dashboards
- Click "Dashboards" (四 icon) in left sidebar
- Browse to "RAG System" folder
- Select desired dashboard

## Common Use Cases

### Use Case 1: Identify Performance Bottleneck

**Dashboard**: Pipeline Internals
**Steps**:
1. Look at "Stage Duration Over Time" panel
2. Identify which stage (embedding/retrieval/generation) takes longest
3. Check stage-specific metrics below:
   - **Embedding slow?** → Check "Embedding Throughput" and "Batch Utilization"
   - **Retrieval slow?** → Check "Retrieval Scores" and database performance
   - **Generation slow?** → Check "Token Generation Speed" and "Context Utilization"

**Actions**:
- Embedding: Increase batch size or enable GPU
- Retrieval: Add indexes, tune similarity threshold
- Generation: Reduce context window, optimize prompts

### Use Case 2: Monitor System Health

**Dashboard**: System Resources
**Steps**:
1. Check "System Overview" gauges (CPU, Memory, Process CPU)
2. Review "Memory Breakdown" to see memory pressure
3. Monitor "Garbage Collection Activity" for Python performance
4. Check "Disk Usage" and "Network Traffic"

**Alert Thresholds**:
- CPU >80% sustained → Consider scaling or optimization
- Memory >85% → Risk of swapping, reduce batch sizes
- Swap usage >10% → System under memory pressure
- GC collections >10/sec → Memory churn, check for leaks

### Use Case 3: Improve Retrieval Quality

**Dashboard**: Quality & Performance
**Steps**:
1. Check "Average Retrieval Score" (target: >0.75)
2. Look at "Retrieval Score Trends" for consistency
3. Review "Score Variance" (lower is better)
4. Check "Relevant Documents" user feedback

**Actions**:
- Low scores → Improve embeddings, tune chunk size
- High variance → Inconsistent results, check data quality
- Few relevant docs → Adjust TOP_K, improve chunking

### Use Case 4: Optimize Cache Performance

**Dashboard**: Quality & Performance
**Steps**:
1. Check "Cache Hit Rate" (target: >50%)
2. Look at "Cache Operations" (hits vs misses)
3. Review "Cache Size" and "Evictions"
4. Check "Semantic Similarity Threshold"

**Actions**:
- Low hit rate → Increase cache size, lower similarity threshold
- High evictions → Increase cache size or TTL
- No hits → Enable semantic caching, tune threshold

### Use Case 5: Track Query Performance Over Time

**Dashboard**: Quality & Performance
**Steps**:
1. Review "Query Latency Percentiles" panel
2. Check p50, p95, p99 latencies
3. Look at "Query Latency Distribution" heatmap
4. Monitor "Query Throughput"

**SLO Targets**:
- p50 latency <1s
- p95 latency <3s
- p99 latency <5s
- Success rate >99%

## Panel Quick Reference

### Pipeline Internals Dashboard

| Panel | Metric | Good | Warning | Critical |
|-------|--------|------|---------|----------|
| Total Query Time | Average latency | <1s | 1-3s | >3s |
| Success Rate | % successful | >99% | 95-99% | <95% |
| Embedding Throughput | Emb/sec | >60 | 30-60 | <30 |
| Token Generation Speed | Tokens/sec | >10 | 5-10 | <5 |
| Context Utilization | % used | <70% | 70-90% | >90% |
| Retrieval Score Avg | Similarity | >0.8 | 0.6-0.8 | <0.6 |

### System Resources Dashboard

| Panel | Metric | Good | Warning | Critical |
|-------|--------|------|---------|----------|
| System CPU | % usage | <60% | 60-80% | >80% |
| System Memory | % usage | <70% | 70-85% | >85% |
| RAG Process CPU | % usage | <50% | 50-80% | >80% |
| Swap Usage | % used | <5% | 5-20% | >20% |
| Disk Usage | % full | <70% | 70-90% | >90% |

### Quality & Performance Dashboard

| Panel | Metric | Good | Warning | Critical |
|-------|--------|------|---------|----------|
| Cache Hit Rate | % hits | >80% | 50-80% | <50% |
| User Satisfaction | 0-5 scale | >4 | 3-4 | <3 |
| DB Error Rate | % errors | <0.1% | 0.1-1% | >1% |
| Query Throughput | QPS | Varies | - | - |

## Keyboard Shortcuts

- `d` + `k` → Dashboard search
- `Esc` → Exit fullscreen panel
- `t` + `z` → Zoom out time range
- `t` + `l` → Toggle legend
- `v` + `d` → View dashboard settings

## Time Range Presets

- **Real-time monitoring**: Last 5 minutes, refresh 5s
- **Recent performance**: Last 1 hour, refresh 30s
- **Daily trends**: Last 24 hours, refresh 5m
- **Weekly analysis**: Last 7 days, refresh 1h

## Common Queries

### Find slow queries
```promql
# Queries taking >5 seconds
histogram_quantile(0.99, rate(rag_query_duration_seconds_bucket[5m])) > 5
```

### Identify memory leaks
```promql
# Memory growing over time
deriv(rag_process_memory_rss_bytes[30m]) > 0
```

### Cache effectiveness
```promql
# Cache hit rate trending
rate(rag_cache_hits_total[5m]) / rate(rag_cache_requests_total[5m])
```

### Retrieval quality degradation
```promql
# Retrieval scores dropping
(avg_over_time(rag_retrieval_score_avg[1h]) - avg_over_time(rag_retrieval_score_avg[1h] offset 1h)) < -0.1
```

## Troubleshooting Quick Fixes

### No Data in Dashboard
1. Check Prometheus: `curl http://localhost:9090/api/v1/targets`
2. Check metrics endpoint: `curl http://localhost:9102/metrics`
3. Verify time range is appropriate
4. Refresh dashboard (Ctrl+R)

### Wrong Time Zone
1. Dashboard Settings → Time options
2. Set timezone to "Browser" or specific zone

### Panel Shows "N/A"
1. Metric not collected yet (wait a few minutes)
2. Check PromQL query syntax
3. Verify metric name matches exported metrics

### Slow Dashboard Loading
1. Reduce time range
2. Increase refresh interval
3. Simplify complex queries
4. Check Prometheus performance

## Best Practices

1. **Start with Pipeline Internals** to understand flow
2. **Use System Resources** during load testing
3. **Monitor Quality & Performance** for production
4. **Set up alerts** for critical metrics
5. **Create snapshots** before optimization changes
6. **Use annotations** to mark deployments
7. **Share dashboards** with team via links

## Metric Export Verification

Test metrics are being exported:
```bash
# Check file-based export
cat metrics/rag_app.prom

# Check HTTP endpoint
curl http://localhost:9102/metrics | grep rag_

# Check Prometheus has data
curl 'http://localhost:9090/api/v1/query?query=rag_query_total'
```

## Dashboard Customization

### Add Custom Panel
1. Click "Add" → "Visualization"
2. Select metric from dropdown
3. Configure visualization type
4. Set thresholds and colors
5. Save dashboard

### Create Alert
1. Edit panel → Alert tab
2. Set condition (e.g., `value > threshold`)
3. Configure notification channel
4. Test alert
5. Save

### Export Dashboard
1. Dashboard Settings (gear icon)
2. JSON Model → Copy to clipboard
3. Save to file
4. Share with team

## Integration with Other Tools

### Slack Notifications
Configure in Grafana → Alerting → Notification channels

### Webhook Integration
Send alerts to custom endpoints for automation

### Jupyter Notebooks
Query Prometheus data for analysis:
```python
from prometheus_api_client import PrometheusConnect
prom = PrometheusConnect(url="http://localhost:9090")
result = prom.custom_query(query="rag_query_total")
```

## Performance Optimization Workflow

1. **Baseline**: Record current metrics
2. **Change**: Make optimization (code, config, hardware)
3. **Measure**: Compare dashboards before/after
4. **Validate**: Ensure no regressions
5. **Document**: Annotate dashboard with change

## Support Resources

- Implementation Guide: `GRAFANA_DASHBOARDS_IMPLEMENTATION.md`
- Design Document: `GRAFANA_DASHBOARD_DESIGN.md`
- Metrics Module: `utils/metrics.py`
- Prometheus Docs: https://prometheus.io/docs/
- Grafana Docs: https://grafana.com/docs/

## Quick Commands

```bash
# Start full monitoring stack
docker-compose up -d prometheus grafana

# Start macOS exporter
python macos_exporter.py --port 9101 &

# Check metrics
curl http://localhost:9102/metrics | grep -E "rag_query|rag_retrieval|rag_llm"

# Test metric export
python -c "from utils.metrics import RAGMetrics; m = RAGMetrics(); m.export(); print('OK')"

# View exported metrics
cat metrics/rag_app.prom | head -50
```

## Dashboard Access Matrix

| Role | Pipeline Internals | System Resources | Quality & Performance |
|------|-------------------|------------------|----------------------|
| Developer | Primary | Secondary | Tertiary |
| SRE | Secondary | Primary | Secondary |
| Product Manager | Tertiary | No | Primary |
| Data Scientist | Primary | No | Primary |
| QA Engineer | Secondary | No | Primary |

## Recommended Views by Scenario

### Development
- Pipeline Internals (focus on stage breakdown)
- Look for optimization opportunities

### Load Testing
- System Resources (full view)
- Monitor resource limits

### Production Monitoring
- Quality & Performance (overview)
- Set up alerts for anomalies

### Troubleshooting
- All three dashboards
- Correlate metrics across views

## Metrics Retention

- **Real-time**: 1 hour (full resolution)
- **Short-term**: 7 days (1m aggregation)
- **Long-term**: 90 days (5m aggregation)

Configure in Prometheus retention settings.
