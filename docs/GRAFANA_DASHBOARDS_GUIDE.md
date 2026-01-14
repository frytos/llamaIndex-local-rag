# Grafana Dashboards Guide for RAG Pipeline

## Overview

Your RAG system now has **comprehensive Grafana dashboards** that provide deep visibility into every aspect of the pipeline - from resource utilization to query quality. This guide explains each dashboard and how to use them for monitoring, optimization, and troubleshooting.

## Available Dashboards

### 1. **RAG Pipeline Internals** (`rag_pipeline_internals`)
Deep dive into the RAG pipeline stages: embedding, retrieval, and LLM generation.

**Use this dashboard for:**
- Understanding pipeline bottlenecks
- Monitoring each stage's performance
- Optimizing throughput and latency

**Key Panels:**

#### Pipeline Overview
- **Total Query Time (Avg)**: End-to-end latency
  - 🟢 Green: < 2s (good)
  - 🟡 Yellow: 2-5s (acceptable)
  - 🔴 Red: > 5s (needs optimization)

- **Success Rate**: Query success percentage
  - 🟢 Green: > 99%
  - 🟡 Yellow: 95-99%
  - 🔴 Red: < 95%

- **Queries/Second**: Current throughput

#### Embedding Stage
- **Embedding Duration**: Time spent computing embeddings
- **Embedding Throughput**: Embeddings per second
- **Embedding Batch Utilization**: How efficiently batches are used
  - Target: > 80% for optimal GPU/CPU usage

#### Retrieval Stage
- **Retrieval Scores Distribution**: P50, P95, P99, and top score
  - High variance = inconsistent retrieval quality
  - P50 < 0.5 = poor document relevance

- **Documents Retrieved per Query**: Tracks TOP_K effectiveness

#### LLM Generation Stage
- **LLM Token Generation Rate**: Tokens/second
  - M1 Mac (llama.cpp): 5-15 tokens/sec typical
  - GPU (vLLM): 30-100+ tokens/sec

- **LLM Generation Duration**: Time spent generating answers
- **Context Window Utilization**: % of context window used
  - 🟢 Green: < 70% (safe)
  - 🟡 Yellow: 70-90% (monitor)
  - 🔴 Red: > 90% (risk of truncation)

#### Chunking & Indexing
- **Chunk Size Distribution**: P50, P95, P99 chunk sizes
- **Indexing Throughput**: Documents and chunks per second

#### Database Operations
- **Total Vector Store Rows**: Current index size
- **Database Operations Rate**: Ops/second
- **Database Errors**: Critical issues requiring attention

---

### 2. **RAG System Resources (M1 Mac)** (`rag_system_resources`)
M1-specific resource monitoring: CPU, memory, Metal GPU, disk I/O, and process metrics.

**Use this dashboard for:**
- Identifying resource constraints
- Optimizing for 16GB memory limit
- Monitoring M1-specific bottlenecks

**Key Panels:**

#### System Overview
- **System CPU Usage**: Overall macOS CPU (all cores)
  - M1 has 8 cores (4 performance + 4 efficiency)
  - Target: < 80% for headroom

- **System Memory Usage**: Overall memory pressure
  - 🟢 Green: < 70% (healthy)
  - 🟡 Yellow: 70-85% (monitor for swapping)
  - 🔴 Red: > 85% (performance degradation)

- **RAG Process CPU**: CPU used by RAG process specifically
- **RAG Process Memory**: RSS memory (Resident Set Size)

#### M1-Specific Metrics
- **Load Average** (1m, 5m, 15m): System load trends
  - Target: < 4.0 (50% of 8 cores)
  - High load (> 6) = CPU bottleneck

- **CPU Core Breakdown**: Performance vs Efficiency cores
- **Memory Breakdown**: Active, Wired, Inactive, Free
  - **Wired**: Can't be paged out (kernel, drivers)
  - **Active**: Recently used memory
  - **Inactive**: Cache that can be freed

#### Process Resources
- **Thread Count**: Active Python threads
  - Typical: 5-20 threads
  - Spike = potential thread pool issue

- **Python Heap Memory**: Python object memory
- **Model Memory**: Embedding model + LLM memory usage
  - Embedding model (bge-small): ~50-100MB
  - LLM (Mistral 7B GGUF): ~4-6GB

#### Disk & Swap
- **Disk I/O**: Read/Write rates
- **Swap Usage**: Paging to disk (should be minimal)
  - Any swap usage = memory pressure

- **PostgreSQL Disk Space**: Vector store database size

#### Garbage Collection
- **GC Collections**: Python garbage collection by generation
- **GC Duration**: Time spent in GC

---

### 3. **RAG Quality & Performance** (`rag_quality_performance`)
Query quality, retrieval effectiveness, cache performance, and user experience metrics.

**Use this dashboard for:**
- Monitoring RAG answer quality
- Optimizing cache effectiveness
- Understanding user experience

**Key Panels:**

#### Quality Overview
- **Overall Success Rate**: Long-term success percentage
- **Average Retrieval Score**: Mean similarity score
  - > 0.8 = excellent relevance
  - 0.6-0.8 = good
  - < 0.6 = poor (consider reindexing or adjusting TOP_K)

- **Cache Hit Rate**: % of queries served from cache
  - Target: > 50% for typical workloads
  - > 80% = excellent caching

- **User Satisfaction Score**: If collecting feedback

#### Retrieval Quality Trends
- **Retrieval Score Over Time**: Tracks score trends
  - Declining scores = index degradation or query drift

- **Score Distribution Heatmap**: Visual distribution of scores
- **Retrieval Variance**: Score consistency
  - Low variance = consistent quality
  - High variance = unpredictable results

#### Query Patterns
- **Query Length Distribution**: Character/token counts
- **Queries by Hour**: Time-based patterns
- **Query Types**: If using categories

#### Cache Performance
- **Cache Hit Rate Trend**: Over time
- **Cache Size Growth**: Entries and bytes
- **Cache Evictions**: How often cache is cleared
- **Semantic Similarity Threshold**: Current threshold setting

#### Answer Quality
- **Answer Length Distribution**: Generated response lengths
- **Sources Cited per Answer**: How many retrieved docs are used
- **Token Efficiency**: Tokens generated per query

#### Performance Percentiles
- **Query Latency Heatmap**: P50, P75, P90, P95, P99, P99.9
  - P50: Typical user experience
  - P99: Worst 1% of queries

- **Stage Breakdown**: Time spent in each pipeline stage
  - Embedding: ~0.1-0.5s
  - Retrieval: ~0.2-0.5s
  - Generation: ~5-15s (M1), ~1-3s (GPU)

---

## Quick Start

### 1. Start Monitoring Stack
```bash
# Start Prometheus, Grafana, and exporters
docker-compose -f config/docker-compose.yml up -d

# Or use the minimal stack
docker-compose -f config/docker-compose.minimal.yml up -d

# Start macOS exporter (for M1-specific metrics)
python macos_exporter.py --port 9101
```

### 2. Access Grafana
- **URL**: http://localhost:3000
- **Default credentials**: admin / admin (change on first login)

### 3. Configure Data Source
Grafana should auto-configure Prometheus, but if needed:
1. Go to Configuration > Data Sources
2. Add Prometheus
3. URL: `http://prometheus:9090` (Docker) or `http://localhost:9090` (local)

### 4. Import Dashboards
Dashboards are in `config/grafana/dashboards/`:
- `rag_pipeline_internals.json`
- `rag_system_resources.json`
- `rag_quality_performance.json`
- `rag_overview.json` (simpler overview)
- `rag_application.json` (application-level)

**Auto-provisioning**: Dashboards are automatically loaded if using the Docker stack.

**Manual import**:
1. Go to Dashboards > Import
2. Upload JSON file
3. Select Prometheus data source
4. Click Import

### 5. Enable Metrics in RAG Code

Add metrics instrumentation to your queries:

```python
from utils.metrics import get_metrics

# Get global metrics instance
metrics = get_metrics()

# During query execution
with metrics.query_timer():
    # Embedding stage
    with metrics.embedding_timer():
        embeddings = embed_query(query)

    # Retrieval stage
    results = retriever.retrieve(query)
    scores = [node.score for node in results]
    metrics.record_retrieval(len(results), scores)

    # LLM generation
    start_time = time.time()
    answer = llm.generate(prompt)
    duration = time.time() - start_time

    tokens_generated = len(answer.split())  # Rough estimate
    metrics.record_llm_generation(
        tokens_generated=tokens_generated,
        duration_seconds=duration,
        context_tokens_used=context_size,
        context_tokens_available=8192
    )

metrics.record_query_success()

# Export metrics for Prometheus
metrics.export("rag_app.prom")
```

---

## Common Use Cases

### Scenario 1: Queries are Slow
**Check**: RAG Pipeline Internals → Stage Breakdown

**Analysis**:
- Embedding slow? → Check `Embedding Duration` and `Batch Utilization`
- Retrieval slow? → Check database performance and index size
- Generation slow? → Check `LLM Token Generation Rate` and context utilization

**Solutions**:
- Increase embedding batch size: `EMBED_BATCH=128`
- Reduce TOP_K: `TOP_K=3` instead of 5
- Reduce context: Lower `CHUNK_SIZE` or `TOP_K`
- Use vLLM for GPU acceleration

### Scenario 2: High Memory Usage
**Check**: RAG System Resources → Memory Breakdown

**Analysis**:
- **Python Heap** high? → Memory leak, check object retention
- **Model Memory** high? → Models not unloaded
- **Swap usage** > 0? → Insufficient RAM

**Solutions**:
```python
# Unload models when not needed
del embed_model
gc.collect()

# Reduce model size
N_GPU_LAYERS=16  # Instead of 24

# Reduce batch sizes
EMBED_BATCH=32
N_BATCH=128
```

### Scenario 3: Poor Retrieval Quality
**Check**: RAG Quality & Performance → Retrieval Quality

**Analysis**:
- **Average score** < 0.6? → Index quality issue
- **High variance**? → Inconsistent chunk quality
- **Low top score**? → Missing relevant documents

**Solutions**:
- Reindex with better chunking:
  ```bash
  CHUNK_SIZE=700 CHUNK_OVERLAP=150 RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
  ```
- Try different embedding model:
  ```bash
  EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
  ```
- Increase TOP_K to retrieve more candidates

### Scenario 4: Cache Not Effective
**Check**: RAG Quality & Performance → Cache Performance

**Analysis**:
- **Hit rate** < 20%? → Queries too unique
- **Semantic threshold** too strict? → Widen threshold

**Solutions**:
```python
from utils.query_cache import QueryCache

cache = QueryCache(
    similarity_threshold=0.85  # Lower from 0.95
)
```

---

## Alerting Rules

Configure alerts in Prometheus (`config/monitoring/prometheus.yml`):

### Critical Alerts
```yaml
groups:
  - name: rag_critical
    interval: 30s
    rules:
      - alert: RAGHighErrorRate
        expr: (rate(rag_query_errors_total[5m]) / rate(rag_query_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "RAG error rate > 5%"

      - alert: RAGHighLatency
        expr: rate(rag_query_duration_seconds_sum[5m]) / rate(rag_query_duration_seconds_count[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RAG query latency > 10s"

      - alert: HighMemoryPressure
        expr: macos_memory_percent > 90
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Memory usage > 90%"
```

---

## Dashboard Customization

### Add Custom Panels

**Example: Add "Slow Queries" Panel**
```json
{
  "type": "timeseries",
  "title": "Slow Queries (>10s)",
  "targets": [{
    "expr": "increase(rag_query_duration_seconds_bucket{le=\"+Inf\"}[1m]) - increase(rag_query_duration_seconds_bucket{le=\"10\"}[1m])",
    "legendFormat": "Slow Queries"
  }]
}
```

### Modify Thresholds
Edit JSON files to adjust color thresholds:
```json
"thresholds": {
  "steps": [
    {"value": 0, "color": "green"},
    {"value": 70, "color": "yellow"},
    {"value": 90, "color": "red"}
  ]
}
```

---

## Troubleshooting

### Metrics Not Showing
1. **Check Prometheus targets**: http://localhost:9090/targets
   - All targets should be "UP"

2. **Check metrics export**:
   ```bash
   ls -l metrics/rag_app.prom
   # Should show recent timestamp
   ```

3. **Verify Prometheus scraping**:
   ```bash
   curl http://localhost:9090/api/v1/query?query=rag_query_total
   ```

### Dashboard Shows "No Data"
1. **Check time range**: Adjust dashboard time picker
2. **Run some queries** to generate metrics
3. **Check data source**: Configuration > Data Sources > Test

### macOS Metrics Missing
1. **Start macOS exporter**:
   ```bash
   python macos_exporter.py --port 9101
   ```
2. **Check Prometheus config** includes:
   ```yaml
   - job_name: 'macos-host'
     static_configs:
       - targets: ['host.docker.internal:9101']
   ```

---

## Performance Tips

### For M1 Mac (16GB)
- Keep **memory usage** < 70%
- Monitor **swap usage** (should be near 0)
- Use **Performance cores** for heavy compute
- Set `N_GPU_LAYERS=24` for optimal Metal usage

### For Production
- Use **vLLM** for GPU acceleration (15-30x faster)
- Enable **query caching** (50-80% hit rate typical)
- Set appropriate **refresh intervals**:
  - Real-time monitoring: 5-10s
  - General use: 30s-1m
  - Historical analysis: 5m-15m

### Dashboard Performance
- Limit time ranges for heavy queries
- Use **recording rules** in Prometheus for expensive calculations
- Reduce panel count on overview dashboards

---

## Related Documentation

- **Metrics API**: See `utils/metrics.py` for full API
- **Prometheus Config**: `config/monitoring/prometheus.yml`
- **Docker Stack**: `config/docker-compose.yml`
- **macOS Exporter**: `macos_exporter.py --help`
- **Grafana Setup**: `docs/GRAFANA_MCP_SETUP.md`

---

## Support

For issues or questions:
1. Check Grafana logs: `docker logs grafana`
2. Check Prometheus logs: `docker logs prometheus`
3. Review metrics export: `cat metrics/rag_app.prom`
4. File an issue with dashboard export and error logs
