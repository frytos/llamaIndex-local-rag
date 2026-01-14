# Grafana Dashboards Implementation Guide

**Date**: 2026-01-09
**Author**: Performance Engineer
**Version**: 1.0.0

## Overview

This document provides a complete implementation guide for the three new comprehensive Grafana dashboards designed for the RAG system.

## Dashboards Created

### 1. RAG Pipeline Internals (`rag_pipeline_internals.json`)
**Purpose**: Deep visibility into each pipeline stage
**Refresh Rate**: 10 seconds
**Key Features**:
- Pipeline stage breakdown (embedding, retrieval, generation)
- LLM token generation metrics
- Retrieval quality scores with percentiles
- Chunking and indexing analytics
- Bottleneck identification

**Sections**:
- Pipeline Overview (4 stat panels)
- Stage Breakdown (pie chart + time series)
- Embedding Performance (throughput, duration, batch utilization)
- Retrieval Performance (scores, variance, relevance)
- LLM Generation (tokens/sec, context utilization)
- Chunking & Indexing (chunk sizes, throughput)

### 2. RAG System Resources (`rag_system_resources.json`)
**Purpose**: M1-specific resource monitoring
**Refresh Rate**: 5 seconds
**Key Features**:
- System-level metrics (CPU, memory, disk, network)
- RAG process-specific resources
- Memory breakdown by component
- Python garbage collection tracking
- Disk I/O monitoring

**Sections**:
- System Overview (CPU, memory gauges)
- CPU & Load (time series, load averages)
- Memory Breakdown (active, wired, inactive, free)
- Component Memory (embedding model, LLM, cache, heap)
- Python & GC (garbage collection activity)
- Disk I/O (usage, throughput, network traffic)
- Resource Alerts (health status table)

### 3. RAG Quality & Performance (`rag_quality_performance.json`)
**Purpose**: Quality metrics and user experience
**Refresh Rate**: 30 seconds
**Key Features**:
- Query success rate and satisfaction
- Retrieval quality percentiles
- Cache performance analytics
- Query/answer length trends
- Database health monitoring

**Sections**:
- Quality Overview (success rate, retrieval score, cache hit rate)
- Performance Metrics (latency percentiles, heatmap)
- Retrieval Quality (score trends, variance)
- Cache Performance (hit rate, size, evictions)
- Query & Answer Analytics (length trends, sources cited)
- Throughput & Volume (QPS, cumulative queries)
- Database Health (rows, operations, error rate)

## Implementation Steps

### Phase 1: Enhanced Metrics Module (COMPLETED)

The `utils/metrics.py` file has been enhanced with:
- Pipeline stage timing (embedding, retrieval, generation)
- LLM metrics (tokens/sec, context utilization)
- Retrieval percentiles (p50, p95, p99)
- Resource monitoring (CPU, memory, GC)
- Quality metrics (query/answer length, satisfaction)
- Chunking metrics (size percentiles, overlap)
- Cache analytics (size, evictions, similarity threshold)

### Phase 2: Integration with RAG Pipeline

To integrate the enhanced metrics into your RAG pipeline:

#### 1. Import Metrics Module
```python
from utils.metrics import RAGMetrics, get_metrics

# Get global metrics instance
metrics = get_metrics()
```

#### 2. Instrument Query Pipeline
```python
# In your query function
def run_query(engine, question):
    # Record query info
    metrics.record_query_info(question)

    # Time entire query
    with metrics.query_timer():
        # Stage 1: Embedding
        with metrics.stage_timer("embedding"):
            query_embedding = embed_query(question)

        # Stage 2: Retrieval
        with metrics.stage_timer("retrieval"):
            docs = retrieve_documents(query_embedding)
            scores = [doc.score for doc in docs]
            metrics.record_retrieval(len(docs), scores)

        # Stage 3: Generation
        with metrics.stage_timer("generation"):
            start_time = time.time()
            response = llm.generate(docs, question)
            duration = time.time() - start_time

            # Record LLM metrics
            tokens_generated = len(response.split())  # Rough estimate
            metrics.record_llm_generation(
                tokens_generated=tokens_generated,
                duration_seconds=duration,
                context_tokens_used=sum(len(d.text) for d in docs) // 4,
                context_tokens_available=8192  # Your context window
            )

            # Record answer info
            metrics.record_answer_info(
                answer_text=response,
                sources_cited=len(docs)
            )

    # Record success
    metrics.record_query_success()

    # Export metrics periodically
    metrics.export()

    return response
```

#### 3. Instrument Indexing Pipeline
```python
def index_documents(doc_path):
    start_time = time.time()

    # Load and chunk
    docs = load_documents(doc_path)
    chunk_start = time.time()
    chunks, chunk_sizes = chunk_documents(docs)
    chunk_duration = time.time() - chunk_start

    # Record chunking metrics
    metrics.record_chunking(
        duration_seconds=chunk_duration,
        chunk_sizes=chunk_sizes,
        overlap_bytes=chunk_overlap
    )

    # Embed with batch tracking
    with metrics.embedding_timer():
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_start = time.time()
            emb = embed_batch(batch)
            batch_duration = time.time() - batch_start

            metrics.record_embedding_batch(
                batch_size=len(batch),
                max_batch_size=batch_size,
                duration_seconds=batch_duration
            )
            embeddings.extend(emb)

    # Record indexing
    total_duration = time.time() - start_time
    metrics.record_indexing(
        num_documents=len(docs),
        num_chunks=len(chunks),
        duration_seconds=total_duration
    )

    metrics.export()
```

#### 4. Add Resource Monitoring (Optional)
```python
import threading
import time

def resource_monitor(metrics, interval=5):
    """Background thread to update resource metrics"""
    while True:
        metrics.update_resource_metrics()
        metrics.update_gc_metrics()
        metrics.export()
        time.sleep(interval)

# Start monitoring thread
monitor_thread = threading.Thread(
    target=resource_monitor,
    args=(metrics, 5),
    daemon=True
)
monitor_thread.start()
```

### Phase 3: Dashboard Deployment

#### 1. Copy Dashboard Files
```bash
# Dashboards are already in place:
ls -la config/grafana/dashboards/
# - rag_pipeline_internals.json
# - rag_system_resources.json
# - rag_quality_performance.json
```

#### 2. Update Grafana Provisioning
Edit `config/grafana/provisioning/dashboards/dashboards.yml`:
```yaml
apiVersion: 1

providers:
  - name: 'RAG Dashboards'
    orgId: 1
    folder: 'RAG System'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

#### 3. Restart Grafana
```bash
# If using Docker Compose
docker-compose restart grafana

# Or rebuild
docker-compose up -d --build grafana
```

### Phase 4: Metrics Export Configuration

#### Option A: File-Based Metrics (Current)
The metrics are exported to `metrics/rag_app.prom` and Prometheus scrapes this file.

Current limitation: Prometheus needs a push gateway or HTTP endpoint.

#### Option B: HTTP Metrics Endpoint (Recommended)
Add Prometheus HTTP server to your application:

```python
# Add to your main application file
from prometheus_client import start_http_server
import threading

# Start Prometheus metrics server
def start_metrics_server(port=9102):
    start_http_server(port)
    print(f"Metrics server running on port {port}")

# In your main() or __init__
metrics_thread = threading.Thread(
    target=start_metrics_server,
    args=(9102,),
    daemon=True
)
metrics_thread.start()
```

Update `config/monitoring/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'rag-app'
    static_configs:
      - targets: ['host.docker.internal:9102']
        labels:
          service: 'rag-application'
          environment: 'local'
    scrape_interval: 15s
    scrape_timeout: 10s
```

## Metrics Reference

### Pipeline Metrics
```
rag_query_total                                    # Total queries
rag_query_success_total                            # Successful queries
rag_query_errors_total                             # Failed queries
rag_query_duration_seconds_sum                     # Total query time
rag_query_duration_seconds_count                   # Query count
rag_pipeline_stage_duration_seconds_sum{stage}     # Stage timing
rag_pipeline_stage_duration_seconds_count{stage}   # Stage count
rag_pipeline_stage_errors_total{stage}             # Stage errors
```

### Retrieval Metrics
```
rag_retrieval_total                                # Total retrievals
rag_retrieval_score_avg                            # Average score
rag_retrieval_score_p50                            # Median score
rag_retrieval_score_p95                            # 95th percentile
rag_retrieval_score_p99                            # 99th percentile
rag_retrieval_top_score                            # Highest score
rag_retrieval_score_variance                       # Score variance
rag_retrieval_documents_total                      # Documents retrieved
rag_retrieval_documents_relevant_total             # Relevant docs (feedback)
```

### LLM Metrics
```
rag_llm_tokens_generated_total                     # Total tokens
rag_llm_tokens_per_second                          # Generation speed
rag_llm_context_tokens_used                        # Context used
rag_llm_context_tokens_available                   # Context available
rag_llm_context_utilization_percent                # Context usage %
rag_llm_generation_duration_seconds_sum            # Generation time
```

### Cache Metrics
```
rag_cache_requests_total                           # Cache requests
rag_cache_hits_total                               # Cache hits
rag_cache_misses_total                             # Cache misses
rag_cache_hit_rate                                 # Hit rate (0-1)
rag_cache_size_entries                             # Cache entries
rag_cache_size_bytes                               # Cache memory
rag_cache_evictions_total                          # Evictions
rag_cache_semantic_similarity_threshold            # Similarity threshold
```

### Resource Metrics (requires psutil)
```
rag_process_cpu_percent                            # Process CPU
rag_process_memory_rss_bytes                       # Resident memory
rag_process_memory_vms_bytes                       # Virtual memory
rag_process_threads_active                         # Active threads
rag_memory_embedding_model_bytes                   # Embedding model mem
rag_memory_llm_model_bytes                         # LLM model mem
rag_memory_cache_bytes                             # Cache mem
rag_memory_python_heap_bytes                       # Python heap
rag_process_gc_collections_total{generation}       # GC collections
rag_process_gc_duration_seconds                    # GC duration
```

### Quality Metrics
```
rag_query_length_chars                             # Query length
rag_query_tokens                                   # Query tokens
rag_answer_length_chars                            # Answer length
rag_answer_sources_cited                           # Sources cited
rag_answer_satisfaction_score                      # User satisfaction
```

### Chunking Metrics
```
rag_chunking_duration_seconds                      # Chunking time
rag_chunks_per_document                            # Avg chunks/doc
rag_chunk_size_bytes_p50                           # Median chunk size
rag_chunk_size_bytes_p95                           # 95th percentile
rag_chunk_size_bytes_p99                           # 99th percentile
rag_chunk_overlap_bytes                            # Overlap size
```

## Troubleshooting

### Issue: Dashboards not showing data
**Solutions**:
1. Check Prometheus is scraping metrics:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```
2. Verify metrics are being exported:
   ```bash
   curl http://localhost:9102/metrics | grep rag_
   ```
3. Check Grafana data source:
   - Navigate to Configuration > Data Sources
   - Test Prometheus connection

### Issue: Missing system metrics
**Solutions**:
1. Ensure macOS exporter is running:
   ```bash
   python macos_exporter.py --port 9101
   ```
2. Check Prometheus can reach exporter:
   ```bash
   curl http://localhost:9101/metrics | grep macos_
   ```

### Issue: Percentile metrics not updating
**Solutions**:
1. Need at least 10 samples for percentile calculation
2. Check score_buffer is being populated:
   ```python
   print(f"Buffer size: {len(metrics._score_buffer)}")
   ```
3. Ensure metrics.export() is called after operations

### Issue: High memory usage from metrics
**Solutions**:
1. Reduce buffer size in metrics initialization:
   ```python
   metrics._buffer_max_size = 500  # Default is 1000
   ```
2. Export metrics less frequently
3. Use sampling for expensive metrics

## Performance Impact

Expected overhead from comprehensive metrics:
- **CPU**: <1% additional usage
- **Memory**: ~10-20MB for buffers and state
- **Disk I/O**: Minimal (periodic export)
- **Network**: ~5KB per scrape interval

Optimization tips:
- Use background thread for resource metrics
- Sample expensive operations (GC stats)
- Batch metric updates
- Use lazy evaluation for percentiles

## Next Steps

1. **Week 1**: Deploy enhanced metrics and Pipeline Internals dashboard
2. **Week 2**: Add system resources monitoring and deploy Resources dashboard
3. **Week 3**: Implement quality feedback collection and deploy Quality dashboard
4. **Week 4**: Set up alerting rules and SLO tracking

## Alerting Rules (Future)

Example alert rules to add to Prometheus:

```yaml
groups:
  - name: rag_alerts
    interval: 30s
    rules:
      - alert: HighQueryLatency
        expr: rate(rag_query_duration_seconds_sum[5m]) / rate(rag_query_duration_seconds_count[5m]) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High query latency detected"
          description: "Average query latency is {{ $value }}s"

      - alert: LowRetrievalQuality
        expr: rag_retrieval_score_avg < 0.6
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low retrieval quality"
          description: "Average retrieval score is {{ $value }}"

      - alert: LowCacheHitRate
        expr: rag_cache_hit_rate < 0.3
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value }}"

      - alert: HighMemoryUsage
        expr: rag_process_memory_rss_bytes > 14000000000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Process using {{ $value | humanize }}B"
```

## Validation Checklist

Before production deployment:
- [ ] All metrics are being collected and exported
- [ ] Dashboards load without errors
- [ ] Real data is flowing into panels
- [ ] Thresholds are appropriate for your system
- [ ] Alerting rules are configured
- [ ] Documentation is updated
- [ ] Team is trained on dashboard usage
- [ ] Runbook created for common issues

## Support

For questions or issues:
1. Check this implementation guide
2. Review `GRAFANA_DASHBOARD_DESIGN.md` for design decisions
3. Examine metric definitions in `utils/metrics.py`
4. Test metrics export: `python utils/metrics.py`

## References

- Prometheus Query Language: https://prometheus.io/docs/prometheus/latest/querying/basics/
- Grafana Dashboard Best Practices: https://grafana.com/docs/grafana/latest/dashboards/
- RAG Pipeline Documentation: `docs/PROJECT_EXPLANATION.md`
- Metrics Module: `utils/metrics.py`
