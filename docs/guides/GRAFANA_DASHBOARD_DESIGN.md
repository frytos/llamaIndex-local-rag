# Grafana Dashboard Design for RAG System

**Date**: 2026-01-09
**Author**: Performance Engineer
**System**: Local RAG Pipeline (M1 Mac 16GB)

## Executive Summary

Comprehensive Grafana dashboard design for RAG system with focus on:
1. Pipeline internals (embedding, retrieval, LLM generation)
2. M1-specific resource monitoring
3. Quality and performance tracking

## Current Metrics Analysis

### Existing Metrics (utils/metrics.py)

**Strengths:**
- Query timing and success tracking
- Retrieval score monitoring
- Basic embedding metrics
- Cache hit rate tracking
- Database operation counts

**Gaps Identified:**

1. **Pipeline Breakdown Missing**
   - No individual stage timings (embed vs retrieve vs generate)
   - No token throughput metrics
   - No chunking performance data
   - Missing LLM-specific metrics (tokens/sec, context usage)

2. **Resource Metrics Incomplete**
   - No GPU (Metal) utilization tracking
   - No memory breakdown by component
   - No disk I/O patterns
   - Missing Python GC metrics

3. **Quality Metrics Limited**
   - No retrieval relevance tracking
   - No answer quality indicators
   - No semantic similarity metrics
   - Missing context window utilization

4. **Performance Insights Lacking**
   - No percentile distributions (p50, p95, p99)
   - No batch processing efficiency
   - No throughput metrics
   - Missing bottleneck identification

## Dashboard Specifications

### Dashboard 1: RAG Pipeline Internals
**Purpose**: Deep visibility into each pipeline stage
**Refresh**: 10s
**Target Audience**: Developers, Performance Engineers

### Dashboard 2: RAG System Resources (M1)
**Purpose**: M1-specific resource monitoring
**Refresh**: 5s
**Target Audience**: SREs, System Administrators

### Dashboard 3: RAG Quality & Performance
**Purpose**: Quality metrics and user experience
**Refresh**: 30s
**Target Audience**: Product Managers, Data Scientists

## New Metrics Required

### Pipeline Metrics
```python
# Stage timing
rag_pipeline_stage_duration_seconds{stage="embedding|retrieval|generation"}
rag_pipeline_stage_errors_total{stage="embedding|retrieval|generation"}

# LLM metrics
rag_llm_tokens_generated_total
rag_llm_tokens_per_second
rag_llm_context_tokens_used
rag_llm_context_tokens_available
rag_llm_context_utilization_percent
rag_llm_generation_duration_seconds

# Chunking metrics
rag_chunking_duration_seconds
rag_chunks_per_document
rag_chunk_size_bytes{percentile="p50|p95|p99"}
rag_chunk_overlap_bytes
```

### Resource Metrics (M1-Specific)
```python
# Metal GPU (from IOKit or system_profiler)
rag_metal_gpu_utilization_percent
rag_metal_gpu_memory_used_bytes
rag_metal_gpu_memory_total_bytes
rag_metal_gpu_active

# Memory breakdown
rag_memory_embedding_model_bytes
rag_memory_llm_model_bytes
rag_memory_cache_bytes
rag_memory_python_heap_bytes

# Process metrics
rag_process_cpu_percent
rag_process_memory_rss_bytes
rag_process_threads_active
rag_process_gc_collections_total{generation="0|1|2"}
rag_process_gc_duration_seconds
```

### Quality Metrics
```python
# Retrieval quality
rag_retrieval_score_percentile{percentile="p50|p95|p99"}
rag_retrieval_top_score
rag_retrieval_score_variance
rag_retrieval_documents_relevant_total  # Manual feedback
rag_retrieval_rerank_improvement  # If using reranker

# Answer quality (requires feedback)
rag_answer_satisfaction_score  # User feedback
rag_answer_length_chars
rag_answer_sources_cited

# Semantic cache
rag_cache_semantic_similarity_threshold
rag_cache_size_entries
rag_cache_evictions_total
```

### Performance Metrics
```python
# Throughput
rag_queries_per_second
rag_documents_indexed_per_second
rag_embeddings_per_second

# Batch efficiency
rag_embedding_batch_size
rag_embedding_batch_utilization_percent

# Query complexity
rag_query_length_chars
rag_query_tokens
rag_response_sources_used
```

## Implementation Priority

### Phase 1: Core Pipeline Metrics (Week 1)
- Stage timing breakdown
- LLM token metrics
- Basic resource tracking
- Dashboard 1 deployment

### Phase 2: M1-Specific Resources (Week 2)
- Metal GPU monitoring
- Memory breakdown
- Python profiling
- Dashboard 2 deployment

### Phase 3: Quality & Optimization (Week 3)
- Retrieval quality tracking
- Percentile distributions
- Cache analytics
- Dashboard 3 deployment

## Technical Notes

### M1 Metal GPU Monitoring
Metal GPU metrics on M1 are challenging:
- No direct API like NVIDIA's nvidia-smi
- Options:
  1. Parse `sudo powermetrics --samplers gpu_power` (requires root)
  2. Use Activity Monitor data via `ioreg`
  3. Sample via `system_profiler SPDisplaysDataType`
  4. Use third-party tools (iStat Menus API)

**Recommendation**: Implement lightweight sampling via `ioreg` or `powermetrics` with caching.

### Prometheus Integration
Current setup uses file-based metrics export:
- Metrics written to `metrics/rag_app.prom`
- Prometheus scrapes file or uses Pushgateway
- 15s scrape interval

**Enhancement**: Add HTTP endpoint using `prometheus_client` for real-time scraping.

### Performance Overhead
Target: <1% overhead from metrics collection
- Use sampling for expensive metrics (GPU)
- Batch metric updates
- Async metric export
- Lazy evaluation for percentiles

## Dashboard Visualization Guidelines

### Color Scheme
- **Green**: Healthy (>95% success, <2s latency)
- **Yellow**: Warning (90-95% success, 2-5s latency)
- **Red**: Critical (<90% success, >5s latency)

### Panel Types
- **Stat**: Current values with thresholds
- **Time Series**: Trends over time
- **Gauge**: Resource utilization
- **Heatmap**: Distribution analysis
- **Table**: Detailed breakdowns
- **Bar Gauge**: Comparative metrics

### Thresholds
- Query latency: Green <1s, Yellow <3s, Red >3s
- Success rate: Green >99%, Yellow >95%, Red <95%
- CPU: Green <60%, Yellow <80%, Red >80%
- Memory: Green <70%, Yellow <85%, Red >90%
- Cache hit rate: Green >80%, Yellow >50%, Red <50%

## Success Metrics

Dashboards should enable:
1. Identify bottleneck in <30 seconds
2. Track performance trends over time
3. Correlate resource usage with performance
4. Optimize cache and retrieval strategies
5. Plan capacity and scaling

## Next Steps

1. Enhance `utils/metrics.py` with new metrics
2. Create dashboard JSON configurations
3. Set up Metal GPU monitoring
4. Deploy dashboards to Grafana
5. Document alert thresholds
6. Create runbook for common issues
