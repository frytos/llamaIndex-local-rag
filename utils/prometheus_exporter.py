"""
Prometheus Metrics Exporter for RAG Pipeline

Exports metrics in Prometheus format for scraping by Prometheus server.
Integrates with Grafana for visualization and alerting.

Usage:
    # In your RAG pipeline or FastAPI app
    from utils.prometheus_exporter import (
        track_query, track_cache, track_validation,
        export_metrics, update_all_metrics
    )

    # Track a query
    track_query(
        query_type="factual",
        latency_s=2.5,
        cache_hit=False,
        confidence=0.85
    )

    # Export metrics for Prometheus scraping
    metrics_text = export_metrics()

    # FastAPI integration
    from fastapi import FastAPI, Response
    app = FastAPI()

    @app.get("/metrics")
    def metrics():
        update_all_metrics()  # Update from module stats
        return Response(content=export_metrics(), media_type="text/plain")

Environment Variables:
    PROMETHEUS_PORT=8000              # Metrics endpoint port
    PROMETHEUS_UPDATE_INTERVAL=15     # Auto-update interval (seconds)

Requirements:
    pip install prometheus_client
"""

import logging
import os
import time
from typing import Optional

# Prometheus client
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        generate_latest, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning(
        "prometheus_client not available. "
        "Install with: pip install prometheus-client"
    )

# Import stats modules
from utils.query_cache import semantic_cache, cache
from utils.conversation_memory import session_manager

log = logging.getLogger(__name__)

if not PROMETHEUS_AVAILABLE:
    # Provide no-op functions if prometheus not available
    def track_query(*args, **kwargs):
        pass

    def track_cache(*args, **kwargs):
        pass

    def track_validation(*args, **kwargs):
        pass

    def export_metrics():
        return "# Prometheus not available\n"

    def update_all_metrics():
        pass

else:
    # ========================================================================
    # METRIC DEFINITIONS
    # ========================================================================

    # Query metrics
    query_counter = Counter(
        'rag_queries_total',
        'Total number of queries processed',
        ['query_type', 'cache_hit']
    )

    query_latency = Histogram(
        'rag_query_latency_seconds',
        'Query latency in seconds',
        ['component'],  # embedding, retrieval, reranking, generation, total
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    )

    # Cache metrics
    cache_hit_rate = Gauge(
        'rag_cache_hit_rate',
        'Semantic cache hit rate (0-1)'
    )

    cache_size = Gauge(
        'rag_cache_size',
        'Number of cached queries'
    )

    cache_disk_mb = Gauge(
        'rag_cache_disk_mb',
        'Cache disk usage in MB'
    )

    cache_hits_total = Counter(
        'rag_cache_hits_total',
        'Total cache hits'
    )

    cache_misses_total = Counter(
        'rag_cache_misses_total',
        'Total cache misses'
    )

    # Answer quality metrics
    answer_confidence = Histogram(
        'rag_answer_confidence',
        'Answer confidence score (0-1)',
        buckets=[0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    )

    hallucination_counter = Counter(
        'rag_hallucinations_total',
        'Total number of hallucinations detected'
    )

    low_confidence_counter = Counter(
        'rag_low_confidence_total',
        'Queries with confidence below threshold'
    )

    # Retrieval metrics
    retrieval_mrr = Histogram(
        'rag_retrieval_mrr',
        'Retrieval Mean Reciprocal Rank',
        buckets=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    )

    retrieval_ndcg = Histogram(
        'rag_retrieval_ndcg',
        'Retrieval nDCG@k score',
        buckets=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    )

    # Conversation metrics
    active_sessions = Gauge(
        'rag_active_sessions',
        'Number of active conversation sessions'
    )

    session_turns = Histogram(
        'rag_session_turns',
        'Number of turns per conversation session',
        buckets=[1, 2, 3, 5, 10, 20, 50, 100]
    )

    # System metrics
    memory_usage_bytes = Gauge(
        'rag_memory_usage_bytes',
        'Process memory usage in bytes'
    )

    # Configuration info
    rag_config = Info('rag_config', 'RAG configuration information')

    # Set static configuration (call once at startup)
    rag_config.info({
        'cache_enabled': os.getenv('ENABLE_SEMANTIC_CACHE', '1'),
        'cache_threshold': os.getenv('SEMANTIC_CACHE_THRESHOLD', '0.92'),
        'routing_enabled': os.getenv('ENABLE_QUERY_ROUTING', '0'),
        'routing_method': os.getenv('ROUTING_METHOD', 'pattern'),
        'reranking_enabled': os.getenv('ENABLE_RERANKING', '0'),
        'hyde_enabled': os.getenv('ENABLE_HYDE', '0'),
    })

    # ========================================================================
    # TRACKING FUNCTIONS
    # ========================================================================

    def track_query(
        query_type: str,
        latency_s: float,
        cache_hit: bool,
        confidence: Optional[float] = None,
        component: str = "total"
    ):
        """
        Track a query execution.

        Args:
            query_type: Type of query (factual, conceptual, etc.)
            latency_s: Query latency in seconds
            cache_hit: Whether query was served from cache
            confidence: Answer confidence score (0-1)
            component: Component being timed (total, embedding, retrieval, etc.)
        """
        query_counter.labels(
            query_type=query_type,
            cache_hit=str(cache_hit)
        ).inc()

        query_latency.labels(component=component).observe(latency_s)

        if confidence is not None:
            answer_confidence.observe(confidence)

            # Track low confidence
            if confidence < 0.7:
                low_confidence_counter.inc()

    def track_cache(hit: bool):
        """
        Track cache access.

        Args:
            hit: True if cache hit, False if miss
        """
        if hit:
            cache_hits_total.inc()
        else:
            cache_misses_total.inc()

    def track_validation(
        confidence: float,
        has_hallucinations: bool,
        hallucination_count: int = 0
    ):
        """
        Track answer validation results.

        Args:
            confidence: Confidence score (0-1)
            has_hallucinations: Whether hallucinations were detected
            hallucination_count: Number of hallucinations found
        """
        answer_confidence.observe(confidence)

        if has_hallucinations:
            hallucination_counter.inc(hallucination_count)

    def track_retrieval(mrr: float, ndcg: float):
        """
        Track retrieval quality metrics.

        Args:
            mrr: Mean Reciprocal Rank
            ndcg: Normalized Discounted Cumulative Gain
        """
        retrieval_mrr.observe(mrr)
        retrieval_ndcg.observe(ndcg)

    def track_conversation(turns: int):
        """
        Track conversation session metrics.

        Args:
            turns: Number of turns in the conversation
        """
        session_turns.observe(turns)

    # ========================================================================
    # UPDATE FUNCTIONS
    # ========================================================================

    def update_all_metrics():
        """
        Update all gauge metrics from module stats.

        Call this periodically or before exporting metrics.
        """
        # Update cache metrics
        cache_stats = semantic_cache.stats()

        total_requests = cache_stats['hits'] + cache_stats['misses']
        if total_requests > 0:
            cache_hit_rate.set(cache_stats['hit_rate'])

        cache_size.set(cache_stats['count'])
        cache_disk_mb.set(cache_stats['size_mb'])

        # Update session metrics
        conv_stats = session_manager.stats()
        active_sessions.set(conv_stats['active_sessions'])

        # Update system metrics (if psutil available)
        try:
            import psutil
            process = psutil.Process()
            memory_usage_bytes.set(process.memory_info().rss)
        except ImportError:
            pass

        log.debug("Updated all Prometheus metrics")

    def export_metrics() -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        update_all_metrics()
        return generate_latest(REGISTRY)


# ============================================================================
# AUTO-UPDATE DAEMON (Optional)
# ============================================================================

def start_metrics_updater(interval: int = 15):
    """
    Start background thread to auto-update metrics.

    Args:
        interval: Update interval in seconds
    """
    import threading

    def update_loop():
        while True:
            try:
                update_all_metrics()
                time.sleep(interval)
            except Exception as e:
                log.error(f"Error updating metrics: {e}")
                time.sleep(interval)

    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()
    log.info(f"Started metrics updater (interval: {interval}s)")


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def example_integration():
    """
    Example: How to integrate Prometheus metrics into your RAG pipeline.

    This shows the integration pattern for rag_low_level_m1_16gb_verbose.py
    """

    print("=" * 70)
    print("Prometheus Integration Example")
    print("=" * 70)

    print("\n# In your RAG pipeline code:")
    print("""
from utils.prometheus_exporter import (
    track_query, track_cache, track_validation,
    update_all_metrics, export_metrics
)

# At startup (if using FastAPI)
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/metrics")
def metrics():
    '''Prometheus metrics endpoint'''
    update_all_metrics()
    return Response(content=export_metrics(), media_type="text/plain")

# In your query function
def query_with_metrics(query_text: str):
    start = time.time()

    # Check cache
    cached = semantic_cache.get_semantic(query, embedding)
    track_cache(hit=cached is not None)

    if cached:
        track_query("unknown", time.time() - start, cache_hit=True)
        return cached

    # Route query
    routing = router.route(query_text)
    query_type = routing.query_type.value

    # Retrieve and generate
    response = query_engine.query(query_text)

    # Validate
    validation = validator.validate_answer(response, query_text, chunks)

    # Track metrics
    track_query(
        query_type=query_type,
        latency_s=time.time() - start,
        cache_hit=False,
        confidence=validation['confidence_score']
    )

    track_validation(
        confidence=validation['confidence_score'],
        has_hallucinations=validation['hallucination_count'] > 0,
        hallucination_count=validation['hallucination_count']
    )

    return response
    """)

    print("\n# Access metrics")
    print("curl http://localhost:8000/metrics")
    print("\n# Example output:")
    print("rag_queries_total{query_type=\"factual\",cache_hit=\"false\"} 45")
    print("rag_cache_hit_rate 0.35")
    print("rag_query_latency_seconds_bucket{component=\"total\",le=\"2.0\"} 38")
    print("rag_answer_confidence_bucket{le=\"0.8\"} 42")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if PROMETHEUS_AVAILABLE:
        example_integration()
    else:
        print("❌ prometheus_client not installed")
        print("\nInstall with:")
        print("  pip install prometheus-client")
        print("\nThen rerun this script to see integration examples.")
