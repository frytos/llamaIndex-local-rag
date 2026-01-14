"""
metrics.py - Prometheus metrics instrumentation for RAG pipeline

Provides:
  - Query metrics (latency, success rate, error rate)
  - Pipeline stage metrics (embedding, retrieval, generation)
  - Retrieval metrics (scores, document count, quality)
  - Embedding metrics (time, batch size, throughput)
  - Cache metrics (hit rate, miss rate, semantic similarity)
  - LLM metrics (tokens/sec, context usage)
  - Resource metrics (CPU, memory, GPU)
  - Quality metrics (retrieval scores, answer quality)
  - File-based metrics export for Prometheus

Usage:
    from utils.metrics import RAGMetrics

    metrics = RAGMetrics()

    # Record query with stage breakdown
    with metrics.query_timer():
        with metrics.stage_timer("embedding"):
            embeddings = embed_query(...)
        with metrics.stage_timer("retrieval"):
            docs = retrieve(...)
        with metrics.stage_timer("generation"):
            result = generate(...)

    metrics.record_query_success()
    metrics.record_retrieval_score(0.85)
    metrics.record_llm_generation(tokens=150, duration=2.3)

    # Export metrics
    metrics.export()
"""

import gc
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Optional: psutil for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MetricValue:
    """Single metric value with metadata"""
    name: str
    value: float
    metric_type: str  # gauge, counter, histogram
    help_text: str
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def to_prometheus(self) -> str:
        """Convert to Prometheus text format"""
        lines = []

        # HELP line
        lines.append(f"# HELP {self.name} {self.help_text}")

        # TYPE line
        lines.append(f"# TYPE {self.name} {self.metric_type}")

        # Metric line with labels
        if self.labels:
            label_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
            lines.append(f"{self.name}{{{label_str}}} {self.value}")
        else:
            lines.append(f"{self.name} {self.value}")

        # Timestamp (optional)
        if self.timestamp:
            lines[-1] += f" {int(self.timestamp * 1000)}"

        return "\n".join(lines)


class RAGMetrics:
    """Metrics collector for RAG pipeline"""

    def __init__(self, export_dir: Optional[str] = None):
        """
        Initialize metrics collector

        Args:
            export_dir: Directory to export metrics (default: ./metrics)
        """
        self.export_dir = Path(export_dir or os.getenv("METRICS_DIR", "metrics"))
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self._metrics: Dict[str, MetricValue] = {}
        self._lock = threading.Lock()

        # Initialize counters
        self._init_metrics()

    def _init_metrics(self):
        """Initialize all metrics with default values"""
        # Query metrics
        self._set_metric("rag_query_total", 0, "counter", "Total number of queries")
        self._set_metric("rag_query_success_total", 0, "counter", "Total successful queries")
        self._set_metric("rag_query_errors_total", 0, "counter", "Total failed queries")

        # Latency metrics (will use histogram buckets)
        self._set_metric("rag_query_duration_seconds_sum", 0, "counter", "Total query duration")
        self._set_metric("rag_query_duration_seconds_count", 0, "counter", "Total query count")

        # Pipeline stage metrics (NEW)
        for stage in ["embedding", "retrieval", "generation"]:
            self._set_metric(f"rag_pipeline_stage_duration_seconds_sum", 0, "counter",
                           f"Pipeline stage duration", {"stage": stage})
            self._set_metric(f"rag_pipeline_stage_duration_seconds_count", 0, "counter",
                           f"Pipeline stage count", {"stage": stage})
            self._set_metric(f"rag_pipeline_stage_errors_total", 0, "counter",
                           f"Pipeline stage errors", {"stage": stage})

        # Retrieval metrics (enhanced)
        self._set_metric("rag_retrieval_total", 0, "counter", "Total retrievals")
        self._set_metric("rag_retrieval_score_sum", 0, "counter", "Sum of retrieval scores")
        self._set_metric("rag_retrieval_score_count", 0, "counter", "Count of retrieval scores")
        self._set_metric("rag_retrieval_score_avg", 0, "gauge", "Average retrieval score")
        self._set_metric("rag_retrieval_score_p50", 0, "gauge", "Retrieval score p50")
        self._set_metric("rag_retrieval_score_p95", 0, "gauge", "Retrieval score p95")
        self._set_metric("rag_retrieval_score_p99", 0, "gauge", "Retrieval score p99")
        self._set_metric("rag_retrieval_top_score", 0, "gauge", "Top retrieval score")
        self._set_metric("rag_retrieval_score_variance", 0, "gauge", "Retrieval score variance")
        self._set_metric("rag_retrieval_documents_total", 0, "counter", "Total documents retrieved")
        self._set_metric("rag_retrieval_documents_relevant_total", 0, "counter", "Relevant documents (user feedback)")

        # Embedding metrics (enhanced)
        self._set_metric("rag_embedding_duration_seconds_sum", 0, "counter", "Total embedding time")
        self._set_metric("rag_embedding_duration_seconds_count", 0, "counter", "Embedding count")
        self._set_metric("rag_embedding_errors_total", 0, "counter", "Embedding errors")
        self._set_metric("rag_embedding_batch_size", 0, "gauge", "Current embedding batch size")
        self._set_metric("rag_embedding_batch_utilization_percent", 0, "gauge", "Embedding batch utilization")
        self._set_metric("rag_embeddings_per_second", 0, "gauge", "Embedding throughput")

        # LLM metrics (NEW)
        self._set_metric("rag_llm_tokens_generated_total", 0, "counter", "Total tokens generated")
        self._set_metric("rag_llm_tokens_per_second", 0, "gauge", "LLM token generation speed")
        self._set_metric("rag_llm_context_tokens_used", 0, "gauge", "Context tokens used")
        self._set_metric("rag_llm_context_tokens_available", 0, "gauge", "Context tokens available")
        self._set_metric("rag_llm_context_utilization_percent", 0, "gauge", "Context utilization")
        self._set_metric("rag_llm_generation_duration_seconds_sum", 0, "counter", "Total LLM generation time")
        self._set_metric("rag_llm_generation_duration_seconds_count", 0, "counter", "LLM generation count")

        # Cache metrics (enhanced)
        self._set_metric("rag_cache_requests_total", 0, "counter", "Total cache requests")
        self._set_metric("rag_cache_hits_total", 0, "counter", "Cache hits")
        self._set_metric("rag_cache_misses_total", 0, "counter", "Cache misses")
        self._set_metric("rag_cache_hit_rate", 0, "gauge", "Cache hit rate")
        self._set_metric("rag_cache_size_entries", 0, "gauge", "Cache size in entries")
        self._set_metric("rag_cache_size_bytes", 0, "gauge", "Cache size in bytes")
        self._set_metric("rag_cache_evictions_total", 0, "counter", "Cache evictions")
        self._set_metric("rag_cache_semantic_similarity_threshold", 0, "gauge", "Semantic cache threshold")

        # Indexing/Chunking metrics (enhanced)
        self._set_metric("rag_documents_indexed_total", 0, "counter", "Total documents indexed")
        self._set_metric("rag_chunks_created_total", 0, "counter", "Total chunks created")
        self._set_metric("rag_index_duration_seconds", 0, "gauge", "Last indexing duration")
        self._set_metric("rag_chunking_duration_seconds", 0, "gauge", "Last chunking duration")
        self._set_metric("rag_chunks_per_document", 0, "gauge", "Average chunks per document")
        self._set_metric("rag_chunk_size_bytes_p50", 0, "gauge", "Chunk size p50")
        self._set_metric("rag_chunk_size_bytes_p95", 0, "gauge", "Chunk size p95")
        self._set_metric("rag_chunk_size_bytes_p99", 0, "gauge", "Chunk size p99")
        self._set_metric("rag_chunk_overlap_bytes", 0, "gauge", "Chunk overlap size")

        # Database metrics
        self._set_metric("rag_db_rows_total", 0, "gauge", "Total rows in vector store")
        self._set_metric("rag_db_operations_total", 0, "counter", "Total database operations")
        self._set_metric("rag_db_errors_total", 0, "counter", "Database errors")

        # Throughput metrics (NEW)
        self._set_metric("rag_queries_per_second", 0, "gauge", "Query throughput")
        self._set_metric("rag_documents_indexed_per_second", 0, "gauge", "Indexing throughput")

        # Resource metrics (NEW) - only if psutil available
        if PSUTIL_AVAILABLE:
            self._set_metric("rag_process_cpu_percent", 0, "gauge", "Process CPU usage")
            self._set_metric("rag_process_memory_rss_bytes", 0, "gauge", "Process memory RSS")
            self._set_metric("rag_process_memory_vms_bytes", 0, "gauge", "Process memory VMS")
            self._set_metric("rag_process_threads_active", 0, "gauge", "Active threads")
            self._set_metric("rag_memory_embedding_model_bytes", 0, "gauge", "Embedding model memory")
            self._set_metric("rag_memory_llm_model_bytes", 0, "gauge", "LLM model memory")
            self._set_metric("rag_memory_cache_bytes", 0, "gauge", "Cache memory")
            self._set_metric("rag_memory_python_heap_bytes", 0, "gauge", "Python heap memory")

        # Python GC metrics (NEW)
        for gen in [0, 1, 2]:
            self._set_metric(f"rag_process_gc_collections_total", 0, "counter",
                           "GC collections", {"generation": str(gen)})
        self._set_metric("rag_process_gc_duration_seconds", 0, "gauge", "Last GC duration")

        # Quality metrics (NEW)
        self._set_metric("rag_answer_satisfaction_score", 0, "gauge", "User satisfaction score")
        self._set_metric("rag_answer_length_chars", 0, "gauge", "Answer length in characters")
        self._set_metric("rag_answer_sources_cited", 0, "gauge", "Sources cited in answer")
        self._set_metric("rag_query_length_chars", 0, "gauge", "Query length in characters")
        self._set_metric("rag_query_tokens", 0, "gauge", "Query length in tokens")

        # System metrics
        self._set_metric("rag_info", 1, "gauge", "RAG system info")

        # Storage for percentile calculation
        self._score_buffer: List[float] = []
        self._chunk_size_buffer: List[int] = []
        self._buffer_max_size = 1000  # Keep last 1000 values

    def _set_metric(
        self,
        name: str,
        value: float,
        metric_type: str,
        help_text: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a metric value"""
        with self._lock:
            self._metrics[name] = MetricValue(
                name=name,
                value=value,
                metric_type=metric_type,
                help_text=help_text,
                labels=labels or {},
                timestamp=time.time()
            )

    def _increment_metric(self, name: str, value: float = 1.0):
        """Increment a counter metric"""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].value += value
                self._metrics[name].timestamp = time.time()

    def _get_metric(self, name: str) -> Optional[float]:
        """Get metric value"""
        with self._lock:
            return self._metrics.get(name, MetricValue(name, 0, "gauge", "")).value

    # ==================== Query Metrics ====================

    @contextmanager
    def query_timer(self):
        """Context manager to time queries"""
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.record_query_duration(duration)
        except Exception as e:
            duration = time.time() - start_time
            self.record_query_duration(duration)
            raise

    def record_query_duration(self, duration_seconds: float):
        """Record query duration"""
        self._increment_metric("rag_query_total")
        self._increment_metric("rag_query_duration_seconds_sum", duration_seconds)
        self._increment_metric("rag_query_duration_seconds_count")

        # Create histogram buckets
        buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        for bucket in buckets:
            if duration_seconds <= bucket:
                bucket_name = f"rag_query_duration_seconds_bucket{{le=\"{bucket}\"}}"
                self._increment_metric(bucket_name)

        # +Inf bucket
        self._increment_metric('rag_query_duration_seconds_bucket{le="+Inf"}')

    def record_query_success(self):
        """Record successful query"""
        self._increment_metric("rag_query_success_total")

    def record_query_error(self, error_type: Optional[str] = None):
        """Record query error"""
        self._increment_metric("rag_query_errors_total")

        if error_type:
            metric_name = f'rag_query_errors_total{{type="{error_type}"}}'
            self._increment_metric(metric_name)

    # ==================== Pipeline Stage Metrics ====================

    @contextmanager
    def stage_timer(self, stage: str):
        """
        Context manager to time pipeline stages

        Args:
            stage: Stage name (embedding, retrieval, generation)
        """
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.record_stage_duration(stage, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.record_stage_duration(stage, duration)
            self.record_stage_error(stage)
            raise

    def record_stage_duration(self, stage: str, duration_seconds: float):
        """Record pipeline stage duration"""
        with self._lock:
            # Find metric with matching stage label
            for name, metric in self._metrics.items():
                if "pipeline_stage_duration_seconds_sum" in name and metric.labels.get("stage") == stage:
                    metric.value += duration_seconds
                    metric.timestamp = time.time()
                elif "pipeline_stage_duration_seconds_count" in name and metric.labels.get("stage") == stage:
                    metric.value += 1
                    metric.timestamp = time.time()

    def record_stage_error(self, stage: str):
        """Record pipeline stage error"""
        with self._lock:
            for name, metric in self._metrics.items():
                if "pipeline_stage_errors_total" in name and metric.labels.get("stage") == stage:
                    metric.value += 1
                    metric.timestamp = time.time()

    # ==================== Retrieval Metrics ====================

    def record_retrieval(self, num_documents: int, scores: list):
        """
        Record retrieval operation with enhanced metrics

        Args:
            num_documents: Number of documents retrieved
            scores: List of similarity scores
        """
        self._increment_metric("rag_retrieval_total")
        self._increment_metric("rag_retrieval_documents_total", num_documents)

        if scores:
            # Basic statistics
            avg_score = sum(scores) / len(scores)
            self._increment_metric("rag_retrieval_score_sum", sum(scores))
            self._increment_metric("rag_retrieval_score_count", len(scores))

            # Update average
            total_sum = self._get_metric("rag_retrieval_score_sum")
            total_count = self._get_metric("rag_retrieval_score_count")
            if total_count > 0:
                self._set_metric(
                    "rag_retrieval_score_avg",
                    total_sum / total_count,
                    "gauge",
                    "Average retrieval score"
                )

            # Top score
            self._set_metric("rag_retrieval_top_score", max(scores), "gauge", "Top retrieval score")

            # Variance
            if len(scores) > 1:
                variance = np.var(scores)
                self._set_metric("rag_retrieval_score_variance", variance, "gauge", "Retrieval score variance")

            # Update score buffer for percentiles
            self._score_buffer.extend(scores)
            if len(self._score_buffer) > self._buffer_max_size:
                self._score_buffer = self._score_buffer[-self._buffer_max_size:]

            # Calculate percentiles
            if len(self._score_buffer) >= 10:  # Need at least 10 samples
                p50 = np.percentile(self._score_buffer, 50)
                p95 = np.percentile(self._score_buffer, 95)
                p99 = np.percentile(self._score_buffer, 99)
                self._set_metric("rag_retrieval_score_p50", p50, "gauge", "Retrieval score p50")
                self._set_metric("rag_retrieval_score_p95", p95, "gauge", "Retrieval score p95")
                self._set_metric("rag_retrieval_score_p99", p99, "gauge", "Retrieval score p99")

    def record_retrieval_score(self, score: float):
        """Record single retrieval score"""
        self.record_retrieval(1, [score])

    def record_retrieval_relevance(self, is_relevant: bool):
        """
        Record user feedback on retrieval relevance

        Args:
            is_relevant: Whether retrieved documents were relevant
        """
        if is_relevant:
            self._increment_metric("rag_retrieval_documents_relevant_total")

    # ==================== Embedding Metrics ====================

    @contextmanager
    def embedding_timer(self):
        """Context manager to time embedding operations"""
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.record_embedding_duration(duration)
        except Exception as e:
            self.record_embedding_error()
            raise

    def record_embedding_duration(self, duration_seconds: float):
        """Record embedding operation duration"""
        self._increment_metric("rag_embedding_duration_seconds_sum", duration_seconds)
        self._increment_metric("rag_embedding_duration_seconds_count")

    def record_embedding_error(self):
        """Record embedding error"""
        self._increment_metric("rag_embedding_errors_total")

    def record_embedding_batch(self, batch_size: int, max_batch_size: int, duration_seconds: float):
        """
        Record embedding batch operation

        Args:
            batch_size: Actual batch size used
            max_batch_size: Maximum batch size configured
            duration_seconds: Time taken for batch
        """
        self._set_metric("rag_embedding_batch_size", batch_size, "gauge", "Current embedding batch size")

        if max_batch_size > 0:
            utilization = (batch_size / max_batch_size) * 100
            self._set_metric("rag_embedding_batch_utilization_percent", utilization, "gauge",
                           "Embedding batch utilization")

        if duration_seconds > 0:
            throughput = batch_size / duration_seconds
            self._set_metric("rag_embeddings_per_second", throughput, "gauge", "Embedding throughput")

    # ==================== LLM Metrics ====================

    def record_llm_generation(
        self,
        tokens_generated: int,
        duration_seconds: float,
        context_tokens_used: Optional[int] = None,
        context_tokens_available: Optional[int] = None
    ):
        """
        Record LLM generation metrics

        Args:
            tokens_generated: Number of tokens generated
            duration_seconds: Time taken for generation
            context_tokens_used: Tokens used in context (prompt + retrieved docs)
            context_tokens_available: Total context window size
        """
        self._increment_metric("rag_llm_tokens_generated_total", tokens_generated)
        self._increment_metric("rag_llm_generation_duration_seconds_sum", duration_seconds)
        self._increment_metric("rag_llm_generation_duration_seconds_count")

        # Tokens per second
        if duration_seconds > 0:
            tokens_per_sec = tokens_generated / duration_seconds
            self._set_metric("rag_llm_tokens_per_second", tokens_per_sec, "gauge",
                           "LLM token generation speed")

        # Context utilization
        if context_tokens_used is not None:
            self._set_metric("rag_llm_context_tokens_used", context_tokens_used, "gauge",
                           "Context tokens used")

        if context_tokens_available is not None:
            self._set_metric("rag_llm_context_tokens_available", context_tokens_available, "gauge",
                           "Context tokens available")

        if context_tokens_used is not None and context_tokens_available is not None and context_tokens_available > 0:
            utilization = (context_tokens_used / context_tokens_available) * 100
            self._set_metric("rag_llm_context_utilization_percent", utilization, "gauge",
                           "Context utilization")

    # ==================== Cache Metrics ====================

    def record_cache_hit(self):
        """Record cache hit"""
        self._increment_metric("rag_cache_requests_total")
        self._increment_metric("rag_cache_hits_total")
        self._update_cache_hit_rate()

    def record_cache_miss(self):
        """Record cache miss"""
        self._increment_metric("rag_cache_requests_total")
        self._increment_metric("rag_cache_misses_total")
        self._update_cache_hit_rate()

    def _update_cache_hit_rate(self):
        """Update cache hit rate"""
        hits = self._get_metric("rag_cache_hits_total")
        total = self._get_metric("rag_cache_requests_total")

        if total > 0:
            hit_rate = hits / total
            self._set_metric("rag_cache_hit_rate", hit_rate, "gauge", "Cache hit rate")

    def update_cache_stats(
        self,
        size_entries: int,
        size_bytes: int,
        similarity_threshold: Optional[float] = None
    ):
        """
        Update cache statistics

        Args:
            size_entries: Number of entries in cache
            size_bytes: Total size of cache in bytes
            similarity_threshold: Semantic similarity threshold for cache hits
        """
        self._set_metric("rag_cache_size_entries", size_entries, "gauge", "Cache size in entries")
        self._set_metric("rag_cache_size_bytes", size_bytes, "gauge", "Cache size in bytes")

        if similarity_threshold is not None:
            self._set_metric("rag_cache_semantic_similarity_threshold", similarity_threshold, "gauge",
                           "Semantic cache threshold")

    def record_cache_eviction(self):
        """Record cache eviction"""
        self._increment_metric("rag_cache_evictions_total")

    # ==================== Indexing Metrics ====================

    def record_indexing(
        self,
        num_documents: int,
        num_chunks: int,
        duration_seconds: float
    ):
        """Record indexing operation"""
        self._increment_metric("rag_documents_indexed_total", num_documents)
        self._increment_metric("rag_chunks_created_total", num_chunks)
        self._set_metric(
            "rag_index_duration_seconds",
            duration_seconds,
            "gauge",
            "Last indexing duration"
        )

        # Throughput
        if duration_seconds > 0:
            docs_per_sec = num_documents / duration_seconds
            self._set_metric("rag_documents_indexed_per_second", docs_per_sec, "gauge",
                           "Indexing throughput")

        # Chunks per document
        if num_documents > 0:
            chunks_per_doc = num_chunks / num_documents
            self._set_metric("rag_chunks_per_document", chunks_per_doc, "gauge",
                           "Average chunks per document")

    def record_chunking(
        self,
        duration_seconds: float,
        chunk_sizes: List[int],
        overlap_bytes: int
    ):
        """
        Record chunking operation

        Args:
            duration_seconds: Time taken for chunking
            chunk_sizes: List of chunk sizes in bytes
            overlap_bytes: Overlap size in bytes
        """
        self._set_metric("rag_chunking_duration_seconds", duration_seconds, "gauge",
                       "Last chunking duration")
        self._set_metric("rag_chunk_overlap_bytes", overlap_bytes, "gauge", "Chunk overlap size")

        # Update chunk size buffer
        self._chunk_size_buffer.extend(chunk_sizes)
        if len(self._chunk_size_buffer) > self._buffer_max_size:
            self._chunk_size_buffer = self._chunk_size_buffer[-self._buffer_max_size:]

        # Calculate percentiles
        if len(self._chunk_size_buffer) >= 10:
            p50 = np.percentile(self._chunk_size_buffer, 50)
            p95 = np.percentile(self._chunk_size_buffer, 95)
            p99 = np.percentile(self._chunk_size_buffer, 99)
            self._set_metric("rag_chunk_size_bytes_p50", p50, "gauge", "Chunk size p50")
            self._set_metric("rag_chunk_size_bytes_p95", p95, "gauge", "Chunk size p95")
            self._set_metric("rag_chunk_size_bytes_p99", p99, "gauge", "Chunk size p99")

    # ==================== Database Metrics ====================

    def record_db_operation(self):
        """Record database operation"""
        self._increment_metric("rag_db_operations_total")

    def record_db_error(self):
        """Record database error"""
        self._increment_metric("rag_db_errors_total")

    def update_db_row_count(self, count: int):
        """Update total row count"""
        self._set_metric(
            "rag_db_rows_total",
            count,
            "gauge",
            "Total rows in vector store"
        )

    # ==================== Resource Metrics ====================

    def update_resource_metrics(self):
        """
        Update process resource metrics (CPU, memory, threads)
        Only works if psutil is available
        """
        if not PSUTIL_AVAILABLE:
            return

        try:
            process = psutil.Process()

            # CPU
            cpu_percent = process.cpu_percent(interval=0.1)
            self._set_metric("rag_process_cpu_percent", cpu_percent, "gauge", "Process CPU usage")

            # Memory
            mem_info = process.memory_info()
            self._set_metric("rag_process_memory_rss_bytes", mem_info.rss, "gauge", "Process memory RSS")
            self._set_metric("rag_process_memory_vms_bytes", mem_info.vms, "gauge", "Process memory VMS")

            # Threads
            num_threads = process.num_threads()
            self._set_metric("rag_process_threads_active", num_threads, "gauge", "Active threads")

            # Python heap (approximate)
            import sys
            heap_size = sys.getsizeof(gc.get_objects())
            self._set_metric("rag_memory_python_heap_bytes", heap_size, "gauge", "Python heap memory")

        except Exception as e:
            pass  # Silently fail if resource metrics unavailable

    def update_model_memory(self, embedding_model_bytes: int = 0, llm_model_bytes: int = 0):
        """
        Update model memory usage

        Args:
            embedding_model_bytes: Memory used by embedding model
            llm_model_bytes: Memory used by LLM model
        """
        if embedding_model_bytes > 0:
            self._set_metric("rag_memory_embedding_model_bytes", embedding_model_bytes, "gauge",
                           "Embedding model memory")
        if llm_model_bytes > 0:
            self._set_metric("rag_memory_llm_model_bytes", llm_model_bytes, "gauge",
                           "LLM model memory")

    def update_gc_metrics(self):
        """Update Python garbage collection metrics"""
        try:
            # GC counts by generation
            gc_counts = gc.get_count()
            for gen in range(3):
                if gen < len(gc_counts):
                    with self._lock:
                        for name, metric in self._metrics.items():
                            if "gc_collections_total" in name and metric.labels.get("generation") == str(gen):
                                metric.value = gc_counts[gen]
                                metric.timestamp = time.time()

            # GC stats (if available)
            stats = gc.get_stats()
            if stats:
                # Last collection duration (approximate)
                latest_stat = stats[-1]
                collected = latest_stat.get('collected', 0)
                if collected > 0:
                    # Approximate duration based on collected objects
                    duration = collected / 1000000.0  # Very rough approximation
                    self._set_metric("rag_process_gc_duration_seconds", duration, "gauge",
                                   "Last GC duration")
        except Exception:
            pass  # Silently fail

    # ==================== Quality Metrics ====================

    def record_query_info(self, query_text: str, num_tokens: Optional[int] = None):
        """
        Record query information

        Args:
            query_text: Query text
            num_tokens: Number of tokens in query (optional)
        """
        query_length = len(query_text)
        self._set_metric("rag_query_length_chars", query_length, "gauge", "Query length in characters")

        if num_tokens is not None:
            self._set_metric("rag_query_tokens", num_tokens, "gauge", "Query length in tokens")

    def record_answer_info(
        self,
        answer_text: str,
        sources_cited: int,
        satisfaction_score: Optional[float] = None
    ):
        """
        Record answer information

        Args:
            answer_text: Generated answer
            sources_cited: Number of sources cited
            satisfaction_score: User satisfaction (0-1 or 1-5 scale)
        """
        answer_length = len(answer_text)
        self._set_metric("rag_answer_length_chars", answer_length, "gauge", "Answer length in characters")
        self._set_metric("rag_answer_sources_cited", sources_cited, "gauge", "Sources cited in answer")

        if satisfaction_score is not None:
            self._set_metric("rag_answer_satisfaction_score", satisfaction_score, "gauge",
                           "User satisfaction score")

    def update_throughput_metrics(self, window_seconds: float = 60.0):
        """
        Calculate and update throughput metrics

        Args:
            window_seconds: Time window for rate calculation
        """
        # Queries per second
        query_count = self._get_metric("rag_query_total")
        if query_count > 0 and window_seconds > 0:
            qps = query_count / window_seconds
            self._set_metric("rag_queries_per_second", qps, "gauge", "Query throughput")

    # ==================== Export ====================

    def export(self, filename: str = "rag_app.prom") -> Path:
        """
        Export metrics to Prometheus text format

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_path = self.export_dir / filename

        with self._lock:
            lines = []

            # Group metrics by base name (remove bucket suffixes)
            grouped_metrics = {}
            for metric in self._metrics.values():
                base_name = metric.name.split("{")[0]
                if base_name not in grouped_metrics:
                    grouped_metrics[base_name] = []
                grouped_metrics[base_name].append(metric)

            # Export each group
            for base_name, metrics_list in grouped_metrics.items():
                # HELP and TYPE (from first metric in group)
                first_metric = metrics_list[0]
                lines.append(f"# HELP {base_name} {first_metric.help_text}")
                lines.append(f"# TYPE {base_name} {first_metric.metric_type}")

                # All metric values
                for metric in metrics_list:
                    if metric.labels:
                        label_str = ",".join([f'{k}="{v}"' for k, v in metric.labels.items()])
                        lines.append(f"{metric.name}{{{label_str}}} {metric.value}")
                    else:
                        lines.append(f"{metric.name} {metric.value}")

                lines.append("")  # Blank line between metrics

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            total_queries = self._get_metric("rag_query_total")
            successful_queries = self._get_metric("rag_query_success_total")
            failed_queries = self._get_metric("rag_query_errors_total")

            total_duration = self._get_metric("rag_query_duration_seconds_sum")
            query_count = self._get_metric("rag_query_duration_seconds_count")
            avg_duration = total_duration / query_count if query_count > 0 else 0

            summary = {
                "queries": {
                    "total": int(total_queries),
                    "successful": int(successful_queries),
                    "failed": int(failed_queries),
                    "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                    "avg_duration_seconds": avg_duration,
                    "queries_per_second": self._get_metric("rag_queries_per_second")
                },
                "retrieval": {
                    "total": int(self._get_metric("rag_retrieval_total")),
                    "avg_score": self._get_metric("rag_retrieval_score_avg"),
                    "score_p50": self._get_metric("rag_retrieval_score_p50"),
                    "score_p95": self._get_metric("rag_retrieval_score_p95"),
                    "score_p99": self._get_metric("rag_retrieval_score_p99"),
                    "top_score": self._get_metric("rag_retrieval_top_score"),
                    "total_documents": int(self._get_metric("rag_retrieval_documents_total"))
                },
                "llm": {
                    "tokens_generated": int(self._get_metric("rag_llm_tokens_generated_total")),
                    "tokens_per_second": self._get_metric("rag_llm_tokens_per_second"),
                    "context_utilization_percent": self._get_metric("rag_llm_context_utilization_percent")
                },
                "cache": {
                    "requests": int(self._get_metric("rag_cache_requests_total")),
                    "hits": int(self._get_metric("rag_cache_hits_total")),
                    "misses": int(self._get_metric("rag_cache_misses_total")),
                    "hit_rate": self._get_metric("rag_cache_hit_rate"),
                    "size_entries": int(self._get_metric("rag_cache_size_entries"))
                },
                "database": {
                    "total_rows": int(self._get_metric("rag_db_rows_total")),
                    "operations": int(self._get_metric("rag_db_operations_total")),
                    "errors": int(self._get_metric("rag_db_errors_total"))
                },
                "indexing": {
                    "documents_indexed": int(self._get_metric("rag_documents_indexed_total")),
                    "chunks_created": int(self._get_metric("rag_chunks_created_total")),
                    "chunks_per_document": self._get_metric("rag_chunks_per_document"),
                    "documents_per_second": self._get_metric("rag_documents_indexed_per_second")
                }
            }

            # Add resource metrics if available
            if PSUTIL_AVAILABLE:
                summary["resources"] = {
                    "cpu_percent": self._get_metric("rag_process_cpu_percent"),
                    "memory_rss_mb": self._get_metric("rag_process_memory_rss_bytes") / 1024 / 1024,
                    "threads_active": int(self._get_metric("rag_process_threads_active"))
                }

            return summary


# Global metrics instance
_metrics_instance = None


def get_metrics() -> RAGMetrics:
    """Get global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = RAGMetrics()
    return _metrics_instance


# CLI for testing
if __name__ == "__main__":
    import json

    metrics = RAGMetrics()

    # Simulate some operations
    with metrics.query_timer():
        time.sleep(0.1)
    metrics.record_query_success()
    metrics.record_retrieval(3, [0.85, 0.78, 0.92])

    with metrics.query_timer():
        time.sleep(0.2)
    metrics.record_query_success()
    metrics.record_retrieval(2, [0.91, 0.88])

    metrics.record_cache_hit()
    metrics.record_cache_miss()
    metrics.record_cache_hit()

    # Export
    output_path = metrics.export()
    print(f"Metrics exported to: {output_path}")

    # Print summary
    print("\nMetrics Summary:")
    print(json.dumps(metrics.get_summary(), indent=2))
