"""
metrics.py - Prometheus metrics instrumentation for RAG pipeline

Provides:
  - Query metrics (latency, success rate, error rate)
  - Retrieval metrics (scores, document count)
  - Embedding metrics (time, batch size)
  - Cache metrics (hit rate, miss rate)
  - File-based metrics export for Prometheus

Usage:
    from utils.metrics import RAGMetrics

    metrics = RAGMetrics()

    # Record query
    with metrics.query_timer():
        result = run_query(...)

    metrics.record_query_success()
    metrics.record_retrieval_score(0.85)

    # Export metrics
    metrics.export()
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime


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

        # Retrieval metrics
        self._set_metric("rag_retrieval_total", 0, "counter", "Total retrievals")
        self._set_metric("rag_retrieval_score_sum", 0, "counter", "Sum of retrieval scores")
        self._set_metric("rag_retrieval_score_count", 0, "counter", "Count of retrieval scores")
        self._set_metric("rag_retrieval_score_avg", 0, "gauge", "Average retrieval score")
        self._set_metric("rag_retrieval_documents_total", 0, "counter", "Total documents retrieved")

        # Embedding metrics
        self._set_metric("rag_embedding_duration_seconds_sum", 0, "counter", "Total embedding time")
        self._set_metric("rag_embedding_duration_seconds_count", 0, "counter", "Embedding count")
        self._set_metric("rag_embedding_errors_total", 0, "counter", "Embedding errors")

        # Cache metrics
        self._set_metric("rag_cache_requests_total", 0, "counter", "Total cache requests")
        self._set_metric("rag_cache_hits_total", 0, "counter", "Cache hits")
        self._set_metric("rag_cache_misses_total", 0, "counter", "Cache misses")
        self._set_metric("rag_cache_hit_rate", 0, "gauge", "Cache hit rate")

        # Indexing metrics
        self._set_metric("rag_documents_indexed_total", 0, "counter", "Total documents indexed")
        self._set_metric("rag_chunks_created_total", 0, "counter", "Total chunks created")
        self._set_metric("rag_index_duration_seconds", 0, "gauge", "Last indexing duration")

        # Database metrics
        self._set_metric("rag_db_rows_total", 0, "gauge", "Total rows in vector store")
        self._set_metric("rag_db_operations_total", 0, "counter", "Total database operations")
        self._set_metric("rag_db_errors_total", 0, "counter", "Database errors")

        # System metrics
        self._set_metric("rag_info", 1, "gauge", "RAG system info")

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

    # ==================== Retrieval Metrics ====================

    def record_retrieval(self, num_documents: int, scores: list):
        """Record retrieval operation"""
        self._increment_metric("rag_retrieval_total")
        self._increment_metric("rag_retrieval_documents_total", num_documents)

        if scores:
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

    def record_retrieval_score(self, score: float):
        """Record single retrieval score"""
        self.record_retrieval(1, [score])

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
        """Get metrics summary"""
        with self._lock:
            total_queries = self._get_metric("rag_query_total")
            successful_queries = self._get_metric("rag_query_success_total")
            failed_queries = self._get_metric("rag_query_errors_total")

            total_duration = self._get_metric("rag_query_duration_seconds_sum")
            query_count = self._get_metric("rag_query_duration_seconds_count")
            avg_duration = total_duration / query_count if query_count > 0 else 0

            return {
                "queries": {
                    "total": int(total_queries),
                    "successful": int(successful_queries),
                    "failed": int(failed_queries),
                    "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                    "avg_duration_seconds": avg_duration
                },
                "retrieval": {
                    "total": int(self._get_metric("rag_retrieval_total")),
                    "avg_score": self._get_metric("rag_retrieval_score_avg"),
                    "total_documents": int(self._get_metric("rag_retrieval_documents_total"))
                },
                "cache": {
                    "requests": int(self._get_metric("rag_cache_requests_total")),
                    "hits": int(self._get_metric("rag_cache_hits_total")),
                    "misses": int(self._get_metric("rag_cache_misses_total")),
                    "hit_rate": self._get_metric("rag_cache_hit_rate")
                },
                "database": {
                    "total_rows": int(self._get_metric("rag_db_rows_total")),
                    "operations": int(self._get_metric("rag_db_operations_total")),
                    "errors": int(self._get_metric("rag_db_errors_total"))
                }
            }


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
