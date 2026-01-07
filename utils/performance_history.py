"""
Performance History Database

SQLite-based time-series storage for performance metrics tracking.
Supports multi-platform baselines, git metadata tracking, and trend analysis.

Usage:
    from utils.performance_history import PerformanceHistory

    history = PerformanceHistory()

    # Record a performance run
    history.record_run(
        metrics={"embedding_throughput": 67.0, "query_latency_no_vllm": 8.0},
        metadata={"platform": "M1_Mac_16GB", "git_commit": "abc123"}
    )

    # Get baseline for a metric
    baseline = history.get_baseline("M1_Mac_16GB", "embedding_throughput")

    # Get trend data
    trend = history.get_trend("M1_Mac_16GB", "query_latency_no_vllm", days=30)

    # Compare current metrics to baseline
    comparison = history.compare_to_baseline(current_metrics, "M1_Mac_16GB")

Environment Variables:
    PERFORMANCE_DB_PATH=./benchmarks/history/performance.db
    REGRESSION_THRESHOLD=0.20
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Tracked metrics
TRACKED_METRICS = [
    "embedding_throughput",
    "vector_search_latency",
    "query_latency_no_vllm",
    "query_latency_vllm",
    "db_insertion_throughput",
    "peak_memory_gb",
    "avg_memory_gb",
    "cache_hit_rate",
    "tokens_per_second",
    "avg_mrr",
    "avg_ndcg",
]


class PerformanceHistory:
    """
    SQLite database for storing performance metrics over time.

    Features:
    - Multi-platform support (M1 Mac, RTX 4090, GitHub Actions, RunPod)
    - Git metadata tracking (commit, branch)
    - Automatic baseline calculation
    - Trend analysis
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize performance history database.

        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = os.getenv(
                "PERFORMANCE_DB_PATH", "benchmarks/history/performance.db"
            )

        self.db_path = db_path

        # Create directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Create database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create performance_runs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                git_commit TEXT,
                git_branch TEXT,
                platform TEXT NOT NULL,
                python_version TEXT,
                run_type TEXT NOT NULL,

                -- Core Metrics
                embedding_throughput REAL,
                vector_search_latency REAL,
                query_latency_no_vllm REAL,
                query_latency_vllm REAL,
                db_insertion_throughput REAL,
                peak_memory_gb REAL,
                avg_memory_gb REAL,
                cache_hit_rate REAL,
                tokens_per_second REAL,
                avg_mrr REAL,
                avg_ndcg REAL,

                -- Metadata
                num_test_queries INTEGER,
                test_duration_seconds REAL,
                notes TEXT
            )
        """
        )

        # Create indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_platform_timestamp
            ON performance_runs(platform, timestamp)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_commit
            ON performance_runs(git_commit)
        """
        )

        conn.commit()
        conn.close()

        log.debug(f"Initialized performance database at {self.db_path}")

    def record_run(self, metrics: Dict, metadata: Dict) -> int:
        """
        Record a performance run with git metadata.

        Args:
            metrics: Dictionary of metric_name -> value
            metadata: Dictionary with platform, git_commit, git_branch, etc.

        Returns:
            Run ID

        Example:
            run_id = history.record_run(
                metrics={
                    "embedding_throughput": 67.0,
                    "query_latency_no_vllm": 8.0,
                    "cache_hit_rate": 0.42
                },
                metadata={
                    "platform": "M1_Mac_16GB",
                    "git_commit": "abc123",
                    "git_branch": "main",
                    "python_version": "3.11.9",
                    "run_type": "ci"
                }
            )
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build column names and values
        columns = ["timestamp", "platform", "run_type"]
        values = [
            datetime.now().isoformat(),
            metadata.get("platform", "unknown"),
            metadata.get("run_type", "manual"),
        ]

        # Add optional metadata
        for key in ["git_commit", "git_branch", "python_version"]:
            if key in metadata:
                columns.append(key)
                values.append(metadata[key])

        # Add metrics
        for metric_name, value in metrics.items():
            if metric_name in TRACKED_METRICS:
                columns.append(metric_name)
                values.append(value)

        # Add other metadata
        if "num_test_queries" in metadata:
            columns.append("num_test_queries")
            values.append(metadata["num_test_queries"])

        if "test_duration_seconds" in metadata:
            columns.append("test_duration_seconds")
            values.append(metadata["test_duration_seconds"])

        if "notes" in metadata:
            columns.append("notes")
            values.append(metadata["notes"])

        # Insert
        placeholders = ", ".join(["?" for _ in values])
        columns_str = ", ".join(columns)

        cursor.execute(
            f"INSERT INTO performance_runs ({columns_str}) VALUES ({placeholders})",
            values,
        )

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        log.info(f"Recorded performance run {run_id} for {metadata.get('platform')}")

        return run_id

    def get_recent_runs(self, platform: str, limit: int = 50) -> List[Dict]:
        """
        Get recent runs for a platform.

        Args:
            platform: Platform identifier (e.g., "M1_Mac_16GB")
            limit: Maximum number of runs to return

        Returns:
            List of run dictionaries (most recent first)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM performance_runs
            WHERE platform = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (platform, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        # Convert to dicts
        runs = [dict(row) for row in rows]

        return runs

    def get_baseline(
        self, platform: str, metric: str, min_runs: int = 5
    ) -> Optional[float]:
        """
        Get baseline value for a metric (median of last N runs).

        Args:
            platform: Platform identifier
            metric: Metric name
            min_runs: Minimum runs needed to calculate baseline

        Returns:
            Baseline value (median) or None if insufficient data
        """
        if metric not in TRACKED_METRICS:
            log.warning(f"Metric {metric} not in tracked metrics")
            return None

        recent_runs = self.get_recent_runs(platform, limit=min_runs)

        if len(recent_runs) < min_runs:
            log.debug(
                f"Insufficient runs for baseline: {len(recent_runs)} < {min_runs}"
            )
            return None

        # Extract values (skip None)
        values = [run[metric] for run in recent_runs if run[metric] is not None]

        if len(values) < min_runs:
            log.debug(f"Insufficient non-null values for {metric}")
            return None

        # Return median
        baseline = float(np.median(values))

        return baseline

    def get_trend(
        self, platform: str, metric: str, days: int = 30
    ) -> List[Tuple[datetime, float]]:
        """
        Get time-series data for trend charts.

        Args:
            platform: Platform identifier
            metric: Metric name
            days: Number of days to look back

        Returns:
            List of (timestamp, value) tuples
        """
        if metric not in TRACKED_METRICS:
            log.warning(f"Metric {metric} not in tracked metrics")
            return []

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT timestamp, {metric}
            FROM performance_runs
            WHERE platform = ? AND timestamp >= ? AND {metric} IS NOT NULL
            ORDER BY timestamp ASC
        """,
            (platform, cutoff_date),
        )

        rows = cursor.fetchall()
        conn.close()

        # Convert to (datetime, value) tuples
        trend = [(datetime.fromisoformat(row[0]), row[1]) for row in rows]

        return trend

    def compare_to_baseline(
        self, current_metrics: Dict, platform: str, threshold: float = None
    ) -> Dict:
        """
        Compare current metrics to baseline.

        Args:
            current_metrics: Dictionary of current metric values
            platform: Platform identifier
            threshold: Regression threshold (default from env or 0.20)

        Returns:
            {
                'regressions': [(metric, current, baseline, pct_change), ...],
                'improvements': [...],
                'stable': [...]
            }
        """
        if threshold is None:
            threshold = float(os.getenv("REGRESSION_THRESHOLD", "0.20"))

        regressions = []
        improvements = []
        stable = []

        for metric, current_value in current_metrics.items():
            if current_value is None:
                continue

            baseline = self.get_baseline(platform, metric)

            if baseline is None:
                log.debug(f"No baseline for {metric}, skipping comparison")
                continue

            # Calculate percentage change
            # For latency: lower is better
            # For throughput: higher is better
            if metric in [
                "vector_search_latency",
                "query_latency_no_vllm",
                "query_latency_vllm",
                "peak_memory_gb",
                "avg_memory_gb",
            ]:
                # Lower is better
                pct_change = (current_value - baseline) / baseline
                is_regression = pct_change > threshold
                is_improvement = pct_change < -0.05  # >5% improvement
            else:
                # Higher is better (throughput, hit rate, etc.)
                pct_change = (current_value - baseline) / baseline
                is_regression = pct_change < -threshold
                is_improvement = pct_change > 0.05  # >5% improvement

            if is_regression:
                regressions.append((metric, current_value, baseline, pct_change))
            elif is_improvement:
                improvements.append((metric, current_value, baseline, pct_change))
            else:
                stable.append((metric, current_value, baseline, pct_change))

        return {
            "regressions": regressions,
            "improvements": improvements,
            "stable": stable,
        }

    def get_available_platforms(self) -> List[str]:
        """
        Get list of platforms with recorded data.

        Returns:
            List of platform identifiers
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT platform
            FROM performance_runs
            ORDER BY platform
        """
        )

        platforms = [row[0] for row in cursor.fetchall()]
        conn.close()

        return platforms

    def get_run_by_id(self, run_id: int) -> Optional[Dict]:
        """
        Get a specific run by ID.

        Args:
            run_id: Run ID

        Returns:
            Run dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM performance_runs
            WHERE id = ?
        """,
            (run_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_stats(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with stats (total_runs, platforms, date_range)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total runs
        cursor.execute("SELECT COUNT(*) FROM performance_runs")
        total_runs = cursor.fetchone()[0]

        # Platforms
        cursor.execute("SELECT COUNT(DISTINCT platform) FROM performance_runs")
        num_platforms = cursor.fetchone()[0]

        # Date range
        cursor.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM performance_runs"
        )
        min_date, max_date = cursor.fetchone()

        conn.close()

        return {
            "total_runs": total_runs,
            "num_platforms": num_platforms,
            "first_run": min_date,
            "last_run": max_date,
            "db_path": self.db_path,
        }

    def export_to_json(self, platform: str, output_file: str):
        """
        Export platform data to JSON.

        Args:
            platform: Platform identifier
            output_file: Output file path
        """
        runs = self.get_recent_runs(platform, limit=10000)  # All runs

        with open(output_file, "w") as f:
            json.dump(runs, f, indent=2)

        log.info(f"Exported {len(runs)} runs to {output_file}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize
    history = PerformanceHistory()

    # Show stats
    stats = history.get_stats()
    print(f"Performance History Stats:")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Platforms: {stats['num_platforms']}")
    print(f"  Date range: {stats['first_run']} to {stats['last_run']}")

    # Get available platforms
    platforms = history.get_available_platforms()
    print(f"\nAvailable platforms: {platforms}")

    # Example: Record a run
    print("\nExample: Recording a test run...")
    run_id = history.record_run(
        metrics={
            "embedding_throughput": 67.0,
            "query_latency_no_vllm": 8.0,
            "cache_hit_rate": 0.42,
        },
        metadata={
            "platform": "M1_Mac_16GB",
            "git_commit": "test123",
            "git_branch": "main",
            "python_version": "3.11.9",
            "run_type": "manual",
        },
    )
    print(f"Recorded run ID: {run_id}")

    # Example: Get baseline
    if platforms:
        platform = platforms[0]
        baseline = history.get_baseline(platform, "embedding_throughput")
        print(f"\nBaseline for {platform} embedding_throughput: {baseline}")

        # Example: Get trend
        trend = history.get_trend(platform, "embedding_throughput", days=30)
        print(f"Trend data points: {len(trend)}")

        # Example: Compare
        comparison = history.compare_to_baseline(
            {"embedding_throughput": 65.0}, platform
        )
        print(f"\nComparison results:")
        print(f"  Regressions: {len(comparison['regressions'])}")
        print(f"  Improvements: {len(comparison['improvements'])}")
        print(f"  Stable: {len(comparison['stable'])}")
