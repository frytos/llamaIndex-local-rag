"""
Unit tests for performance history database.

Tests:
- Database initialization
- Recording performance runs
- Retrieving recent runs
- Baseline calculation
- Trend analysis
- Regression detection
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from utils.performance_history import PerformanceHistory, TRACKED_METRICS


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_performance.db")
        yield db_path


@pytest.fixture
def history(temp_db):
    """Create a PerformanceHistory instance with temp database."""
    return PerformanceHistory(db_path=temp_db)


class TestPerformanceHistoryInit:
    """Test database initialization."""

    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database file."""
        history = PerformanceHistory(db_path=temp_db)
        assert os.path.exists(temp_db)

    def test_init_creates_directory(self):
        """Test that initialization creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "subdir", "performance.db")
            history = PerformanceHistory(db_path=db_path)
            assert os.path.exists(db_path)

    def test_init_idempotent(self, temp_db):
        """Test that multiple initializations don't fail."""
        history1 = PerformanceHistory(db_path=temp_db)
        history2 = PerformanceHistory(db_path=temp_db)
        # Should not raise


class TestRecordRun:
    """Test recording performance runs."""

    def test_record_basic_run(self, history):
        """Test recording a basic performance run."""
        run_id = history.record_run(
            metrics={"embedding_throughput": 67.0, "query_latency_no_vllm": 8.0},
            metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
        )

        assert run_id > 0

    def test_record_all_metrics(self, history):
        """Test recording all tracked metrics."""
        metrics = {
            "embedding_throughput": 67.0,
            "vector_search_latency": 0.011,
            "query_latency_no_vllm": 8.0,
            "query_latency_vllm": 1.5,
            "db_insertion_throughput": 1250.0,
            "peak_memory_gb": 12.5,
            "avg_memory_gb": 8.3,
            "cache_hit_rate": 0.42,
            "tokens_per_second": 15.2,
            "avg_mrr": 0.85,
            "avg_ndcg": 0.78,
        }

        run_id = history.record_run(
            metrics=metrics, metadata={"platform": "M1_Mac_16GB", "run_type": "ci"}
        )

        assert run_id > 0

        # Verify stored
        run = history.get_run_by_id(run_id)
        assert run is not None
        assert run["embedding_throughput"] == 67.0
        assert run["query_latency_no_vllm"] == 8.0

    def test_record_with_git_metadata(self, history):
        """Test recording with git metadata."""
        run_id = history.record_run(
            metrics={"embedding_throughput": 67.0},
            metadata={
                "platform": "M1_Mac_16GB",
                "run_type": "ci",
                "git_commit": "abc123",
                "git_branch": "main",
                "python_version": "3.11.9",
            },
        )

        run = history.get_run_by_id(run_id)
        assert run["git_commit"] == "abc123"
        assert run["git_branch"] == "main"
        assert run["python_version"] == "3.11.9"

    def test_record_with_notes(self, history):
        """Test recording with notes."""
        run_id = history.record_run(
            metrics={"embedding_throughput": 67.0},
            metadata={
                "platform": "M1_Mac_16GB",
                "run_type": "manual",
                "notes": "Test run after optimization",
            },
        )

        run = history.get_run_by_id(run_id)
        assert run["notes"] == "Test run after optimization"

    def test_record_ignores_unknown_metrics(self, history):
        """Test that unknown metrics are ignored."""
        run_id = history.record_run(
            metrics={
                "embedding_throughput": 67.0,
                "unknown_metric": 999.0,  # Should be ignored
            },
            metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
        )

        # Should not raise, unknown metric ignored
        assert run_id > 0


class TestGetRecentRuns:
    """Test retrieving recent runs."""

    def test_get_recent_runs_empty(self, history):
        """Test getting recent runs when database is empty."""
        runs = history.get_recent_runs("M1_Mac_16GB", limit=10)
        assert runs == []

    def test_get_recent_runs_single_platform(self, history):
        """Test getting runs for a single platform."""
        # Record 5 runs
        for i in range(5):
            history.record_run(
                metrics={"embedding_throughput": 67.0 + i},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        runs = history.get_recent_runs("M1_Mac_16GB", limit=10)
        assert len(runs) == 5

        # Should be ordered by timestamp DESC (most recent first)
        assert runs[0]["embedding_throughput"] == 71.0  # Last recorded

    def test_get_recent_runs_respects_limit(self, history):
        """Test that limit parameter works."""
        # Record 10 runs
        for i in range(10):
            history.record_run(
                metrics={"embedding_throughput": 67.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        runs = history.get_recent_runs("M1_Mac_16GB", limit=5)
        assert len(runs) == 5

    def test_get_recent_runs_filters_by_platform(self, history):
        """Test that runs are filtered by platform."""
        # Record runs for two platforms
        for i in range(3):
            history.record_run(
                metrics={"embedding_throughput": 67.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        for i in range(2):
            history.record_run(
                metrics={"embedding_throughput": 50.0},
                metadata={"platform": "GitHub_Actions_macOS", "run_type": "ci"},
            )

        m1_runs = history.get_recent_runs("M1_Mac_16GB", limit=10)
        ci_runs = history.get_recent_runs("GitHub_Actions_macOS", limit=10)

        assert len(m1_runs) == 3
        assert len(ci_runs) == 2


class TestGetBaseline:
    """Test baseline calculation."""

    def test_get_baseline_no_data(self, history):
        """Test getting baseline when no data exists."""
        baseline = history.get_baseline("M1_Mac_16GB", "embedding_throughput")
        assert baseline is None

    def test_get_baseline_insufficient_runs(self, history):
        """Test baseline when fewer than min_runs."""
        # Record only 2 runs (min_runs defaults to 5)
        for i in range(2):
            history.record_run(
                metrics={"embedding_throughput": 67.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        baseline = history.get_baseline("M1_Mac_16GB", "embedding_throughput")
        assert baseline is None

    def test_get_baseline_median_calculation(self, history):
        """Test that baseline is median of recent runs."""
        # Record 5 runs with different values
        values = [65.0, 67.0, 70.0, 68.0, 66.0]
        for val in values:
            history.record_run(
                metrics={"embedding_throughput": val},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        baseline = history.get_baseline("M1_Mac_16GB", "embedding_throughput")

        # Median of [65, 66, 67, 68, 70] = 67.0
        assert baseline == 67.0

    def test_get_baseline_with_min_runs(self, history):
        """Test baseline with custom min_runs."""
        # Record 3 runs
        for i in range(3):
            history.record_run(
                metrics={"embedding_throughput": 67.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        # Should fail with min_runs=5
        baseline = history.get_baseline("M1_Mac_16GB", "embedding_throughput", min_runs=5)
        assert baseline is None

        # Should succeed with min_runs=3
        baseline = history.get_baseline("M1_Mac_16GB", "embedding_throughput", min_runs=3)
        assert baseline == 67.0

    def test_get_baseline_ignores_none_values(self, history):
        """Test that None values are handled correctly in baseline calculation."""
        # Record 7 runs: first 2 without the metric, then 5 with it
        for i in range(2):
            history.record_run(
                metrics={"query_latency_no_vllm": 8.0},  # Different metric
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        for i in range(5):
            history.record_run(
                metrics={"embedding_throughput": 67.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        # get_recent_runs(limit=5) will return the last 5 runs (all with embedding_throughput)
        baseline = history.get_baseline("M1_Mac_16GB", "embedding_throughput", min_runs=5)
        assert baseline == 67.0

        # Also test that it skips runs without the metric when calculating
        # get_recent_runs(limit=7) will return all 7, but only 5 have the metric
        baseline_with_more_runs = history.get_baseline("M1_Mac_16GB", "embedding_throughput", min_runs=3)
        assert baseline_with_more_runs == 67.0  # Should still work with at least 3


class TestGetTrend:
    """Test trend analysis."""

    def test_get_trend_empty(self, history):
        """Test getting trend when no data exists."""
        trend = history.get_trend("M1_Mac_16GB", "embedding_throughput", days=30)
        assert trend == []

    def test_get_trend_returns_tuples(self, history):
        """Test that trend returns (datetime, value) tuples."""
        history.record_run(
            metrics={"embedding_throughput": 67.0},
            metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
        )

        trend = history.get_trend("M1_Mac_16GB", "embedding_throughput", days=30)

        assert len(trend) == 1
        timestamp, value = trend[0]
        assert isinstance(timestamp, datetime)
        assert value == 67.0

    def test_get_trend_filters_by_days(self, history):
        """Test that trend respects days parameter."""
        # Can't easily test time-based filtering without mocking timestamps
        # This is a placeholder for integration test
        pass

    def test_get_trend_ordered_by_time(self, history):
        """Test that trend is ordered chronologically."""
        # Record 3 runs
        for i in range(3):
            history.record_run(
                metrics={"embedding_throughput": 67.0 + i},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        trend = history.get_trend("M1_Mac_16GB", "embedding_throughput", days=30)

        assert len(trend) == 3
        # Should be ordered by timestamp ASC (earliest first)
        assert trend[0][1] == 67.0  # First recorded
        assert trend[-1][1] == 69.0  # Last recorded


class TestCompareToBaseline:
    """Test regression detection."""

    def test_compare_no_baseline(self, history):
        """Test comparison when no baseline exists."""
        comparison = history.compare_to_baseline(
            {"embedding_throughput": 67.0}, "M1_Mac_16GB"
        )

        # Should skip metrics without baselines
        assert comparison["regressions"] == []
        assert comparison["improvements"] == []
        assert comparison["stable"] == []

    def test_compare_regression_latency(self, history):
        """Test regression detection for latency metrics (lower is better)."""
        # Establish baseline: 8.0s
        for i in range(5):
            history.record_run(
                metrics={"query_latency_no_vllm": 8.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        # Test regression (>20% slower)
        comparison = history.compare_to_baseline(
            {"query_latency_no_vllm": 10.0},  # +25% = regression
            "M1_Mac_16GB",
            threshold=0.20,
        )

        assert len(comparison["regressions"]) == 1
        metric, current, baseline, pct = comparison["regressions"][0]
        assert metric == "query_latency_no_vllm"
        assert current == 10.0
        assert baseline == 8.0
        assert pct > 0.20

    def test_compare_improvement_latency(self, history):
        """Test improvement detection for latency metrics."""
        # Establish baseline: 8.0s
        for i in range(5):
            history.record_run(
                metrics={"query_latency_no_vllm": 8.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        # Test improvement (>5% faster)
        comparison = history.compare_to_baseline(
            {"query_latency_no_vllm": 7.0},  # -12.5% = improvement
            "M1_Mac_16GB",
        )

        assert len(comparison["improvements"]) == 1
        metric, current, baseline, pct = comparison["improvements"][0]
        assert metric == "query_latency_no_vllm"
        assert current == 7.0

    def test_compare_regression_throughput(self, history):
        """Test regression detection for throughput metrics (higher is better)."""
        # Establish baseline: 67.0
        for i in range(5):
            history.record_run(
                metrics={"embedding_throughput": 67.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        # Test regression (<-20% slower)
        comparison = history.compare_to_baseline(
            {"embedding_throughput": 50.0},  # -25.4% = regression
            "M1_Mac_16GB",
            threshold=0.20,
        )

        assert len(comparison["regressions"]) == 1
        metric, current, baseline, pct = comparison["regressions"][0]
        assert metric == "embedding_throughput"

    def test_compare_stable(self, history):
        """Test stable metrics (within threshold)."""
        # Establish baseline: 67.0
        for i in range(5):
            history.record_run(
                metrics={"embedding_throughput": 67.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        # Test stable (+3% - not regression and not improvement)
        # Improvement threshold is +5%, regression is -20%
        comparison = history.compare_to_baseline(
            {"embedding_throughput": 69.0},  # +3% = stable
            "M1_Mac_16GB",
            threshold=0.20,
        )

        assert len(comparison["stable"]) == 1
        assert len(comparison["regressions"]) == 0
        assert len(comparison["improvements"]) == 0

    def test_compare_multiple_metrics(self, history):
        """Test comparison with multiple metrics."""
        # Establish baselines
        for i in range(5):
            history.record_run(
                metrics={
                    "embedding_throughput": 67.0,
                    "query_latency_no_vllm": 8.0,
                    "cache_hit_rate": 0.40,
                },
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        # Mixed results
        comparison = history.compare_to_baseline(
            {
                "embedding_throughput": 50.0,  # Regression (-25%)
                "query_latency_no_vllm": 7.0,  # Improvement (-12.5%)
                "cache_hit_rate": 0.42,  # Stable (+5%)
            },
            "M1_Mac_16GB",
        )

        assert len(comparison["regressions"]) == 1
        assert len(comparison["improvements"]) == 1
        assert len(comparison["stable"]) == 1


class TestMiscellaneous:
    """Test miscellaneous methods."""

    def test_get_available_platforms(self, history):
        """Test getting list of platforms."""
        # Record runs for multiple platforms
        history.record_run(
            metrics={"embedding_throughput": 67.0},
            metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
        )

        history.record_run(
            metrics={"embedding_throughput": 50.0},
            metadata={"platform": "GitHub_Actions_macOS", "run_type": "ci"},
        )

        platforms = history.get_available_platforms()
        assert "M1_Mac_16GB" in platforms
        assert "GitHub_Actions_macOS" in platforms

    def test_get_stats(self, history):
        """Test getting database statistics."""
        # Record some runs
        for i in range(3):
            history.record_run(
                metrics={"embedding_throughput": 67.0},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        stats = history.get_stats()

        assert stats["total_runs"] == 3
        assert stats["num_platforms"] == 1
        assert stats["first_run"] is not None
        assert stats["last_run"] is not None

    def test_export_to_json(self, history, temp_db):
        """Test exporting data to JSON."""
        # Record runs
        for i in range(3):
            history.record_run(
                metrics={"embedding_throughput": 67.0 + i},
                metadata={"platform": "M1_Mac_16GB", "run_type": "manual"},
            )

        # Export
        output_file = str(Path(temp_db).parent / "export.json")
        history.export_to_json("M1_Mac_16GB", output_file)

        # Verify export
        assert os.path.exists(output_file)

        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["embedding_throughput"] == 69.0  # Most recent first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
