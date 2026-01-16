#!/usr/bin/env python3
"""
Test Sentry Integration - Comprehensive Test

Tests all Sentry features:
- Error tracking
- Performance monitoring
- Logging
- Metrics
- Profiling
- Breadcrumbs
- Context

Run with: python test_sentry_integration.py
"""

from dotenv import load_dotenv
load_dotenv()

from utils.sentry_config import (
    init_sentry,
    capture_exception,
    add_breadcrumb,
    set_context,
    measure_performance,
    log_info,
    log_warning,
    log_error,
    increment_counter,
    set_gauge,
    record_distribution,
    record_set,
    start_profiler,
    stop_profiler,
    profile_block,
)
import time
import logging

# Set up Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def slow_function():
    """Simulate slow operation"""
    time.sleep(0.05)
    return "done"


def fast_function():
    """Simulate fast operation"""
    time.sleep(0.01)
    return "done"


def main():
    print("=" * 60)
    print("Sentry Integration Test".center(60))
    print("=" * 60)

    # Initialize Sentry
    if not init_sentry():
        print("\n✗ Sentry initialization failed")
        print("  Check SENTRY_DSN and ENABLE_SENTRY in .env")
        return

    print("\n1. Testing Context...")
    set_context("test", {
        "run_id": "test-123",
        "version": "1.0",
        "environment": "test"
    })
    print("   ✓ Context set")

    print("\n2. Testing Breadcrumbs...")
    add_breadcrumb("Test started", category="test", level="info")
    add_breadcrumb("Loading configuration", category="test", level="info")
    print("   ✓ Breadcrumbs added")

    print("\n3. Testing Direct Logging (sentry_sdk.logger)...")
    log_info("Test info log", test_id="123", status="running")
    log_warning("Test warning log", metric="high_latency", value=5000)
    log_error("Test error log", error_code="TEST001")
    print("   ✓ Direct logs sent")

    print("\n4. Testing Python Logging (automatic capture)...")
    logger.info("Info: Test operation started")
    logger.warning("Warning: High memory usage detected")
    logger.error("Error: Database connection timeout")
    print("   ✓ Python logs sent (automatically)")

    print("\n5. Testing Metrics...")
    # Counter
    increment_counter("test.query.count")
    increment_counter("test.error.count", tags={"error_type": "timeout"})
    increment_counter("test.documents.processed", value=100)
    print("   ✓ Counters incremented")

    # Gauge
    set_gauge("test.database.connections", 42)
    set_gauge("test.memory.usage_mb", 1024.5)
    set_gauge("test.queue.depth", 100, tags={"queue": "embeddings"})
    print("   ✓ Gauges set")

    # Distribution
    record_distribution("test.query.latency_ms", 234.5)
    record_distribution("test.chunk.size_chars", 700)
    record_distribution("test.embedding.time_ms", 150.3, tags={"model": "bge-small"})
    print("   ✓ Distributions recorded")

    # Set
    record_set("test.users.active", "user_123")
    record_set("test.queries.unique", "hash_abc")
    print("   ✓ Sets recorded")

    print("\n6. Testing Performance Measurement...")
    with measure_performance("test_operation", operation_type="embedding"):
        time.sleep(0.1)  # Simulate work
    print("   ✓ Performance measured")

    print("\n7. Testing Profiling (manual control)...")
    start_profiler()
    for i in range(5):
        slow_function()
        fast_function()
    stop_profiler()
    print("   ✓ Manual profiling complete")

    print("\n8. Testing Profiling (context manager)...")
    with profile_block():
        for i in range(5):
            slow_function()
    print("   ✓ Context profiling complete")

    print("\n9. Testing Error Capture (handled exception)...")
    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError as e:
        capture_exception(
            e,
            extra={
                "operation": "division",
                "numerator": 1,
                "denominator": 0,
            },
            tags={
                "test": "true",
                "error_type": "zero_division",
            }
        )
        print("   ✓ Handled exception captured")

    print("\n10. Testing Error Capture (unhandled exception simulation)...")
    print("    Skipping unhandled exception (would crash script)")
    print("    To test: remove try/except and run 1/0")

    # Give Sentry time to send all data
    print("\n" + "=" * 60)
    print("Waiting 3 seconds for Sentry to send all data...")
    time.sleep(3)

    print("\n" + "=" * 60)
    print("✓ Test Complete!".center(60))
    print("=" * 60)

    print("\nCheck your Sentry dashboard at: https://sentry.io/")
    print("\nYou should see:")
    print("  📋 Logs Tab:")
    print("     - 3 direct logs (info, warning, error)")
    print("     - 3 Python logs (info, warning, error)")
    print("")
    print("  📊 Metrics Tab:")
    print("     - 3 counters (query.count, error.count, documents.processed)")
    print("     - 3 gauges (database.connections, memory.usage_mb, queue.depth)")
    print("     - 3 distributions (query.latency_ms, chunk.size_chars, embedding.time_ms)")
    print("     - 2 sets (users.active, queries.unique)")
    print("")
    print("  🐛 Issues Tab:")
    print("     - 1 error (ZeroDivisionError)")
    print("     - Breadcrumbs showing test flow")
    print("     - Custom context (test data)")
    print("")
    print("  ⚡ Performance Tab:")
    print("     - 1 transaction (test_operation)")
    print("     - Profiling data (CPU/memory usage)")
    print("")


if __name__ == "__main__":
    main()
