"""
Sentry Configuration and Initialization

Centralizes Sentry setup for error tracking, performance monitoring,
and debugging across all entry points (CLI, web UI, scripts).

Usage:
    from utils.sentry_config import init_sentry

    init_sentry()  # Call once at application startup

Environment Variables:
    SENTRY_DSN: Sentry project DSN (required)
    SENTRY_ENVIRONMENT: Environment name (development/staging/production)
    SENTRY_TRACES_SAMPLE_RATE: Performance monitoring sample rate (0.0-1.0)
    SENTRY_PROFILES_SAMPLE_RATE: Profiling sample rate (0.0-1.0)
    SENTRY_ENABLE_LOGS: Enable sending logs to Sentry (0 or 1)
    ENABLE_SENTRY: Enable/disable Sentry (0 or 1)
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def get_git_release() -> Optional[str]:
    """Get current git commit hash for release tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short commit hash
    except Exception:
        pass
    return None


def get_git_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def before_send(event, hint):
    """
    Filter and modify events before sending to Sentry.

    Use this to:
    - Remove sensitive data (passwords, tokens, API keys)
    - Filter out noisy errors
    - Add custom context
    """
    # Filter out expected/noisy errors
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]

        # Ignore KeyboardInterrupt (user cancelled)
        if isinstance(exc_value, KeyboardInterrupt):
            return None

        # Ignore database connection errors in development
        env = os.getenv("SENTRY_ENVIRONMENT", "development")
        if env == "development":
            error_msg = str(exc_value).lower()
            if any(x in error_msg for x in ["connection refused", "could not connect"]):
                return None

    # Scrub sensitive data from request headers
    if "request" in event:
        headers = event["request"].get("headers", {})
        for key in list(headers.keys()):
            if key.lower() in ["authorization", "cookie", "x-api-key", "api-key"]:
                headers[key] = "[Filtered]"

    # Scrub sensitive environment variables
    if "contexts" in event and "runtime" in event["contexts"]:
        env_vars = event["contexts"]["runtime"].get("env", {})
        sensitive_keys = [
            "PGPASSWORD", "PGUSER", "SENTRY_DSN",
            "OPENAI_API_KEY", "HF_TOKEN", "API_KEY"
        ]
        for key in sensitive_keys:
            if key in env_vars:
                env_vars[key] = "[Filtered]"

    return event


def init_sentry() -> bool:
    """
    Initialize Sentry SDK with RAG-specific configuration.

    Returns:
        bool: True if Sentry was initialized, False if disabled/skipped
    """
    # Check if Sentry is enabled
    if os.getenv("ENABLE_SENTRY", "1") == "0":
        log.debug("Sentry is disabled (ENABLE_SENTRY=0)")
        return False

    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        log.warning("Sentry DSN not configured. Skipping Sentry initialization.")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.threading import ThreadingIntegration

        # Try to import optional integrations
        try:
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            fastapi_integration = FastApiIntegration()
        except ImportError:
            fastapi_integration = None

        # Environment configuration
        environment = os.getenv("SENTRY_ENVIRONMENT", "development")
        release = get_git_release()

        # Performance monitoring sample rates
        # traces_sample_rate: % of transactions to capture (0.0 = none, 1.0 = all)
        # profiles_sample_rate: % of sampled transactions to profile (CPU/memory)
        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
        profiles_sample_rate = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1"))

        # Enable logs (send Python logs to Sentry)
        enable_logs = os.getenv("SENTRY_ENABLE_LOGS", "1") == "1"

        # Adjust sample rates based on environment
        if environment == "production":
            # Lower sample rate in production to reduce overhead
            traces_sample_rate = min(traces_sample_rate, 0.2)
            profiles_sample_rate = min(profiles_sample_rate, 0.1)
        elif environment == "development":
            # Higher sample rate for local testing
            traces_sample_rate = 1.0
            profiles_sample_rate = 0.0  # Disable profiling in dev

        # Build integrations list
        # Configure logging integration based on enable_logs setting
        if enable_logs:
            logging_integration = LoggingIntegration(
                level=logging.INFO,        # Capture INFO and above
                event_level=logging.ERROR  # Send ERROR and above to Sentry
            )
        else:
            logging_integration = LoggingIntegration(
                level=logging.ERROR,       # Only capture ERROR and above
                event_level=logging.ERROR
            )

        integrations = [
            logging_integration,
            SqlalchemyIntegration(),
            ThreadingIntegration(propagate_hub=True),
        ]

        if fastapi_integration:
            integrations.append(fastapi_integration)

        # Initialize Sentry
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            traces_sample_rate=traces_sample_rate,
            profiles_sample_rate=profiles_sample_rate,
            send_default_pii=False,  # Don't send PII by default
            before_send=before_send,
            integrations=integrations,
            # Enable distributed tracing
            enable_tracing=True,
            # Set max breadcrumbs (default: 100)
            max_breadcrumbs=50,
            # Debug mode (useful for testing)
            debug=os.getenv("SENTRY_DEBUG", "0") == "1",
        )

        # Set global tags for filtering in Sentry UI
        sentry_sdk.set_tag("project", "local-rag")
        sentry_sdk.set_tag("git_branch", get_git_branch())

        # Set user context (non-PII)
        sentry_sdk.set_user({
            "id": os.getenv("USER", "unknown"),
            "environment": environment,
        })

        log.info(
            f"✓ Sentry initialized "
            f"(env={environment}, release={release}, traces={traces_sample_rate}, "
            f"profiling={profiles_sample_rate}, logs={enable_logs})"
        )
        return True

    except ImportError:
        log.error(
            "sentry-sdk not installed. "
            "Install with: pip install 'sentry-sdk[fastapi,sqlalchemy]'"
        )
        return False
    except Exception as e:
        log.error(f"Failed to initialize Sentry: {e}")
        return False


def capture_exception(error: Exception, **context):
    """
    Manually capture an exception with additional context.

    Args:
        error: Exception to capture
        **context: Additional context to attach (tags, extra data)

    Example:
        capture_exception(
            error,
            extra={"query": query, "table": table_name},
            tags={"operation": "query"}
        )
    """
    try:
        import sentry_sdk

        # Add context
        if "extra" in context:
            sentry_sdk.set_context("custom", context["extra"])

        if "tags" in context:
            for key, value in context["tags"].items():
                sentry_sdk.set_tag(key, value)

        sentry_sdk.capture_exception(error)
    except ImportError:
        log.debug("Sentry SDK not available, skipping error capture")
    except Exception as e:
        log.warning(f"Failed to capture exception in Sentry: {e}")


def add_breadcrumb(message: str, category: str = "custom", level: str = "info", **data):
    """
    Add a breadcrumb for debugging context.

    Breadcrumbs are logged events that help debug errors by showing
    what happened before the error occurred.

    Args:
        message: Breadcrumb message
        category: Category for filtering (e.g., "query", "database", "llm")
        level: Log level (debug/info/warning/error)
        **data: Additional structured data

    Example:
        add_breadcrumb(
            "Executing RAG query",
            category="query",
            level="info",
            query="What is RAG?",
            top_k=5
        )
    """
    try:
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data,
        )
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to add breadcrumb: {e}")


def set_context(name: str, data: dict):
    """
    Set custom context for Sentry events.

    Args:
        name: Context name
        data: Context data dictionary

    Example:
        set_context("rag_config", {
            "chunk_size": 700,
            "top_k": 5,
            "model": "mistral-7b"
        })
    """
    try:
        import sentry_sdk
        sentry_sdk.set_context(name, data)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to set context: {e}")


# Convenience function for measuring performance
class measure_performance:
    """
    Context manager for measuring operation performance.

    Usage:
        with measure_performance("query_embedding"):
            embeddings = embed_model.get_text_embedding(query)
    """

    def __init__(self, operation: str, **tags):
        self.operation = operation
        self.tags = tags

    def __enter__(self):
        try:
            import sentry_sdk
            self.transaction = sentry_sdk.start_transaction(
                op=self.operation,
                name=self.operation
            )
            for key, value in self.tags.items():
                self.transaction.set_tag(key, value)
            self.transaction.__enter__()
        except (ImportError, Exception):
            self.transaction = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.transaction:
            try:
                self.transaction.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass


# ============================================================================
# Logging Functions
# ============================================================================

def log_info(message: str, **extra):
    """
    Send an info log to Sentry.

    Args:
        message: Log message
        **extra: Additional context data

    Example:
        log_info("User query processed", query="What is RAG?", latency_ms=234)
    """
    try:
        import sentry_sdk
        if extra:
            sentry_sdk.set_context("log_extra", extra)
        sentry_sdk.logger.info(message)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to send info log: {e}")


def log_warning(message: str, **extra):
    """
    Send a warning log to Sentry.

    Args:
        message: Warning message
        **extra: Additional context data

    Example:
        log_warning("Slow query detected", duration_ms=5000, query="...")
    """
    try:
        import sentry_sdk
        if extra:
            sentry_sdk.set_context("log_extra", extra)
        sentry_sdk.logger.warning(message)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to send warning log: {e}")


def log_error(message: str, **extra):
    """
    Send an error log to Sentry.

    Args:
        message: Error message
        **extra: Additional context data

    Example:
        log_error("Database connection failed", host="localhost", port=5432)
    """
    try:
        import sentry_sdk
        if extra:
            sentry_sdk.set_context("log_extra", extra)
        sentry_sdk.logger.error(message)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to send error log: {e}")


# ============================================================================
# Metrics Functions
# ============================================================================

def increment_counter(metric_name: str, value: int = 1, tags: dict = None):
    """
    Increment a counter metric in Sentry.

    Counters track the number of times an event occurs.

    Args:
        metric_name: Name of the metric
        value: Amount to increment (default: 1)
        tags: Optional tags for filtering

    Example:
        increment_counter("rag.query.count")
        increment_counter("rag.error.count", tags={"error_type": "timeout"})
        increment_counter("documents.indexed", value=100)
    """
    try:
        import sentry_sdk
        from sentry_sdk import metrics
        metrics.incr(metric_name, value=value, tags=tags)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to increment counter: {e}")


def set_gauge(metric_name: str, value: float, tags: dict = None):
    """
    Set a gauge metric in Sentry.

    Gauges track the current value of something (like queue depth, memory usage).

    Args:
        metric_name: Name of the metric
        value: Current value
        tags: Optional tags for filtering

    Example:
        set_gauge("database.connections", 42)
        set_gauge("memory.usage_mb", 1024.5)
        set_gauge("queue.depth", 100, tags={"queue": "embeddings"})
    """
    try:
        import sentry_sdk
        from sentry_sdk import metrics
        metrics.gauge(metric_name, value=value, tags=tags)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to set gauge: {e}")


def record_distribution(metric_name: str, value: float, tags: dict = None):
    """
    Record a distribution metric in Sentry.

    Distributions track the statistical distribution of values
    (like response times, chunk sizes).

    Args:
        metric_name: Name of the metric
        value: Value to record
        tags: Optional tags for filtering

    Example:
        record_distribution("query.latency_ms", 234.5)
        record_distribution("chunk.size_chars", 700)
        record_distribution("embedding.time_ms", 150.3, tags={"model": "bge-small"})
    """
    try:
        import sentry_sdk
        from sentry_sdk import metrics
        metrics.distribution(metric_name, value=value, tags=tags)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to record distribution: {e}")


def record_set(metric_name: str, value: str, tags: dict = None):
    """
    Record a set metric in Sentry.

    Sets track the count of unique values (like unique users, unique queries).

    Args:
        metric_name: Name of the metric
        value: Unique value to track
        tags: Optional tags for filtering

    Example:
        record_set("users.active", user_id)
        record_set("queries.unique", query_hash)
    """
    try:
        import sentry_sdk
        from sentry_sdk import metrics
        metrics.set(metric_name, value=value, tags=tags)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to record set: {e}")


# ============================================================================
# Profiler Control Functions
# ============================================================================

def start_profiler():
    """
    Start continuous profiling.

    Profiles CPU and memory usage of your application code.
    Call stop_profiler() to stop profiling, or it will continue until
    the process exits.

    Example:
        from utils.sentry_config import start_profiler, stop_profiler

        start_profiler()

        # Your code to profile
        for i in range(100):
            process_document(doc)

        stop_profiler()
    """
    try:
        import sentry_sdk
        sentry_sdk.profiler.start_profiler()
        log.debug("Sentry profiler started")
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to start profiler: {e}")


def stop_profiler():
    """
    Stop continuous profiling.

    Optional - if not called, profiler will continue until process exits.

    Example:
        stop_profiler()
    """
    try:
        import sentry_sdk
        sentry_sdk.profiler.stop_profiler()
        log.debug("Sentry profiler stopped")
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Failed to stop profiler: {e}")


class profile_block:
    """
    Context manager for profiling a code block.

    Usage:
        with profile_block():
            # Code to profile
            for i in range(100):
                slow_function()
    """

    def __enter__(self):
        start_profiler()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_profiler()
