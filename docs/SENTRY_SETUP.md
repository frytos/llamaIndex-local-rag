# Sentry Setup Guide

Complete guide for configuring Sentry error tracking and performance monitoring in your Local RAG Pipeline.

## Quick Start

### 1. Install Dependencies

```bash
pip install "sentry-sdk[fastapi,sqlalchemy]==2.20.0"
```

This is already in `requirements.txt`, so you can just run:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Add to your `.env` file (copy from `config/.env.example`):

```bash
# Enable Sentry
ENABLE_SENTRY=1

# Your Sentry DSN (from Sentry project settings)
SENTRY_DSN=https://dd5bfe2c25694280914ca28a090f8104@o4510190896414721.ingest.de.sentry.io/4510721807417424

# Environment (development/staging/production)
SENTRY_ENVIRONMENT=development

# Performance monitoring (0.0-1.0)
SENTRY_TRACES_SAMPLE_RATE=1.0  # 100% in development
SENTRY_PROFILES_SAMPLE_RATE=0.0  # Disabled in development

# Enable logs (send Python logs to Sentry)
SENTRY_ENABLE_LOGS=1  # Enabled

# Debug mode (only for troubleshooting)
SENTRY_DEBUG=0
```

### 3. Initialize Sentry in Your Code

Add this to the top of your entry point files (after imports):

```python
from utils.sentry_config import init_sentry

# Initialize Sentry early
init_sentry()
```

## Integration Examples

### CLI Scripts (`rag_interactive.py`, `rag_low_level_m1_16gb_verbose.py`)

Add after imports, before main logic:

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Sentry early (before any potential errors)
from utils.sentry_config import init_sentry
init_sentry()

# Rest of your code...
def main():
    # Your main logic
    pass

if __name__ == "__main__":
    main()
```

### Web UI (`rag_web.py` - Streamlit)

Add at the top of the file:

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Load .env file FIRST
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Initialize Sentry EARLY (before streamlit imports)
from utils.sentry_config import init_sentry
init_sentry()

# Now import Streamlit and other modules
import streamlit as st
# ... rest of imports
```

### FastAPI Services (if you use FastAPI)

The Sentry SDK automatically integrates with FastAPI when initialized:

```python
from fastapi import FastAPI
from utils.sentry_config import init_sentry

# Initialize Sentry BEFORE creating FastAPI app
init_sentry()

app = FastAPI()

# Your routes...
```

## Advanced Usage

### Manual Error Capture

Capture specific errors with additional context:

```python
from utils.sentry_config import capture_exception

try:
    result = complex_operation()
except Exception as e:
    capture_exception(
        e,
        extra={
            "query": user_query,
            "table": table_name,
            "chunk_size": chunk_size,
        },
        tags={
            "operation": "embedding",
            "model": "bge-small-en",
        }
    )
    raise  # Re-raise if you want the error to propagate
```

### Adding Breadcrumbs

Log events that help debug errors:

```python
from utils.sentry_config import add_breadcrumb

# Before each major operation
add_breadcrumb(
    "Loading documents",
    category="document_processing",
    level="info",
    path=pdf_path,
    file_count=len(files)
)

# Load documents...

add_breadcrumb(
    "Chunking documents",
    category="document_processing",
    chunk_size=700,
    chunk_overlap=150
)

# Chunk documents...
```

### Set Custom Context

Add structured data to all events:

```python
from utils.sentry_config import set_context

# Set RAG configuration context
set_context("rag_config", {
    "chunk_size": 700,
    "chunk_overlap": 150,
    "top_k": 5,
    "embed_model": "BAAI/bge-small-en",
    "embed_dim": 384,
})

# Set database context
set_context("database", {
    "host": os.getenv("PGHOST"),
    "port": os.getenv("PGPORT"),
    "database": os.getenv("DB_NAME"),
    "table": table_name,
})
```

### Measure Performance

Track timing of specific operations:

```python
from utils.sentry_config import measure_performance

# Measure embedding performance
with measure_performance("embedding", model="bge-small-en"):
    embeddings = embed_model.get_text_embedding_batch(texts)

# Measure query performance
with measure_performance("database_query", table=table_name):
    results = retriever.retrieve(query)

# Measure LLM inference
with measure_performance("llm_inference", model="mistral-7b"):
    response = query_engine.query(question)
```

### Send Logs to Sentry

Use the logging helper functions:

```python
from utils.sentry_config import log_info, log_warning, log_error

# Info logs (for tracking key events)
log_info("Document indexing started", doc_count=100, table="my_index")

# Warning logs (for issues that don't stop execution)
log_warning("Slow query detected", duration_ms=5000, query="...")

# Error logs (for errors you handle)
log_error("Database connection failed", host="localhost", port=5432)
```

Or use Python's built-in logging (automatically sent to Sentry):

```python
import logging

logger = logging.getLogger(__name__)

# These logs are automatically sent to Sentry
logger.info("Query processed successfully")
logger.warning("High memory usage detected")
logger.error("Failed to load model")
```

### Track Metrics

Send custom metrics to Sentry:

```python
from utils.sentry_config import (
    increment_counter,
    set_gauge,
    record_distribution,
    record_set
)

# Count events
increment_counter("rag.query.count")
increment_counter("rag.error.count", tags={"error_type": "timeout"})
increment_counter("documents.indexed", value=100)

# Track current values
set_gauge("database.connections", 42)
set_gauge("memory.usage_mb", 1024.5)
set_gauge("queue.depth", 100, tags={"queue": "embeddings"})

# Track distributions (for percentiles)
record_distribution("query.latency_ms", 234.5)
record_distribution("chunk.size_chars", 700)
record_distribution("embedding.time_ms", 150.3, tags={"model": "bge-small"})

# Track unique values
record_set("users.active", user_id)
record_set("queries.unique", query_hash)
```

### Profile Code Performance

Use continuous profiling to find performance bottlenecks:

```python
from utils.sentry_config import start_profiler, stop_profiler, profile_block

# Manual control
start_profiler()

for i in range(100):
    slow_function()
    fast_function()

stop_profiler()

# Or use context manager
with profile_block():
    # Code to profile
    process_large_dataset()
```

## Configuration Recommendations

### Development Environment

```bash
SENTRY_ENVIRONMENT=development
SENTRY_TRACES_SAMPLE_RATE=1.0   # Capture all transactions
SENTRY_PROFILES_SAMPLE_RATE=0.0  # Disable profiling (expensive)
SENTRY_ENABLE_LOGS=1             # Send logs to Sentry
SENTRY_DEBUG=0                   # No debug output
```

### Staging Environment

```bash
SENTRY_ENVIRONMENT=staging
SENTRY_TRACES_SAMPLE_RATE=0.2   # 20% sample
SENTRY_PROFILES_SAMPLE_RATE=0.1  # 10% profiling
SENTRY_ENABLE_LOGS=1             # Send logs
SENTRY_DEBUG=0
```

### Production Environment

```bash
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1   # 10% sample (lower overhead)
SENTRY_PROFILES_SAMPLE_RATE=0.05 # 5% profiling
SENTRY_ENABLE_LOGS=1             # Send logs
SENTRY_DEBUG=0
```

## What Gets Tracked

### Automatic Tracking

When Sentry is initialized, it automatically captures:

- **Unhandled Exceptions**: Any uncaught errors
- **Database Queries**: SQLAlchemy query performance (with SQLAlchemy integration)
- **API Requests**: FastAPI request/response timing (with FastAPI integration)
- **Logs**: INFO, WARNING, ERROR level logs from Python logging (when `SENTRY_ENABLE_LOGS=1`)
- **Threading**: Errors in background threads

### Manual Tracking

Use the helper functions for:

- **Custom Errors**: `capture_exception()` for handled errors you want to track
- **Breadcrumbs**: `add_breadcrumb()` for debugging context
- **Context**: `set_context()` for structured metadata
- **Performance**: `measure_performance()` for custom timing
- **Logs**: `log_info()`, `log_warning()`, `log_error()` for direct Sentry logging
- **Metrics**: `increment_counter()`, `set_gauge()`, `record_distribution()` for business metrics
- **Profiling**: `start_profiler()`, `stop_profiler()`, `profile_block()` for performance profiling

## Filtered Data

The `before_send` hook automatically filters sensitive data:

- **Headers**: Authorization, Cookie, API keys
- **Environment Variables**: Database passwords, API tokens, Sentry DSN
- **Errors**: KeyboardInterrupt (user cancelled)
- **Development**: Connection errors (in development only)

## Sentry Dashboard Features

### Issues Tab

View all captured exceptions:
- Error type and message
- Stack trace
- Breadcrumbs (events leading to error)
- Custom context and tags
- Affected users count
- First seen / last seen timestamps

### Performance Tab

View transaction performance:
- Average response times
- P50/P75/P95/P99 percentiles
- Slow transactions
- Database query performance
- LLM inference timing

### Releases Tab

Track errors by git commit:
- Errors introduced in each release
- Regression detection
- Deploy tracking

## Troubleshooting

### Sentry Not Capturing Errors

1. Check DSN is configured:
   ```bash
   echo $SENTRY_DSN
   ```

2. Check Sentry is enabled:
   ```bash
   echo $ENABLE_SENTRY  # Should be 1
   ```

3. Test with a manual error:
   ```python
   from utils.sentry_config import init_sentry
   import sentry_sdk

   init_sentry()
   sentry_sdk.capture_message("Test message from RAG pipeline")

   # Or trigger an error
   1 / 0
   ```

4. Enable debug mode:
   ```bash
   SENTRY_DEBUG=1 python your_script.py
   ```

### Performance Data Not Showing

1. Check `traces_sample_rate` is > 0:
   ```bash
   echo $SENTRY_TRACES_SAMPLE_RATE  # Should be 0.1 or higher
   ```

2. Verify performance monitoring is enabled in Sentry project settings

### Too Much Noise

1. Lower sample rates:
   ```bash
   SENTRY_TRACES_SAMPLE_RATE=0.05  # 5% instead of 10%
   ```

2. Add custom filtering in `before_send()` (edit `utils/sentry_config.py`)

3. Use Sentry's ignore rules in project settings

## Cost Optimization

Sentry pricing is based on:
- **Events**: Errors captured (very cheap)
- **Transactions**: Performance samples (more expensive)
- **Profiles**: CPU/memory profiles (most expensive)

To reduce costs:

1. **Lower sample rates in production**:
   ```bash
   SENTRY_TRACES_SAMPLE_RATE=0.05  # 5% vs 10%
   SENTRY_PROFILES_SAMPLE_RATE=0.0  # Disable profiling
   ```

2. **Use conditional sampling**: Sample 100% of errors, 10% of successes
   (requires custom code in `before_send`)

3. **Filter noisy endpoints**: Ignore health checks, static assets
   (add to `before_send` in `utils/sentry_config.py`)

## Security Best Practices

1. **Never commit `.env`**: DSN is semi-sensitive (can't send data without project access, but still keep private)

2. **Use environment-specific projects**: Separate Sentry projects for dev/staging/prod

3. **Enable IP filtering**: In Sentry project settings, only allow your server IPs

4. **Review PII settings**: Ensure `send_default_pii=False` (default in our config)

5. **Audit before_send hook**: Review what data is being filtered in `utils/sentry_config.py`

## Testing Your Setup

Run this test script to verify Sentry is working:

```python
#!/usr/bin/env python3
"""Test Sentry integration - comprehensive test of all features"""
from dotenv import load_dotenv
load_dotenv()

from utils.sentry_config import (
    init_sentry,
    capture_exception,
    add_breadcrumb,
    set_context,
    log_info,
    log_warning,
    log_error,
    increment_counter,
    set_gauge,
    record_distribution,
    start_profiler,
    stop_profiler,
)
import time

# Initialize
if init_sentry():
    print("✓ Sentry initialized successfully")

    # Test context
    set_context("test", {"run_id": "test-123", "version": "1.0"})
    print("✓ Context set")

    # Test breadcrumbs
    add_breadcrumb("Starting test", category="test", level="info")
    print("✓ Breadcrumb added")

    # Test logging
    log_info("Test info log", test_id="123")
    print("✓ Info log sent")

    log_warning("Test warning log", metric="high_latency")
    print("✓ Warning log sent")

    # Test metrics
    increment_counter("test.counter")
    print("✓ Counter incremented")

    set_gauge("test.gauge", 42.5)
    print("✓ Gauge set")

    record_distribution("test.latency_ms", 123.4)
    print("✓ Distribution recorded")

    # Test profiling
    start_profiler()
    time.sleep(0.1)  # Simulate work
    stop_profiler()
    print("✓ Profiler tested")

    # Test error capture
    try:
        raise ValueError("Test error from RAG pipeline")
    except Exception as e:
        capture_exception(e, tags={"test": "true"})
        print("✓ Exception captured")

    # Give Sentry time to send
    print("\nWaiting 3 seconds for Sentry to send data...")
    time.sleep(3)

    print("\n✓ Test complete! Check your Sentry dashboard:")
    print("  https://sentry.io/")
    print("\nYou should see:")
    print("  - 1 error (ValueError)")
    print("  - 3 log messages (info, warning)")
    print("  - 3 metrics (counter, gauge, distribution)")
    print("  - 1 breadcrumb")
    print("  - Custom context (test)")
else:
    print("✗ Sentry initialization failed")
    print("  Check SENTRY_DSN and ENABLE_SENTRY in .env")
```

Save as `test_sentry.py` and run:

```bash
python test_sentry.py
```

Check your Sentry dashboard at https://sentry.io/ to see the test error.

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` with your Sentry DSN
3. Add `init_sentry()` to your entry points
4. Run the test script to verify
5. Monitor your Sentry dashboard for errors and performance data

## Resources

- [Sentry Python SDK Docs](https://docs.sentry.io/platforms/python/)
- [Sentry Performance Monitoring](https://docs.sentry.io/product/performance/)
- [FastAPI Integration](https://docs.sentry.io/platforms/python/integrations/fastapi/)
- [SQLAlchemy Integration](https://docs.sentry.io/platforms/python/integrations/sqlalchemy/)
