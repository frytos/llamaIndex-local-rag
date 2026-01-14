# RAG Web Backend Management

**Last Updated**: January 2026

Backend infrastructure classes for the Streamlit RAG Web UI, providing performance tracking, configuration management, cache statistics, and conversation management.

## Overview

The `rag_web_backend.py` module provides four core management classes:

1. **PerformanceTracker** - Track and persist indexing/query performance metrics
2. **ConfigurationManager** - Save/load/export RAG configurations
3. **CacheManager** - Manage semantic cache with statistics
4. **ConversationManager** - Multi-turn dialogue support

## Installation

No additional dependencies required beyond the main project requirements.

```bash
# The backend module is part of the project
python rag_web_backend.py  # Run tests
```

## Quick Start

```python
from rag_web_backend import (
    PerformanceTracker,
    ConfigurationManager,
    CacheManager,
    ConversationManager,
)

# Initialize managers
tracker = PerformanceTracker()
config_mgr = ConfigurationManager()
cache_mgr = CacheManager()
conv_mgr = ConversationManager()

# Track performance
tracker.record_indexing(duration=120.5, chunks=1000, config={...})
tracker.record_query(duration=8.2, results_count=4, cache_hit=False, config={...})

# Manage configurations
config_mgr.save("production", {"chunk_size": 700, "top_k": 4})
loaded = config_mgr.load("production")

# Cache statistics
stats = cache_mgr.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")

# Conversation context
conv_mgr.add_turn("session_1", "user", "What is RAG?")
context = conv_mgr.get_context("session_1")
```

## PerformanceTracker

### Purpose

Track and persist performance metrics for indexing and query operations with thread-safe JSON-based storage.

### Features

- Thread-safe metric recording
- JSON-based persistence (survives restarts)
- Summary statistics and aggregations
- CSV export for analysis
- Time-windowed queries (last N days)

### Basic Usage

```python
tracker = PerformanceTracker()

# Record indexing operation
tracker.record_indexing(
    duration=120.5,          # seconds
    chunks=1000,             # number of chunks indexed
    config={
        "chunk_size": 700,
        "chunk_overlap": 150,
        "embed_model": "bge-small-en",
        "table_name": "my_index"
    }
)

# Record query operation
tracker.record_query(
    duration=8.2,            # seconds
    results_count=4,         # number of results returned
    cache_hit=False,         # whether query was served from cache
    config={
        "table_name": "my_index",
        "top_k": 4,
        "model": "mistral-7b"
    }
)
```

### Getting Statistics

```python
# All-time summary
summary = tracker.get_summary()

print(f"Indexing operations: {summary['indexing']['count']}")
print(f"Avg throughput: {summary['indexing']['avg_throughput']:.1f} chunks/s")
print(f"Query operations: {summary['queries']['count']}")
print(f"Avg query time: {summary['queries']['avg_duration']:.2f}s")
print(f"Cache hit rate: {summary['cache']['hit_rate']:.2%}")

# Last 7 days only
summary_7d = tracker.get_summary(days=7)
```

### Summary Structure

```python
{
    "indexing": {
        "count": 5,
        "total_chunks": 5000,
        "avg_duration": 115.2,
        "min_duration": 98.5,
        "max_duration": 142.3,
        "avg_throughput": 67.3,  # chunks/second
        "max_throughput": 89.1
    },
    "queries": {
        "count": 42,
        "avg_duration": 7.8,
        "min_duration": 0.5,     # cache hit
        "max_duration": 15.3,
        "cache_hits": 18,
        "cache_hit_rate": 0.4286  # 42.86%
    },
    "cache": {
        "hits": 18,
        "misses": 24,
        "total": 42,
        "hit_rate": 0.4286
    },
    "time_window_days": None  # or 7, 30, etc.
}
```

### Export to CSV

```python
tracker.export_to_csv("performance_report.csv")
```

CSV format:
```csv
timestamp,operation,duration_seconds,chunks,throughput,cache_hit,results_count,config
2026-01-08T10:30:00,indexing,120.5,1000,8.3,,,"{...}"
2026-01-08T10:32:15,query,8.2,,,False,4,"{...}"
```

### Storage Location

```
.cache/performance/
├── indexing_history.json
├── query_history.json
└── cache_stats.json
```

### Thread Safety

All methods are thread-safe using `threading.RLock()`. Safe to use in multi-threaded Streamlit environment.

## ConfigurationManager

### Purpose

Save, load, and export RAG configurations with support for named presets and .env format export.

### Features

- JSON-based configuration storage
- Named configuration presets
- Export to .env format
- Import from .env files
- Thread-safe operations
- Automatic metadata tracking (created_at, updated_at)

### Basic Usage

```python
config_mgr = ConfigurationManager()

# Save configuration
config_mgr.save("production", {
    "chunk_size": 700,
    "chunk_overlap": 150,
    "embed_model": "bge-small-en",
    "top_k": 4,
    "temperature": 0.1,
    "max_new_tokens": 256
})

# Load configuration
config = config_mgr.load("production")
print(config["chunk_size"])  # 700

# List all saved configs
configs = config_mgr.list_saved()
for cfg in configs:
    print(f"{cfg['name']}: {cfg['created_at']}")

# Delete configuration
config_mgr.delete("old_config")
```

### Export to .env Format

```python
config = config_mgr.load("production")
env_content = config_mgr.export_to_env(config)

print(env_content)
# Output:
# # RAG Configuration
# # Generated: 2026-01-08T10:30:00
#
# CHUNK_SIZE=700
# CHUNK_OVERLAP=150
# EMBED_MODEL=bge-small-en
# TOP_K=4
# TEMP=0.1
# MAX_NEW_TOKENS=256
```

### Import from .env Format

```python
env_content = """
CHUNK_SIZE=700
CHUNK_OVERLAP=150
EMBED_MODEL=bge-small-en
TOP_K=4
"""

config = config_mgr.import_from_env(env_content)
# Returns: {"chunk_size": 700, "chunk_overlap": 150, ...}

# Save imported config
config_mgr.save("imported_config", config)
```

### Configuration Mapping

The manager automatically maps between config keys and environment variable names:

| Config Key | Environment Variable |
|------------|---------------------|
| chunk_size | CHUNK_SIZE |
| chunk_overlap | CHUNK_OVERLAP |
| embed_model | EMBED_MODEL |
| embed_dim | EMBED_DIM |
| top_k | TOP_K |
| table_name | PGTABLE |
| reset_table | RESET_TABLE |
| context_window | CTX |
| max_new_tokens | MAX_NEW_TOKENS |
| temperature | TEMP |
| n_gpu_layers | N_GPU_LAYERS |
| n_batch | N_BATCH |

### Storage Location

```
.cache/configs/
├── production.json
├── development.json
└── testing.json
```

### Configuration File Format

```json
{
  "name": "production",
  "created_at": "2026-01-08T10:00:00",
  "updated_at": "2026-01-08T10:30:00",
  "config": {
    "chunk_size": 700,
    "chunk_overlap": 150,
    "embed_model": "bge-small-en",
    "top_k": 4
  }
}
```

## CacheManager

### Purpose

Manage semantic cache with statistics tracking, integrating with the existing `utils.query_cache` module.

### Features

- Track cache hits/misses
- Recent entry tracking
- Integration with SemanticQueryCache
- Thread-safe operations
- UI-friendly statistics

### Basic Usage

```python
cache_mgr = CacheManager()

# Record cache events
cache_mgr.record_hit()
cache_mgr.record_miss()

# Get statistics
stats = cache_mgr.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2f}%")
print(f"Total requests: {stats['total']}")

# Get recent cache entries
recent = cache_mgr.get_recent_entries(limit=10)
for entry in recent:
    print(f"Query: {entry['query'][:50]}...")
    print(f"Similarity: {entry['similarity']:.4f}")

# Clear cache
cache_mgr.clear()
```

### Statistics Structure

```python
{
    "hits": 18,
    "misses": 24,
    "total": 42,
    "hit_rate": 42.86,  # percentage (0-100)
    "entries_count": 100,
    "semantic_cache": {  # if available
        "enabled": True,
        "count": 256,
        "hits": 18,
        "misses": 24,
        "hit_rate": 0.4286,  # decimal (0-1)
        "threshold": 0.92,
        "max_size": 1000,
        "ttl": 86400,
        "size_mb": 12.5
    }
}
```

### Integration with SemanticQueryCache

The CacheManager automatically integrates with the existing `utils.query_cache.semantic_cache`:

```python
# In your query code
from utils.query_cache import semantic_cache
from rag_web_backend import CacheManager

cache_mgr = CacheManager()

# Check cache
cached = semantic_cache.get_semantic(query, query_embedding)

if cached:
    cache_mgr.record_hit()
    return cached
else:
    cache_mgr.record_miss()
    # Run full RAG pipeline
    result = run_rag_query(query)
    semantic_cache.set_semantic(query, query_embedding, result)
    return result
```

## ConversationManager

### Purpose

Multi-turn conversation management for RAG with session-based tracking and context window management.

### Features

- Session-based conversation tracking
- Configurable context window (last N turns)
- Thread-safe operations
- Automatic session cleanup
- Export conversation history
- Formatted context for LLM prompts

### Basic Usage

```python
conv_mgr = ConversationManager(max_turns=10)

# Add conversation turns
conv_mgr.add_turn("session_1", "user", "What is RAG?")
conv_mgr.add_turn(
    "session_1",
    "assistant",
    "RAG is Retrieval-Augmented Generation...",
    metadata={"sources": ["doc1.pdf"], "confidence": 0.95}
)

# Get conversation context
context = conv_mgr.get_context("session_1", last_n=5)

# Get formatted context for LLM prompt
formatted = conv_mgr.get_formatted_context("session_1", last_n=5)
print(formatted)
# Output:
# User: What is RAG?
# Assistant: RAG is Retrieval-Augmented Generation...
```

### Managing Sessions

```python
# List all active sessions
sessions = conv_mgr.list_sessions()
for session in sessions:
    print(f"{session['session_id']}: {session['turn_count']} turns")
    print(f"  Last activity: {session['last_activity']}")

# Clear specific session
conv_mgr.clear_conversation("session_1")

# Cleanup old sessions (older than 24 hours)
removed = conv_mgr.cleanup_old_sessions(hours=24)
print(f"Removed {removed} old sessions")
```

### Export Conversation

```python
# Export to JSON
conv_mgr.export_session("session_1", "conversation_export.json")
```

Export format:
```json
{
  "session_id": "session_1",
  "metadata": {
    "created_at": "2026-01-08T10:00:00",
    "turn_count": 5,
    "last_activity": "2026-01-08T10:30:00"
  },
  "conversation": [
    {
      "role": "user",
      "content": "What is RAG?",
      "timestamp": "2026-01-08T10:00:00",
      "metadata": {}
    },
    {
      "role": "assistant",
      "content": "RAG is...",
      "timestamp": "2026-01-08T10:00:15",
      "metadata": {
        "sources": ["doc1.pdf"],
        "confidence": 0.95
      }
    }
  ]
}
```

### Context Window Management

The manager automatically maintains a sliding window of the last N turns:

```python
# Max 10 turns kept in memory
conv_mgr = ConversationManager(max_turns=10)

# Add 15 turns
for i in range(15):
    conv_mgr.add_turn("session_1", "user", f"Question {i}")
    conv_mgr.add_turn("session_1", "assistant", f"Answer {i}")

# Only last 10 turns are kept
context = conv_mgr.get_context("session_1")
assert len(context) == 10
```

## Streamlit Integration

### Initialize in Session State

```python
# In your init_session_state() function
def init_session_state():
    """Initialize session state with backend managers."""
    defaults = {
        # Existing state...
        "db_host": os.environ.get("PGHOST", "localhost"),

        # Backend managers
        "performance_tracker": PerformanceTracker(),
        "config_manager": ConfigurationManager(),
        "cache_manager": CacheManager(),
        "conversation_manager": ConversationManager(max_turns=10),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

### Track Indexing Performance

```python
def run_indexing(...):
    """Indexing with performance tracking."""
    tracker = st.session_state.performance_tracker

    start_time = time.time()

    # Your indexing code here...
    docs = load_documents(path)
    chunks, doc_idxs = chunk_documents(docs)
    nodes = build_nodes(docs, chunks, doc_idxs)
    # ... embedding and storage

    duration = time.time() - start_time

    # Record performance
    tracker.record_indexing(
        duration=duration,
        chunks=len(nodes),
        config={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embed_model": embed_model_name,
            "table_name": table_name
        }
    )

    st.success(f"✓ Indexed {len(nodes)} chunks in {duration:.2f}s")
```

### Track Query Performance

```python
def run_query(...):
    """Query with performance tracking."""
    tracker = st.session_state.performance_tracker
    cache_mgr = st.session_state.cache_manager

    start_time = time.time()

    # Check cache
    cached = semantic_cache.get_semantic(query, query_embedding)

    if cached:
        cache_mgr.record_hit()
        duration = time.time() - start_time
        tracker.record_query(duration, 0, True, {...})
        return cached

    cache_mgr.record_miss()

    # Run RAG pipeline
    results = retriever.retrieve(query)
    response = query_engine.query(query)

    duration = time.time() - start_time

    # Record performance
    tracker.record_query(
        duration=duration,
        results_count=len(results),
        cache_hit=False,
        config={"table_name": table_name, "top_k": top_k}
    )

    return response
```

### Performance Dashboard

```python
def page_performance():
    """Performance dashboard page."""
    st.header("Performance Dashboard")

    tracker = st.session_state.performance_tracker
    cache_mgr = st.session_state.cache_manager

    # Time window selector
    window = st.selectbox("Time Window:", ["All Time", "Last 7 Days", "Last 30 Days"])
    days = None if window == "All Time" else int(window.split()[1])

    # Get summary
    summary = tracker.get_summary(days=days)

    # Display metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Indexing Operations", summary["indexing"].get("count", 0))
        if summary["indexing"]:
            st.metric(
                "Avg Throughput",
                f"{summary['indexing']['avg_throughput']:.1f} chunks/s"
            )

    with col2:
        st.metric("Query Operations", summary["queries"].get("count", 0))
        if summary["queries"]:
            st.metric(
                "Avg Query Time",
                f"{summary['queries']['avg_duration']:.2f}s"
            )

    with col3:
        cache_stats = cache_mgr.get_stats()
        st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
        st.metric("Cache Entries", cache_stats['entries_count'])

    # Export button
    if st.button("📊 Export to CSV"):
        tracker.export_to_csv("performance_export.csv")
        st.success("Exported to performance_export.csv")
```

## Example Application

See `examples/rag_web_backend_integration.py` for a complete Streamlit demo showing all integration patterns.

Run the demo:
```bash
streamlit run examples/rag_web_backend_integration.py
```

## Best Practices

### Performance Tracking

1. **Always record both operations**: Track both indexing and query operations for complete metrics
2. **Include relevant config**: Store configuration details to correlate performance with settings
3. **Regular exports**: Export to CSV periodically for long-term analysis
4. **Time windows**: Use time windows for dashboard displays to avoid overwhelming users

### Configuration Management

1. **Use descriptive names**: Name configs descriptively (e.g., "production_high_quality", "dev_fast")
2. **Version control**: Consider storing configs in git for team collaboration
3. **Export before changes**: Export current config before making changes
4. **Validate after load**: Validate loaded configs before applying them

### Cache Management

1. **Monitor hit rate**: Track cache hit rate to optimize similarity threshold
2. **Periodic cleanup**: Implement automatic cleanup of old cache entries
3. **Size monitoring**: Monitor cache size and adjust max_size if needed
4. **Log cache decisions**: Log cache hits/misses for debugging

### Conversation Management

1. **Session ID generation**: Use UUIDs for session IDs to avoid collisions
2. **Regular cleanup**: Clean up old sessions periodically (24-48 hours)
3. **Context window size**: Balance context window size with LLM token limits
4. **Metadata tracking**: Store relevant metadata (sources, confidence) with responses

## Thread Safety

All classes are thread-safe using `threading.RLock()`:

- Multiple concurrent reads are allowed
- Writes are serialized
- No risk of data corruption
- Safe for use in Streamlit's multi-threaded environment

## Storage and Performance

### Storage Locations

```
.cache/
├── performance/
│   ├── indexing_history.json    (~10KB per 100 operations)
│   ├── query_history.json       (~10KB per 100 operations)
│   └── cache_stats.json         (~1KB)
├── configs/
│   ├── production.json          (~2KB each)
│   └── development.json
└── semantic_queries/            (managed by SemanticQueryCache)
    └── *.json                   (~100KB each)
```

### Performance Characteristics

| Operation | Time | Memory |
|-----------|------|--------|
| record_indexing | ~1ms | ~200 bytes |
| record_query | ~1ms | ~200 bytes |
| get_summary | ~5-10ms | N/A |
| save config | ~2-5ms | ~2KB |
| load config | ~2-5ms | N/A |
| cache stats | <1ms | N/A |
| add_turn | <1ms | ~500 bytes |
| get_context | <1ms | N/A |

## Troubleshooting

### Issue: Performance data not persisting

**Cause**: Directory permissions or disk space
**Solution**: Check that `.cache/` directory is writable and has space

### Issue: Configuration not loading

**Cause**: JSON corruption or invalid filename
**Solution**: Check `.cache/configs/` for valid JSON files

### Issue: Cache stats mismatch

**Cause**: Multiple instances or manual cache clearing
**Solution**: Use single CacheManager instance per application

### Issue: Conversation context too large

**Cause**: max_turns too high for LLM context window
**Solution**: Reduce max_turns or use last_n parameter in get_context()

## Testing

Run built-in tests:
```bash
python rag_web_backend.py
```

Output:
```
======================================================================
Testing PerformanceTracker
======================================================================
✓ PerformanceTracker tests passed

======================================================================
Testing ConfigurationManager
======================================================================
✓ ConfigurationManager tests passed

======================================================================
Testing CacheManager
======================================================================
✓ CacheManager tests passed

======================================================================
Testing ConversationManager
======================================================================
✓ ConversationManager tests passed

======================================================================
✓ All backend management tests passed!
======================================================================
```

## Future Enhancements

Potential future additions:

1. **Database backend**: SQLite storage for better querying
2. **Metrics export**: Prometheus/Grafana integration
3. **A/B testing**: Track performance across different configurations
4. **Alerts**: Automated alerts for performance regressions
5. **Async operations**: Async versions for high-throughput scenarios

## Related Documentation

- `docs/START_HERE.md` - Project overview
- `docs/PERFORMANCE_QUICK_START.md` - Performance optimization
- `utils/query_cache.py` - Semantic cache implementation
- `utils/performance_history.py` - SQLite-based performance tracking
- `rag_web.py` - Main Streamlit web UI

## Support

For issues or questions:
1. Check this documentation
2. Review `examples/rag_web_backend_integration.py`
3. Run tests with `python rag_web_backend.py`
4. Check logs in application output
