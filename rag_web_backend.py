#!/usr/bin/env python3
"""
Backend Management Classes for RAG Web UI

Provides backend infrastructure for the Streamlit web UI including:
- PerformanceTracker: Track and persist indexing/query performance metrics
- ConfigurationManager: Save/load/export RAG configurations
- CacheManager: Manage semantic cache with statistics
- ConversationManager: Multi-turn dialogue support

Usage:
    from rag_web_backend import (
        PerformanceTracker,
        ConfigurationManager,
        CacheManager,
        ConversationManager,
    )

    # Track performance
    tracker = PerformanceTracker()
    tracker.record_indexing(duration=120.5, chunks=1000, config={...})

    # Manage configurations
    config_mgr = ConfigurationManager()
    config_mgr.save("my_config", {...})

    # Cache statistics
    cache_mgr = CacheManager()
    stats = cache_mgr.get_stats()

    # Conversation context
    conv_mgr = ConversationManager()
    conv_mgr.add_turn("session_1", "user", "What is RAG?")
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


# =============================================================================
# PerformanceTracker
# =============================================================================


class PerformanceTracker:
    """
    Track and persist performance metrics for indexing and query operations.

    Features:
    - Thread-safe metric recording
    - JSON-based persistence
    - Summary statistics and aggregations
    - CSV export for analysis
    - Automatic time-windowed retention

    Example:
        tracker = PerformanceTracker()

        # Record indexing operation
        tracker.record_indexing(
            duration=120.5,
            chunks=1000,
            config={"chunk_size": 700, "model": "bge-small-en"}
        )

        # Record query operation
        tracker.record_query(
            duration=8.2,
            results_count=4,
            cache_hit=False,
            config={"top_k": 4, "model": "mistral-7b"}
        )

        # Get summary
        summary = tracker.get_summary()
        print(f"Avg query time: {summary['queries']['avg_duration']:.2f}s")

        # Export to CSV
        tracker.export_to_csv("performance_report.csv")
    """

    def __init__(self, storage_dir: str = ".cache/performance"):
        """
        Initialize performance tracker.

        Args:
            storage_dir: Directory to store performance data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe storage
        self._lock = threading.RLock()

        # In-memory history
        self.indexing_history: List[Dict[str, Any]] = []
        self.query_history: List[Dict[str, Any]] = []
        self.cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}

        # File paths
        self.indexing_file = self.storage_dir / "indexing_history.json"
        self.query_file = self.storage_dir / "query_history.json"
        self.cache_file = self.storage_dir / "cache_stats.json"

        # Load existing data
        self._load_from_disk()

        log.debug(f"PerformanceTracker initialized: {self.storage_dir}")

    def _load_from_disk(self):
        """Load historical data from disk."""
        try:
            if self.indexing_file.exists():
                with open(self.indexing_file) as f:
                    self.indexing_history = json.load(f)
                log.debug(f"Loaded {len(self.indexing_history)} indexing records")

            if self.query_file.exists():
                with open(self.query_file) as f:
                    self.query_history = json.load(f)
                log.debug(f"Loaded {len(self.query_history)} query records")

            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    self.cache_stats = json.load(f)
                log.debug(f"Loaded cache stats: {self.cache_stats}")

        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Error loading performance data: {e}")

    def _save_to_disk(self):
        """Persist data to disk (atomic writes)."""
        try:
            # Indexing history
            temp_file = self.indexing_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.indexing_history, f, indent=2)
            temp_file.rename(self.indexing_file)

            # Query history
            temp_file = self.query_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.query_history, f, indent=2)
            temp_file.rename(self.query_file)

            # Cache stats
            temp_file = self.cache_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.cache_stats, f, indent=2)
            temp_file.rename(self.cache_file)

        except (IOError, OSError) as e:
            log.error(f"Error saving performance data: {e}")

    def record_indexing(
        self, duration: float, chunks: int, config: Dict[str, Any]
    ) -> None:
        """
        Record an indexing operation.

        Args:
            duration: Duration in seconds
            chunks: Number of chunks indexed
            config: Configuration dict (chunk_size, model, etc.)
        """
        with self._lock:
            record = {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "chunks": chunks,
                "throughput": chunks / duration if duration > 0 else 0,
                "config": config,
            }

            self.indexing_history.append(record)
            self._save_to_disk()

            log.info(
                f"Recorded indexing: {chunks} chunks in {duration:.2f}s "
                f"({record['throughput']:.1f} chunks/s)"
            )

    def record_query(
        self,
        duration: float,
        results_count: int,
        cache_hit: bool,
        config: Dict[str, Any],
    ) -> None:
        """
        Record a query operation.

        Args:
            duration: Duration in seconds
            results_count: Number of results returned
            cache_hit: Whether query was served from cache
            config: Configuration dict (top_k, model, etc.)
        """
        with self._lock:
            record = {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "results_count": results_count,
                "cache_hit": cache_hit,
                "config": config,
            }

            self.query_history.append(record)

            # Update cache stats
            if cache_hit:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1

            self._save_to_disk()

            log.info(
                f"Recorded query: {results_count} results in {duration:.2f}s "
                f"(cache_hit={cache_hit})"
            )

    def get_summary(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics.

        Args:
            days: Limit to last N days (None = all time)

        Returns:
            Dictionary with summary statistics for indexing and queries
        """
        with self._lock:
            cutoff = None
            if days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            # Filter by time window
            indexing = self.indexing_history
            queries = self.query_history

            if cutoff:
                indexing = [r for r in indexing if r["timestamp"] >= cutoff]
                queries = [r for r in queries if r["timestamp"] >= cutoff]

            # Indexing stats
            indexing_stats = {}
            if indexing:
                durations = [r["duration_seconds"] for r in indexing]
                throughputs = [r["throughput"] for r in indexing]
                total_chunks = sum(r["chunks"] for r in indexing)

                indexing_stats = {
                    "count": len(indexing),
                    "total_chunks": total_chunks,
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "avg_throughput": sum(throughputs) / len(throughputs),
                    "max_throughput": max(throughputs),
                }

            # Query stats
            query_stats = {}
            if queries:
                durations = [r["duration_seconds"] for r in queries]
                cache_hits = sum(1 for r in queries if r["cache_hit"])

                query_stats = {
                    "count": len(queries),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "cache_hits": cache_hits,
                    "cache_hit_rate": cache_hits / len(queries) if queries else 0,
                }

            # Cache stats
            total_cache_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            cache_hit_rate = (
                self.cache_stats["hits"] / total_cache_requests
                if total_cache_requests > 0
                else 0
            )

            return {
                "indexing": indexing_stats,
                "queries": query_stats,
                "cache": {
                    "hits": self.cache_stats["hits"],
                    "misses": self.cache_stats["misses"],
                    "total": total_cache_requests,
                    "hit_rate": cache_hit_rate,
                },
                "time_window_days": days,
            }

    def export_to_csv(self, output_path: str = "performance_export.csv") -> None:
        """
        Export performance metrics to CSV.

        Args:
            output_path: Output CSV file path
        """
        with self._lock:
            # Combine indexing and query data
            records = []

            for record in self.indexing_history:
                records.append(
                    {
                        "timestamp": record["timestamp"],
                        "operation": "indexing",
                        "duration_seconds": record["duration_seconds"],
                        "chunks": record["chunks"],
                        "throughput": record["throughput"],
                        "cache_hit": None,
                        "results_count": None,
                        "config": json.dumps(record["config"]),
                    }
                )

            for record in self.query_history:
                records.append(
                    {
                        "timestamp": record["timestamp"],
                        "operation": "query",
                        "duration_seconds": record["duration_seconds"],
                        "chunks": None,
                        "throughput": None,
                        "cache_hit": record["cache_hit"],
                        "results_count": record["results_count"],
                        "config": json.dumps(record["config"]),
                    }
                )

            # Create DataFrame and export
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp")
            df.to_csv(output_path, index=False)

            log.info(f"Exported {len(records)} performance records to {output_path}")

    def clear(self) -> None:
        """Clear all performance history."""
        with self._lock:
            self.indexing_history.clear()
            self.query_history.clear()
            self.cache_stats = {"hits": 0, "misses": 0}
            self._save_to_disk()
            log.info("Cleared all performance history")


# =============================================================================
# ConfigurationManager
# =============================================================================


class ConfigurationManager:
    """
    Save/load/export RAG configurations.

    Features:
    - JSON-based configuration storage
    - Named configuration presets
    - Export to .env format
    - Import from .env files
    - Thread-safe operations

    Example:
        config_mgr = ConfigurationManager()

        # Save configuration
        config_mgr.save("production", {
            "chunk_size": 700,
            "chunk_overlap": 150,
            "embed_model": "bge-small-en",
            "top_k": 4
        })

        # Load configuration
        config = config_mgr.load("production")

        # List saved configs
        configs = config_mgr.list_saved()

        # Export as .env
        env_content = config_mgr.export_to_env(config)
    """

    def __init__(self, config_dir: str = ".cache/configs"):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        log.debug(f"ConfigurationManager initialized: {self.config_dir}")

    def save(self, name: str, config: Dict[str, Any]) -> None:
        """
        Save configuration to JSON file.

        Args:
            name: Configuration name (used as filename)
            config: Configuration dictionary

        Raises:
            ValueError: If name is empty or contains invalid characters
        """
        if not name or not name.strip():
            raise ValueError("Configuration name cannot be empty")

        # Sanitize filename
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
        safe_name = safe_name.strip()

        if not safe_name:
            raise ValueError(f"Invalid configuration name: {name}")

        config_file = self.config_dir / f"{safe_name}.json"

        with self._lock:
            try:
                # Add metadata
                full_config = {
                    "name": name,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "config": config,
                }

                # Check if updating existing config
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            existing = json.load(f)
                            full_config["created_at"] = existing.get(
                                "created_at", full_config["created_at"]
                            )
                    except (json.JSONDecodeError, IOError):
                        pass  # Use new created_at

                # Atomic write
                temp_file = config_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    json.dump(full_config, f, indent=2)
                temp_file.rename(config_file)

                log.info(f"Saved configuration: {name}")

            except (IOError, OSError) as e:
                log.error(f"Error saving configuration: {e}")
                raise

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from JSON file.

        Args:
            name: Configuration name

        Returns:
            Configuration dictionary or None if not found
        """
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
        safe_name = safe_name.strip()
        config_file = self.config_dir / f"{safe_name}.json"

        with self._lock:
            if not config_file.exists():
                log.warning(f"Configuration not found: {name}")
                return None

            try:
                with open(config_file) as f:
                    data = json.load(f)
                    log.info(f"Loaded configuration: {name}")
                    return data["config"]

            except (json.JSONDecodeError, IOError, KeyError) as e:
                log.error(f"Error loading configuration: {e}")
                return None

    def list_saved(self) -> List[Dict[str, Any]]:
        """
        List all saved configurations.

        Returns:
            List of configuration metadata dicts with keys:
            - name: Configuration name
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - file_size: File size in bytes
        """
        with self._lock:
            configs = []

            for config_file in sorted(self.config_dir.glob("*.json")):
                try:
                    with open(config_file) as f:
                        data = json.load(f)

                    configs.append(
                        {
                            "name": data.get("name", config_file.stem),
                            "created_at": data.get("created_at", "unknown"),
                            "updated_at": data.get("updated_at", "unknown"),
                            "file_size": config_file.stat().st_size,
                        }
                    )

                except (json.JSONDecodeError, IOError) as e:
                    log.warning(f"Error reading {config_file}: {e}")

            return configs

    def delete(self, name: str) -> bool:
        """
        Delete a configuration.

        Args:
            name: Configuration name

        Returns:
            True if deleted, False if not found
        """
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
        safe_name = safe_name.strip()
        config_file = self.config_dir / f"{safe_name}.json"

        with self._lock:
            if not config_file.exists():
                log.warning(f"Configuration not found: {name}")
                return False

            try:
                config_file.unlink()
                log.info(f"Deleted configuration: {name}")
                return True

            except OSError as e:
                log.error(f"Error deleting configuration: {e}")
                return False

    def export_to_env(self, config: Dict[str, Any]) -> str:
        """
        Export configuration as .env format.

        Args:
            config: Configuration dictionary

        Returns:
            String in .env format
        """
        lines = [
            "# RAG Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            "",
        ]

        # Map common config keys to environment variable names
        env_mapping = {
            "chunk_size": "CHUNK_SIZE",
            "chunk_overlap": "CHUNK_OVERLAP",
            "embed_model": "EMBED_MODEL",
            "embed_model_name": "EMBED_MODEL",
            "embed_dim": "EMBED_DIM",
            "top_k": "TOP_K",
            "pdf_path": "PDF_PATH",
            "table": "PGTABLE",
            "table_name": "PGTABLE",
            "reset_table": "RESET_TABLE",
            "context_window": "CTX",
            "max_new_tokens": "MAX_NEW_TOKENS",
            "temperature": "TEMP",
            "n_gpu_layers": "N_GPU_LAYERS",
            "n_batch": "N_BATCH",
        }

        for key, value in config.items():
            env_key = env_mapping.get(key, key.upper())
            lines.append(f"{env_key}={value}")

        return "\n".join(lines)

    def import_from_env(self, env_content: str) -> Dict[str, Any]:
        """
        Parse .env file content to configuration dict.

        Args:
            env_content: Content of .env file

        Returns:
            Configuration dictionary
        """
        config = {}

        # Reverse mapping
        reverse_mapping = {
            "CHUNK_SIZE": "chunk_size",
            "CHUNK_OVERLAP": "chunk_overlap",
            "EMBED_MODEL": "embed_model",
            "EMBED_DIM": "embed_dim",
            "TOP_K": "top_k",
            "PDF_PATH": "pdf_path",
            "PGTABLE": "table_name",
            "RESET_TABLE": "reset_table",
            "CTX": "context_window",
            "MAX_NEW_TOKENS": "max_new_tokens",
            "TEMP": "temperature",
            "N_GPU_LAYERS": "n_gpu_layers",
            "N_BATCH": "n_batch",
        }

        for line in env_content.split("\n"):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Map to config key
                config_key = reverse_mapping.get(key, key.lower())

                # Try to parse as int/float/bool
                try:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

                config[config_key] = value

        return config


# =============================================================================
# CacheManager
# =============================================================================


class CacheManager:
    """
    Manage semantic cache with statistics tracking.

    Wraps the existing SemanticQueryCache with additional UI-friendly
    statistics and management methods.

    Example:
        cache_mgr = CacheManager()

        # Record cache events
        cache_mgr.record_hit()
        cache_mgr.record_miss()

        # Get statistics
        stats = cache_mgr.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")

        # Get recent entries
        entries = cache_mgr.get_recent_entries(limit=10)
    """

    def __init__(self):
        """Initialize cache manager."""
        self._lock = threading.RLock()
        self.hits: int = 0
        self.misses: int = 0
        self.entries: List[Dict[str, Any]] = []

        # Try to import and use existing semantic cache
        self.semantic_cache = None
        try:
            from utils.query_cache import semantic_cache

            self.semantic_cache = semantic_cache
            log.debug("CacheManager using existing semantic_cache")
        except ImportError:
            log.warning("semantic_cache not available, using standalone mode")

    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1

    def add_entry(
        self, query: str, similarity: float, cached_query: str, timestamp: float
    ) -> None:
        """
        Add cache entry metadata for tracking.

        Args:
            query: Query that was served from cache
            similarity: Similarity score
            cached_query: Original cached query
            timestamp: Timestamp of cache hit
        """
        with self._lock:
            self.entries.append(
                {
                    "query": query,
                    "similarity": similarity,
                    "cached_query": cached_query,
                    "timestamp": timestamp,
                }
            )

            # Keep only recent entries (last 100)
            if len(self.entries) > 100:
                self.entries = self.entries[-100:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - total: Total cache requests
            - hit_rate: Hit rate percentage (0-100)
            - entries_count: Number of tracked entries
            - semantic_cache_stats: Stats from semantic_cache if available
        """
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (100 * self.hits / total) if total > 0 else 0

            stats = {
                "hits": self.hits,
                "misses": self.misses,
                "total": total,
                "hit_rate": hit_rate,
                "entries_count": len(self.entries),
            }

            # Add semantic cache stats if available
            if self.semantic_cache:
                try:
                    semantic_stats = self.semantic_cache.stats()
                    stats["semantic_cache"] = semantic_stats
                except Exception as e:
                    log.warning(f"Error getting semantic cache stats: {e}")

            return stats

    def get_recent_entries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent cache entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent cache entry dictionaries
        """
        with self._lock:
            return self.entries[-limit:]

    def clear(self) -> None:
        """Clear cache statistics and entries."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.entries.clear()

            # Clear semantic cache if available
            if self.semantic_cache:
                try:
                    self.semantic_cache.clear()
                    log.info("Cleared semantic cache")
                except Exception as e:
                    log.warning(f"Error clearing semantic cache: {e}")

            log.info("Cleared cache manager statistics")


# =============================================================================
# ConversationManager
# =============================================================================


class ConversationManager:
    """
    Multi-turn conversation management for RAG.

    Features:
    - Session-based conversation tracking
    - Configurable context window (last N turns)
    - Thread-safe operations
    - Automatic session cleanup
    - Export conversation history

    Example:
        conv_mgr = ConversationManager(max_turns=10)

        # Add conversation turns
        conv_mgr.add_turn("session_1", "user", "What is RAG?")
        conv_mgr.add_turn("session_1", "assistant", "RAG is...")

        # Get conversation context
        context = conv_mgr.get_context("session_1", last_n=5)

        # Clear conversation
        conv_mgr.clear_conversation("session_1")
    """

    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation manager.

        Args:
            max_turns: Maximum turns to keep per conversation
        """
        self.max_turns = max_turns
        self._lock = threading.RLock()

        # Session storage: session_id -> list of turns
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}

        # Track session metadata
        self.session_metadata: Dict[str, Dict[str, Any]] = {}

        log.debug(f"ConversationManager initialized: max_turns={max_turns}")

    def add_turn(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a conversation turn.

        Args:
            session_id: Session identifier
            role: Role (user, assistant, system)
            content: Message content
            metadata: Optional metadata (sources, confidence, etc.)
        """
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {role}. Must be user/assistant/system")

        with self._lock:
            # Initialize session if needed
            if session_id not in self.conversations:
                self.conversations[session_id] = []
                self.session_metadata[session_id] = {
                    "created_at": datetime.now().isoformat(),
                    "turn_count": 0,
                }

            # Add turn
            turn = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            self.conversations[session_id].append(turn)
            self.session_metadata[session_id]["turn_count"] += 1
            self.session_metadata[session_id]["last_activity"] = datetime.now().isoformat()

            # Trim to max_turns
            if len(self.conversations[session_id]) > self.max_turns:
                self.conversations[session_id] = self.conversations[session_id][
                    -self.max_turns :
                ]

            log.debug(
                f"Added turn to session {session_id}: {role} ({len(content)} chars)"
            )

    def get_context(
        self, session_id: str, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation context for a session.

        Args:
            session_id: Session identifier
            last_n: Number of most recent turns to return (None = all)

        Returns:
            List of conversation turns (oldest to newest)
        """
        with self._lock:
            if session_id not in self.conversations:
                log.debug(f"No conversation found for session: {session_id}")
                return []

            turns = self.conversations[session_id]

            if last_n is not None and last_n > 0:
                turns = turns[-last_n:]

            return turns

    def get_formatted_context(
        self, session_id: str, last_n: Optional[int] = None
    ) -> str:
        """
        Get conversation context formatted as a string.

        Args:
            session_id: Session identifier
            last_n: Number of most recent turns to return

        Returns:
            Formatted conversation string
        """
        turns = self.get_context(session_id, last_n)

        if not turns:
            return ""

        lines = []
        for turn in turns:
            role = turn["role"].capitalize()
            content = turn["content"]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session existed and was cleared, False otherwise
        """
        with self._lock:
            if session_id in self.conversations:
                del self.conversations[session_id]
                del self.session_metadata[session_id]
                log.info(f"Cleared conversation: {session_id}")
                return True

            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions with metadata.

        Returns:
            List of session info dictionaries
        """
        with self._lock:
            sessions = []

            for session_id, metadata in self.session_metadata.items():
                turn_count = len(self.conversations.get(session_id, []))

                sessions.append(
                    {
                        "session_id": session_id,
                        "turn_count": turn_count,
                        "created_at": metadata.get("created_at"),
                        "last_activity": metadata.get("last_activity"),
                    }
                )

            # Sort by last activity (most recent first)
            sessions.sort(key=lambda x: x["last_activity"], reverse=True)

            return sessions

    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """
        Remove sessions older than specified hours.

        Args:
            hours: Maximum age in hours

        Returns:
            Number of sessions removed
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            cutoff_iso = cutoff.isoformat()

            sessions_to_remove = []

            for session_id, metadata in self.session_metadata.items():
                last_activity = metadata.get("last_activity", metadata.get("created_at"))

                if last_activity < cutoff_iso:
                    sessions_to_remove.append(session_id)

            # Remove old sessions
            for session_id in sessions_to_remove:
                del self.conversations[session_id]
                del self.session_metadata[session_id]

            if sessions_to_remove:
                log.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

            return len(sessions_to_remove)

    def export_session(self, session_id: str, output_path: str) -> bool:
        """
        Export conversation to JSON file.

        Args:
            session_id: Session identifier
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if session_id not in self.conversations:
                log.warning(f"Session not found: {session_id}")
                return False

            try:
                data = {
                    "session_id": session_id,
                    "metadata": self.session_metadata[session_id],
                    "conversation": self.conversations[session_id],
                }

                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)

                log.info(f"Exported session {session_id} to {output_path}")
                return True

            except (IOError, OSError) as e:
                log.error(f"Error exporting session: {e}")
                return False


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("Testing PerformanceTracker")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = PerformanceTracker(storage_dir=tmpdir)

        # Record some operations
        tracker.record_indexing(
            duration=120.5,
            chunks=1000,
            config={"chunk_size": 700, "model": "bge-small-en"},
        )

        tracker.record_query(
            duration=8.2,
            results_count=4,
            cache_hit=False,
            config={"top_k": 4},
        )

        tracker.record_query(
            duration=0.5,
            results_count=4,
            cache_hit=True,
            config={"top_k": 4},
        )

        # Get summary
        summary = tracker.get_summary()
        print("\nSummary:")
        print(f"  Indexing ops: {summary['indexing'].get('count', 0)}")
        print(f"  Query ops: {summary['queries'].get('count', 0)}")
        print(
            f"  Cache hit rate: {summary['cache']['hit_rate']:.2%}"
        )

        print("✓ PerformanceTracker tests passed")

    print("\n" + "=" * 70)
    print("Testing ConfigurationManager")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_mgr = ConfigurationManager(config_dir=tmpdir)

        # Save configuration
        config_mgr.save(
            "test_config",
            {
                "chunk_size": 700,
                "chunk_overlap": 150,
                "embed_model": "bge-small-en",
                "top_k": 4,
            },
        )

        # Load configuration
        loaded_config = config_mgr.load("test_config")
        assert loaded_config is not None
        assert loaded_config["chunk_size"] == 700

        # List configs
        configs = config_mgr.list_saved()
        assert len(configs) == 1

        # Export to .env
        env_content = config_mgr.export_to_env(loaded_config)
        assert "CHUNK_SIZE=700" in env_content

        # Import from .env
        imported = config_mgr.import_from_env(env_content)
        assert imported["chunk_size"] == 700

        print("✓ ConfigurationManager tests passed")

    print("\n" + "=" * 70)
    print("Testing CacheManager")
    print("=" * 70)

    cache_mgr = CacheManager()

    # Record hits/misses
    cache_mgr.record_hit()
    cache_mgr.record_hit()
    cache_mgr.record_miss()

    # Get stats
    stats = cache_mgr.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert abs(stats["hit_rate"] - 66.67) < 0.1

    print(f"  Hit rate: {stats['hit_rate']:.2f}%")
    print("✓ CacheManager tests passed")

    print("\n" + "=" * 70)
    print("Testing ConversationManager")
    print("=" * 70)

    conv_mgr = ConversationManager(max_turns=5)

    # Add turns
    conv_mgr.add_turn("session_1", "user", "What is RAG?")
    conv_mgr.add_turn("session_1", "assistant", "RAG is Retrieval-Augmented Generation...")
    conv_mgr.add_turn("session_1", "user", "How does it work?")

    # Get context
    context = conv_mgr.get_context("session_1")
    assert len(context) == 3

    # Get formatted
    formatted = conv_mgr.get_formatted_context("session_1", last_n=2)
    assert "assistant" in formatted.lower()

    # List sessions
    sessions = conv_mgr.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["turn_count"] == 3

    # Clear
    conv_mgr.clear_conversation("session_1")
    assert len(conv_mgr.get_context("session_1")) == 0

    print("✓ ConversationManager tests passed")

    print("\n" + "=" * 70)
    print("✓ All backend management tests passed!")
    print("=" * 70)
