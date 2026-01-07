#!/usr/bin/env python3
"""
Real-time RAG Monitoring (Terminal-based)

Simple text-based monitoring that auto-refreshes in your terminal.
No dependencies beyond the standard library + your utils modules.

Usage:
    python scripts/monitor_live.py

    # With custom refresh interval
    python scripts/monitor_live.py --interval 10
"""

import time
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.query_cache import semantic_cache, cache
from utils.conversation_memory import session_manager

# Optional: system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_bytes(bytes_val):
    """Format bytes to human-readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def get_status_emoji(value, thresholds):
    """Get status emoji based on value and thresholds"""
    if value >= thresholds['good']:
        return "✅"
    elif value >= thresholds['fair']:
        return "⚠️"
    else:
        return "❌"


def print_metrics(show_details=False):
    """Print current metrics"""
    clear_screen()

    # Header
    print("=" * 70)
    print(f"RAG PIPELINE LIVE MONITORING")
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ========================================================================
    # CACHE PERFORMANCE
    # ========================================================================
    print("\n📦 CACHE PERFORMANCE")
    print("-" * 70)

    cache_stats = semantic_cache.stats()
    total_requests = cache_stats['hits'] + cache_stats['misses']

    # Hit rate with status
    hit_rate = cache_stats['hit_rate']
    status = get_status_emoji(hit_rate, {'good': 0.3, 'fair': 0.1})

    print(f"  Hit Rate:       {hit_rate:>6.1%}  {status}")
    print(f"  Total Queries:  {total_requests:>6d}")
    print(f"  Cache Hits:     {cache_stats['hits']:>6d}  (🚀 ~{cache_stats['hits'] * 10:.0f}s saved)")
    print(f"  Cache Misses:   {cache_stats['misses']:>6d}")
    print(f"  Cache Size:     {cache_stats['count']:>6d} entries")
    print(f"  Disk Usage:     {cache_stats['size_mb']:>6.1f} MB")

    if cache_stats['hits'] > 0:
        speedup = total_requests / cache_stats['misses'] if cache_stats['misses'] > 0 else float('inf')
        print(f"  Effective Speedup: {speedup:>3.1f}x")

    # Embedding cache
    embed_stats = cache.stats()
    print(f"\n  Embedding Cache: {embed_stats['count']:>5d} cached ({embed_stats['size_mb']:.1f} MB)")

    # ========================================================================
    # CONVERSATION SESSIONS
    # ========================================================================
    print("\n💬 CONVERSATION SESSIONS")
    print("-" * 70)

    conv_stats = session_manager.stats()

    print(f"  Active:         {conv_stats['active_sessions']:>6d}")
    print(f"  Total Created:  {conv_stats['total_created']:>6d}")
    print(f"  Total Expired:  {conv_stats['total_expired']:>6d}")
    print(f"  Timeout:        {conv_stats['session_timeout']:>6d}s ({conv_stats['session_timeout']//60}min)")

    if show_details and conv_stats['active_sessions'] > 0:
        print(f"\n  Active Session Details:")
        active_sessions = session_manager.list_active_sessions()

        for i, session_id in enumerate(active_sessions[:5], 1):  # Show first 5
            memory = session_manager.get(session_id)
            if memory:
                mem_stats = memory.stats()
                idle_min = mem_stats['idle_seconds'] // 60
                print(f"    {i}. {session_id[:12]}... → {mem_stats['total_turns']} turns, "
                      f"{idle_min}min idle")

        if len(active_sessions) > 5:
            print(f"    ... and {len(active_sessions) - 5} more sessions")

    # ========================================================================
    # SYSTEM HEALTH
    # ========================================================================
    print("\n🖥️  SYSTEM HEALTH")
    print("-" * 70)

    # Overall status
    health_score = 0
    max_score = 0

    # Cache health (weight: 3)
    max_score += 3
    if hit_rate > 0.3:
        health_score += 3
        cache_health = "✅ GOOD"
    elif hit_rate > 0.1:
        health_score += 2
        cache_health = "⚠️  FAIR"
    else:
        health_score += 1
        cache_health = "❌ POOR"

    print(f"  Cache Performance:    {cache_health}")

    # Cache size health (weight: 2)
    max_score += 2
    if cache_stats['size_mb'] < 500:
        health_score += 2
        size_health = "✅ NORMAL"
    elif cache_stats['size_mb'] < 1000:
        health_score += 1
        size_health = "⚠️  GROWING"
    else:
        size_health = "❌ HIGH"

    print(f"  Cache Size:           {size_health} ({cache_stats['size_mb']:.1f} MB)")

    # Session health (weight: 2)
    max_score += 2
    if conv_stats['active_sessions'] < 50:
        health_score += 2
        session_health = "✅ NORMAL"
    elif conv_stats['active_sessions'] < 100:
        health_score += 1
        session_health = "⚠️  GROWING"
    else:
        session_health = "❌ HIGH"

    print(f"  Session Count:        {session_health} ({conv_stats['active_sessions']})")

    # System resources (if available)
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_available_gb = mem.available / (1024**3)

        max_score += 3
        if mem_percent < 80:
            health_score += 3
            mem_health = "✅ NORMAL"
        elif mem_percent < 90:
            health_score += 2
            mem_health = "⚠️  HIGH"
        else:
            health_score += 1
            mem_health = "❌ CRITICAL"

        print(f"  Memory Usage:         {mem_health} ({mem_percent:.1f}%, {mem_available_gb:.1f} GB free)")

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        print(f"  CPU Usage:            {cpu_percent:>5.1f}%")

    # Overall health score
    health_percentage = (health_score / max_score) * 100 if max_score > 0 else 0

    print("\n  Overall Health Score:")
    if health_percentage > 80:
        overall_status = "✅ EXCELLENT"
    elif health_percentage > 60:
        overall_status = "⚠️  GOOD"
    elif health_percentage > 40:
        overall_status = "⚠️  FAIR"
    else:
        overall_status = "❌ POOR"

    print(f"    {overall_status} ({health_percentage:.0f}%)")

    # ========================================================================
    # QUICK ACTIONS
    # ========================================================================
    print("\n💡 QUICK ACTIONS")
    print("-" * 70)

    actions = []

    if hit_rate < 0.2:
        actions.append("⚠️  Lower cache threshold: export SEMANTIC_CACHE_THRESHOLD=0.90")

    if cache_stats['size_mb'] > 1000:
        actions.append("⚠️  Reduce cache size: export SEMANTIC_CACHE_MAX_SIZE=500")

    if conv_stats['active_sessions'] > 80:
        actions.append("⚠️  Cleanup sessions: python -c \"from utils.conversation_memory import session_manager; session_manager.cleanup_expired()\"")

    if not actions:
        actions.append("✅ No actions needed - all systems healthy!")

    for action in actions:
        print(f"  {action}")

    # ========================================================================
    # FOOTER
    # ========================================================================
    print("\n" + "=" * 70)
    print("Press Ctrl+C to exit | Auto-refresh enabled")
    print("Use --help for options")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Real-time RAG monitoring")
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed information (session details, etc.)"
    )
    args = parser.parse_args()

    print(f"Starting live monitoring (refresh every {args.interval}s)...\n")
    time.sleep(1)

    try:
        while True:
            print_metrics(show_details=args.details)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        clear_screen()
        print("\n" + "=" * 70)
        print("Live Monitoring Stopped")
        print("=" * 70)

        # Show final summary
        cache_stats = semantic_cache.stats()
        print(f"\nFinal Summary:")
        print(f"  Total Queries Processed: {cache_stats['hits'] + cache_stats['misses']}")
        print(f"  Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Time Saved (estimated): ~{cache_stats['hits'] * 10:.0f} seconds")
        print()


if __name__ == "__main__":
    main()
