#!/usr/bin/env python3
"""
HNSW Performance Validation Script

Validates query performance improvements from HNSW indices.
Can be run as part of CI/CD or monitoring to ensure optimal performance.

Usage:
    python scripts/validate_hnsw_performance.py [table_name]
    python scripts/validate_hnsw_performance.py --all
    python scripts/validate_hnsw_performance.py --check-thresholds
"""
import os
import sys
import time
import argparse
import psycopg2
from psycopg2 import sql
from typing import Dict, Optional
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Performance thresholds based on table size
PERFORMANCE_THRESHOLDS = {
    'small': {  # < 10K rows
        'max_latency_ms': 50,
        'expected_speedup': 2.0,
        'description': 'Small tables (< 10K rows)'
    },
    'medium': {  # 10K - 100K rows
        'max_latency_ms': 200,
        'expected_speedup': 10.0,
        'description': 'Medium tables (10K - 100K rows)'
    },
    'large': {  # > 100K rows
        'max_latency_ms': 500,
        'expected_speedup': 50.0,
        'description': 'Large tables (> 100K rows)'
    }
}


def connect_db():
    """Connect to PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        database=os.getenv("DB_NAME", "vector_db"),
    )


def get_table_size_category(row_count: int) -> str:
    """Categorize table by size."""
    if row_count < 10000:
        return 'small'
    elif row_count < 100000:
        return 'medium'
    else:
        return 'large'


def has_hnsw_index(conn, table_name: str) -> bool:
    """Check if table has HNSW index."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*)
            FROM pg_indexes
            WHERE tablename = %s
              AND indexdef LIKE '%%hnsw%%'
              AND indexdef LIKE '%%embedding%%';
        """, (table_name,))
        return cur.fetchone()[0] > 0


def get_index_stats(conn, table_name: str) -> Dict:
    """Get index statistics and table info."""
    stats = {}

    with conn.cursor() as cur:
        # Row count
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
        stats['row_count'] = cur.fetchone()[0]

        # Embedding dimensions
        cur.execute(sql.SQL("""
            SELECT vector_dims(embedding) FROM {} WHERE embedding IS NOT NULL LIMIT 1
        """).format(sql.Identifier(table_name)))
        result = cur.fetchone()
        stats['dimensions'] = result[0] if result else None

        # Check for HNSW index
        stats['has_hnsw'] = has_hnsw_index(conn, table_name)

        # Get index size if it exists
        if stats['has_hnsw']:
            cur.execute("""
                SELECT pg_size_pretty(pg_relation_size(indexname::regclass))
                FROM pg_indexes
                WHERE tablename = %s
                  AND indexdef LIKE '%%hnsw%%'
                  AND indexdef LIKE '%%embedding%%';
            """, (table_name,))
            result = cur.fetchone()
            stats['index_size'] = result[0] if result else 'Unknown'

    return stats


def benchmark_query_performance(conn, table_name: str, num_trials: int = 10) -> Dict:
    """
    Benchmark query performance with detailed metrics.

    Returns:
        Dict with latency statistics and query plan info
    """
    with conn.cursor() as cur:
        # Get embedding dimensions
        cur.execute(sql.SQL("""
            SELECT vector_dims(embedding) FROM {} WHERE embedding IS NOT NULL LIMIT 1
        """).format(sql.Identifier(table_name)))

        result = cur.fetchone()
        if not result:
            return {'error': 'No embeddings found'}

        embed_dim = result[0]
        query_vector = [0.1] * embed_dim

        # Warm-up query (populate cache)
        cur.execute(sql.SQL("""
            SELECT id FROM {} WHERE embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT 10
        """).format(sql.Identifier(table_name)), (query_vector,))
        cur.fetchall()

        # Benchmark trials
        latencies = []
        for _ in range(num_trials):
            start = time.time()

            cur.execute(sql.SQL("""
                SELECT id, 1 - (embedding <=> %s::vector) as similarity
                FROM {}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 10
            """).format(sql.Identifier(table_name)), (query_vector, query_vector))

            cur.fetchall()
            latency = time.time() - start
            latencies.append(latency)

        # Get query plan
        cur.execute(sql.SQL("""
            EXPLAIN (FORMAT JSON)
            SELECT id FROM {}
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 10
        """).format(sql.Identifier(table_name)), (query_vector,))

        plan = cur.fetchone()[0]

        # Check if index is being used
        plan_str = json.dumps(plan)
        uses_index = 'Index Scan' in plan_str and 'hnsw' in plan_str.lower()

        return {
            'avg_latency_ms': sum(latencies) / len(latencies) * 1000,
            'min_latency_ms': min(latencies) * 1000,
            'max_latency_ms': max(latencies) * 1000,
            'p50_latency_ms': sorted(latencies)[len(latencies)//2] * 1000,
            'p95_latency_ms': sorted(latencies)[int(len(latencies)*0.95)] * 1000,
            'trials': num_trials,
            'uses_index': uses_index,
            'query_plan': plan
        }


def validate_table(conn, table_name: str, verbose: bool = True) -> Dict:
    """
    Validate HNSW performance for a single table.

    Returns:
        Dict with validation results and status
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Validating: {table_name}")
        print(f"{'='*70}")

    # Get table stats
    stats = get_index_stats(conn, table_name)

    if verbose:
        print(f"  Rows: {stats['row_count']:,}")
        print(f"  Dimensions: {stats['dimensions']}")
        print(f"  HNSW Index: {'✅ Yes' if stats['has_hnsw'] else '❌ No'}")
        if stats['has_hnsw']:
            print(f"  Index Size: {stats.get('index_size', 'Unknown')}")

    # Determine size category and thresholds
    size_category = get_table_size_category(stats['row_count'])
    thresholds = PERFORMANCE_THRESHOLDS[size_category]

    if verbose:
        print(f"  Category: {size_category} ({thresholds['description']})")
        print(f"  Expected max latency: {thresholds['max_latency_ms']}ms")

    # Benchmark performance
    if verbose:
        print(f"\n  📊 Benchmarking query performance...")

    perf = benchmark_query_performance(conn, table_name)

    if 'error' in perf:
        return {
            'table': table_name,
            'status': 'error',
            'error': perf['error']
        }

    if verbose:
        print(f"    Avg latency: {perf['avg_latency_ms']:.1f}ms")
        print(f"    P50 latency: {perf['p50_latency_ms']:.1f}ms")
        print(f"    P95 latency: {perf['p95_latency_ms']:.1f}ms")
        print(f"    Uses index: {'✅ Yes' if perf['uses_index'] else '❌ No'}")

    # Validation checks
    checks = {
        'has_index': stats['has_hnsw'],
        'uses_index': perf['uses_index'],
        'meets_latency_threshold': perf['avg_latency_ms'] <= thresholds['max_latency_ms']
    }

    all_passed = all(checks.values())

    if verbose:
        print(f"\n  Validation:")
        print(f"    Index exists: {'✅' if checks['has_index'] else '❌'}")
        print(f"    Index used in queries: {'✅' if checks['uses_index'] else '❌'}")
        print(f"    Latency within threshold: {'✅' if checks['meets_latency_threshold'] else '❌'}")

        if not all_passed:
            print(f"\n  ⚠️  Validation issues detected:")
            if not checks['has_index']:
                print(f"    • No HNSW index - run: python migrate_add_hnsw_indices.py")
            if not checks['uses_index']:
                print(f"    • Index exists but not used - check query patterns")
            if not checks['meets_latency_threshold']:
                print(f"    • Latency {perf['avg_latency_ms']:.1f}ms exceeds threshold {thresholds['max_latency_ms']}ms")
                print(f"    • Consider tuning ef_search: SET hnsw.ef_search = 200;")

    return {
        'table': table_name,
        'status': 'pass' if all_passed else 'fail',
        'stats': stats,
        'performance': perf,
        'checks': checks,
        'thresholds': thresholds
    }


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate HNSW index performance')
    parser.add_argument('table', nargs='?', help='Table name to validate')
    parser.add_argument('--all', action='store_true', help='Validate all tables')
    parser.add_argument('--check-thresholds', action='store_true', help='Show performance thresholds')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()

    if args.check_thresholds:
        print("Performance Thresholds by Table Size:")
        print()
        for category, thresholds in PERFORMANCE_THRESHOLDS.items():
            print(f"{category.upper()}: {thresholds['description']}")
            print(f"  Max latency: {thresholds['max_latency_ms']}ms")
            print(f"  Expected speedup: {thresholds['expected_speedup']}x")
            print()
        return

    # Connect to database
    try:
        conn = connect_db()
        conn.autocommit = True
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

    # Get tables to validate
    if args.all:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public' AND tablename LIKE 'data_%'
                ORDER BY tablename
            """)
            tables = [row[0] for row in cur.fetchall()]
    elif args.table:
        tables = [args.table]
    else:
        parser.print_help()
        conn.close()
        return

    # Validate each table
    results = []
    for table in tables:
        result = validate_table(conn, table, verbose=not args.json)
        results.append(result)

    # Summary
    if not args.json:
        print(f"\n{'='*70}")
        print("Validation Summary")
        print(f"{'='*70}\n")

        passed = sum(1 for r in results if r['status'] == 'pass')
        failed = sum(1 for r in results if r['status'] == 'fail')
        errors = sum(1 for r in results if r['status'] == 'error')

        print(f"  Total: {len(results)} tables")
        print(f"  Passed: {passed} ✅")
        print(f"  Failed: {failed} ❌")
        if errors:
            print(f"  Errors: {errors} ⚠️")

        if failed > 0:
            print(f"\n  Tables needing attention:")
            for r in results:
                if r['status'] == 'fail':
                    print(f"    • {r['table']}")

    else:
        # JSON output for programmatic use
        print(json.dumps(results, indent=2, default=str))

    conn.close()

    # Exit code
    sys.exit(0 if all(r['status'] == 'pass' for r in results) else 1)


if __name__ == "__main__":
    main()
