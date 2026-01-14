#!/usr/bin/env python3
"""
HNSW Index Migration Script

Adds HNSW indices to existing RAG tables and validates performance improvement.

Features:
- Automatic detection of tables without HNSW indices
- Before/after performance benchmarking
- Detailed progress tracking and reporting
- Safe execution with rollback support
"""
import os
import sys
import time
import psycopg2
from psycopg2 import sql
from typing import List, Dict, Tuple
import numpy as np

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def connect_db():
    """Connect to PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        database=os.getenv("DB_NAME", "vector_db"),
    )


def list_tables(conn) -> List[str]:
    """List all data tables (with 'data_' prefix)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE 'data_%'
            ORDER BY tablename;
        """)
        return [row[0] for row in cur.fetchall()]


def has_hnsw_index(conn, table_name: str) -> bool:
    """Check if table has HNSW index on embedding column."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*)
            FROM pg_indexes
            WHERE tablename = %s
              AND indexdef LIKE '%%hnsw%%'
              AND indexdef LIKE '%%embedding%%';
        """, (table_name,))
        return cur.fetchone()[0] > 0


def get_table_info(conn, table_name: str) -> Dict:
    """Get table row count and size."""
    info = {}
    with conn.cursor() as cur:
        # Row count
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
        info['row_count'] = cur.fetchone()[0]

        # Table size
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size(%s))", (table_name,))
        info['size'] = cur.fetchone()[0]

        # Embedding dimensions
        cur.execute(sql.SQL("""
            SELECT vector_dims(embedding) as dim
            FROM {}
            WHERE embedding IS NOT NULL
            LIMIT 1
        """).format(sql.Identifier(table_name)))
        result = cur.fetchone()
        info['dimensions'] = result[0] if result else None

    return info


def benchmark_query(conn, table_name: str, num_trials: int = 5) -> Dict:
    """Benchmark query performance with multiple trials."""
    with conn.cursor() as cur:
        # Get embedding dimensions
        cur.execute(sql.SQL("""
            SELECT vector_dims(embedding) as dim
            FROM {}
            WHERE embedding IS NOT NULL
            LIMIT 1
        """).format(sql.Identifier(table_name)))

        result = cur.fetchone()
        if not result:
            return {'error': 'No embeddings found'}

        embed_dim = result[0]

        # Create dummy query vector
        query_vector = [0.1] * embed_dim

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

            cur.fetchall()  # Consume results
            latency = time.time() - start
            latencies.append(latency)

        return {
            'avg_latency': np.mean(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'std_latency': np.std(latencies),
            'trials': num_trials
        }


def create_hnsw_index(conn, table_name: str, m: int = 16, ef_construction: int = 64) -> Dict:
    """
    Create HNSW index on table.

    Args:
        conn: Database connection
        table_name: Table name (with data_ prefix)
        m: Number of connections per layer (default: 16)
            - Higher = better recall, more memory
            - Range: 4-64, recommended: 16-32
        ef_construction: Build-time search width (default: 64)
            - Higher = better quality, slower build
            - Range: 16-512, recommended: 64-200

    Returns:
        Dict with creation time and success status
    """
    index_name = f"{table_name}_hnsw_idx"

    try:
        with conn.cursor() as cur:
            print(f"  Creating HNSW index: {index_name}")
            print(f"    Parameters: m={m}, ef_construction={ef_construction}")

            start = time.time()

            cur.execute(sql.SQL("""
                CREATE INDEX {}
                ON {}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = %s, ef_construction = %s)
            """).format(sql.Identifier(index_name), sql.Identifier(table_name)),
            (m, ef_construction))

            conn.commit()

            creation_time = time.time() - start

            return {
                'success': True,
                'creation_time': creation_time,
                'index_name': index_name
            }

    except Exception as e:
        conn.rollback()
        return {
            'success': False,
            'error': str(e)
        }


def format_speedup(before: float, after: float) -> str:
    """Format speedup as a friendly string."""
    if after == 0:
        return "∞x faster"

    speedup = before / after

    if speedup < 1.1:
        return "~same"
    elif speedup < 2:
        return f"{speedup:.1f}x faster"
    elif speedup < 10:
        return f"{speedup:.1f}x faster"
    else:
        return f"{speedup:.0f}x faster"


def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(description='Add HNSW indices to RAG tables')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    args = parser.parse_args()

    print("=" * 80)
    print("HNSW Index Migration & Performance Validation")
    print("=" * 80)
    print()

    # Connect to database
    try:
        conn = connect_db()
        conn.autocommit = False  # Use transactions
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("\nMake sure to set environment variables:")
        print("  PGHOST, PGPORT, PGUSER, PGPASSWORD, DB_NAME")
        sys.exit(1)

    # List all tables
    tables = list_tables(conn)

    if not tables:
        print("❌ No tables found in database")
        sys.exit(1)

    print(f"📊 Found {len(tables)} tables to analyze")
    print()

    # Analyze tables
    tables_to_migrate = []
    tables_already_indexed = []

    for table in tables:
        has_index = has_hnsw_index(conn, table)
        info = get_table_info(conn, table)

        if has_index:
            tables_already_indexed.append((table, info))
            print(f"  ✅ {table}")
            print(f"      {info['row_count']:,} rows, {info['size']}, HNSW index exists")
        else:
            tables_to_migrate.append((table, info))
            print(f"  ⚠️  {table}")
            print(f"      {info['row_count']:,} rows, {info['size']}, NO HNSW index")

    print()

    if not tables_to_migrate:
        print("✅ All tables already have HNSW indices!")
        print()
        print("Performance tip: Query speed can be tuned with ef_search parameter:")
        print("  SET hnsw.ef_search = 200;  -- Higher = better recall, slower queries")
        conn.close()
        return

    # Confirm migration
    print(f"🔧 Found {len(tables_to_migrate)} tables that need HNSW indices:")
    for table, info in tables_to_migrate:
        print(f"  • {table} ({info['row_count']:,} rows)")
    print()

    if args.dry_run:
        print("🔍 Dry run mode - no changes will be made")
        print("   Run without --dry-run to perform migration")
        conn.close()
        return

    if not args.yes:
        response = input("Proceed with migration? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Migration cancelled")
            conn.close()
            return
    else:
        print("✅ Auto-confirmed (--yes flag)")
        print()

    print()
    print("=" * 80)
    print("Starting Migration")
    print("=" * 80)
    print()

    # Migrate each table
    results = []

    for i, (table, info) in enumerate(tables_to_migrate, 1):
        print(f"[{i}/{len(tables_to_migrate)}] Processing: {table}")
        print(f"  Rows: {info['row_count']:,}")
        print(f"  Size: {info['size']}")
        print(f"  Dimensions: {info['dimensions']}")
        print()

        # Benchmark BEFORE
        print("  📊 Benchmarking query performance WITHOUT HNSW...")
        before = benchmark_query(conn, table)

        if 'error' in before:
            print(f"  ⚠️  Benchmark failed: {before['error']}")
            print(f"  Skipping table {table}")
            print()
            continue

        print(f"    Avg latency: {before['avg_latency']*1000:.1f}ms ({before['trials']} trials)")
        print()

        # Create HNSW index
        print("  🔨 Creating HNSW index...")

        # Estimate time based on row count
        estimated_time = info['row_count'] / 10000  # ~10K rows/second
        if estimated_time > 10:
            print(f"    ⏳ Estimated time: {estimated_time:.0f}s (large table)")

        creation_result = create_hnsw_index(conn, table)

        if not creation_result['success']:
            print(f"  ❌ Failed to create index: {creation_result['error']}")
            print()
            continue

        print(f"    ✅ Created in {creation_result['creation_time']:.1f}s")
        print(f"    Throughput: {info['row_count'] / creation_result['creation_time']:.0f} vectors/sec")
        print()

        # Benchmark AFTER
        print("  📊 Benchmarking query performance WITH HNSW...")
        after = benchmark_query(conn, table)

        if 'error' in after:
            print(f"  ⚠️  Benchmark failed: {after['error']}")
        else:
            print(f"    Avg latency: {after['avg_latency']*1000:.1f}ms ({after['trials']} trials)")

            speedup = format_speedup(before['avg_latency'], after['avg_latency'])
            print(f"    🚀 Improvement: {speedup}")

        print()

        # Store results
        results.append({
            'table': table,
            'info': info,
            'before': before,
            'after': after,
            'creation_time': creation_result['creation_time']
        })

    # Summary report
    print("=" * 80)
    print("Migration Complete")
    print("=" * 80)
    print()

    if not results:
        print("❌ No tables were successfully migrated")
        conn.close()
        return

    print(f"✅ Successfully migrated {len(results)} tables")
    print()

    # Performance summary table
    print("Performance Improvements:")
    print()
    print(f"{'Table':<50} {'Before':<12} {'After':<12} {'Speedup'}")
    print("-" * 90)

    for result in results:
        table = result['table']
        before_ms = result['before']['avg_latency'] * 1000
        after_ms = result['after']['avg_latency'] * 1000
        speedup = format_speedup(result['before']['avg_latency'], result['after']['avg_latency'])

        print(f"{table:<50} {before_ms:>10.1f}ms {after_ms:>10.1f}ms {speedup}")

    print()

    # Overall statistics
    total_before = sum(r['before']['avg_latency'] for r in results)
    total_after = sum(r['after']['avg_latency'] for r in results)
    overall_speedup = format_speedup(total_before, total_after)

    print(f"Overall average speedup: {overall_speedup}")
    print()

    # Save report
    report_file = f"hnsw_migration_report_{int(time.time())}.txt"
    with open(report_file, 'w') as f:
        f.write("HNSW Index Migration Report\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"Table: {result['table']}\n")
            f.write(f"  Rows: {result['info']['row_count']:,}\n")
            f.write(f"  Size: {result['info']['size']}\n")
            f.write(f"  Index creation time: {result['creation_time']:.1f}s\n")
            f.write(f"  Before: {result['before']['avg_latency']*1000:.1f}ms\n")
            f.write(f"  After: {result['after']['avg_latency']*1000:.1f}ms\n")
            speedup = format_speedup(result['before']['avg_latency'], result['after']['avg_latency'])
            f.write(f"  Speedup: {speedup}\n")
            f.write("\n")

    print(f"📄 Report saved to: {report_file}")
    print()

    # Tuning tips
    print("🎯 Performance Tuning Tips:")
    print()
    print("  1. Adjust ef_search for query-time quality/speed trade-off:")
    print("     SET hnsw.ef_search = 200;  -- Default: 40")
    print()
    print("  2. Monitor query performance:")
    print("     EXPLAIN ANALYZE SELECT ... ORDER BY embedding <=> query LIMIT 10;")
    print()
    print("  3. For very large tables (>1M rows), consider:")
    print("     - Higher m (32-64) for better recall")
    print("     - Higher ef_construction (128-256) for better quality")
    print()

    conn.close()


if __name__ == "__main__":
    main()
