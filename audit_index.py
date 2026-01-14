#!/usr/bin/env python3
"""
RAG Index Audit Script

Performs comprehensive audit of RAG index health, consistency, and quality.
"""
import os
import sys
import json
import psycopg2
from psycopg2 import sql
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, assume env vars are set


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
    """List all tables in the database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """)
        return [row[0] for row in cur.fetchall()]


def get_table_info(conn, table_name: str) -> Dict:
    """Get basic table information."""
    info = {}

    with conn.cursor() as cur:
        # Row count
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
        info['row_count'] = cur.fetchone()[0]

        # Table size
        cur.execute("""
            SELECT pg_size_pretty(pg_total_relation_size(%s)) as size
        """, (table_name,))
        info['size'] = cur.fetchone()[0]

        # Column info
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table_name,))
        info['columns'] = {row[0]: row[1] for row in cur.fetchall()}

    return info


def check_configuration(conn, table_name: str) -> Dict:
    """Check index configuration consistency."""
    config = {
        'consistent': True,
        'configs': [],
        'warnings': []
    }

    with conn.cursor() as cur:
        # Check if metadata column exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'metadata_'
            );
        """, (table_name,))

        has_metadata = cur.fetchone()[0]

        if not has_metadata:
            config['consistent'] = False
            config['warnings'].append("Legacy index: No metadata column found")
            return config

        # Get distinct configurations
        cur.execute(sql.SQL("""
            SELECT
                metadata_->>'_chunk_size' as chunk_size,
                metadata_->>'_chunk_overlap' as overlap,
                metadata_->>'_embed_model' as model,
                metadata_->>'_embed_dim' as dim,
                COUNT(*) as count
            FROM {}
            WHERE metadata_ IS NOT NULL
            GROUP BY chunk_size, overlap, model, dim
            ORDER BY count DESC;
        """).format(sql.Identifier(table_name)))

        configs = cur.fetchall()

        if not configs:
            config['consistent'] = False
            config['warnings'].append("No configuration metadata found")
            return config

        for row in configs:
            config['configs'].append({
                'chunk_size': row[0],
                'overlap': row[1],
                'model': row[2],
                'dim': row[3],
                'count': row[4]
            })

        if len(configs) > 1:
            config['consistent'] = False
            config['warnings'].append(f"Mixed configurations detected: {len(configs)} different configs")

    return config


def analyze_chunks(conn, table_name: str) -> Dict:
    """Analyze chunk quality and statistics."""
    stats = {}

    with conn.cursor() as cur:
        # Chunk size statistics
        cur.execute(sql.SQL("""
            SELECT
                AVG(LENGTH(text))::int as avg_size,
                MIN(LENGTH(text)) as min_size,
                MAX(LENGTH(text)) as max_size,
                STDDEV(LENGTH(text))::int as stddev,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(text))::int as median
            FROM {};
        """).format(sql.Identifier(table_name)))

        row = cur.fetchone()
        stats['size'] = {
            'avg': row[0],
            'min': row[1],
            'max': row[2],
            'stddev': row[3],
            'median': row[4]
        }

        # Sample chunks
        cur.execute(sql.SQL("""
            SELECT text, LENGTH(text) as len
            FROM {}
            ORDER BY RANDOM()
            LIMIT 3;
        """).format(sql.Identifier(table_name)))

        stats['samples'] = [{'text': row[0][:200] + '...', 'length': row[1]} for row in cur.fetchall()]

        # Check for documents metadata
        cur.execute(sql.SQL("""
            SELECT
                COUNT(DISTINCT metadata_->>'file_name') as num_docs,
                COUNT(*) as total_chunks
            FROM {}
            WHERE metadata_ IS NOT NULL;
        """).format(sql.Identifier(table_name)))

        row = cur.fetchone()
        if row[0] and row[0] > 0:
            stats['chunks_per_doc'] = round(row[1] / row[0], 1)
        else:
            stats['chunks_per_doc'] = None

    return stats


def check_embeddings(conn, table_name: str) -> Dict:
    """Verify embedding health."""
    emb_stats = {}

    with conn.cursor() as cur:
        # Check for null embeddings
        cur.execute(sql.SQL("""
            SELECT COUNT(*)
            FROM {}
            WHERE embedding IS NULL;
        """).format(sql.Identifier(table_name)))

        emb_stats['null_count'] = cur.fetchone()[0]

        # Get embedding dimensions using vector_dims()
        cur.execute(sql.SQL("""
            SELECT vector_dims(embedding) as dim
            FROM {}
            WHERE embedding IS NOT NULL
            LIMIT 1;
        """).format(sql.Identifier(table_name)))

        result = cur.fetchone()
        emb_stats['dimensions'] = result[0] if result else None

        # Check dimension consistency
        cur.execute(sql.SQL("""
            SELECT COUNT(DISTINCT vector_dims(embedding))
            FROM {}
            WHERE embedding IS NOT NULL;
        """).format(sql.Identifier(table_name)))

        emb_stats['dim_consistent'] = cur.fetchone()[0] == 1

    return emb_stats


def test_query(conn, table_name: str, embed_dim: int = 384) -> Dict:
    """Test query performance with a sample."""
    results = {}

    # Create a dummy query vector
    query_vector = [0.1] * embed_dim

    with conn.cursor() as cur:
        import time

        # Test retrieval performance
        start = time.time()
        cur.execute(sql.SQL("""
            SELECT
                text,
                1 - (embedding <=> %s::vector) as similarity
            FROM {}
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
        """).format(sql.Identifier(table_name)), (query_vector, query_vector))

        latency = time.time() - start
        rows = cur.fetchall()

        if rows:
            similarities = [row[1] for row in rows]
            results['latency'] = round(latency, 3)
            results['avg_similarity'] = round(np.mean(similarities), 3)
            results['top_similarity'] = round(max(similarities), 3)
            results['min_similarity'] = round(min(similarities), 3)
        else:
            results['error'] = "No results returned"

    return results


def generate_report(table_name: str, table_info: Dict, config: Dict,
                   chunk_stats: Dict, emb_stats: Dict, query_results: Dict) -> str:
    """Generate formatted audit report."""

    report = []
    report.append("=" * 80)
    report.append(f"RAG INDEX AUDIT REPORT: {table_name}")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary
    status = "✅ HEALTHY"
    if not config['consistent'] or emb_stats['null_count'] > 0:
        status = "⚠️  WARNINGS"
    if emb_stats['null_count'] > 0:
        status = "❌ ISSUES"

    report.append("## SUMMARY")
    report.append(f"Status: {status}")
    report.append(f"Rows: {table_info['row_count']:,}")
    report.append(f"Storage Size: {table_info['size']}")
    report.append("")

    # Configuration
    report.append("## CONFIGURATION")
    if config['consistent']:
        cfg = config['configs'][0]
        report.append("✅ Consistent configuration across all chunks")
        report.append(f"  - Chunk Size: {cfg['chunk_size']}")
        report.append(f"  - Overlap: {cfg['overlap']}")
        report.append(f"  - Model: {cfg['model']}")
        report.append(f"  - Dimensions: {cfg['dim']}")
    else:
        report.append("⚠️  Configuration Issues:")
        for warning in config['warnings']:
            report.append(f"  - {warning}")
        if config['configs']:
            report.append("")
            report.append("Found configurations:")
            for i, cfg in enumerate(config['configs'], 1):
                report.append(f"  Config {i}: cs={cfg['chunk_size']}, ov={cfg['overlap']}, "
                            f"model={cfg['model']}, rows={cfg['count']}")
    report.append("")

    # Chunk Statistics
    report.append("## CHUNK STATISTICS")
    if chunk_stats['size']:
        s = chunk_stats['size']
        report.append(f"  - Average Size: {s['avg']} characters")
        report.append(f"  - Median Size: {s['median']} characters")
        report.append(f"  - Range: {s['min']} - {s['max']} characters")
        report.append(f"  - Std Dev: {s['stddev']} characters")
        if chunk_stats['chunks_per_doc']:
            report.append(f"  - Chunks per Document: {chunk_stats['chunks_per_doc']:.1f} avg")

    if chunk_stats['samples']:
        report.append("")
        report.append("Sample chunks:")
        for i, sample in enumerate(chunk_stats['samples'], 1):
            report.append(f"  {i}. ({sample['length']} chars) {sample['text']}")
    report.append("")

    # Embedding Health
    report.append("## EMBEDDING HEALTH")
    if emb_stats['null_count'] == 0:
        report.append("✅ All chunks have embeddings")
    else:
        report.append(f"❌ {emb_stats['null_count']} chunks missing embeddings")

    report.append(f"  - Dimensions: {emb_stats['dimensions']}")

    if emb_stats['dim_consistent']:
        report.append("✅ Embedding dimensions consistent")
    else:
        report.append("⚠️  Inconsistent embedding dimensions detected")
    report.append("")

    # Query Performance
    report.append("## QUERY PERFORMANCE")
    if 'error' in query_results:
        report.append(f"❌ {query_results['error']}")
    else:
        report.append("Test retrieval (dummy query):")
        report.append(f"  - Latency: {query_results['latency']}s")
        report.append(f"  - Top Similarity: {query_results['top_similarity']}")
        report.append(f"  - Avg Similarity: {query_results['avg_similarity']}")
        report.append(f"  - Min Similarity: {query_results['min_similarity']}")

        # Quality assessment
        if query_results['avg_similarity'] > 0.5:
            report.append("  ✅ Good retrieval quality expected")
        elif query_results['avg_similarity'] > 0.3:
            report.append("  ⚠️  Moderate retrieval quality")
        else:
            report.append("  ⚠️  Low similarity scores detected")
    report.append("")

    # Recommendations
    report.append("## RECOMMENDATIONS")
    recommendations = []

    if not config['consistent']:
        recommendations.append("🔧 Rebuild index with RESET_TABLE=1 for consistent configuration")

    if emb_stats['null_count'] > 0:
        recommendations.append("🔧 Re-run indexing to generate missing embeddings")

    if chunk_stats['size']['stddev'] > chunk_stats['size']['avg'] * 0.5:
        recommendations.append("💡 High chunk size variance - consider adjusting chunk_size")

    if chunk_stats['size']['avg'] < 200:
        recommendations.append("💡 Small chunks may lose context - consider increasing CHUNK_SIZE")
    elif chunk_stats['size']['avg'] > 1500:
        recommendations.append("💡 Large chunks may be too broad - consider decreasing CHUNK_SIZE")

    if 'latency' in query_results and query_results['latency'] > 1.0:
        recommendations.append("⚡ Slow query performance - consider adding vector index")

    if not recommendations:
        recommendations.append("✅ Index is healthy and ready for production use")

    for rec in recommendations:
        report.append(f"  {rec}")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main audit function."""
    # Check for table name argument
    table_name = sys.argv[1] if len(sys.argv) > 1 else None

    # Connect to database
    try:
        conn = connect_db()
        conn.autocommit = True
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("\nMake sure to set environment variables:")
        print("  PGHOST, PGPORT, PGUSER, PGPASSWORD, DB_NAME")
        print("\nOr load from .env file:")
        print("  source .env")
        sys.exit(1)

    # List available tables
    tables = list_tables(conn)

    if not tables:
        print("❌ No tables found in database")
        sys.exit(1)

    # If no table specified, show available tables
    if not table_name:
        print("📊 Available tables:")
        print()
        for i, table in enumerate(tables, 1):
            print(f"  {i}. {table}")
        print()
        print("Usage: python audit_index.py <table_name>")
        sys.exit(0)

    # Validate table exists
    if table_name not in tables:
        print(f"❌ Table '{table_name}' not found")
        print(f"\nAvailable tables: {', '.join(tables)}")
        sys.exit(1)

    # Run audit
    print(f"🔍 Auditing index: {table_name}")
    print()

    try:
        # Gather information
        table_info = get_table_info(conn, table_name)
        config = check_configuration(conn, table_name)
        chunk_stats = analyze_chunks(conn, table_name)
        emb_stats = check_embeddings(conn, table_name)

        # Determine embedding dimensions for query test
        embed_dim = emb_stats.get('dimensions', 384)
        query_results = test_query(conn, table_name, embed_dim)

        # Generate report
        report = generate_report(table_name, table_info, config, chunk_stats,
                                emb_stats, query_results)

        print(report)

        # Save report to file
        report_file = f"audit_report_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print()
        print(f"📄 Report saved to: {report_file}")

    except Exception as e:
        print(f"❌ Audit failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
