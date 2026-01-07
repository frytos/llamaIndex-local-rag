#!/usr/bin/env python3
"""
Performance Analysis Tool for RAG Pipeline (M1 Mac Mini 16GB)

Analyzes all performance aspects of the RAG pipeline:
- Query latency breakdown
- Indexing throughput
- Resource utilization
- Database optimization
- Configuration tuning recommendations

Usage:
    python performance_analysis.py --analyze-all
    python performance_analysis.py --query-latency
    python performance_analysis.py --index-performance
    python performance_analysis.py --database-check
"""

import os
import sys
import time
import argparse
import json
import psycopg2
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import platform

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - install with: pip install psutil")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  torch not available - GPU detection disabled")


@dataclass
class PerformanceMetrics:
    """Container for performance measurements"""
    # Query Performance
    query_embedding_time: float = 0.0
    vector_search_time: float = 0.0
    context_formatting_time: float = 0.0
    llm_generation_time: float = 0.0
    total_query_time: float = 0.0
    tokens_per_second: float = 0.0

    # Indexing Performance
    document_loading_throughput: float = 0.0  # files/sec
    chunking_throughput: float = 0.0  # docs/sec
    embedding_throughput: float = 0.0  # chunks/sec
    database_insert_throughput: float = 0.0  # nodes/sec
    total_indexing_time: float = 0.0

    # Resource Utilization
    cpu_usage_percent: float = 0.0
    memory_usage_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_available: bool = False
    gpu_type: str = ""

    # Database Performance
    table_size_mb: float = 0.0
    index_exists: bool = False
    index_type: str = ""
    row_count: int = 0
    avg_vector_search_ms: float = 0.0

    # Configuration
    chunk_size: int = 0
    chunk_overlap: int = 0
    top_k: int = 0
    embed_batch_size: int = 0
    embed_model: str = ""
    embed_dim: int = 0


class PerformanceAnalyzer:
    """Comprehensive performance analyzer for RAG pipeline"""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.db_config = self._load_db_config()

    def _load_db_config(self) -> Dict:
        """Load database configuration from environment"""
        return {
            "host": os.getenv("PGHOST", "localhost"),
            "port": int(os.getenv("PGPORT", "5432")),
            "user": os.getenv("PGUSER"),
            "password": os.getenv("PGPASSWORD"),
            "database": os.getenv("DB_NAME", "vector_db"),
        }

    def analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze system resources and capabilities"""
        print("\n" + "="*70)
        print("SYSTEM RESOURCES ANALYSIS")
        print("="*70)

        results = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
        }

        # CPU and Memory
        if PSUTIL_AVAILABLE:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()

            results.update({
                "cpu_cores": cpu_count,
                "cpu_usage_percent": cpu_percent,
                "memory_total_gb": round(mem.total / 1e9, 2),
                "memory_available_gb": round(mem.available / 1e9, 2),
                "memory_used_percent": mem.percent,
            })

            self.metrics.cpu_usage_percent = cpu_percent
            self.metrics.memory_usage_gb = round((mem.total - mem.available) / 1e9, 2)
            self.metrics.memory_available_gb = round(mem.available / 1e9, 2)

            print(f"✓ CPU: {cpu_count} cores @ {cpu_percent}% usage")
            print(f"✓ Memory: {results['memory_total_gb']}GB total, "
                  f"{results['memory_available_gb']}GB available ({mem.percent}% used)")
        else:
            print("⚠️  System metrics unavailable (psutil not installed)")

        # GPU Detection
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                results["gpu_type"] = "Apple Metal (MPS)"
                results["gpu_available"] = True
                self.metrics.gpu_available = True
                self.metrics.gpu_type = "Apple Metal (MPS)"
                print(f"✓ GPU: Apple Metal (MPS) available")
            elif torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                results["gpu_type"] = gpu_name
                results["gpu_available"] = True
                self.metrics.gpu_available = True
                self.metrics.gpu_type = gpu_name
                print(f"✓ GPU: {gpu_name} available")
            else:
                results["gpu_available"] = False
                print("⚠️  No GPU acceleration detected")
        else:
            results["gpu_available"] = False
            print("⚠️  GPU detection unavailable (torch not installed)")

        return results

    def analyze_database_performance(self, table_name: str = None) -> Dict[str, Any]:
        """Analyze PostgreSQL + pgvector performance"""
        print("\n" + "="*70)
        print("DATABASE PERFORMANCE ANALYSIS")
        print("="*70)

        if table_name is None:
            table_name = os.getenv("PGTABLE", "inbox_clean_cs700_ov150_bge")

        actual_table = f"data_{table_name}"
        results = {}

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                );
            """, (actual_table,))
            table_exists = cur.fetchone()[0]

            if not table_exists:
                print(f"⚠️  Table '{actual_table}' does not exist")
                return {"error": "Table not found"}

            # Row count
            from psycopg2 import sql
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(actual_table))
            )
            row_count = cur.fetchone()[0]
            self.metrics.row_count = row_count
            results["row_count"] = row_count
            print(f"✓ Table: {actual_table}")
            print(f"  • Rows: {row_count:,}")

            # Table size
            cur.execute(
                sql.SQL("SELECT pg_size_pretty(pg_total_relation_size(%s));"),
                (actual_table,)
            )
            size_pretty = cur.fetchone()[0]

            cur.execute(
                sql.SQL("SELECT pg_total_relation_size(%s) / (1024.0 * 1024.0);"),
                (actual_table,)
            )
            size_mb = cur.fetchone()[0]
            self.metrics.table_size_mb = size_mb
            results["table_size"] = size_pretty
            results["table_size_mb"] = round(size_mb, 2)
            print(f"  • Size: {size_pretty} ({size_mb:.2f} MB)")

            # Check for indexes
            cur.execute(f"""
                SELECT
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE tablename = %s
                AND indexname LIKE '%%hnsw%%' OR indexname LIKE '%%ivfflat%%';
            """, (actual_table,))
            indexes = cur.fetchall()

            if indexes:
                self.metrics.index_exists = True
                for idx_name, idx_def in indexes:
                    if 'hnsw' in idx_name.lower():
                        self.metrics.index_type = "HNSW"
                        results["index_type"] = "HNSW"
                    elif 'ivfflat' in idx_name.lower():
                        self.metrics.index_type = "IVFFlat"
                        results["index_type"] = "IVFFlat"
                    print(f"  • Index: {idx_name} ({self.metrics.index_type})")
                    results["index_name"] = idx_name
            else:
                print(f"  ⚠️  No vector index found (HNSW/IVFFlat)")
                print(f"     Vector searches will be slow (sequential scan)")
                results["index_type"] = "None (Sequential Scan)"

            # Benchmark vector search speed
            print("\n  Testing vector search performance...")
            cur.execute(
                sql.SQL("SELECT embedding FROM {} LIMIT 1;").format(sql.Identifier(actual_table))
            )
            sample_embedding = cur.fetchone()

            if sample_embedding and sample_embedding[0]:
                embedding_str = sample_embedding[0]

                # Run 10 test queries
                search_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    cur.execute(f"""
                        SELECT * FROM {actual_table}
                        ORDER BY embedding <=> %s
                        LIMIT 5;
                    """, (embedding_str,))
                    cur.fetchall()
                    search_times.append((time.perf_counter() - start) * 1000)

                avg_search_ms = sum(search_times) / len(search_times)
                min_search_ms = min(search_times)
                max_search_ms = max(search_times)

                self.metrics.avg_vector_search_ms = avg_search_ms
                results["avg_vector_search_ms"] = round(avg_search_ms, 2)
                results["min_vector_search_ms"] = round(min_search_ms, 2)
                results["max_vector_search_ms"] = round(max_search_ms, 2)

                print(f"  • Vector search (avg): {avg_search_ms:.2f}ms")
                print(f"  • Vector search (min): {min_search_ms:.2f}ms")
                print(f"  • Vector search (max): {max_search_ms:.2f}ms")

                # Performance assessment
                if avg_search_ms < 50:
                    print(f"  ✓ Excellent performance (<50ms)")
                elif avg_search_ms < 200:
                    print(f"  ✓ Good performance (<200ms)")
                elif avg_search_ms < 500:
                    print(f"  ⚠️  Moderate performance (200-500ms)")
                else:
                    print(f"  ❌ Poor performance (>500ms) - consider adding HNSW index")

            cur.close()
            conn.close()

        except Exception as e:
            print(f"❌ Database analysis failed: {e}")
            results["error"] = str(e)

        return results

    def benchmark_embedding_performance(self, model_name: str = None) -> Dict[str, Any]:
        """Benchmark embedding generation performance"""
        print("\n" + "="*70)
        print("EMBEDDING PERFORMANCE BENCHMARK")
        print("="*70)

        if model_name is None:
            model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")

        print(f"Model: {model_name}")

        try:
            # Import after potential env var changes
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Detect device
            device = "cpu"
            if TORCH_AVAILABLE:
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"

            print(f"Device: {device}")

            # Load model
            print("Loading model...")
            start = time.perf_counter()
            model = HuggingFaceEmbedding(model_name=model_name, device=device)
            load_time = time.perf_counter() - start
            print(f"✓ Model loaded in {load_time:.2f}s")

            # Single query embedding (typical query scenario)
            query_text = "What are the main findings in the document?"
            print(f"\n1. Single Query Embedding:")

            # Warmup
            _ = model.get_query_embedding(query_text)

            # Benchmark
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = model.get_query_embedding(query_text)
                times.append((time.perf_counter() - start) * 1000)

            avg_query_ms = sum(times) / len(times)
            self.metrics.query_embedding_time = avg_query_ms / 1000
            print(f"  • Average: {avg_query_ms:.2f}ms")
            print(f"  • Min: {min(times):.2f}ms")
            print(f"  • Max: {max(times):.2f}ms")

            # Batch embedding (indexing scenario)
            batch_sizes = [16, 32, 64, 128]
            batch_results = {}

            print(f"\n2. Batch Embedding (Indexing):")
            for batch_size in batch_sizes:
                texts = [f"Sample document chunk number {i} for testing embedding throughput"
                         for i in range(batch_size)]

                # Warmup
                _ = model.get_text_embedding_batch(texts)

                # Benchmark
                start = time.perf_counter()
                _ = model.get_text_embedding_batch(texts)
                elapsed = time.perf_counter() - start

                throughput = batch_size / elapsed
                per_item_ms = (elapsed / batch_size) * 1000

                batch_results[batch_size] = {
                    "throughput": throughput,
                    "per_item_ms": per_item_ms,
                    "elapsed": elapsed
                }

                print(f"  • Batch {batch_size:3d}: {throughput:6.1f} emb/sec "
                      f"({per_item_ms:.2f}ms/item)")

            # Find optimal batch size
            optimal_batch = max(batch_results.keys(),
                              key=lambda k: batch_results[k]["throughput"])
            optimal_throughput = batch_results[optimal_batch]["throughput"]
            self.metrics.embedding_throughput = optimal_throughput

            print(f"\n  ✓ Optimal batch size: {optimal_batch} ({optimal_throughput:.1f} emb/sec)")

            # Estimate indexing time for various corpus sizes
            print(f"\n3. Estimated Indexing Time:")
            corpus_sizes = [1000, 5000, 10000, 50000]
            for size in corpus_sizes:
                time_seconds = size / optimal_throughput
                time_minutes = time_seconds / 60
                print(f"  • {size:6,} chunks: {time_minutes:.1f} minutes")

            return {
                "model": model_name,
                "device": device,
                "load_time_s": round(load_time, 2),
                "query_embedding_ms": round(avg_query_ms, 2),
                "batch_results": {k: {
                    "throughput": round(v["throughput"], 1),
                    "per_item_ms": round(v["per_item_ms"], 2)
                } for k, v in batch_results.items()},
                "optimal_batch_size": optimal_batch,
                "optimal_throughput": round(optimal_throughput, 1)
            }

        except Exception as e:
            print(f"❌ Embedding benchmark failed: {e}")
            return {"error": str(e)}

    def analyze_query_latency(self) -> Dict[str, Any]:
        """Analyze query latency breakdown"""
        print("\n" + "="*70)
        print("QUERY LATENCY BREAKDOWN")
        print("="*70)

        print("""
Query pipeline stages:
1. Query Embedding: Convert user question to vector
2. Vector Search: Find similar chunks in pgvector
3. Context Formatting: Prepare retrieved chunks
4. LLM Generation: Generate answer from context

Expected latency (M1 Mac Mini):
""")

        # Expected performance baselines
        baseline = {
            "query_embedding_ms": {
                "value": 50,
                "description": "bge-small-en on MPS"
            },
            "vector_search_ms": {
                "value": 100,
                "description": "With HNSW index, 10k rows"
            },
            "context_formatting_ms": {
                "value": 10,
                "description": "Minimal overhead"
            },
            "llm_generation_s": {
                "value": 8,
                "description": "llama.cpp, 150 tokens @ 20 tok/s"
            },
            "total_query_s": {
                "value": 8.2,
                "description": "Total end-to-end"
            }
        }

        print(f"  • Query Embedding: ~{baseline['query_embedding_ms']['value']}ms "
              f"({baseline['query_embedding_ms']['description']})")
        print(f"  • Vector Search: ~{baseline['vector_search_ms']['value']}ms "
              f"({baseline['vector_search_ms']['description']})")
        print(f"  • Context Format: ~{baseline['context_formatting_ms']['value']}ms "
              f"({baseline['context_formatting_ms']['description']})")
        print(f"  • LLM Generation: ~{baseline['llm_generation_s']['value']}s "
              f"({baseline['llm_generation_s']['description']})")
        print(f"  • TOTAL: ~{baseline['total_query_s']['value']}s")

        print(f"\nBottleneck: LLM generation (97% of query time)")
        print(f"\nOptimization options:")
        print(f"  1. Use vLLM server (2-3s queries, 3-4x faster)")
        print(f"  2. Reduce max_tokens (faster but shorter answers)")
        print(f"  3. Use smaller model (faster but lower quality)")
        print(f"  4. GPU acceleration (if available)")

        return baseline

    def generate_optimization_recommendations(self) -> List[str]:
        """Generate specific optimization recommendations"""
        recommendations = []

        # Database optimizations
        if not self.metrics.index_exists:
            recommendations.append({
                "priority": "HIGH",
                "category": "Database",
                "issue": "No vector index detected",
                "impact": "50-100x slower retrieval on large tables",
                "solution": "Create HNSW index",
                "command": "python rag_low_level_m1_16gb_verbose.py --create-hnsw-index",
                "estimated_improvement": "Reduce vector search from 500ms to 50ms"
            })

        if self.metrics.row_count > 50000 and self.metrics.index_type == "IVFFlat":
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Database",
                "issue": "Using IVFFlat index on large table",
                "impact": "HNSW provides better performance for >50k rows",
                "solution": "Migrate to HNSW index",
                "command": "DROP INDEX old_idx; CREATE INDEX USING hnsw...",
                "estimated_improvement": "20-30% faster vector search"
            })

        # Embedding optimizations
        if self.metrics.embed_batch_size < 64:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Embeddings",
                "issue": f"Small embedding batch size ({self.metrics.embed_batch_size})",
                "impact": "Suboptimal GPU utilization",
                "solution": "Increase batch size to 64-128",
                "command": "export EMBED_BATCH=64",
                "estimated_improvement": "1.5-2x faster embedding throughput"
            })

        # LLM optimizations
        if not os.getenv("USE_VLLM"):
            recommendations.append({
                "priority": "HIGH",
                "category": "LLM",
                "issue": "Using llama.cpp (CPU-based)",
                "impact": "Slow query generation (8-15s per query)",
                "solution": "Use vLLM server mode",
                "command": "./scripts/start_vllm_server.sh && USE_VLLM=1 python rag_low_level_m1_16gb_verbose.py",
                "estimated_improvement": "3-4x faster queries (8s → 2-3s)"
            })

        # Memory optimizations
        if PSUTIL_AVAILABLE and self.metrics.memory_available_gb < 4:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Memory",
                "issue": f"Low available memory ({self.metrics.memory_available_gb}GB)",
                "impact": "May cause swapping and slowdowns",
                "solution": "Reduce batch sizes or close other applications",
                "command": "export EMBED_BATCH=32 DB_INSERT_BATCH=100",
                "estimated_improvement": "Prevent memory pressure"
            })

        # Chunk size optimizations
        if self.metrics.chunk_size > 1000:
            recommendations.append({
                "priority": "LOW",
                "category": "Chunking",
                "issue": f"Large chunk size ({self.metrics.chunk_size})",
                "impact": "May exceed context window with high TOP_K",
                "solution": "Reduce chunk size or TOP_K",
                "command": "export CHUNK_SIZE=700 TOP_K=3",
                "estimated_improvement": "Prevent context overflow"
            })

        return recommendations

    def generate_performance_report(self, output_file: str = None):
        """Generate comprehensive performance report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*70)

        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": self.analyze_system_resources(),
            "database": self.analyze_database_performance(),
            "query_latency": self.analyze_query_latency(),
            "recommendations": self.generate_optimization_recommendations()
        }

        # Print recommendations
        print("\n" + "="*70)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*70)

        recommendations = report["recommendations"]
        if not recommendations:
            print("✓ No critical optimizations needed - system is well-configured!")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. [{rec['priority']}] {rec['category']}: {rec['issue']}")
                print(f"   Impact: {rec['impact']}")
                print(f"   Solution: {rec['solution']}")
                print(f"   Command: {rec['command']}")
                print(f"   Estimated improvement: {rec['estimated_improvement']}")

        # Save report
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n✓ Report saved to: {output_file}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Performance analysis tool for RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--analyze-all", action="store_true",
                       help="Run all performance analyses")
    parser.add_argument("--query-latency", action="store_true",
                       help="Analyze query latency breakdown")
    parser.add_argument("--database-check", action="store_true",
                       help="Check database performance")
    parser.add_argument("--embedding-benchmark", action="store_true",
                       help="Benchmark embedding performance")
    parser.add_argument("--system-resources", action="store_true",
                       help="Analyze system resources")
    parser.add_argument("--table", type=str,
                       help="Table name to analyze (default: from PGTABLE env)")
    parser.add_argument("--model", type=str,
                       help="Embedding model to benchmark (default: from EMBED_MODEL env)")
    parser.add_argument("--output", type=str,
                       help="Save report to JSON file")

    args = parser.parse_args()

    # If no specific analysis requested, show help
    if not any([args.analyze_all, args.query_latency, args.database_check,
                args.embedding_benchmark, args.system_resources]):
        parser.print_help()
        return

    analyzer = PerformanceAnalyzer()

    if args.analyze_all:
        analyzer.generate_performance_report(output_file=args.output)
    else:
        if args.system_resources:
            analyzer.analyze_system_resources()

        if args.database_check:
            analyzer.analyze_database_performance(table_name=args.table)

        if args.embedding_benchmark:
            analyzer.benchmark_embedding_performance(model_name=args.model)

        if args.query_latency:
            analyzer.analyze_query_latency()


if __name__ == "__main__":
    main()
