"""
rag_low_level_m1_16gb_verbose.py

Goal
-----
A low-level RAG pipeline, fully local, with Postgres+pgvector for vectors, and llama.cpp GGUF for generation.
This is tuned to be "reasonable" on a 16GB Mac mini M1.

What you will learn by reading logs
-----------------------------------
1) How many Documents the PDF became (often pages)
2) How many chunks were produced + why overlap matters
3) How embeddings are computed (batched) + time per batch
4) How many rows are stored in Postgres + table reset behavior
5) What retrieval returns (scores, metadata, text previews)
6) What the LLM answers given retrieved evidence
"""

import os
import sys
import time
import platform
import logging
import argparse
import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, List, Optional, Iterable, Tuple
from pathlib import Path

import psycopg2
from psycopg2 import OperationalError as PgOperationalError

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    logging.warning("tqdm not installed, progress bars disabled. Install with: pip install tqdm")

from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.llama_cpp import LlamaCPP


# -----------------------
# Optional memory logging (nice to have, not required)
# -----------------------
try:
    import psutil  # pip install psutil
except ImportError as e:
    psutil = None
    # Note: psutil import failed, memory stats will not be available
except Exception as e:
    psutil = None
    logging.warning(f"Unexpected error importing psutil: {type(e).__name__}: {e}")


# -----------------------
# Logging configuration
# -----------------------
def setup_logging() -> logging.Logger:
    """
    LOG_LEVEL can be: DEBUG, INFO, WARNING, ERROR
    Default INFO is already pretty chatty.
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("rag")


log = setup_logging()


def log_system_info():
    log.info("System info:")
    log.info(f"  Python: {sys.version.split()[0]}")
    log.info(f"  Platform: {platform.platform()}")
    if psutil:
        vm = psutil.virtual_memory()
        log.info(f"  RAM: total={vm.total/1e9:.1f}GB available={vm.available/1e9:.1f}GB used={vm.percent}%")
    else:
        log.info("  RAM: (psutil not installed; skipping memory stats)")


def now_ms() -> int:
    return int(time.time() * 1000)


def dur_s(start_ms: int) -> float:
    return (now_ms() - start_ms) / 1000.0


def chunked(it: List[Any], n: int) -> Iterable[List[Any]]:
    """Yield list chunks of size n."""
    for i in range(0, len(it), n):
        yield it[i : i + n]


def preview(text: str, n: int = 220) -> str:
    """Small helper to keep logs readable."""
    t = (text or "").replace("\n", " ").strip()
    return (t[:n] + "…") if len(t) > n else t


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for tracking."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Local RAG pipeline with PostgreSQL + pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (index + query)
  python %(prog)s

  # Query existing index only (skip re-indexing)
  python %(prog)s --query-only

  # Interactive mode (ask multiple questions)
  python %(prog)s --interactive

  # Single query via CLI
  python %(prog)s --query "What are the main findings?"

  # Use different document
  PDF_PATH=my_doc.pdf PGTABLE=my_doc python %(prog)s

Environment Variables:
  See README.md for full list of configuration options
        """
    )

    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Skip document ingestion, only run query on existing index"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive REPL mode for multiple queries"
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Single query to run (overrides QUESTION env var)"
    )

    parser.add_argument(
        "--doc",
        "-d",
        type=str,
        help="Document path (overrides PDF_PATH env var)"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip startup validation (use with caution)"
    )

    return parser.parse_args()


# -----------------------
# Configuration (all overrideable via env vars)
# -----------------------
@dataclass
class Settings:
    # Postgres
    db_name: str = os.getenv("DB_NAME", "vector_db")
    host: str = os.getenv("PGHOST", "localhost")
    port: str = os.getenv("PGPORT", "5432")
    user: str = os.getenv("PGUSER", "fryt")
    password: str = os.getenv("PGPASSWORD", "frytos")
    table: str = os.getenv("PGTABLE", "llama2_paper")

    # Input
    pdf_path: str = os.getenv("PDF_PATH", "data/llama2.pdf")

    # Reset behaviors
    # RESET_TABLE=1 is useful while iterating so you don't duplicate rows every run
    reset_table: bool = os.getenv("RESET_TABLE", "0") == "1"
    # RESET_DB=1 is more nuclear; only use if you want a fresh DB
    reset_db: bool = os.getenv("RESET_DB", "0") == "1"

    # Chunking knobs (RAG quality knobs)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval knobs
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Embeddings knobs
    embed_model_name: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    embed_batch: int = int(os.getenv("EMBED_BATCH", "16"))

    # LLM knobs (llama.cpp)
    # Default: Mistral 7B Instruct GGUF Q4_K_M (good for 16GB M1)
    model_url: str = os.getenv(
        "MODEL_URL",
        "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    )
    # If you download manually, set MODEL_PATH to a local file and it will skip model_url.
    model_path: str = os.getenv("MODEL_PATH", "")

    temperature: float = float(os.getenv("TEMP", "0.1"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    context_window: int = int(os.getenv("CTX", "3072"))

    # On Apple Silicon, these matter a lot:
    # - N_GPU_LAYERS: offload more layers to Metal can speed up, but too high can crash or thrash
    # - N_BATCH: affects prompt processing throughput + peak memory
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "16"))
    n_batch: int = int(os.getenv("N_BATCH", "128"))

    # Question
    question: str = os.getenv(
        "QUESTION",
        "Summarize the key safety-related training ideas described in this paper.",
    )

    def validate(self) -> None:
        """
        Validate settings and provide helpful error messages.
        Raises ValueError with actionable error messages if validation fails.
        """
        errors = []

        # Validate PDF path
        pdf_file = Path(self.pdf_path)
        if not pdf_file.exists():
            errors.append(
                f"PDF file not found: {self.pdf_path}\n"
                f"  Fix: Set PDF_PATH environment variable to an existing PDF file"
            )
        elif not pdf_file.is_file():
            errors.append(f"PDF path is not a file: {self.pdf_path}")
        elif pdf_file.suffix.lower() != ".pdf":
            errors.append(
                f"File is not a PDF: {self.pdf_path}\n"
                f"  Current extension: {pdf_file.suffix}\n"
                f"  This script currently only supports PDF files"
            )

        # Validate chunk settings
        if self.chunk_size <= 0:
            errors.append(f"CHUNK_SIZE must be positive, got: {self.chunk_size}")
        if self.chunk_overlap < 0:
            errors.append(f"CHUNK_OVERLAP cannot be negative, got: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than CHUNK_SIZE ({self.chunk_size})\n"
                f"  Recommended: CHUNK_OVERLAP should be 10-20% of CHUNK_SIZE"
            )

        # Validate retrieval settings
        if self.top_k <= 0:
            errors.append(f"TOP_K must be positive, got: {self.top_k}")

        # Validate embedding settings
        if self.embed_dim <= 0:
            errors.append(f"EMBED_DIM must be positive, got: {self.embed_dim}")
        if self.embed_batch <= 0:
            errors.append(f"EMBED_BATCH must be positive, got: {self.embed_batch}")

        # Validate LLM settings
        if self.temperature < 0 or self.temperature > 2:
            errors.append(
                f"TEMP should be between 0 and 2, got: {self.temperature}\n"
                f"  Recommended: 0.0-0.1 for factual RAG, 0.7-1.0 for creative tasks"
            )
        if self.max_new_tokens <= 0:
            errors.append(f"MAX_NEW_TOKENS must be positive, got: {self.max_new_tokens}")
        if self.context_window <= 0:
            errors.append(f"CTX must be positive, got: {self.context_window}")
        if self.n_gpu_layers < 0:
            errors.append(f"N_GPU_LAYERS cannot be negative, got: {self.n_gpu_layers}")
        if self.n_batch <= 0:
            errors.append(f"N_BATCH must be positive, got: {self.n_batch}")

        # Validate model configuration
        if self.model_path and not Path(self.model_path).exists():
            errors.append(
                f"MODEL_PATH specified but file not found: {self.model_path}\n"
                f"  Fix: Either download the model or unset MODEL_PATH to auto-download"
            )

        # Validate database settings
        if not self.db_name or not self.db_name.strip():
            errors.append("DB_NAME cannot be empty")
        if not self.table or not self.table.strip():
            errors.append("PGTABLE cannot be empty")

        try:
            port_int = int(self.port)
            if port_int < 1 or port_int > 65535:
                errors.append(f"PGPORT must be between 1 and 65535, got: {port_int}")
        except ValueError:
            errors.append(f"PGPORT must be a valid integer, got: {self.port}")

        if errors:
            error_msg = "\n\n".join([f"❌ {err}" for err in errors])
            raise ValueError(
                f"\n\n{'='*70}\n"
                f"Configuration Validation Failed\n"
                f"{'='*70}\n\n"
                f"{error_msg}\n\n"
                f"{'='*70}\n"
            )


S = Settings()
# Validation will be called in main() after CLI args are parsed


# -----------------------
# DB helpers with retry logic
# -----------------------
def retry_with_backoff(func, max_retries=3, initial_delay=1.0, backoff_factor=2.0, exception_types=(Exception,)):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry (should take no arguments)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exception_types: Tuple of exception types to catch and retry

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except exception_types as e:
            last_exception = e
            if attempt < max_retries - 1:
                log.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                log.error(f"All {max_retries} attempts failed. Giving up.")

    raise last_exception


def admin_conn():
    """
    Connect to the 'postgres' admin DB (useful to create/drop databases).
    Includes retry logic with helpful error messages.
    """
    def _connect():
        try:
            return psycopg2.connect(
                dbname="postgres",
                host=S.host,
                port=S.port,
                user=S.user,
                password=S.password,
                connect_timeout=10,
            )
        except PgOperationalError as e:
            error_msg = str(e).lower()
            if "could not connect" in error_msg or "connection refused" in error_msg:
                raise PgOperationalError(
                    f"Cannot connect to PostgreSQL at {S.host}:{S.port}\n"
                    f"  Fix: Make sure PostgreSQL is running:\n"
                    f"    docker-compose up -d\n"
                    f"    docker-compose ps  # Check status\n"
                    f"  Original error: {e}"
                )
            elif "authentication failed" in error_msg or "password" in error_msg:
                raise PgOperationalError(
                    f"PostgreSQL authentication failed for user '{S.user}'\n"
                    f"  Fix: Check PGUSER and PGPASSWORD environment variables\n"
                    f"  Original error: {e}"
                )
            else:
                raise

    return retry_with_backoff(_connect, max_retries=3, exception_types=(PgOperationalError,))


def db_conn():
    """
    Connect to the target DB where pgvector tables live.
    Includes retry logic with helpful error messages.
    """
    def _connect():
        try:
            return psycopg2.connect(
                dbname=S.db_name,
                host=S.host,
                port=S.port,
                user=S.user,
                password=S.password,
                connect_timeout=10,
            )
        except PgOperationalError as e:
            error_msg = str(e).lower()
            if "database" in error_msg and "does not exist" in error_msg:
                raise PgOperationalError(
                    f"Database '{S.db_name}' does not exist\n"
                    f"  Fix: The script will try to create it automatically, or create manually:\n"
                    f"    docker-compose exec db psql -U {S.user} -c 'CREATE DATABASE {S.db_name};'\n"
                    f"  Original error: {e}"
                )
            elif "could not connect" in error_msg or "connection refused" in error_msg:
                raise PgOperationalError(
                    f"Cannot connect to PostgreSQL at {S.host}:{S.port}\n"
                    f"  Fix: Make sure PostgreSQL is running:\n"
                    f"    docker-compose up -d\n"
                    f"  Original error: {e}"
                )
            else:
                raise

    return retry_with_backoff(_connect, max_retries=3, exception_types=(PgOperationalError,))


def ensure_db_exists():
    """
    For Docker Compose setups where POSTGRES_DB already creates the DB, this is unnecessary.
    But it's safe-ish if user has permissions.
    """
    start = now_ms()
    try:
        conn = admin_conn()
        conn.autocommit = True
        with conn.cursor() as c:
            if S.reset_db:
                log.warning(f"RESET_DB=1 -> Dropping database {S.db_name} (data loss).")
                c.execute(f"DROP DATABASE IF EXISTS {S.db_name}")
            # Create DB (will fail if exists -> we catch below)
            c.execute(f"CREATE DATABASE {S.db_name}")
        conn.close()
        log.info(f"DB ensure/create done in {dur_s(start):.2f}s (created new DB).")
    except Exception as e:
        # Most common: "already exists" or no permission.
        log.info(f"DB ensure/create skipped/failed harmlessly: {type(e).__name__}: {e} ({dur_s(start):.2f}s)")


def ensure_pgvector_extension():
    """
    pgvector must be enabled per database:
      CREATE EXTENSION vector;
    Without it, type 'vector' doesn't exist (your earlier error).
    """
    start = now_ms()
    conn = db_conn()
    conn.autocommit = True
    with conn.cursor() as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.close()
    log.info(f"pgvector extension ensured in {dur_s(start):.2f}s")


def reset_table_if_requested():
    """
    RESET_TABLE=1 drops the vector store table so re-running does not duplicate rows.
    """
    if not S.reset_table:
        log.info("RESET_TABLE=0 -> keeping existing table (may duplicate on re-ingest).")
        return

    start = now_ms()
    conn = db_conn()
    conn.autocommit = True
    with conn.cursor() as c:
        log.warning(f"RESET_TABLE=1 -> Dropping table '{S.table}' if it exists.")
        c.execute(f'DROP TABLE IF EXISTS "{S.table}";')
    conn.close()
    log.info(f"Table reset done in {dur_s(start):.2f}s")


def count_rows() -> Optional[int]:
    """
    Useful to see ingestion effect. If table doesn't exist yet, return None.
    Note: PGVectorStore adds 'data_' prefix to table names.
    """
    try:
        conn = db_conn()
        # PGVectorStore uses "data_{table_name}" format
        actual_table = f"data_{S.table}"
        with conn.cursor() as c:
            c.execute(f'SELECT COUNT(*) FROM "{actual_table}";')
            n = int(c.fetchone()[0])
        conn.close()
        return n
    except psycopg2.errors.UndefinedTable:
        # Table doesn't exist yet - this is expected on first run
        log.debug(f"Table 'data_{S.table}' does not exist yet (normal for first run)")
        return None
    except PgOperationalError as e:
        log.warning(f"Failed to count rows (database connection issue): {e}")
        return None
    except Exception as e:
        log.warning(f"Failed to count rows: {type(e).__name__}: {e}")
        return None


def check_index_configuration() -> Optional[dict]:
    """
    Check the configuration of existing index by sampling stored metadata.
    Returns dict with configuration info, or None if table doesn't exist or is empty.

    This helps detect "mixed index" scenarios where the table contains chunks
    created with different chunk_size/overlap settings.
    """
    try:
        conn = db_conn()
        actual_table = f"data_{S.table}"

        with conn.cursor() as c:
            # Sample some rows to check their metadata
            # We check both old and new rows in case there's a mix
            c.execute(f'''
                SELECT metadata, id FROM "{actual_table}"
                ORDER BY id LIMIT 10
            ''')
            rows = c.fetchall()

        conn.close()

        if not rows:
            return None

        # Extract configurations from sampled rows
        configs = []
        for metadata, row_id in rows:
            if metadata and isinstance(metadata, dict):
                config = {
                    "chunk_size": metadata.get("_chunk_size"),
                    "chunk_overlap": metadata.get("_chunk_overlap"),
                    "embed_model": metadata.get("_embed_model"),
                    "index_signature": metadata.get("_index_signature"),
                }
                configs.append(config)

        if not configs:
            # Old index without metadata - can't determine configuration
            return {"legacy": True, "has_metadata": False}

        # Check if all configs are the same
        first_config = configs[0]
        all_same = all(c == first_config for c in configs)

        result = {
            "legacy": False,
            "has_metadata": True,
            "chunk_size": first_config.get("chunk_size"),
            "chunk_overlap": first_config.get("chunk_overlap"),
            "embed_model": first_config.get("embed_model"),
            "index_signature": first_config.get("index_signature"),
            "is_consistent": all_same,
            "sampled_configs": len(set(str(c) for c in configs)),
        }

        return result

    except psycopg2.errors.UndefinedTable:
        return None
    except Exception as e:
        log.warning(f"Failed to check index configuration: {type(e).__name__}: {e}")
        return None


def save_query_log(
    question: str,
    answer: str,
    retrieved_chunks: List[Any],
    retrieval_time: float,
    generation_time: float,
    parameters: dict
) -> Optional[str]:
    """
    Save query results to a structured JSON log file.

    Args:
        question: The query text
        answer: The generated answer
        retrieved_chunks: List of retrieved NodeWithScore objects
        retrieval_time: Time taken for retrieval (seconds)
        generation_time: Time taken for generation (seconds)
        parameters: Dictionary of all relevant parameters

    Returns:
        Path to the log file, or None if logging is disabled
    """
    # Check if logging is enabled
    if os.getenv("LOG_QUERIES", "0") != "1":
        return None

    try:
        # Create log directory structure
        log_dir = Path("query_logs") / S.table
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"{timestamp}_query.json"

        # Process retrieved chunks
        chunks_data = []
        for i, chunk in enumerate(retrieved_chunks):
            chunk_info = {
                "rank": i + 1,
                "similarity_score": float(chunk.score) if hasattr(chunk, 'score') and chunk.score is not None else None,
                "text": chunk.node.get_content(),
                "text_length": len(chunk.node.get_content()),
                "metadata": chunk.node.metadata if hasattr(chunk.node, 'metadata') else {}
            }
            chunks_data.append(chunk_info)

        # Calculate retrieval quality metrics
        scores = [c["similarity_score"] for c in chunks_data if c["similarity_score"] is not None]
        quality_metrics = {}
        if scores:
            quality_metrics = {
                "best_score": max(scores),
                "worst_score": min(scores),
                "average_score": sum(scores) / len(scores),
                "score_range": max(scores) - min(scores)
            }

        # Build complete log structure
        log_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "document_table": S.table,
                "pdf_path": S.pdf_path,
                "log_file": str(log_file)
            },
            "query": {
                "question": question,
                "question_length": len(question),
                "question_words": len(question.split())
            },
            "parameters": parameters,
            "retrieval": {
                "retrieved_chunks": chunks_data,
                "num_chunks": len(chunks_data),
                "retrieval_time_seconds": round(retrieval_time, 3),
                "quality_metrics": quality_metrics
            },
            "generation": {
                "answer": answer,
                "answer_length": len(answer),
                "answer_words": len(answer.split()),
                "generation_time_seconds": round(generation_time, 3),
                "tokens_per_second": round(len(answer.split()) / generation_time, 2) if generation_time > 0 else 0
            },
            "performance": {
                "total_time_seconds": round(retrieval_time + generation_time, 3),
                "retrieval_percentage": round((retrieval_time / (retrieval_time + generation_time)) * 100, 1) if (retrieval_time + generation_time) > 0 else 0,
                "generation_percentage": round((generation_time / (retrieval_time + generation_time)) * 100, 1) if (retrieval_time + generation_time) > 0 else 0
            }
        }

        # Write to file with pretty formatting
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        log.info(f"\n💾 Query log saved to: {log_file}")
        log.info(f"   • Question: {len(question)} chars")
        log.info(f"   • Answer: {len(answer)} chars")
        log.info(f"   • Chunks: {len(chunks_data)}")
        log.info(f"   • Total time: {retrieval_time + generation_time:.2f}s")

        return str(log_file)

    except Exception as e:
        log.warning(f"Failed to save query log: {type(e).__name__}: {e}")
        return None


# -----------------------
# Retriever with verbose logs (this is where you "see retrieval")
# -----------------------
class VectorDBRetriever(BaseRetriever):
    """
    A retriever is "query -> relevant nodes".
    We implement it manually so you see the real steps:
      - embed query text
      - vector similarity search in Postgres
      - return NodeWithScore objects
    """

    def __init__(self, vector_store: PGVectorStore, embed_model: Any, similarity_top_k: int):
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        self.last_retrieval_time = 0.0  # Track last retrieval time for logging
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        q = query_bundle.query_str
        retrieval_start = now_ms()  # Track total retrieval time

        log.info(f"\n{'='*70}")
        log.info(f"STEP 4: RETRIEVAL - Finding relevant chunks via vector similarity")
        log.info(f"{'='*70}")
        log.info(f"❓ Query: \"{q}\"")

        log.info(f"\n💡 How retrieval works:")
        log.info(f"  1. Convert query to embedding vector (same model as documents)")
        log.info(f"  2. Calculate cosine similarity with all stored vectors")
        log.info(f"  3. Return top-{self._similarity_top_k} most similar chunks")
        log.info(f"  4. Similarity score: 1.0 = identical, 0.0 = unrelated")

        t0 = now_ms()
        q_emb = self._embed_model.get_query_embedding(q)
        log.info(f"\n🔢 Query embedding computed in {dur_s(t0):.2f}s ({len(q_emb)} dimensions)")

        t1 = now_ms()
        vsq = VectorStoreQuery(
            query_embedding=q_emb,
            similarity_top_k=self._similarity_top_k,
            mode="default",
        )
        res = self._vector_store.query(vsq)
        search_time = dur_s(t1)
        log.info(f"🔍 Vector search complete in {search_time:.2f}s")
        log.info(f"  • Searched through stored embeddings")
        log.info(f"  • Found top-{self._similarity_top_k} most similar chunks")

        out: List[NodeWithScore] = []
        for i, node in enumerate(res.nodes):
            score: Optional[float] = res.similarities[i] if res.similarities is not None else None
            nws = NodeWithScore(node=node, score=score)
            out.append(nws)

        # Calculate score distribution for quality insights
        scores = [nws.score for nws in out if isinstance(nws.score, (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score

            log.info(f"\n📊 Retrieval Quality Metrics:")
            log.info(f"  • Best match score: {max_score:.4f}")
            log.info(f"  • Worst match score: {min_score:.4f}")
            log.info(f"  • Average score: {avg_score:.4f}")
            log.info(f"  • Score range: {score_range:.4f}")

            if max_score > 0.8:
                log.info(f"  ✓ Excellent: Found highly relevant chunks (>0.8)")
            elif max_score > 0.6:
                log.info(f"  ✓ Good: Found relevant chunks (0.6-0.8)")
            elif max_score > 0.4:
                log.info(f"  ⚠️  Fair: Moderate relevance (0.4-0.6) - answer may be vague")
            else:
                log.info(f"  ⚠️  Poor: Low relevance (<0.4) - may not answer question")

        # Log retrieved chunks with details
        log.info(f"\n📄 Retrieved Chunks (these will be sent to LLM):")

        # Check if retrieved chunks have configuration metadata
        chunk_configs_seen = set()
        for i, nws in enumerate(out, start=1):
            md = nws.node.metadata or {}
            score_str = f"{nws.score:.4f}" if isinstance(nws.score, (int, float)) else "None"
            page = md.get("page_label") or md.get("page") or md.get("source") or "?"

            # Track chunk configurations
            chunk_config = md.get("_index_signature", "unknown")
            chunk_configs_seen.add(chunk_config)

            # Show configuration for first chunk or if it differs
            config_info = ""
            if i == 1 or len(chunk_configs_seen) > 1:
                chunk_size = md.get("_chunk_size", "?")
                chunk_overlap = md.get("_chunk_overlap", "?")
                config_info = f" [cs={chunk_size}, ov={chunk_overlap}]"

            log.info(f"\n  {i}. Similarity: {score_str} | Source: {page}{config_info}")
            log.info(f"     Text: \"{preview(nws.node.get_content(), 200)}\"")

        # Warn if mixed configurations detected in retrieval
        if len(chunk_configs_seen) > 1:
            log.warning(f"\n  ⚠️  WARNING: Retrieved chunks from {len(chunk_configs_seen)} different configurations!")
            log.warning(f"  ⚠️  This indicates a mixed index - results may be unpredictable")
            log.warning(f"  ⚠️  Consider rebuilding with RESET_TABLE=1")

        # Store total retrieval time for logging
        self.last_retrieval_time = dur_s(retrieval_start)

        return out


# -----------------------
# Main pipeline
# -----------------------
def build_embed_model() -> HuggingFaceEmbedding:
    """
    Embeddings transform text -> vector.
    This is what makes vector search possible.
    """
    log.info(f"Embedding model: {S.embed_model_name} (expected dim={S.embed_dim})")
    t = now_ms()
    model = HuggingFaceEmbedding(model_name=S.embed_model_name)
    log.info(f"Embedding model loaded in {dur_s(t):.2f}s")
    return model


def build_llm() -> LlamaCPP:
    """
    LLM is only for "answer synthesis".
    Retrieval quality is dominated by embedding/chunking/top_k;
    LLM mainly affects answer style + reasoning and speed.
    """
    src = f"MODEL_PATH={S.model_path}" if S.model_path else f"MODEL_URL={S.model_url}"
    log.info(f"LLM: llama.cpp GGUF ({src})")
    log.info(f"LLM params: CTX={S.context_window} MAX_NEW_TOKENS={S.max_new_tokens} TEMP={S.temperature} N_GPU_LAYERS={S.n_gpu_layers} N_BATCH={S.n_batch}")

    llm = LlamaCPP(
        model_url=None if S.model_path else S.model_url,
        model_path=S.model_path or None,
        temperature=S.temperature,
        max_new_tokens=S.max_new_tokens,
        context_window=S.context_window,
        model_kwargs={
            "n_gpu_layers": S.n_gpu_layers,
            "n_batch": S.n_batch,
        },
        verbose=True,  # llama.cpp will emit its own logs too
    )
    return llm


def load_documents(doc_path: str) -> List[Any]:
    """
    Load documents from various formats (PDF, DOCX, TXT, MD).
    Returns LlamaIndex Documents.
    """
    path = Path(doc_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {doc_path}\n"
            f"  Fix: Place your document at this path or set PDF_PATH environment variable\n"
            f"  Example: PDF_PATH=/path/to/your/document.pdf python {sys.argv[0]}"
        )

    ext = path.suffix.lower()
    file_size_mb = path.stat().st_size / (1024 * 1024)

    log.info(f"Loading document: {doc_path}")
    log.info(f"  Format: {ext}, Size: {file_size_mb:.1f} MB")

    t = now_ms()
    docs = []

    try:
        if ext == ".pdf":
            # PDF: Use PyMuPDFReader (page-per-document)
            docs = PyMuPDFReader().load(file_path=str(path))

        elif ext == ".docx":
            # DOCX: Load as single document
            try:
                from docx import Document as DocxDocument
            except ImportError:
                raise ImportError(
                    "python-docx not installed. Install with: pip install python-docx"
                )

            docx_doc = DocxDocument(str(path))
            text = "\n".join([p.text for p in docx_doc.paragraphs])

            from llama_index.core.schema import Document
            docs = [Document(text=text, metadata={"source": str(path), "format": "docx"})]

        elif ext in (".txt", ".md"):
            # TXT/MD: Load as single document
            text = path.read_text(encoding="utf-8", errors="replace")

            from llama_index.core.schema import Document
            docs = [Document(text=text, metadata={"source": str(path), "format": ext[1:]})]

        else:
            raise ValueError(
                f"Unsupported file format: {ext}\n"
                f"  Supported formats: .pdf, .docx, .txt, .md\n"
                f"  File: {doc_path}"
            )

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError, ImportError)):
            raise
        raise RuntimeError(
            f"Failed to load document: {doc_path}\n"
            f"  Error: {type(e).__name__}: {e}\n"
            f"  Fix: Ensure the file is valid and not corrupted"
        )

    if not docs:
        raise ValueError(
            f"Document loaded but contains no content: {doc_path}\n"
            f"  The document may be empty or corrupted"
        )

    # Calculate total text statistics
    total_chars = sum(len(doc.text) for doc in docs)
    total_words = sum(len(doc.text.split()) for doc in docs)
    avg_chars_per_doc = total_chars / len(docs) if docs else 0

    log.info(f"✓ Loaded {len(docs)} document(s) in {dur_s(t):.2f}s")
    log.info(f"  📊 Statistics:")
    log.info(f"    • Total characters: {total_chars:,}")
    log.info(f"    • Total words: {total_words:,}")
    log.info(f"    • Avg chars/doc: {avg_chars_per_doc:.0f}")
    if ext == ".pdf":
        log.info(f"  ℹ️  Note: PDFs are split per-page for better citation accuracy")
    else:
        log.info(f"  ℹ️  Note: {ext.upper()} loaded as single document for context preservation")

    return docs


def chunk_documents(docs: List[Any]) -> Tuple[List[str], List[int]]:
    """
    Turn documents into text chunks.
    We keep doc_idxs so each chunk knows which doc/page it came from (for metadata/citations).
    """
    log.info(f"\n{'='*70}")
    log.info(f"STEP 1: CHUNKING - Breaking documents into retrievable pieces")
    log.info(f"{'='*70}")
    log.info(f"⚙️  Configuration:")
    log.info(f"  • chunk_size={S.chunk_size} chars")
    log.info(f"  • chunk_overlap={S.chunk_overlap} chars ({S.chunk_overlap/S.chunk_size*100:.1f}% overlap)")

    overlap_ratio = S.chunk_overlap / S.chunk_size
    if overlap_ratio < 0.1:
        log.info(f"  ⚠️  Low overlap may break context across chunks")
    elif overlap_ratio > 0.3:
        log.info(f"  ℹ️  High overlap preserves context but increases storage")
    else:
        log.info(f"  ✓ Good overlap ratio for context preservation")

    log.info(f"\n💡 Why chunking matters:")
    log.info(f"  • Smaller chunks = more precise retrieval, less context")
    log.info(f"  • Larger chunks = more context, less precise matching")
    log.info(f"  • Overlap prevents splitting important information")

    splitter = SentenceSplitter(chunk_size=S.chunk_size, chunk_overlap=S.chunk_overlap)

    t = now_ms()
    chunks: List[str] = []
    doc_idxs: List[int] = []
    chunk_sizes: List[int] = []

    for doc_idx, doc in enumerate(docs):
        # Split this page's text into chunks
        cs = splitter.split_text(doc.text)
        chunks.extend(cs)
        doc_idxs.extend([doc_idx] * len(cs))
        chunk_sizes.extend([len(c) for c in cs])

        # Lightweight progress log every ~25 pages
        if (doc_idx + 1) % 25 == 0:
            log.info(f"  📄 Processed {doc_idx+1}/{len(docs)} documents → {len(chunks)} chunks so far")

    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
    max_chunk_size = max(chunk_sizes) if chunk_sizes else 0

    log.info(f"\n✓ Chunking complete in {dur_s(t):.2f}s")
    log.info(f"  📊 Chunk Statistics:")
    log.info(f"    • Total chunks: {len(chunks)}")
    log.info(f"    • Avg chunk size: {avg_chunk_size:.0f} chars")
    log.info(f"    • Min chunk size: {min_chunk_size} chars")
    log.info(f"    • Max chunk size: {max_chunk_size} chars")
    log.info(f"    • Chunks per document: {len(chunks)/len(docs):.1f} average")

    if chunks:
        log.info(f"\n  📝 Example chunk (first):")
        log.info(f"    \"{preview(chunks[0], 150)}\"")

    return chunks, doc_idxs


def build_nodes(docs: List[Any], chunks: List[str], doc_idxs: List[int]) -> List[TextNode]:
    """
    Create TextNode objects (text + metadata).
    Embedding is added later.

    IMPORTANT: We store chunking parameters in metadata so we can:
    - Detect mixed-index scenarios (different chunk_size in same table)
    - Debug retrieval quality issues
    - Track what configuration produced each chunk
    """
    log.info("Building TextNode objects (text + metadata)")
    t = now_ms()

    # Create an index signature to track chunking configuration
    index_signature = f"cs{S.chunk_size}_ov{S.chunk_overlap}_{S.embed_model_name.replace('/', '_')}"

    nodes: List[TextNode] = []
    for i, chunk in enumerate(chunks):
        n = TextNode(text=chunk)

        # Start with source document metadata
        src_doc = docs[doc_idxs[i]]
        n.metadata = src_doc.metadata.copy() if src_doc.metadata else {}

        # Add chunking parameters to metadata
        # This allows detection of mixed indexes and better debugging
        n.metadata["_chunk_size"] = S.chunk_size
        n.metadata["_chunk_overlap"] = S.chunk_overlap
        n.metadata["_embed_model"] = S.embed_model_name
        n.metadata["_index_signature"] = index_signature

        nodes.append(n)

        if (i + 1) % 500 == 0:
            log.info(f"  built {i+1}/{len(chunks)} nodes")

    log.info(f"Built {len(nodes)} nodes in {dur_s(t):.2f}s")
    log.info(f"Index signature: {index_signature}")
    return nodes


def embed_nodes(embed_model: HuggingFaceEmbedding, nodes: List[TextNode]) -> None:
    """
    Compute embeddings for each node.
    This is often the longest step after LLM inference.
    We do batching for speed and steadier memory usage.
    """
    log.info(f"\n{'='*70}")
    log.info(f"STEP 2: EMBEDDING - Converting text to semantic vectors")
    log.info(f"{'='*70}")
    log.info(f"⚙️  Configuration:")
    log.info(f"  • Model: {S.embed_model_name}")
    log.info(f"  • Dimension: {S.embed_dim} (vector size)")
    log.info(f"  • Batch size: {S.embed_batch} chunks/batch")
    log.info(f"  • Total nodes: {len(nodes)}")

    log.info(f"\n💡 What are embeddings?")
    log.info(f"  • Embeddings convert text into {S.embed_dim}-dimensional vectors")
    log.info(f"  • Similar text → similar vectors (measured by cosine similarity)")
    log.info(f"  • This enables semantic search (meaning-based, not just keywords)")
    log.info(f"  • Example: 'cat' and 'feline' have similar embeddings")

    total_bytes = len(nodes) * S.embed_dim * 4  # 4 bytes per float32
    log.info(f"\n📊 Storage Impact:")
    log.info(f"  • {len(nodes)} vectors × {S.embed_dim} dims × 4 bytes = {total_bytes/1024/1024:.1f} MB")

    t = now_ms()
    total = len(nodes)

    # We embed only the node text (metadata_mode="none") to keep embeddings "pure".
    # You can switch to metadata_mode="all" to include metadata in embeddings (sometimes helps, sometimes adds noise).
    texts = [n.get_content(metadata_mode="none") for n in nodes]

    done = 0
    batches = list(chunked(texts, S.embed_batch))

    log.info(f"\n🔄 Processing {len(batches)} batches...")

    # Use tqdm if available for better progress visualization
    iterator = tqdm(enumerate(batches, start=1), total=len(batches), desc="Embedding", unit="batch") if tqdm else enumerate(batches, start=1)

    batch_times = []
    for batch_idx, batch_texts in iterator:
        tb = now_ms()

        # Newer versions of HuggingFaceEmbedding support batch embedding
        # If not, this will throw and we fallback to per-item embedding.
        try:
            batch_embs = embed_model.get_text_embedding_batch(batch_texts)
        except Exception:
            batch_embs = [embed_model.get_text_embedding(x) for x in batch_texts]

        # Write embeddings back into the nodes (must align exactly)
        start_i = (batch_idx - 1) * S.embed_batch
        for j, emb in enumerate(batch_embs):
            nodes[start_i + j].embedding = emb

        done += len(batch_texts)
        batch_time = dur_s(tb)
        batch_times.append(batch_time)

        # Progress logs: every batch (but less verbose if using tqdm)
        if not tqdm:
            rate = done / max(dur_s(t), 1e-6)
            log.info(f"  📦 Batch {batch_idx:04d} → {done}/{total} nodes | {batch_time:.2f}s | ~{rate:.1f} nodes/s")
        elif batch_idx % 10 == 0:  # Log every 10 batches with tqdm
            rate = done / max(dur_s(t), 1e-6)
            log.debug(f"  Progress: {done}/{total} nodes | ~{rate:.1f} nodes/s")

    total_time = dur_s(t)
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    throughput = total / total_time if total_time > 0 else 0

    log.info(f"\n✓ Embeddings complete in {total_time:.2f}s")
    log.info(f"  📊 Performance:")
    log.info(f"    • Average batch time: {avg_batch_time:.2f}s")
    log.info(f"    • Throughput: {throughput:.1f} nodes/second")
    log.info(f"    • Time per node: {total_time/total*1000:.1f}ms")
    log.info(f"\n  ℹ️  These embeddings enable semantic similarity search in the next step")


def make_vector_store() -> PGVectorStore:
    """
    Create the vector store client that uses Postgres + pgvector.
    """
    log.info(f"Connecting to Postgres vector store: db={S.db_name} host={S.host}:{S.port} user={S.user} table={S.table}")
    store = PGVectorStore.from_params(
        database=S.db_name,
        host=S.host,
        port=S.port,
        user=S.user,
        password=S.password,
        table_name=S.table,
        embed_dim=S.embed_dim,
    )
    return store


def insert_nodes(vector_store: PGVectorStore, nodes: List[TextNode]) -> None:
    """
    Insert nodes into Postgres.
    Depending on LlamaIndex version, PGVectorStore.add may do its own batching internally.
    We'll still batch to:
      - keep transactions smaller
      - show progress clearly
    """
    log.info(f"\n{'='*70}")
    log.info(f"STEP 3: STORAGE - Saving embeddings to PostgreSQL + pgvector")
    log.info(f"{'='*70}")

    before = count_rows()
    if before is not None:
        log.info(f"📊 Current table state: {before} rows exist")

    batch_size = int(os.getenv("DB_INSERT_BATCH", "250"))
    log.info(f"⚙️  Configuration:")
    log.info(f"  • Table: {S.table}")
    log.info(f"  • Batch size: {batch_size} nodes/batch")
    log.info(f"  • Total to insert: {len(nodes)} nodes")

    log.info(f"\n💡 Why use pgvector?")
    log.info(f"  • Specialized PostgreSQL extension for vector similarity search")
    log.info(f"  • Efficient indexing (IVFFlat, HNSW) for fast retrieval")
    log.info(f"  • Supports cosine distance, L2 distance, inner product")
    log.info(f"  • Production-ready: ACID compliance + SQL features")

    t = now_ms()
    total = len(nodes)
    inserted = 0

    batches = list(chunked(nodes, batch_size))
    log.info(f"\n🔄 Inserting {len(batches)} batches...")
    iterator = tqdm(enumerate(batches, start=1), total=len(batches), desc="Inserting", unit="batch") if tqdm else enumerate(batches, start=1)

    for bidx, batch in iterator:
        tb = now_ms()
        vector_store.add(batch)
        inserted += len(batch)

        if not tqdm:
            log.info(f"  💾 Batch {bidx:04d} → {inserted}/{total} nodes | {dur_s(tb):.2f}s")

    total_time = dur_s(t)
    throughput = total / total_time if total_time > 0 else 0

    after = count_rows()
    log.info(f"\n✓ Storage complete in {total_time:.2f}s")
    log.info(f"  📊 Results:")
    log.info(f"    • Inserted: {total} nodes")
    log.info(f"    • Throughput: {throughput:.1f} nodes/second")
    if after is not None:
        log.info(f"    • Total rows in table: {after}")
        if before is not None:
            log.info(f"    • New rows added: {after - before}")
    log.info(f"\n  ✓ Vector index is now ready for semantic search!")


def health_check() -> None:
    """
    Perform startup health checks to catch common issues early.
    Better to fail fast with clear errors than fail later with cryptic ones.
    """
    log.info("=== HEALTH CHECK ===")
    checks_passed = 0
    checks_total = 0

    # Check 1: PostgreSQL connectivity
    checks_total += 1
    try:
        log.info("✓ Checking PostgreSQL connection...")
        conn = admin_conn()
        conn.close()
        log.info("  ✓ PostgreSQL is reachable")
        checks_passed += 1
    except Exception as e:
        log.error(f"  ✗ PostgreSQL connection failed: {e}")
        log.error("    Fix: Ensure PostgreSQL is running (docker-compose up -d)")

    # Check 2: PDF file exists
    checks_total += 1
    try:
        log.info("✓ Checking PDF file...")
        if Path(S.pdf_path).exists():
            size_mb = os.path.getsize(S.pdf_path) / (1024 * 1024)
            log.info(f"  ✓ PDF file exists ({size_mb:.1f} MB)")
            checks_passed += 1
        else:
            log.error(f"  ✗ PDF file not found: {S.pdf_path}")
    except Exception as e:
        log.error(f"  ✗ PDF check failed: {e}")

    # Check 3: Disk space (if psutil available)
    if psutil:
        checks_total += 1
        try:
            log.info("✓ Checking disk space...")
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            if free_gb < 2:
                log.warning(f"  ⚠ Low disk space: {free_gb:.1f} GB free (recommend at least 5GB)")
            else:
                log.info(f"  ✓ Disk space OK ({free_gb:.1f} GB free)")
                checks_passed += 1
        except Exception as e:
            log.warning(f"  ⚠ Disk space check failed: {e}")

    # Check 4: Memory availability (if psutil available)
    if psutil:
        checks_total += 1
        try:
            log.info("✓ Checking memory...")
            vm = psutil.virtual_memory()
            avail_gb = vm.available / (1024**3)
            if avail_gb < 2:
                log.warning(f"  ⚠ Low memory: {avail_gb:.1f} GB available (recommend at least 4GB)")
                log.warning("    Consider closing other applications or reducing N_GPU_LAYERS")
            else:
                log.info(f"  ✓ Memory OK ({avail_gb:.1f} GB available)")
                checks_passed += 1
        except Exception as e:
            log.warning(f"  ⚠ Memory check failed: {e}")

    log.info(f"Health check: {checks_passed}/{checks_total} checks passed")

    if checks_passed < 2:  # At minimum need DB and PDF
        log.error("Critical health checks failed. Cannot proceed.")
        log.error("Please fix the issues above and try again.")
        sys.exit(1)
    elif checks_passed < checks_total:
        log.warning("Some health checks failed, but proceeding anyway...")
        time.sleep(2)  # Give user time to see warnings

    log.info("Health check complete!\n")


def run_query(query_engine: Any, question: str, show_sources: bool = True, retrieval_time: float = 0.0) -> None:
    """Execute a single query and display results."""
    log.info(f"\n{'='*70}")
    log.info(f"STEP 5: GENERATION - LLM synthesizes answer from retrieved chunks")
    log.info(f"{'='*70}")

    log.info(f"\n💡 How answer generation works:")
    log.info(f"  1. Retrieved chunks are combined into context")
    log.info(f"  2. Context + query sent to local LLM")
    log.info(f"  3. LLM generates answer grounded in provided context")
    log.info(f"  4. This prevents hallucination (making up information)")

    log.info(f"\n⚙️  LLM Configuration:")
    log.info(f"  • Model: Mistral 7B Instruct (Q4_K_M quantized)")
    log.info(f"  • Temperature: {S.temperature} ({'factual/deterministic' if S.temperature < 0.3 else 'balanced' if S.temperature < 0.7 else 'creative'})")
    log.info(f"  • Max tokens: {S.max_new_tokens}")
    log.info(f"  • Context window: {S.context_window}")

    log.info(f"\n🔄 Generating answer...")
    t = now_ms()
    resp = query_engine.query(question)
    generation_time = dur_s(t)

    # Calculate token statistics
    answer_length = len(str(resp))
    answer_words = len(str(resp).split())
    tokens_per_second = answer_words / generation_time if generation_time > 0 else 0

    log.info(f"\n✓ Answer generated in {generation_time:.2f}s")
    log.info(f"  📊 Generation Stats:")
    log.info(f"    • Answer length: {answer_length} characters, {answer_words} words")
    log.info(f"    • Speed: ~{tokens_per_second:.1f} words/second")
    log.info(f"    • Time per word: {generation_time/answer_words*1000:.0f}ms") if answer_words > 0 else None

    print("\n" + "="*70)
    print("✨ FINAL ANSWER:")
    print("="*70)
    print(str(resp))
    print("="*70 + "\n")

    # Show sources for transparency
    if show_sources and resp.source_nodes:
        log.info(f"\n📚 Sources Used ({len(resp.source_nodes)} chunks):")
        for i, node in enumerate(resp.source_nodes[:3], 1):  # Show top 3
            md = node.node.metadata or {}
            page = md.get("page_label") or md.get("page") or md.get("source") or "?"
            score = node.score if hasattr(node, 'score') else None
            score_str = f" (similarity: {score:.4f})" if score else ""
            log.info(f"  {i}. Source: {page}{score_str}")
            log.info(f"     \"{preview(node.node.get_content(), 150)}\"")
        if len(resp.source_nodes) > 3:
            log.info(f"  ... and {len(resp.source_nodes) - 3} more chunks")

        log.info(f"\n  ℹ️  Answer is grounded in these retrieved chunks")

    # Save query log if enabled
    if os.getenv("LOG_QUERIES", "0") == "1":
        parameters = {
            "chunk_size": S.chunk_size,
            "chunk_overlap": S.chunk_overlap,
            "top_k": S.top_k,
            "temperature": S.temperature,
            "max_new_tokens": S.max_new_tokens,
            "context_window": S.context_window,
            "n_gpu_layers": S.n_gpu_layers,
            "n_batch": S.n_batch,
            "embed_model": S.embed_model_name,
            "embed_dim": S.embed_dim,
            "embed_batch": S.embed_batch
        }

        save_query_log(
            question=question,
            answer=str(resp),
            retrieved_chunks=resp.source_nodes if hasattr(resp, 'source_nodes') else [],
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            parameters=parameters
        )


def interactive_mode(query_engine: Any, retriever: VectorDBRetriever) -> None:
    """
    Interactive REPL mode for asking multiple questions.
    Type 'exit' or 'quit' to end session.
    """
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Ask questions about your documents. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            question = input("Question: ").strip()

            if not question:
                continue

            if question.lower() in ('exit', 'quit', 'q'):
                print("Goodbye!")
                break

            run_query(query_engine, question, show_sources=False, retrieval_time=retriever.last_retrieval_time)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            log.error(f"Query failed: {type(e).__name__}: {e}")
            print(f"Error: {e}\n")


def main():
    # Parse CLI arguments first
    args = parse_args()

    # Override settings from CLI args
    if args.doc:
        S.pdf_path = args.doc
    if args.query:
        S.question = args.query

    # Validate settings (unless explicitly skipped)
    if not args.skip_validation:
        S.validate()

    log_system_info()

    log.info("=== SETTINGS ===")
    log.info(f"DB: postgresql://{S.user}:***@{S.host}:{S.port}/{S.db_name} table={S.table}")
    log.info(f"Document: {S.pdf_path}")
    log.info(f"Chunking: chunk_size={S.chunk_size} overlap={S.chunk_overlap}")
    log.info(f"Retrieval: TOP_K={S.top_k}")
    log.info(f"Embeddings: model={S.embed_model_name} dim={S.embed_dim} batch={S.embed_batch}")
    log.info(f"LLM: CTX={S.context_window} MAX_NEW_TOKENS={S.max_new_tokens} TEMP={S.temperature} N_GPU_LAYERS={S.n_gpu_layers} N_BATCH={S.n_batch}")
    log.info(f"Mode: {'Query-only' if args.query_only else 'Interactive' if args.interactive else 'Full pipeline'}")

    # Suggest configuration-specific table names for experimentation
    if S.table in ["llama2_paper", "ethical-slut_paper"]:  # Default table names
        suggested_table = f"{S.table}_cs{S.chunk_size}_ov{S.chunk_overlap}"
        log.info("")
        log.info("💡 TIP: When experimenting with chunk_size, use config-specific table names:")
        log.info(f"   PGTABLE={suggested_table}")
        log.info("   This keeps each configuration in a separate, clean index.")

    if not args.skip_validation:
        # Run health checks before starting heavy operations
        health_check()

    # --- DB prep ---
    ensure_db_exists()
    ensure_pgvector_extension()

    # --- Ingestion pipeline (skip if query-only mode) ---
    if not args.query_only:
        log.info("=== INDEXING DOCUMENTS ===")

        # Check if table already has data
        existing_rows = count_rows()
        if existing_rows and existing_rows > 0 and not S.reset_table:
            log.warning(f"Table '{S.table}' already contains {existing_rows} rows")

            # Check the configuration of existing index
            existing_config = check_index_configuration()
            current_signature = f"cs{S.chunk_size}_ov{S.chunk_overlap}_{S.embed_model_name.replace('/', '_')}"

            if existing_config:
                if existing_config.get("legacy") or not existing_config.get("has_metadata"):
                    log.warning("  ⚠️  LEGACY INDEX: Existing rows don't have configuration metadata")
                    log.warning("  ⚠️  Cannot detect if chunk_size matches current settings")
                elif not existing_config.get("is_consistent"):
                    log.error("  ❌ MIXED INDEX DETECTED: Table contains chunks from multiple configurations!")
                    log.error("  ❌ This will produce unreliable retrieval results")
                    log.error("  ❌ You should set RESET_TABLE=1 to clean up")
                elif existing_config.get("index_signature") != current_signature:
                    log.error("  ❌ CONFIGURATION MISMATCH DETECTED!")
                    log.error(f"  Current config: chunk_size={S.chunk_size}, overlap={S.chunk_overlap}, model={S.embed_model_name}")
                    log.error(f"  Existing index: chunk_size={existing_config.get('chunk_size')}, overlap={existing_config.get('chunk_overlap')}, model={existing_config.get('embed_model')}")
                    log.error("")
                    log.error("  ❌ Proceeding will create a MIXED INDEX (same table, different chunk sizes)")
                    log.error("  ❌ This causes retrieval to return chunks from BOTH configurations")
                    log.error("  ❌ The chunk_size parameter will appear to 'do nothing' because old chunks still exist")
                    log.error("")
                    log.error("  FIX OPTIONS:")
                    log.error("    1. Set RESET_TABLE=1 to rebuild with new config")
                    log.error(f"    2. Use different table name: PGTABLE=ethical-slut_cs{S.chunk_size}_ov{S.chunk_overlap}")
                    log.error("    3. Use --query-only to query existing index without adding more rows")
                    log.error("")
                    sys.exit(1)
                else:
                    log.info(f"  ✓ Configuration matches existing index: {existing_config.get('index_signature')}")

            log.warning("  Set RESET_TABLE=1 to re-index, or use --query-only to skip indexing")
            log.warning("  Proceeding will add MORE rows (incremental indexing)")

        reset_table_if_requested()

        # Load embedding model for indexing
        embed_model = build_embed_model()

        # --- Vector store client ---
        vector_store = make_vector_store()

        # --- Document loading and processing ---
        docs = load_documents(S.pdf_path)
        chunks, doc_idxs = chunk_documents(docs)
        nodes = build_nodes(docs, chunks, doc_idxs)

        embed_nodes(embed_model, nodes)
        insert_nodes(vector_store, nodes)
    else:
        log.info("=== SKIPPING INDEXING (query-only mode) ===")
        # Still need embed model for queries
        embed_model = build_embed_model()
        vector_store = make_vector_store()

        # Verify table exists and has data
        existing_rows = count_rows()
        if not existing_rows or existing_rows == 0:
            log.error(f"Table '{S.table}' is empty or doesn't exist!")
            log.error("  You must index documents first (run without --query-only)")
            sys.exit(1)
        log.info(f"Using existing index with {existing_rows} rows")

        # Check if current parameters match the indexed data
        existing_config = check_index_configuration()
        current_signature = f"cs{S.chunk_size}_ov{S.chunk_overlap}_{S.embed_model_name.replace('/', '_')}"

        if existing_config:
            if existing_config.get("legacy") or not existing_config.get("has_metadata"):
                log.warning("  ⚠️  Querying LEGACY INDEX (no configuration metadata)")
                log.warning("  ⚠️  Cannot verify if chunk_size/overlap match query expectations")
            elif not existing_config.get("is_consistent"):
                log.warning("  ⚠️  MIXED INDEX: Table contains chunks from multiple configurations")
                log.warning("  ⚠️  Results may be inconsistent - consider rebuilding with RESET_TABLE=1")
            elif existing_config.get("index_signature") != current_signature:
                log.warning("  ⚠️  CONFIGURATION MISMATCH:")
                log.warning(f"    Query params: chunk_size={S.chunk_size}, overlap={S.chunk_overlap}")
                log.warning(f"    Index params: chunk_size={existing_config.get('chunk_size')}, overlap={existing_config.get('chunk_overlap')}")
                log.warning("  ⚠️  The CHUNK_SIZE you set doesn't affect retrieval - it only affects indexing!")
                log.warning("  ⚠️  You're querying chunks that were created with the index configuration above")
                log.warning(f"  💡 To query with chunk_size={S.chunk_size}, you need to re-index with RESET_TABLE=1")
            else:
                log.info(f"  ✓ Query configuration matches index: {existing_config.get('index_signature')}")

    # --- Build query engine ---
    log.info("=== LOADING LLM ===")
    llm = build_llm()

    retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=S.top_k)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    # --- Query mode selection ---
    if args.interactive:
        # Interactive REPL mode
        interactive_mode(query_engine, retriever)
    else:
        # Single query mode
        log.info("=== QUESTION ===")
        run_query(query_engine, S.question, retrieval_time=retriever.last_retrieval_time)


if __name__ == "__main__":
    main()
