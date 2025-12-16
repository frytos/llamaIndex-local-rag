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
from dataclasses import dataclass
from typing import Any, List, Optional, Iterable, Tuple
from pathlib import Path

import psycopg2
from psycopg2 import OperationalError as PgOperationalError

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
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

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
S.validate()  # Validate settings immediately after creation


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
    """
    try:
        conn = db_conn()
        with conn.cursor() as c:
            c.execute(f'SELECT COUNT(*) FROM "{S.table}";')
            n = int(c.fetchone()[0])
        conn.close()
        return n
    except psycopg2.errors.UndefinedTable:
        # Table doesn't exist yet - this is expected on first run
        log.debug(f"Table '{S.table}' does not exist yet (normal for first run)")
        return None
    except PgOperationalError as e:
        log.warning(f"Failed to count rows (database connection issue): {e}")
        return None
    except Exception as e:
        log.warning(f"Failed to count rows: {type(e).__name__}: {e}")
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
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        q = query_bundle.query_str
        log.info(f"[RETRIEVE] Query: {q!r}")

        t0 = now_ms()
        q_emb = self._embed_model.get_query_embedding(q)
        log.debug(f"[RETRIEVE] Query embedding dim={len(q_emb)} computed in {dur_s(t0):.2f}s")

        t1 = now_ms()
        vsq = VectorStoreQuery(
            query_embedding=q_emb,
            similarity_top_k=self._similarity_top_k,
            mode="default",
        )
        res = self._vector_store.query(vsq)
        log.info(f"[RETRIEVE] Vector search top_k={self._similarity_top_k} returned {len(res.nodes)} nodes in {dur_s(t1):.2f}s")

        out: List[NodeWithScore] = []
        for i, node in enumerate(res.nodes):
            score: Optional[float] = res.similarities[i] if res.similarities is not None else None
            nws = NodeWithScore(node=node, score=score)
            out.append(nws)

        # Log a compact view of what was retrieved (super important for learning RAG)
        for i, nws in enumerate(out, start=1):
            md = nws.node.metadata or {}
            score_str = f"{nws.score:.4f}" if isinstance(nws.score, (int, float)) else "None"
            # Some PDFs include page info in metadata depending on reader
            page = md.get("page_label") or md.get("page") or md.get("source") or "?"
            log.info(f"[RETRIEVE]  {i}. score={score_str} page={page} text='{preview(nws.node.get_content())}'")

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


def load_pdf(pdf_path: str) -> List[Any]:
    """
    Load PDF pages into LlamaIndex Documents.
    Usually each page becomes one Document.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}\n"
            f"  Fix: Place your PDF at this path or set PDF_PATH environment variable\n"
            f"  Example: PDF_PATH=/path/to/your/document.pdf python {sys.argv[0]}"
        )

    log.info(f"Loading PDF: {pdf_path}")
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    log.info(f"  File size: {file_size_mb:.1f} MB")

    t = now_ms()
    try:
        docs = PyMuPDFReader().load(file_path=pdf_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load PDF: {pdf_path}\n"
            f"  Error: {type(e).__name__}: {e}\n"
            f"  Fix: Ensure the file is a valid PDF (not corrupted)\n"
            f"  Try opening it in a PDF viewer to verify"
        )

    if not docs:
        raise ValueError(
            f"PDF loaded but contains no pages: {pdf_path}\n"
            f"  The PDF may be empty or corrupted"
        )

    log.info(f"Loaded {len(docs)} documents/pages in {dur_s(t):.2f}s")
    return docs


def chunk_documents(docs: List[Any]) -> Tuple[List[str], List[int]]:
    """
    Turn documents into text chunks.
    We keep doc_idxs so each chunk knows which doc/page it came from (for metadata/citations).
    """
    log.info(f"Chunking: chunk_size={S.chunk_size} chunk_overlap={S.chunk_overlap}")
    splitter = SentenceSplitter(chunk_size=S.chunk_size, chunk_overlap=S.chunk_overlap)

    t = now_ms()
    chunks: List[str] = []
    doc_idxs: List[int] = []
    for doc_idx, doc in enumerate(docs):
        # Split this page's text into chunks
        cs = splitter.split_text(doc.text)
        chunks.extend(cs)
        doc_idxs.extend([doc_idx] * len(cs))

        # Lightweight progress log every ~25 pages
        if (doc_idx + 1) % 25 == 0:
            log.info(f"  chunked {doc_idx+1}/{len(docs)} pages -> total_chunks_so_far={len(chunks)}")

    log.info(f"Chunking produced {len(chunks)} chunks in {dur_s(t):.2f}s")
    if chunks:
        log.debug(f"Example chunk: '{preview(chunks[0])}'")
    return chunks, doc_idxs


def build_nodes(docs: List[Any], chunks: List[str], doc_idxs: List[int]) -> List[TextNode]:
    """
    Create TextNode objects (text + metadata).
    Embedding is added later.
    """
    log.info("Building TextNode objects (text + metadata)")
    t = now_ms()

    nodes: List[TextNode] = []
    for i, chunk in enumerate(chunks):
        n = TextNode(text=chunk)

        # Metadata (usually contains source file and page label/number)
        src_doc = docs[doc_idxs[i]]
        n.metadata = src_doc.metadata

        nodes.append(n)

        if (i + 1) % 500 == 0:
            log.info(f"  built {i+1}/{len(chunks)} nodes")

    log.info(f"Built {len(nodes)} nodes in {dur_s(t):.2f}s")
    return nodes


def embed_nodes(embed_model: HuggingFaceEmbedding, nodes: List[TextNode]) -> None:
    """
    Compute embeddings for each node.
    This is often the longest step after LLM inference.
    We do batching for speed and steadier memory usage.
    """
    log.info(f"Embedding nodes: batch_size={S.embed_batch} (this can take a while)")
    t = now_ms()
    total = len(nodes)

    # We embed only the node text (metadata_mode="none") to keep embeddings "pure".
    # You can switch to metadata_mode="all" to include metadata in embeddings (sometimes helps, sometimes adds noise).
    texts = [n.get_content(metadata_mode="none") for n in nodes]

    done = 0
    for batch_idx, batch_texts in enumerate(chunked(texts, S.embed_batch), start=1):
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

        # Progress logs: every batch
        rate = done / max(dur_s(t), 1e-6)
        log.info(f"  embed batch {batch_idx:04d} -> {done}/{total} nodes | {dur_s(tb):.2f}s batch | ~{rate:.1f} nodes/s")

    log.info(f"Embeddings complete in {dur_s(t):.2f}s")


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
    log.info("Inserting nodes into Postgres (vector_store.add)")
    before = count_rows()
    if before is not None:
        log.info(f"Rows before insert: {before}")

    t = now_ms()
    total = len(nodes)
    batch_size = int(os.getenv("DB_INSERT_BATCH", "250"))  # tweak if needed
    inserted = 0

    for bidx, batch in enumerate(chunked(nodes, batch_size), start=1):
        tb = now_ms()
        vector_store.add(batch)
        inserted += len(batch)
        log.info(f"  db insert batch {bidx:04d} -> {inserted}/{total} nodes | {dur_s(tb):.2f}s batch")

    after = count_rows()
    log.info(f"Insert complete in {dur_s(t):.2f}s")
    if after is not None:
        log.info(f"Rows after insert: {after}")


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


def main():
    log_system_info()

    log.info("=== SETTINGS ===")
    log.info(f"DB: postgresql://{S.user}:***@{S.host}:{S.port}/{S.db_name} table={S.table}")
    log.info(f"PDF_PATH: {S.pdf_path}")
    log.info(f"Chunking: chunk_size={S.chunk_size} overlap={S.chunk_overlap}")
    log.info(f"Retrieval: TOP_K={S.top_k}")
    log.info(f"Embeddings: model={S.embed_model_name} dim={S.embed_dim} batch={S.embed_batch}")
    log.info(f"LLM: CTX={S.context_window} MAX_NEW_TOKENS={S.max_new_tokens} TEMP={S.temperature} N_GPU_LAYERS={S.n_gpu_layers} N_BATCH={S.n_batch}")
    log.info(f"Resets: RESET_TABLE={int(S.reset_table)} RESET_DB={int(S.reset_db)}")

    # Run health checks before starting heavy operations
    health_check()

    # --- DB prep ---
    ensure_db_exists()
    ensure_pgvector_extension()
    reset_table_if_requested()

    # --- Models ---
    embed_model = build_embed_model()
    llm = build_llm()

    # --- Vector store client ---
    vector_store = make_vector_store()

    # --- Ingestion pipeline ---
    docs = load_pdf(S.pdf_path)
    chunks, doc_idxs = chunk_documents(docs)
    nodes = build_nodes(docs, chunks, doc_idxs)

    embed_nodes(embed_model, nodes)
    insert_nodes(vector_store, nodes)

    # --- Retrieval + Answer pipeline ---
    retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=S.top_k)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    log.info("=== QUESTION ===")
    log.info(S.question)

    t = now_ms()
    resp = query_engine.query(S.question)
    log.info(f"LLM answered in {dur_s(t):.2f}s")

    log.info("=== ANSWER ===")
    print(str(resp))  # keep answer clean in stdout

    # Also show the top evidence chunk for learning/debugging
    if resp.source_nodes:
        log.info("=== TOP SOURCE CHUNK (most similar) ===")
        print(resp.source_nodes[0].get_content())


if __name__ == "__main__":
    main()
