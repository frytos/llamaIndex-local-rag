#!/usr/bin/env python3
"""
Streamlit Web UI for RAG Pipeline

Launch with: streamlit run rag_web.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Set up logging
log = logging.getLogger(__name__)

# Add the project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file BEFORE importing rag modules
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try to load from config/.env as fallback
    config_env = PROJECT_ROOT / "config" / ".env"
    if config_env.exists():
        load_dotenv(config_env)

# Initialize Sentry early (after dotenv, before streamlit)
from utils.sentry_config import init_sentry
init_sentry()

import streamlit as st
from auth.authenticator import load_authenticator
import plotly.express as px
import plotly.graph_objects as go

# Import shared utilities
from utils.naming import extract_model_short_name, generate_table_name

# Import from existing RAG script
from rag_low_level_m1_16gb_verbose import (
    Settings,
    load_documents,
    chunk_documents,
    build_nodes,
    build_embed_model,
    make_vector_store,
    build_llm,
    VectorDBRetriever,
    ensure_db_exists,
    ensure_pgvector_extension,
    preview,
    chunked,
)

# Database connection for direct queries
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

# For embedding visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# LlamaIndex imports
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle

# RunPod deployment imports
try:
    from utils.runpod_manager import RunPodManager
    from utils.ssh_tunnel import SSHTunnelManager
    from utils.runpod_health import check_vllm_health, check_postgres_health
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"

CHUNK_PRESETS = {
    "Chat messages (300/50)": (300, 50),
    "Ultra-fine (100/20)": (100, 20),
    "Fine-grained (300/60)": (300, 60),
    "Balanced (700/150)": (700, 150),
    "Contextual (1200/240)": (1200, 240),
    "Large context (2000/400)": (2000, 400),
}

EMBED_MODELS = {
    "all-MiniLM-L6-v2 (Fast)": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "bge-small-en (Recommended)": ("BAAI/bge-small-en", 384),
    "bge-base-en (Better)": ("BAAI/bge-base-en-v1.5", 768),
    "bge-large-en (Best)": ("BAAI/bge-large-en-v1.5", 1024),
    "paraphrase-multilingual-MiniLM (Fast, Multilingual)": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
    "bge-m3 (Best, Multilingual - FR/EN/etc)": ("BAAI/bge-m3", 1024),
}

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    # Auto-detect database connection if PGHOST is not set or is "auto"
    pghost = os.environ.get("PGHOST", "")

    if not pghost or pghost == "auto":
        try:
            from utils.runpod_db_config import get_postgres_config
            config = get_postgres_config()
            if config:
                pghost = config["host"]
                pgport = config["port"]
                pguser = config["user"]
                pgpassword = config["password"]
                pgdb = config["database"]
                log.info(f"✅ Auto-detected PostgreSQL for web UI: {pghost}:{pgport}")
            else:
                # Fallback to env vars
                pghost = "localhost"
                pgport = os.environ.get("PGPORT", "5432")
                pguser = os.environ.get("PGUSER")
                pgpassword = os.environ.get("PGPASSWORD")
                pgdb = os.environ.get("DB_NAME", "vector_db")
                log.warning("⚠️ Auto-detection failed, using defaults")
        except Exception as e:
            log.warning(f"⚠️ Auto-detection error: {e}")
            pghost = "localhost"
            pgport = os.environ.get("PGPORT", "5432")
            pguser = os.environ.get("PGUSER")
            pgpassword = os.environ.get("PGPASSWORD")
            pgdb = os.environ.get("DB_NAME", "vector_db")
    else:
        # Use static config from environment
        pgport = os.environ.get("PGPORT", "5432")
        pguser = os.environ.get("PGUSER")
        pgpassword = os.environ.get("PGPASSWORD")
        pgdb = os.environ.get("DB_NAME", "vector_db")

    defaults = {
        # Database settings
        "db_host": pghost,
        "db_port": pgport,
        "db_user": pguser,
        "db_password": pgpassword,
        "db_name": pgdb,

        # Cached resources
        "embed_model": None,
        "embed_model_name": None,
        "llm": None,

        # Indexing state
        "last_indexed_nodes": None,
        "last_chunks": None,
        "last_doc_idxs": None,
        "last_embeddings": None,

        # Query state
        "query_history": [],

        # RunPod deployment state
        "runpod_api_key": os.environ.get("RUNPOD_API_KEY", ""),
        "runpod_manager": None,
        "active_pods": [],
        "selected_pod": None,
        "tunnel_active": False,
        "last_pod_refresh": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================================================
# Database Utilities
# =============================================================================

def get_db_connection(autocommit=False):
    """Get database connection using session state settings."""
    conn = psycopg2.connect(
        host=st.session_state.db_host,
        port=st.session_state.db_port,
        user=st.session_state.db_user,
        password=st.session_state.db_password,
        dbname=st.session_state.db_name,
        sslmode=os.getenv('PGSSLMODE', 'prefer'),
        connect_timeout=10,
    )
    conn.autocommit = autocommit
    return conn

def test_db_connection() -> Tuple[bool, str]:
    """Test database connection."""
    conn = None
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.close()
        conn.close()
        return True, f"Connected! PostgreSQL: {version[:50]}..."
    except Exception as e:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return False, str(e)

def list_vector_tables() -> List[Dict[str, Any]]:
    """List all vector tables with their info."""
    conn = None
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get all tables (pgvector tables have 'embedding' column)
        cur.execute("""
            SELECT table_name
            FROM information_schema.columns
            WHERE column_name = 'embedding'
            AND table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row['table_name'] for row in cur.fetchall()]

        result = []
        for table in tables:
            info = {"name": table, "rows": 0, "chunk_size": "?", "chunk_overlap": "?", "embed_dim": "?"}

            # Get row count
            try:
                cur.execute(
                    sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(table))
                )
                info["rows"] = cur.fetchone()["count"]
            except Exception as e:
                conn.rollback()
                continue

            # Get embedding dimension from column type
            try:
                cur.execute("""
                    SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type
                    FROM pg_catalog.pg_attribute a
                    JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
                    WHERE c.relname = %s
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                """, (table,))
                row = cur.fetchone()
                if row and row["data_type"] and "vector(" in row["data_type"]:
                    # Extract dimension from "vector(N)" format
                    info["embed_dim"] = int(row["data_type"].split("(")[1].split(")")[0])
            except Exception as e:
                pass  # Dimension query failed, keep default

            # Try to get metadata from first row
            try:
                cur.execute(
                    sql.SQL("""
                        SELECT metadata_->>'_chunk_size' as cs,
                               metadata_->>'_chunk_overlap' as co,
                               metadata_->>'_embed_model' as em
                        FROM {}
                        WHERE metadata_->>'_chunk_size' IS NOT NULL
                        LIMIT 1
                    """).format(sql.Identifier(table))
                )
                row = cur.fetchone()
                if row:
                    info["chunk_size"] = row["cs"] or "?"
                    info["chunk_overlap"] = row["co"] or "?"
                    if row["em"]:
                        info["embed_model"] = row["em"]
            except Exception as e:
                pass  # Metadata query failed, keep defaults

            result.append(info)

        cur.close()
        conn.close()
        return result
    except Exception as e:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        st.error(f"Error listing tables: {e}")
        return []

def delete_table(table_name: str) -> bool:
    """Delete a vector table."""
    conn = None
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor()
        cur.execute(
            sql.SQL('DROP TABLE IF EXISTS {} CASCADE').format(sql.Identifier(table_name))
        )
        cur.close()
        conn.close()
        return True
    except Exception as e:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        st.error(f"Error deleting table: {e}")
        return False

def fetch_embeddings_for_viz(table_name: str, limit: int = 500) -> Tuple[np.ndarray, List[str], List[str]]:
    """Fetch embeddings from database for visualization."""
    conn = None
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor()

        cur.execute(
            sql.SQL("""
                SELECT embedding, text, metadata_->>'source' as source
                FROM {}
                LIMIT %s
            """).format(sql.Identifier(table_name)),
            (limit,)
        )

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return np.array([]), [], []

        # Parse embeddings - they may be strings or already arrays
        embeddings_list = []
        for row in rows:
            emb = row[0]
            if isinstance(emb, str):
                # Parse string like "[-0.08, 0.03, ...]" or "[...]"
                import json
                import ast
                try:
                    emb = json.loads(emb)
                except (json.JSONDecodeError, ValueError):
                    # Try ast.literal_eval as safe fallback
                    try:
                        # Clean up numpy representation
                        cleaned = emb.replace('np.str_', '').replace('(', '').replace(')', '')
                        emb = ast.literal_eval(cleaned)
                    except (ValueError, SyntaxError) as e:
                        st.warning(f"Failed to parse embedding: {e}")
                        continue
            embeddings_list.append(emb)

        embeddings = np.array(embeddings_list, dtype=np.float32)
        texts = [row[1][:200] if row[1] else "" for row in rows]
        sources = [Path(row[2]).name if row[2] else "unknown" for row in rows]

        return embeddings, texts, sources
    except Exception as e:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        st.error(f"Error fetching embeddings: {e}")
        return np.array([]), [], []

# =============================================================================
# Document Discovery
# =============================================================================

def list_documents() -> Tuple[List[Path], List[Tuple[Path, int]]]:
    """List available documents and folders in data directory."""
    if not DATA_DIR.exists():
        return [], []

    supported_extensions = {
        ".pdf", ".docx", ".pptx", ".txt", ".md", ".html", ".htm",
        ".json", ".csv", ".xml", ".py", ".js", ".ts", ".java",
    }

    files = []
    folders = []

    for item in sorted(DATA_DIR.iterdir()):
        if item.is_file() and item.suffix.lower() in supported_extensions:
            files.append(item)
        elif item.is_dir() and not item.name.startswith("."):
            # Count files in folder
            count = sum(1 for f in item.rglob("*") if f.is_file())
            folders.append((item, count))

    return files, folders

# Functions extract_model_short_name and generate_table_name now imported from utils.naming

# =============================================================================
# Cached Resources
# =============================================================================

@st.cache_resource
def get_embed_model(model_name: str):
    """Cache embedding model."""
    # Temporarily modify settings
    import rag_low_level_m1_16gb_verbose as rag
    original = rag.S.embed_model_name
    rag.S.embed_model_name = model_name
    model = build_embed_model()
    rag.S.embed_model_name = original
    return model

@st.cache_resource
def get_llm():
    """Cache LLM (expensive to load)."""
    return build_llm()

# =============================================================================
# Visualization Functions
# =============================================================================

def render_chunk_distribution(chunks: List[str]):
    """Render chunk size distribution histogram."""
    sizes = [len(c) for c in chunks]

    fig = px.histogram(
        x=sizes,
        nbins=30,
        labels={"x": "Chunk Size (characters)", "y": "Count"},
        title="Chunk Size Distribution",
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, width='stretch')

def render_embedding_visualization(
    embeddings: np.ndarray,
    texts: List[str],
    sources: List[str],
    method: str = "t-SNE",
    dimensions: int = 2,
):
    """Render 2D/3D embedding visualization."""
    if len(embeddings) < 3:
        st.warning("Need at least 3 embeddings for visualization")
        return

    # Dimensionality reduction
    with st.spinner(f"Computing {method} projection..."):
        if method == "t-SNE":
            perplexity = min(30, len(embeddings) - 1)
            reducer = TSNE(n_components=dimensions, perplexity=perplexity, random_state=42)
            coords = reducer.fit_transform(embeddings)
        elif method == "UMAP":
            try:
                import umap
                n_neighbors = min(15, len(embeddings) - 1)
                reducer = umap.UMAP(n_components=dimensions, n_neighbors=n_neighbors, random_state=42)
                coords = reducer.fit_transform(embeddings)
            except ImportError:
                st.warning("UMAP not available, using t-SNE")
                perplexity = min(30, len(embeddings) - 1)
                reducer = TSNE(n_components=dimensions, perplexity=perplexity, random_state=42)
                coords = reducer.fit_transform(embeddings)
        else:  # PCA
            reducer = PCA(n_components=dimensions)
            coords = reducer.fit_transform(embeddings)

    # Create dataframe
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "text": [t[:100] + "..." if len(t) > 100 else t for t in texts],
        "source": sources,
    })
    if dimensions == 3:
        df["z"] = coords[:, 2]

    # Create plot
    if dimensions == 2:
        fig = px.scatter(
            df, x="x", y="y",
            color="source",
            hover_data=["text"],
            title=f"{method} Projection ({len(embeddings)} chunks)",
        )
    else:
        fig = px.scatter_3d(
            df, x="x", y="y", z="z",
            color="source",
            hover_data=["text"],
            title=f"{method} 3D Projection ({len(embeddings)} chunks)",
        )

    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')

# =============================================================================
# Pages
# =============================================================================

def page_index():
    """Index Documents page."""
    st.header("Index Documents")

    # Document Selection
    st.subheader("1. Select Document or Folder")

    files, folders = list_documents()

    options = []
    option_paths = {}

    # Add upload option first
    options.append("📤 Upload files from your computer")

    for folder, count in folders:
        label = f"📁 {folder.name}/ ({count} files)"
        options.append(label)
        option_paths[label] = folder

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        label = f"📄 {f.name} ({size_mb:.1f} MB)"
        options.append(label)
        option_paths[label] = f

    options.append("📝 Enter custom path")

    selected = st.selectbox("Available documents:", options)

    doc_path = None

    if selected == "📤 Upload files from your computer":
        st.info("📁 Drag and drop files here, or click to browse")

        uploaded_files = st.file_uploader(
            "Choose files to index",
            accept_multiple_files=True,
            type=["pdf", "txt", "html", "md", "py", "js", "json", "csv"],
            help="Supported: PDF, TXT, HTML, MD, PY, JS, JSON, CSV"
        )

        if uploaded_files:
            # Create temporary upload directory
            import tempfile
            upload_dir = Path(tempfile.gettempdir()) / "streamlit_uploads" / f"upload_{int(time.time())}"
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded files
            saved_files = []
            total_size = 0
            for uploaded_file in uploaded_files:
                file_path = upload_dir / uploaded_file.name
                file_path.write_bytes(uploaded_file.read())
                saved_files.append(file_path)
                total_size += file_path.stat().st_size

            st.success(f"✅ Uploaded {len(saved_files)} file(s) ({total_size / (1024*1024):.1f} MB)")

            # Show uploaded files
            with st.expander("📋 Uploaded files"):
                for f in saved_files:
                    st.text(f"• {f.name} ({f.stat().st_size / 1024:.1f} KB)")

            # Use upload directory as doc_path
            doc_path = upload_dir
        else:
            st.warning("Please upload at least one file")
            return

    elif selected == "📝 Enter custom path":
        doc_path = st.text_input("Enter path:", value=str(DATA_DIR))
        doc_path = Path(doc_path) if doc_path else None
    else:
        doc_path = option_paths.get(selected)

    if not doc_path or not doc_path.exists():
        st.warning("Please select a valid document or folder")
        return

    if selected != "📤 Upload files from your computer":
        st.success(f"Selected: `{doc_path}`")

    # Chunking Parameters
    st.subheader("2. Chunking Parameters")

    col1, col2 = st.columns(2)

    with col1:
        preset = st.selectbox("Preset:", list(CHUNK_PRESETS.keys()), index=2)
        default_size, default_overlap = CHUNK_PRESETS[preset]

    with col2:
        use_custom = st.checkbox("Custom values")

    if use_custom:
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk size:", 50, 3000, default_size)
        with col2:
            chunk_overlap = st.slider("Overlap:", 0, 500, default_overlap)
    else:
        chunk_size, chunk_overlap = default_size, default_overlap

    st.info(f"Chunk size: **{chunk_size}** | Overlap: **{chunk_overlap}** ({100*chunk_overlap/chunk_size:.0f}%)")

    # Embedding Model
    st.subheader("3. Embedding Model")

    embed_choice = st.selectbox("Model:", list(EMBED_MODELS.keys()), index=1)
    embed_model_name, embed_dim = EMBED_MODELS[embed_choice]

    # Backend selection
    embed_backend = st.selectbox(
        "Backend:",
        ["huggingface", "mlx"],
        index=0,
        help="MLX is 9x faster on Apple Silicon (M1/M2/M3) - recommended for bge-m3"
    )

    st.info(f"Model: `{embed_model_name}` | Dimensions: **{embed_dim}** | Backend: **{embed_backend}**")

    # Table Name
    st.subheader("4. Index Name")

    suggested_name = generate_table_name(doc_path, chunk_size, chunk_overlap, embed_model_name)
    table_name = st.text_input("Table name:", value=suggested_name)

    reset_table = st.checkbox("Reset table if exists", value=True)

    # Start Indexing
    st.subheader("5. Start Indexing")

    if st.button("🚀 Start Indexing", type="primary", width='stretch'):
        run_indexing(doc_path, table_name, chunk_size, chunk_overlap, embed_model_name, embed_dim, embed_backend, reset_table)

def run_indexing(doc_path: Path, table_name: str, chunk_size: int, chunk_overlap: int,
                 embed_model_name: str, embed_dim: int, embed_backend: str, reset_table: bool):
    """Run the indexing pipeline with progress visualization."""

    # Update settings
    import rag_low_level_m1_16gb_verbose as rag
    rag.S.pdf_path = str(doc_path)
    rag.S.table = table_name
    rag.S.chunk_size = chunk_size
    rag.S.chunk_overlap = chunk_overlap
    rag.S.embed_model_name = embed_model_name
    rag.S.embed_dim = embed_dim
    rag.S.embed_backend = embed_backend
    rag.S.reset_table = reset_table

    # Check if RunPod GPU is configured (auto-detect or static)
    from utils.runpod_db_config import get_embedding_endpoint

    runpod_endpoint = get_embedding_endpoint()
    runpod_api_key = os.getenv("RUNPOD_EMBEDDING_API_KEY")

    if runpod_endpoint and runpod_api_key:
        st.success("🚀 **GPU Acceleration Enabled** - Using RunPod RTX 4090 (~100x faster)")
        st.caption(f"Endpoint: {runpod_endpoint}")
    else:
        st.warning("💻 **CPU Mode** - Embeddings will be slower. Set RUNPOD_EMBEDDING_API_KEY for GPU acceleration.")
        st.caption("GPU embeddings: ~10 seconds | CPU embeddings: ~17 minutes (for 837 chunks)")

    status = st.status("Indexing Pipeline", expanded=True)

    with status:
        # Step 1: Load documents
        st.write("**Step 1: Loading documents...**")
        progress = st.progress(0)

        try:
            docs = load_documents(str(doc_path))
            progress.progress(20)
            st.success(f"✓ Loaded {len(docs)} document(s)")

            total_chars = sum(len(d.text) for d in docs)
            st.caption(f"Total: {total_chars:,} characters")
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            status.update(label="Indexing Failed", state="error")
            return

        # Step 2: Chunk documents
        st.write("**Step 2: Chunking documents...**")

        try:
            chunks, doc_idxs = chunk_documents(docs)
            progress.progress(40)
            st.success(f"✓ Created {len(chunks)} chunks")

            # Store for visualization
            st.session_state.last_chunks = chunks
            st.session_state.last_doc_idxs = doc_idxs

            sizes = [len(c) for c in chunks]
            st.caption(f"Avg size: {sum(sizes)/len(sizes):.0f} chars | Range: {min(sizes)}-{max(sizes)}")
        except Exception as e:
            st.error(f"Error chunking: {e}")
            status.update(label="Indexing Failed", state="error")
            return

        # Step 3: Build nodes and embed
        st.write("**Step 3: Computing embeddings...**")

        try:
            nodes = build_nodes(docs, chunks, doc_idxs)

            # Use the main pipeline's embed_nodes() function which supports RunPod GPU
            embed_model = get_embed_model(embed_model_name)
            rag.embed_nodes(embed_model, nodes)

            progress.progress(80)
            st.success(f"✓ Embedded {len(nodes)} chunks")

            # Store for session state
            st.session_state.last_indexed_nodes = nodes
            embeddings_list = [n.embedding for n in nodes if n.embedding]
            st.session_state.last_embeddings = np.array(embeddings_list) if embeddings_list else None
        except Exception as e:
            st.error(f"Error embedding: {e}")
            status.update(label="Indexing Failed", state="error")
            return

        # Step 4: Store in database
        st.write("**Step 4: Storing in database...**")

        try:
            ensure_db_exists()
            ensure_pgvector_extension()

            if reset_table:
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute(
                        sql.SQL('DROP TABLE IF EXISTS {} CASCADE').format(sql.Identifier(table_name))
                    )
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    pass  # Table might not exist

            vector_store = make_vector_store()

            # Insert in batches
            batch_size = 250
            total_batches = (len(nodes) + batch_size - 1) // batch_size

            for i, batch in enumerate(chunked(nodes, batch_size)):
                vector_store.add(list(batch))
                progress.progress(80 + int(20 * (i + 1) / total_batches))

            st.success(f"✓ Stored {len(nodes)} chunks in `{table_name}`")
        except Exception as e:
            st.error(f"Error storing: {e}")
            status.update(label="Indexing Failed", state="error")
            return

    status.update(label="✅ Indexing Complete!", state="complete")

    # Show visualizations
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Chunk Distribution")
        if st.session_state.last_chunks:
            render_chunk_distribution(st.session_state.last_chunks)

    with col2:
        st.subheader("Chunk Samples")
        if st.session_state.last_chunks:
            for i, chunk in enumerate(st.session_state.last_chunks[:3]):
                with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)

    # Embedding visualization
    st.subheader("Embedding Visualization")

    if st.session_state.last_embeddings is not None and len(st.session_state.last_embeddings) > 2:
        col1, col2 = st.columns(2)
        with col1:
            viz_method = st.selectbox("Method:", ["t-SNE", "PCA", "UMAP"])
        with col2:
            viz_dims = st.radio("Dimensions:", [2, 3], horizontal=True)

        texts = [n.get_content() for n in st.session_state.last_indexed_nodes]
        sources = [Path(n.metadata.get("source", n.metadata.get("file_path", "unknown"))).name
                   for n in st.session_state.last_indexed_nodes]

        # Limit for performance
        limit = min(500, len(st.session_state.last_embeddings))
        render_embedding_visualization(
            st.session_state.last_embeddings[:limit],
            texts[:limit],
            sources[:limit],
            method=viz_method,
            dimensions=viz_dims,
        )

def search_chunks(table_name: str, query: str, top_k: int, embed_model_name: str,
                  embed_dim: int, embed_backend: str):
    """Search and display chunks only (no LLM generation)."""

    import rag_low_level_m1_16gb_verbose as rag
    rag.S.table = table_name
    rag.S.top_k = top_k
    rag.S.embed_model_name = embed_model_name
    rag.S.embed_dim = embed_dim
    rag.S.embed_backend = embed_backend

    st.subheader(f"🔍 Top {top_k} Chunks for: \"{query}\"")

    with st.spinner("Retrieving chunks..."):
        try:
            # Build retriever
            embed_model = get_embed_model(embed_model_name)
            vector_store = make_vector_store()
            retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=top_k)

            # Retrieve chunks
            from llama_index.core import QueryBundle
            results = retriever._retrieve(QueryBundle(query_str=query))

            if not results:
                st.warning("No results found.")
                return

            # Display results
            st.success(f"✅ Found {len(results)} chunks")

            # Statistics
            scores = [r.score for r in results]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Score", f"{max(scores):.4f}")
            with col2:
                st.metric("Avg Score", f"{sum(scores)/len(scores):.4f}")
            with col3:
                st.metric("Worst Score", f"{min(scores):.4f}")
            with col4:
                quality = "🟢 Excellent" if max(scores) > 0.7 else "🟡 Good" if max(scores) > 0.5 else "🟠 Fair" if max(scores) > 0.3 else "🔴 Low"
                st.metric("Quality", quality)

            st.divider()

            # Display each chunk
            for i, result in enumerate(results):
                score = result.score
                meta = result.node.metadata or {}

                # Color coding
                if score > 0.7:
                    badge = "🟢 Excellent"
                    score_color = "green"
                elif score > 0.5:
                    badge = "🟡 Good"
                    score_color = "orange"
                elif score > 0.3:
                    badge = "🟠 Fair"
                    score_color = "orange"
                else:
                    badge = "🔴 Low"
                    score_color = "red"

                with st.expander(f"**Chunk {i+1}:** {badge} (Score: {score:.4f})", expanded=(i < 3)):
                    # Metadata panel
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown("**📊 Metadata:**")
                        st.metric("Similarity Score", f"{score:.4f}")

                        # Show participants if available
                        participants = meta.get('_participants', meta.get('participants', []))
                        if participants:
                            if isinstance(participants, str):
                                import json
                                try:
                                    participants = json.loads(participants)
                                except:
                                    participants = [participants]
                            st.markdown(f"**👥 Participants:**")
                            for p in participants[:5]:
                                st.caption(f"• {p}")

                        # Show dates if available
                        primary_date = meta.get('_primary_date')
                        if primary_date:
                            st.markdown(f"**📅 Date:** {primary_date}")

                        # Content type
                        content_type = meta.get('_content_type')
                        if content_type:
                            st.caption(f"Type: {content_type}")

                        # Chunk config
                        chunk_size = meta.get('_chunk_size')
                        if chunk_size:
                            st.caption(f"Chunk: {chunk_size} chars")

                    with col2:
                        st.markdown("**📝 Content:**")
                        text = result.node.get_content()
                        # Truncate very long chunks for display
                        if len(text) > 1000:
                            st.text_area("", value=text, height=300, disabled=True, label_visibility="collapsed")
                        else:
                            st.text(text)

                        # Source file
                        source = meta.get("file_path", meta.get("source", "Unknown"))
                        if source:
                            st.caption(f"📁 Source: {Path(source).name}")

        except Exception as e:
            st.error(f"Search error: {e}")
            import traceback
            st.code(traceback.format_exc())


def page_query():
    """Query page."""
    st.header("Query Index")

    # Get available indexes
    indexes = list_vector_tables()

    if not indexes:
        st.warning("No indexes found. Please index some documents first.")
        return

    # Compact layout: Index selection + metrics in one row
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        index_options = [f"{idx['name']} ({idx['rows']} chunks, cs={idx['chunk_size']}, dim={idx['embed_dim']})" for idx in indexes]
        selected_idx = st.selectbox("**Index**", index_options, label_visibility="visible")
        selected_index = indexes[index_options.index(selected_idx)]
        table_name = selected_index["name"]

        # Strip 'data_' prefix if present (PGVectorStore adds it automatically)
        if table_name.startswith("data_"):
            table_name = table_name[5:]  # Remove 'data_' prefix

        table_dim = selected_index.get("embed_dim", "?")
        table_model = selected_index.get("embed_model", "unknown")

    with col2:
        st.metric("Chunks", f"{selected_index['rows']:,}", label_visibility="visible")
    with col3:
        st.metric("Chunk Size", selected_index['chunk_size'], label_visibility="visible")
    with col4:
        st.metric("Dim", table_dim, label_visibility="visible")

    # Query Settings: More compact layout
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        # Embedding model selection based on table dimension
        if isinstance(table_dim, int):
            # Filter models by dimension
            compatible_models = {k: v for k, v in EMBED_MODELS.items() if v[1] == table_dim}

            if compatible_models:
                embed_choice = st.selectbox(
                    "**Embedding Model**",
                    list(compatible_models.keys()),
                    help=f"✓ {len(compatible_models)} compatible model(s) for {table_dim}D"
                )
                embed_model_name, embed_dim = compatible_models[embed_choice]
            else:
                st.error(f"⚠️ No models found for {table_dim}D!")
                embed_choice = st.selectbox("**Embedding Model**", list(EMBED_MODELS.keys()), index=1)
                embed_model_name, embed_dim = EMBED_MODELS[embed_choice]
        else:
            embed_choice = st.selectbox("**Embedding Model**", list(EMBED_MODELS.keys()), index=1)
            embed_model_name, embed_dim = EMBED_MODELS[embed_choice]

    with col2:
        embed_backend = st.selectbox(
            "**Backend**",
            ["huggingface", "mlx"],
            index=1 if embed_dim == 1024 else 0,
            help="MLX is 9x faster on Apple Silicon"
        )

    with col3:
        top_k = st.number_input("**TOP_K**", min_value=1, max_value=20, value=4, help="Chunks to retrieve")

    with col4:
        st.write("**Options**")
        show_sources = st.checkbox("Show sources", value=True)

    # Dimension mismatch warning (compact)
    if isinstance(table_dim, int) and embed_dim != table_dim:
        st.error(f"🚨 Dimension mismatch: Table={table_dim}D, Model={embed_dim}D - Select compatible model!")

    # Query input (compact height)
    query = st.text_area("**Your Question:**", height=80, placeholder="What is the main topic of the document?")

    # Search buttons side by side
    col1, col2 = st.columns(2)

    with col1:
        search_and_answer = st.button("🔍 Search & Answer", type="primary", width='stretch')

    with col2:
        search_chunks_only = st.button("📄 Search Chunks Only", width='stretch')

    if search_and_answer or search_chunks_only:
        if not query.strip():
            st.warning("Please enter a question")
            return

        # Check for dimension mismatch before running
        if isinstance(table_dim, int) and embed_dim != table_dim:
            st.error("Cannot run query - dimension mismatch detected. Please select a compatible model.")
            return

        if search_chunks_only:
            search_chunks(table_name, query, top_k, embed_model_name, embed_dim, embed_backend)
        else:
            run_query(table_name, query, top_k, show_sources, embed_model_name, embed_dim, embed_backend)

    # Query history
    if st.session_state.query_history:
        st.subheader("Query History")
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Q: {item['query'][:50]}..."):
                st.write(f"**Answer:** {item['answer']}")
                st.caption(f"Top score: {item['top_score']:.4f} | Chunks: {item['chunks']}")

def run_query(table_name: str, query: str, top_k: int, show_sources: bool,
              embed_model_name: str, embed_dim: int, embed_backend: str):
    """Run a query against the index."""

    import rag_low_level_m1_16gb_verbose as rag
    rag.S.table = table_name
    rag.S.top_k = top_k
    rag.S.embed_model_name = embed_model_name
    rag.S.embed_dim = embed_dim
    rag.S.embed_backend = embed_backend

    with st.spinner("Searching..."):
        try:
            # Build retriever with specified embedding model
            embed_model = get_embed_model(embed_model_name)
            vector_store = make_vector_store()
            retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=top_k)

            # Retrieve
            results = retriever._retrieve(QueryBundle(query_str=query))
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Display retrieved chunks
    if show_sources and results:
        st.subheader("Retrieved Chunks")

        for i, result in enumerate(results):
            score = result.score

            # Color coding
            if score > 0.7:
                badge = "🟢 Excellent"
            elif score > 0.5:
                badge = "🟡 Good"
            elif score > 0.3:
                badge = "🟠 Fair"
            else:
                badge = "🔴 Low"

            with st.expander(f"Chunk {i+1}: {badge} (Score: {score:.4f})", expanded=(i == 0)):
                meta = result.node.metadata or {}
                source = meta.get("source", meta.get("file_path", "Unknown"))

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Similarity", f"{score:.4f}")
                with col2:
                    st.caption(f"Source: {Path(source).name if source else 'Unknown'}")

                st.text(result.node.get_content())

    # Generate answer
    st.subheader("Generated Answer")

    with st.spinner("Generating answer..."):
        try:
            llm = get_llm()
            query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
            response = query_engine.query(query)

            st.success(str(response))

            # Store in history
            st.session_state.query_history.append({
                "query": query,
                "answer": str(response),
                "chunks": len(results),
                "top_score": results[0].score if results else 0,
            })
        except Exception as e:
            st.error(f"Generation error: {e}")

def page_view_indexes():
    """View Indexes page."""
    st.header("View Indexes")

    if st.button("🔄 Refresh"):
        st.rerun()

    indexes = list_vector_tables()

    if not indexes:
        st.info("No indexes found. Index some documents to get started.")
        return

    # Summary stats
    total_chunks = sum(idx["rows"] for idx in indexes)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Indexes", len(indexes))
    with col2:
        st.metric("Total Chunks", f"{total_chunks:,}")
    with col3:
        st.metric("Database", st.session_state.db_name)

    st.divider()

    # Index table
    df = pd.DataFrame(indexes)
    # Rename columns based on available fields
    if "embed_dim" in df.columns:
        df = df[["name", "rows", "chunk_size", "chunk_overlap", "embed_dim"]]
        df.columns = ["Name", "Chunks", "Chunk Size", "Overlap", "Embed Dim"]
    else:
        df = df[["name", "rows", "chunk_size", "chunk_overlap"]]
        df.columns = ["Name", "Chunks", "Chunk Size", "Overlap"]

    st.dataframe(df, width='stretch', hide_index=True)

    # Index details and actions
    st.subheader("Index Actions")

    selected_table = st.selectbox("Select index:", [idx["name"] for idx in indexes])

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👁️ View Embeddings"):
            embeddings, texts, sources = fetch_embeddings_for_viz(selected_table, limit=500)
            if len(embeddings) > 0:
                st.session_state["viz_embeddings"] = embeddings
                st.session_state["viz_texts"] = texts
                st.session_state["viz_sources"] = sources

    with col2:
        if st.button("🗑️ Delete Index", type="secondary"):
            st.session_state["confirm_delete"] = selected_table

    # Confirm delete
    if st.session_state.get("confirm_delete") == selected_table:
        st.warning(f"Are you sure you want to delete `{selected_table}`?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete", type="primary"):
                if delete_table(selected_table):
                    st.success(f"Deleted `{selected_table}`")
                    st.session_state["confirm_delete"] = None
                    st.rerun()
        with col2:
            if st.button("Cancel"):
                st.session_state["confirm_delete"] = None
                st.rerun()

    # Show embeddings visualization if loaded
    if "viz_embeddings" in st.session_state and len(st.session_state["viz_embeddings"]) > 0:
        st.subheader(f"Embeddings from `{selected_table}`")

        col1, col2 = st.columns(2)
        with col1:
            viz_method = st.selectbox("Method:", ["t-SNE", "PCA", "UMAP"], key="view_method")
        with col2:
            viz_dims = st.radio("Dimensions:", [2, 3], horizontal=True, key="view_dims")

        render_embedding_visualization(
            st.session_state["viz_embeddings"],
            st.session_state["viz_texts"],
            st.session_state["viz_sources"],
            method=viz_method,
            dimensions=viz_dims,
        )

def page_settings():
    """Settings page."""
    st.header("Settings")

    # Database settings
    st.subheader("Database Connection")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.db_host = st.text_input("Host:", value=st.session_state.db_host)
        st.session_state.db_user = st.text_input("User:", value=st.session_state.db_user)
        st.session_state.db_name = st.text_input("Database:", value=st.session_state.db_name)

    with col2:
        st.session_state.db_port = st.text_input("Port:", value=st.session_state.db_port)
        st.session_state.db_password = st.text_input("Password:", value=st.session_state.db_password, type="password")

    if st.button("Test Connection"):
        success, message = test_db_connection()
        if success:
            st.success(message)
        else:
            st.error(message)

    st.divider()

    # LLM settings info
    st.subheader("LLM Configuration")
    st.info("""
    LLM settings are configured via environment variables:
    - `MODEL_URL` - Hugging Face model URL
    - `CTX` - Context window size (default: 3072)
    - `TEMP` - Temperature (default: 0.1)
    - `MAX_NEW_TOKENS` - Max generation tokens (default: 256)
    - `N_GPU_LAYERS` - GPU layers for Metal (default: 24)
    """)

    st.divider()

    # Clear cache
    st.subheader("Cache Management")

    if st.button("Clear All Caches"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.embed_model = None
        st.session_state.llm = None
        st.success("Caches cleared!")

def page_deployment():
    """RunPod deployment management page."""
    st.header("☁️ RunPod Deployment")

    if not RUNPOD_AVAILABLE:
        st.error("❌ RunPod utilities not available")
        st.info("Install with: `pip install runpod`")
        return

    # =========================================================================
    # Section 1: API Configuration
    # =========================================================================
    st.subheader("1. API Configuration")

    api_key = st.text_input(
        "RunPod API Key",
        value=st.session_state.runpod_api_key,
        type="password",
        help="Get your API key from https://runpod.io/settings",
        key="api_key_input"
    )

    if api_key and api_key != st.session_state.runpod_api_key:
        st.session_state.runpod_api_key = api_key
        st.session_state.runpod_manager = None  # Reset manager

    if not api_key:
        st.warning("⚠️ Enter your RunPod API key to continue")
        st.markdown("[Get API Key →](https://runpod.io/settings)")
        return

    # Initialize manager
    try:
        if st.session_state.runpod_manager is None:
            with st.spinner("Initializing RunPod API..."):
                st.session_state.runpod_manager = RunPodManager(api_key=api_key)
        manager = st.session_state.runpod_manager
        st.success("✅ API key validated")
    except Exception as e:
        st.error(f"❌ Invalid API key: {e}")
        return

    st.divider()

    # =========================================================================
    # Section 2: Existing Pods
    # =========================================================================
    st.subheader("2. Existing Pods")

    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_pods"):
            st.session_state.last_pod_refresh = time.time()

    # Get pods
    try:
        with st.spinner("Loading pods..."):
            pods = manager.list_pods()
            st.session_state.active_pods = pods
    except Exception as e:
        st.error(f"Failed to load pods: {e}")
        pods = []

    if pods:
        # Initialize SSH host cache if needed
        if 'pod_ssh_hosts' not in st.session_state:
            st.session_state.pod_ssh_hosts = {}

        # Create dataframe for display
        pod_data = []
        for pod in pods:
            # Handle None values from API
            runtime = pod.get('runtime') or {}
            machine = pod.get('machine') or {}

            # Cache SSH host if available
            pod_id = pod.get('id', '')
            ssh_host = machine.get('podHostId', '')
            if pod_id and ssh_host:
                st.session_state.pod_ssh_hosts[pod_id] = ssh_host

            pod_data.append({
                "Name": pod.get('name', 'N/A'),
                "Status": runtime.get('containerState', 'unknown'),
                "GPU": machine.get('gpuTypeId', 'N/A'),
                "Uptime": f"{runtime.get('uptimeInSeconds', 0) // 60}min",
                "Cost/hr": f"${pod.get('costPerHr', 0):.2f}",
                "ID": pod_id[:12] + "..." if pod_id else "N/A"
            })

        df = pd.DataFrame(pod_data)
        st.dataframe(df, width='stretch', hide_index=True)

        # Pod management
        st.write("**Manage Pod:**")

        pod_names = {pod['id']: pod['name'] for pod in pods}
        selected_pod_id = st.selectbox(
            "Select pod",
            options=list(pod_names.keys()),
            format_func=lambda x: pod_names[x],
            key="pod_selector"
        )

        st.session_state.selected_pod = selected_pod_id

        # Get detailed status
        if selected_pod_id:
            with st.spinner("Getting pod status..."):
                status = manager.get_pod_status(selected_pod_id)

            # Display status
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Status", status['status'])
            with col2:
                st.metric("GPU Usage", f"{status['gpu_utilization']}%")
            with col3:
                st.metric("Uptime", f"{status['uptime_seconds'] // 60}min")
            with col4:
                st.metric("Cost/hr", f"${status['cost_per_hour']:.2f}")

            # SSH connection info
            # Try cached SSH host first (more reliable for newly created pods)
            cached_host = st.session_state.get('pod_ssh_hosts', {}).get(selected_pod_id)

            if cached_host:
                # Build SSH command from cached host
                ssh_cmd = f"ssh -i ~/.ssh/runpod_key -N -L 8000:localhost:8000 -L 5432:localhost:5432 {cached_host}@ssh.runpod.io"
            else:
                # Fallback to API query
                ssh_cmd = manager.get_ssh_command(selected_pod_id)

            st.code(ssh_cmd, language="bash")

            # Action buttons
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("▶️ Resume", disabled=status['status'] == 'running', key="btn_resume"):
                    with st.spinner("Resuming pod..."):
                        if manager.resume_pod(selected_pod_id):
                            st.success("✅ Pod resumed!")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to resume")

            with col2:
                if st.button("⏸️ Stop", disabled=status['status'] != 'running', key="btn_stop"):
                    with st.spinner("Stopping pod..."):
                        if manager.stop_pod(selected_pod_id):
                            st.success("✅ Pod stopped! No longer incurring GPU costs.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to stop")

            with col3:
                if st.button("🔄 Restart", disabled=status['status'] != 'running', key="btn_restart"):
                    with st.spinner("Restarting pod..."):
                        if manager.stop_pod(selected_pod_id):
                            time.sleep(5)
                            if manager.resume_pod(selected_pod_id):
                                st.success("✅ Pod restarted!")
                                st.rerun()

            with col4:
                terminate_confirm = st.checkbox("Confirm", key="terminate_confirm")
                if st.button("🗑️ Terminate", disabled=not terminate_confirm, key="btn_terminate"):
                    with st.spinner("Terminating pod..."):
                        if manager.terminate_pod(selected_pod_id):
                            st.success("✅ Pod terminated")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to terminate")

    else:
        st.info("No existing pods found")

    st.divider()

    # =========================================================================
    # Section 3: Create New Pod
    # =========================================================================
    st.subheader("3. Deploy New Pod")

    with st.expander("Pod Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            pod_name = st.text_input(
                "Pod Name",
                value=f"rag-pipeline-{int(time.time())}",
                help="Unique name for your pod"
            )

            # GPU type mapping: display name -> RunPod GPU ID
            gpu_options = {
                "RTX 4090 ($0.50/hr) - Recommended": "NVIDIA GeForce RTX 4090",
                "RTX 4070 Ti ($0.29/hr) - Budget": "NVIDIA GeForce RTX 4070 Ti",
                "RTX 3090 ($0.24/hr) - Cheapest": "NVIDIA GeForce RTX 3090",
                "A100 40GB ($1.50/hr) - Overkill": "NVIDIA A100 40GB PCIe"
            }

            gpu_display = st.selectbox(
                "GPU Type",
                options=list(gpu_options.keys()),
                index=0,
                help="RTX 4090 recommended for best price/performance"
            )

            # Get actual GPU ID for RunPod API
            gpu_type = gpu_options[gpu_display]

        with col2:
            volume_gb = st.number_input(
                "Storage (GB)",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                help="Persistent storage for models and data"
            )

            container_disk_gb = st.number_input(
                "Container Disk (GB)",
                min_value=20,
                max_value=100,
                value=50,
                step=10,
                help="Ephemeral container storage"
            )

    # GitHub Auto-Setup Section
    with st.expander("🚀 GitHub Auto-Setup (Recommended)", expanded=True):
        st.markdown("""
        Enable this to **automatically clone your repo and initialize all services** when the pod starts!

        The pod will:
        1. Clone your GitHub repository
        2. Install PostgreSQL + pgvector
        3. Set up Python environment
        4. Start vLLM server
        5. Be fully ready to use (~10-15 minutes after creation)
        """)

        enable_auto_setup = st.checkbox(
            "Enable automatic GitHub setup",
            value=True,
            help="Highly recommended! Pod will be fully initialized when ready."
        )

        if enable_auto_setup:
            # Get defaults from environment variables
            default_repo = os.getenv("GITHUB_REPO_URL", "https://github.com/frytos/llamaIndex-local-rag.git")
            default_branch = os.getenv("GITHUB_BRANCH", "main")
            default_use_token = os.getenv("USE_GITHUB_TOKEN", "1") == "1"

            github_repo = st.text_input(
                "GitHub Repository URL",
                value=default_repo,
                help="Your repository URL (public or private with token)"
            )

            col1, col2 = st.columns(2)
            with col1:
                github_branch = st.text_input(
                    "Branch",
                    value=default_branch,
                    help="Git branch to clone"
                )
            with col2:
                use_github_token = st.checkbox(
                    "Use GitHub Token (for private repos)",
                    value=default_use_token,
                    help="Uses GH_TOKEN secret from RunPod if available"
                )

            if use_github_token:
                st.info("💡 Make sure you've added GH_TOKEN secret in RunPod settings")
        else:
            # Set defaults when auto-setup is disabled
            github_repo = ""
            github_branch = "main"
            use_github_token = False

    with st.expander("Advanced Configuration"):
        col1, col2 = st.columns(2)

        with col1:
            vllm_model = st.selectbox(
                "vLLM Model",
                options=[
                    "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "TheBloke/Llama-2-7B-Chat-AWQ"
                ],
                index=0
            )

            embed_model = st.selectbox(
                "Embedding Model",
                options=[
                    "BAAI/bge-small-en",
                    "BAAI/bge-base-en-v1.5",
                    "BAAI/bge-m3"
                ],
                index=0
            )

        with col2:
            ctx_size = st.selectbox(
                "Context Window",
                options=[3072, 4096, 8192, 16384],
                index=2
            )

            top_k = st.slider(
                "Top K Retrieval",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of chunks to retrieve"
            )

    # Cost estimation
    st.write("**Cost Estimation:**")

    gpu_costs = {
        "NVIDIA GeForce RTX 4090": 0.50,
        "NVIDIA GeForce RTX 4070 Ti": 0.29,
        "NVIDIA GeForce RTX 3090": 0.24,
        "NVIDIA A100 40GB PCIe": 1.50
    }

    cost_per_hour = gpu_costs.get(gpu_type, 0.50)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hourly", f"${cost_per_hour:.2f}")
    with col2:
        st.metric("Daily (8h)", f"${cost_per_hour * 8:.2f}")
    with col3:
        st.metric("Monthly (8h/day)", f"${cost_per_hour * 8 * 30:.2f}")

    # Deploy button
    if st.button("🚀 Deploy Pod", type="primary", key="deploy_button"):
        with st.spinner("Creating pod... This may take 2-3 minutes"):
            progress = st.progress(0, text="Initializing deployment...")

            try:
                # Create pod
                progress.progress(10, text="Creating pod on RunPod...")

                # Get configuration from Railway environment (will be passed to pod)
                embedding_api_key = os.getenv("RUNPOD_EMBEDDING_API_KEY", "")
                pg_password = os.getenv("PGPASSWORD", "frytos")  # Custom password or default

                custom_env = {
                    "USE_VLLM": "1",
                    "VLLM_MODEL": vllm_model,
                    "EMBED_MODEL": embed_model,
                    "CTX": str(ctx_size),
                    "TOP_K": str(top_k),
                    "RUNPOD_EMBEDDING_API_KEY": embedding_api_key,  # For embedding service authentication
                    "PGPASSWORD": pg_password,  # Custom PostgreSQL password from Railway
                    "PGUSER": os.getenv("PGUSER", "fryt"),
                    "DB_NAME": os.getenv("DB_NAME", "vector_db")
                }

                # Build docker_args for auto-setup
                docker_startup_cmd = None

                if enable_auto_setup and github_repo and "YOUR_USERNAME" not in github_repo:
                    # Build GitHub clone URL with token if needed
                    if use_github_token:
                        # Use GH_TOKEN from RunPod secrets
                        clone_url = github_repo.replace("https://", "https://${GH_TOKEN}@")
                    else:
                        clone_url = github_repo

                    # Build startup command (idempotent with comprehensive logging)
                    docker_startup_cmd = (
                        "bash -c '"
                        "exec > >(tee -a /workspace/startup.log) 2>&1 && "
                        "echo ========================================== && "
                        "echo AUTO-SETUP STARTING: $(date) && "
                        "echo ========================================== && "
                        "echo && "
                        "cd /workspace && "
                        "echo [STEP 1/5] Cloning/updating repository... && "
                        f"if [ -d rag-pipeline ]; then echo Repository exists, pulling updates...; cd rag-pipeline && git pull origin {github_branch}; else echo Cloning fresh repository...; git clone --branch {github_branch} {clone_url} rag-pipeline; fi && "
                        "echo Repository ready! && "
                        "echo && "
                        "cd /workspace/rag-pipeline && "
                        "echo [STEP 2/5] Checking if initialization needed... && "
                        "if [ -f /workspace/.init_complete ]; then echo Already initialized, skipping init script; else echo Running initialization script...; bash scripts/init_runpod_services.sh; echo Creating completion marker...; touch /workspace/.init_complete; echo Initialization complete!; fi && "
                        "echo && "
                        "echo [STEP 3/5] Installing embedding service dependencies... && "
                        "source /workspace/rag-pipeline/.venv/bin/activate && "
                        "pip install --quiet fastapi uvicorn[standard] requests || echo Warning: Dependency installation failed && "
                        "echo && "
                        "echo [STEP 4/5] Starting embedding service... && "
                        "export PORT=8001 && "
                        "nohup /workspace/rag-pipeline/.venv/bin/python -m uvicorn services.embedding_service:app --host 0.0.0.0 --port 8001 --workers 1 > /workspace/embedding_service.log 2>&1 & "
                        "echo Embedding service started on port 8001 && "
                        "sleep 10 && "
                        "curl -s http://localhost:8001/health && echo && echo Embedding service healthy! || echo Warning: Embedding service not responding yet && "
                        "echo && "
                        "echo [STEP 5/5] Verifying services... && "
                        "service postgresql status || echo PostgreSQL not running && "
                        "curl -s http://localhost:8000/health || echo vLLM not ready yet && "
                        "curl -s http://localhost:8001/health || echo Embedding API not ready yet && "
                        "echo && "
                        "echo ========================================== && "
                        "echo AUTO-SETUP COMPLETE: $(date) && "
                        "echo ========================================== && "
                        "echo Pod is ready for use! && "
                        "echo && "
                        "echo Services: && "
                        "echo - PostgreSQL: port 5432 && "
                        "echo - vLLM: port 8000 && "
                        "echo - Embedding API: port 8001 && "
                        "echo && "
                        "echo Logs: && "
                        "echo - Startup: /workspace/startup.log && "
                        "echo - Embedding: /workspace/embedding_service.log && "
                        "echo && "
                        "exec sleep infinity"
                        "'"
                    )

                    st.info(f"🚀 Auto-setup enabled! Cloning from: `{github_repo}` (branch: `{github_branch}`)")

                pod = manager.create_pod(
                    name=pod_name,
                    gpu_type=gpu_type,
                    volume_gb=volume_gb,
                    container_disk_gb=container_disk_gb,
                    env=custom_env,
                    docker_args=docker_startup_cmd  # Auto-setup command
                )

                if not pod:
                    st.error("❌ Failed to create pod")
                    return

                pod_id = pod['id']
                ssh_host = pod['machine']['podHostId']

                # Cache SSH host in session state for later retrieval
                if 'pod_ssh_hosts' not in st.session_state:
                    st.session_state.pod_ssh_hosts = {}
                st.session_state.pod_ssh_hosts[pod_id] = ssh_host

                progress.progress(40, text="Pod created! Waiting for ready state...")

                # Wait for ready
                start_time = time.time()
                timeout = 120  # 2 minutes (shorter timeout)

                pod_ready = False

                while time.time() - start_time < timeout:
                    status = manager.get_pod_status(pod_id)
                    elapsed = time.time() - start_time

                    # Check if pod is running
                    if status['status'] == 'running':
                        pod_ready = True
                        break

                    # If status is unknown but we've waited >60s, assume pod is ready
                    # (RunPod API status lags 1-3 minutes, but pod is usually functional after 60s)
                    if status['status'] == 'unknown' and elapsed > 60:
                        pod_ready = True
                        st.info("✅ Pod deployed successfully! (API status shows 'unknown' but pod is ready - this is normal for new pods)")
                        break

                    progress_pct = min(40 + int((elapsed / timeout) * 50), 90)
                    progress.progress(
                        progress_pct,
                        text=f"Waiting for pod... Status: {status['status']} ({int(elapsed)}s)"
                    )

                    time.sleep(5)

                # Show result
                if pod_ready:
                    progress.progress(100, text="✅ Deployment complete!")

                    st.success("🎉 Pod deployed successfully!")

                    if enable_auto_setup and github_repo and "YOUR_USERNAME" not in github_repo:
                        # Auto-setup enabled
                        # Extract pod ID prefix for HTTP proxy URL
                        pod_id_short = pod_id.split('-')[0] if '-' in pod_id else pod_id

                        st.info(f"""
                        **Pod Details:**
                        - **ID**: `{pod_id}`
                        - **Name**: {pod_name}
                        - **GPU**: {gpu_type}
                        - **Repository**: {github_repo} (branch: {github_branch})
                        - **SSH**: `ssh {ssh_host}@ssh.runpod.io`

                        **🚀 Auto-Setup is Running!**

                        The pod is automatically:
                        1. ✅ Cloning your repository from GitHub
                        2. ✅ Installing PostgreSQL + pgvector (configured for external access)
                        3. ✅ Setting up Python environment
                        4. ✅ Starting vLLM server

                        **This takes 10-15 minutes** (runs in background)

                        **Check Progress:**
                        ```bash
                        ssh {ssh_host}@ssh.runpod.io -i ~/.ssh/runpod_key
                        tail -f /workspace/startup.log
                        ```

                        **After Setup Completes (~15 min), Configure Your Mac:**

                        1. **Get Direct TCP Connection Details:**
                           - Go to RunPod → Your Pod → Connect → "Direct TCP ports"
                           - Note the PostgreSQL port (maps to :5432)

                        2. **Update your .env file:**
                           ```bash
                           # PostgreSQL - Use Direct TCP IP:PORT from RunPod
                           PGHOST=YOUR_TCP_IP
                           PGPORT=YOUR_TCP_PORT  # From RunPod direct TCP ports

                           # vLLM - Use HTTP Proxy (no tunnel needed!)
                           USE_VLLM=1
                           VLLM_API_BASE=https://{ssh_host}-8000.proxy.runpod.net/v1
                           ```

                        3. **Test Services:**
                           ```bash
                           source .env
                           curl https://{ssh_host}-8000.proxy.runpod.net/health
                           psql -h $PGHOST -p $PGPORT -U fryt -d vector_db -c "SELECT 1"
                           ```

                        4. **Run RAG Pipeline:**
                           - Use CLI: `python rag_low_level_m1_16gb_verbose.py --query-only --query "test"`
                           - Or use the "Query" tab in this UI!

                        **No SSH tunnel needed** - direct TCP + HTTPS proxy! 🚀
                        """)
                    else:
                        # Manual setup
                        st.info(f"""
                        **Pod Details:**
                        - **ID**: `{pod_id}`
                        - **Name**: {pod_name}
                        - **GPU**: {gpu_type}
                        - **SSH**: `ssh {ssh_host}@ssh.runpod.io`

                        **Next Steps:**
                        1. Use automated setup script (recommended):
                           ```bash
                           bash scripts/setup_runpod_pod_direct.sh TCP_HOST TCP_PORT ~/.ssh/runpod_key
                           ```
                           Get TCP_HOST and TCP_PORT from RunPod web UI → Your Pod → Connect → "SSH over exposed TCP"

                        2. Or manually SSH and initialize:
                           ```bash
                           ssh {ssh_host}@ssh.runpod.io
                           bash /workspace/rag-pipeline/scripts/init_runpod_services.sh
                           ```

                        3. Create SSH tunnel:
                           ```bash
                           ssh -L 8000:localhost:8000 -L 5432:localhost:5432 {ssh_host}@ssh.runpod.io
                           ```

                        4. Test and run queries from the "Query" tab!
                        """)

                    st.balloons()
                else:
                    progress.progress(100, text="⚠️ Timeout - check manually")

                    st.warning(f"""
                    ⚠️ **Pod created but status check timed out**

                    Your pod was created successfully but the API hasn't updated the status yet.
                    The pod is likely working - you can verify by checking the RunPod web UI.

                    **Pod Details:**
                    - **ID**: `{pod_id}`
                    - **Name**: {pod_name}

                    **Check Status:**
                    1. Go to [RunPod Pods](https://www.runpod.io/console/pods)
                    2. Find pod: {pod_name}
                    3. If it shows "Running", use the automated setup script

                    **Or refresh this page and check "Existing Pods" section above**
                    """)

                    # Refresh pod list
                    time.sleep(2)
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Deployment failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    # =========================================================================
    # Section 4: SSH Tunnel Management
    # =========================================================================
    st.subheader("4. SSH Tunnel Management")

    if pods and selected_pod_id:
        st.write(f"**Tunnel for**: {pod_names[selected_pod_id]}")

        # Get SSH host from cache first (more reliable)
        cached_host = st.session_state.get('pod_ssh_hosts', {}).get(selected_pod_id)

        if cached_host:
            ssh_host = cached_host
        else:
            # Fallback to API query
            status = manager.get_pod_status(selected_pod_id)
            ssh_host = status.get('ssh_host', '')

        # Port selection
        port_options = {
            "vLLM Server (8000)": 8000,
            "PostgreSQL (5432)": 5432,
            "Grafana (3000)": 3000
        }

        selected_ports = st.multiselect(
            "Ports to forward",
            options=list(port_options.keys()),
            default=["vLLM Server (8000)", "PostgreSQL (5432)"],
            key="tunnel_ports"
        )

        ports = [port_options[p] for p in selected_ports]

        # Generate SSH command
        if ports and ssh_host:
            port_forwards = " ".join([f"-L {p}:localhost:{p}" for p in ports])
            ssh_cmd = f"ssh -i ~/.ssh/runpod_key -N {port_forwards} {ssh_host}@ssh.runpod.io"
        elif not ssh_host:
            # Try to get SSH command with retries from manager
            ssh_cmd = manager.get_ssh_command(selected_pod_id, ports=[port_options[p] for p in selected_ports])
        else:
            ssh_cmd = "# Select at least one port to forward"

        st.write("**SSH Tunnel Command:**")
        st.code(ssh_cmd, language="bash")

        st.info("""
        💡 **How to use**:
        1. Copy command above
        2. Run in new terminal
        3. Keep running while using services
        4. Access services at `localhost:PORT`
        """)

        # Quick test buttons
        if 8000 in ports:
            if st.button("Test vLLM (requires tunnel)", key="test_vllm"):
                vllm_status = check_vllm_health()
                if vllm_status['status'] == 'healthy':
                    st.success(f"✅ vLLM is healthy! Latency: {vllm_status['latency_ms']}ms")
                else:
                    st.error(f"❌ vLLM: {vllm_status.get('error', 'Unreachable')}")

        if 5432 in ports:
            if st.button("Test PostgreSQL (requires tunnel)", key="test_pg"):
                pg_status = check_postgres_health()
                if pg_status['status'] == 'healthy':
                    st.success("✅ PostgreSQL is healthy!")
                    if pg_status.get('pgvector'):
                        st.success("✅ pgvector extension installed")
                else:
                    st.error(f"❌ PostgreSQL: {pg_status.get('error', 'Unreachable')}")

    else:
        st.info("Create a pod first to manage SSH tunnels")

    st.divider()

    # =========================================================================
    # Section 5: Cost Dashboard
    # =========================================================================
    st.subheader("5. Cost Dashboard")

    if pods:
        # Calculate total running cost
        total_cost_hr = sum(
            pod.get('costPerHr', 0)
            for pod in pods
            if (pod.get('runtime') or {}).get('containerState') == 'running'
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Pods", len([p for p in pods if (p.get('runtime') or {}).get('containerState') == 'running']))
        with col2:
            st.metric("Hourly Cost", f"${total_cost_hr:.2f}")
        with col3:
            st.metric("Daily (Current)", f"${total_cost_hr * 24:.2f}")
        with col4:
            st.metric("Monthly (Current)", f"${total_cost_hr * 24 * 30:.2f}")

        # Cost breakdown by pod
        st.write("**Cost Breakdown:**")

        cost_data = []
        for pod in pods:
            runtime = pod.get('runtime') or {}
            state = runtime.get('containerState', 'unknown')

            if state == 'running':
                cost = pod.get('costPerHr', 0)
                uptime = runtime.get('uptimeInSeconds', 0)
                hours_running = uptime / 3600

                cost_data.append({
                    "Pod": pod['name'],
                    "Cost/hr": f"${cost:.2f}",
                    "Uptime": f"{uptime // 60}min",
                    "Cost So Far": f"${cost * hours_running:.2f}"
                })

        if cost_data:
            df_cost = pd.DataFrame(cost_data)
            st.dataframe(df_cost, width='stretch', hide_index=True)

            # Projection chart
            st.write("**Cost Projection:**")

            hours_options = list(range(1, 25))
            costs_daily = [total_cost_hr * h for h in hours_options]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours_options,
                y=costs_daily,
                mode='lines+markers',
                name='Daily Cost',
                line=dict(color='#FF4B4B', width=3)
            ))

            fig.update_layout(
                title="Cost vs Usage (Daily)",
                xaxis_title="Hours per Day",
                yaxis_title="Daily Cost ($)",
                template="plotly_white",
                height=300
            )

            st.plotly_chart(fig, width='stretch')

        else:
            st.info("No running pods. No current costs.")

    else:
        st.info("No pods to track costs for")

    st.divider()

    # =========================================================================
    # Section 6: Quick Actions
    # =========================================================================
    st.subheader("6. Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 List Available GPUs", key="list_gpus"):
            with st.spinner("Fetching GPU types..."):
                try:
                    gpus = manager.list_available_gpus()

                    if gpus:
                        st.write("**Available GPUs:**")

                        gpu_data = []
                        for gpu in gpus[:10]:  # Show top 10
                            lowest_price = gpu.get('lowestPrice', {})
                            price = lowest_price.get('uninterruptablePrice', 0)

                            gpu_data.append({
                                "GPU": gpu['displayName'],
                                "Memory": f"{gpu['memoryInGb']}GB",
                                "Price/hr": f"${price:.2f}"
                            })

                        df_gpu = pd.DataFrame(gpu_data)
                        st.dataframe(df_gpu, width='stretch', hide_index=True)
                    else:
                        st.warning("No GPU data available")

                except Exception as e:
                    st.error(f"Failed to fetch GPUs: {e}")

    with col2:
        if st.button("💰 Estimate Costs", key="estimate_costs"):
            st.write("**Monthly Cost Scenarios:**")

            scenarios = [
                ("Development (2h/day)", 2),
                ("Testing (4h/day)", 4),
                ("Production (8h/day)", 8),
                ("24/7", 24)
            ]

            cost_scenarios = []
            for scenario_name, hours in scenarios:
                costs = manager.estimate_cost(hours, cost_per_hour=0.50)
                cost_scenarios.append({
                    "Scenario": scenario_name,
                    "Hours/Day": hours,
                    "Monthly Cost": f"${costs['total_cost']:.2f}"
                })

            df_scenarios = pd.DataFrame(cost_scenarios)
            st.dataframe(df_scenarios, width='stretch', hide_index=True)

    with col3:
        if st.button("🔍 System Health", key="health_check"):
            st.write("**Health Check:**")

            # vLLM
            vllm_health = check_vllm_health()
            if vllm_health['status'] == 'healthy':
                st.success(f"✅ vLLM: {vllm_health['status']} ({vllm_health['latency_ms']}ms)")
            else:
                st.error(f"❌ vLLM: {vllm_health.get('error', 'unreachable')}")

            # PostgreSQL
            pg_health = check_postgres_health()
            if pg_health['status'] == 'healthy':
                st.success("✅ PostgreSQL: healthy")
            else:
                st.error(f"❌ PostgreSQL: {pg_health.get('error', 'unreachable')}")

# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application."""

    # Sidebar
    st.sidebar.title("🔍 RAG Pipeline")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["Index Documents", "Query", "View Indexes", "Settings", "☁️ RunPod Deployment"],
    )

    st.sidebar.divider()

    # Quick stats
    try:
        indexes = list_vector_tables()
        st.sidebar.metric("Indexes", len(indexes))
    except Exception:
        st.sidebar.metric("Indexes", "?")

    st.sidebar.caption(f"DB: {st.session_state.db_name}")

    # Route to page
    if page == "Index Documents":
        page_index()
    elif page == "Query":
        page_query()
    elif page == "View Indexes":
        page_view_indexes()
    elif page == "Settings":
        page_settings()
    elif page == "☁️ RunPod Deployment":
        page_deployment()

# =============================================================================
# Authentication & Main Execution
# =============================================================================

# Initialize authenticator
authenticator = load_authenticator()

# Render login widget (stores results in st.session_state)
try:
    authenticator.login()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.stop()

# Get authentication status from session state
authentication_status = st.session_state.get('authentication_status')
name = st.session_state.get('name')
username = st.session_state.get('username')

# Handle authentication states
if authentication_status:
    # User authenticated - show logout in sidebar
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f'Welcome *{name}*')

    # Initialize session state for authenticated users
    init_session_state()

    # Call main to render the app
    main()

elif authentication_status is False:
    st.error('Username/password is incorrect')
    st.stop()

elif authentication_status is None:
    st.warning('Please enter your username and password')
    st.stop()
