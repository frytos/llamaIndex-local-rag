#!/usr/bin/env python3
"""
Enhanced Streamlit Web UI for RAG Pipeline
Launch with: streamlit run rag_web_enhanced.py

Features:
- Quick Start with presets (Fast/Balanced/Quality)
- Advanced mode exposing ALL parameters
- LLM parameter tuning in UI
- Better organization and evolutive design
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Add project root to path
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

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import shared utilities
from utils.naming import extract_model_short_name, generate_table_name
from utils.metrics import get_metrics

# Import from RAG script
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
    chunked,
)

# Database
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

# Visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# LlamaIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle

# =============================================================================
# Configuration & Presets
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / ".cache" / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Quality presets
PRESETS = {
    "Chat 💬": {
        "chunk_size": 300,
        "chunk_overlap": 50,
        "embed_model": "BAAI/bge-m3",
        "embed_dim": 1024,
        "embed_batch": 64,
        "top_k": 6,
        "description": "Optimized for chat messages & conversations (multilingual)"
    },
    "Fast ⚡": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embed_dim": 384,
        "embed_batch": 128,
        "top_k": 3,
        "description": "Fast indexing, good for quick tests (small chunks, fast model)"
    },
    "Balanced ⚖️": {
        "chunk_size": 700,
        "chunk_overlap": 150,
        "embed_model": "BAAI/bge-small-en",
        "embed_dim": 384,
        "embed_batch": 64,
        "top_k": 4,
        "description": "Balanced quality/speed (recommended for most use cases)"
    },
    "Quality 🎯": {
        "chunk_size": 1200,
        "chunk_overlap": 240,
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_dim": 768,
        "embed_batch": 32,
        "top_k": 5,
        "description": "High quality, slower indexing (large chunks, better model)"
    }
}

EMBED_MODELS = {
    "all-MiniLM-L6-v2 (Fast)": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "bge-small-en (Good)": ("BAAI/bge-small-en", 384),
    "bge-base-en (Better)": ("BAAI/bge-base-en-v1.5", 768),
    "bge-large-en (Best)": ("BAAI/bge-large-en-v1.5", 1024),
    "paraphrase-multilingual-MiniLM (Fast, Multilingual)": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
    "bge-m3 (Best, Multilingual - FR/EN/etc)": ("BAAI/bge-m3", 1024),
}

EMBED_BACKENDS = ["huggingface", "mlx"]

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="RAG Pipeline - Enhanced",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    div[data-testid="stExpander"] details summary {
        font-weight: 600;
    }
    .preset-card {
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        margin: 10px 0;
    }
    .preset-card:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state."""
    defaults = {
        # Database
        "db_host": os.environ.get("PGHOST", "localhost"),
        "db_port": os.environ.get("PGPORT", "5432"),
        "db_user": os.environ.get("PGUSER"),
        "db_password": os.environ.get("PGPASSWORD"),
        "db_name": os.environ.get("DB_NAME", "vector_db"),

        # Cached resources
        "embed_model": None,
        "llm": None,

        # State
        "last_indexed_nodes": None,
        "last_chunks": None,
        "last_embeddings": None,
        "query_history": [],

        # Chat mode
        "chat_history": [],
        "chat_enable_memory": True,
        "chat_max_turns": 10,
        "chat_auto_summarize": False,

        # UI preferences
        "advanced_mode": False,

        # Performance tracking
        "perf_total_sessions": 0,
        "perf_total_queries": 0,
        "perf_cache_hits": 0,
        "perf_cache_misses": 0,
        "perf_chunks_indexed": 0,
        "perf_queries_this_session": 0,
        "perf_chunks_indexed_this_session": 0,
        "perf_query_timings": [],  # List of dicts with query timing data
        "perf_indexing_timings": [],  # List of dicts with indexing timing data
        "perf_session_start": None,

        # Detailed performance metrics
        "perf_query_performance": [],  # Last 50 queries with full details
        "perf_cache_stats": {"hits": 0, "misses": 0, "total": 0},
        "perf_indexing_metrics": [],  # All indexing runs with detailed breakdown

        # Prometheus metrics
        "metrics": None,  # Will be initialized to get_metrics()
        "metrics_export_enabled": True,  # Enable metrics export
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Initialize session start time if not set
    if st.session_state.perf_session_start is None:
        st.session_state.perf_session_start = pd.Timestamp.now()
        st.session_state.perf_total_sessions += 1

    # Initialize Prometheus metrics
    if st.session_state.metrics is None:
        try:
            st.session_state.metrics = get_metrics()
        except Exception as e:
            st.session_state.metrics_export_enabled = False
            # Silently fail - metrics are optional

init_session_state()

# =============================================================================
# Database Utilities
# =============================================================================

def get_db_connection(autocommit=False):
    """Get database connection."""
    conn = psycopg2.connect(
        host=st.session_state.db_host,
        port=st.session_state.db_port,
        user=st.session_state.db_user,
        password=st.session_state.db_password,
        dbname=st.session_state.db_name,
    )
    conn.autocommit = autocommit
    return conn

def test_db_connection() -> Tuple[bool, str]:
    """Test database connection."""
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.close()
        conn.close()
        return True, f"✓ Connected! {version[:60]}..."
    except Exception as e:
        return False, f"✗ Error: {str(e)}"

def list_vector_tables() -> List[Dict[str, Any]]:
    """List all vector tables."""
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get tables with embedding column
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
            info = {"name": table, "rows": 0, "chunk_size": "?", "chunk_overlap": "?", "embed_model": "?"}

            # Get row count
            try:
                cur.execute(sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(table)))
                info["rows"] = cur.fetchone()["count"]
            except:
                conn.rollback()
                continue

            # Get metadata
            try:
                cur.execute(sql.SQL("""
                    SELECT metadata_->>'_chunk_size' as cs,
                           metadata_->>'_chunk_overlap' as co,
                           metadata_->>'_embed_model' as model
                    FROM {}
                    WHERE metadata_->>'_chunk_size' IS NOT NULL
                    LIMIT 1
                """).format(sql.Identifier(table)))
                row = cur.fetchone()
                if row:
                    info["chunk_size"] = row["cs"] or "?"
                    info["chunk_overlap"] = row["co"] or "?"
                    info["embed_model"] = row["model"].split("/")[-1] if row["model"] else "?"
            except:
                pass

            result.append(info)

        cur.close()
        conn.close()
        return result
    except Exception as e:
        st.error(f"Error listing tables: {e}")
        return []

def delete_table(table_name: str) -> bool:
    """Delete a vector table."""
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor()
        cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE').format(sql.Identifier(table_name)))
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting table: {e}")
        return False

def delete_empty_tables() -> int:
    """Delete all empty vector tables."""
    try:
        indexes = list_vector_tables()
        empty_tables = [idx["name"] for idx in indexes if idx["rows"] == 0]

        if not empty_tables:
            st.info("No empty tables to delete")
            return 0

        deleted = 0
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor()

        for table in empty_tables:
            try:
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE').format(sql.Identifier(table)))
                deleted += 1
            except Exception as e:
                st.warning(f"Could not delete `{table}`: {e}")

        cur.close()
        conn.close()

        st.success(f"🧹 Deleted {deleted} empty table(s)")
        return deleted
    except Exception as e:
        st.error(f"Error cleaning tables: {e}")
        return 0

def list_documents() -> Tuple[List[Path], List[Tuple[Path, int]]]:
    """List documents in data directory."""
    if not DATA_DIR.exists():
        return [], []

    supported = {
        ".pdf", ".docx", ".pptx", ".txt", ".md", ".html", ".htm",
        ".json", ".csv", ".xml", ".py", ".js", ".ts", ".java",
    }

    files = []
    folders = []

    for item in sorted(DATA_DIR.iterdir()):
        if item.is_file() and item.suffix.lower() in supported:
            files.append(item)
        elif item.is_dir() and not item.name.startswith("."):
            count = sum(1 for f in item.rglob("*") if f.is_file())
            folders.append((item, count))

    return files, folders

# =============================================================================
# Cached Resources
# =============================================================================

@st.cache_resource
def get_embed_model(model_name: str):
    """Cache embedding model."""
    import rag_low_level_m1_16gb_verbose as rag
    original = rag.S.embed_model_name
    rag.S.embed_model_name = model_name
    model = build_embed_model()
    rag.S.embed_model_name = original
    return model

@st.cache_resource
def get_llm(context_window: int = 3072, max_tokens: int = 256, temperature: float = 0.1):
    """Cache LLM with specific configuration.

    Args:
        context_window: Maximum context window size
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Cached LLM instance for the given configuration
    """
    import rag_low_level_m1_16gb_verbose as rag

    # Set LLM parameters before building
    rag.S.context_window = context_window
    rag.S.max_new_tokens = max_tokens
    rag.S.temperature = temperature

    return build_llm()

# =============================================================================
# Visualization
# =============================================================================

def render_chunk_distribution(chunks: List[str]):
    """Render chunk size distribution."""
    sizes = [len(c) for c in chunks]

    fig = px.histogram(
        x=sizes,
        nbins=30,
        labels={"x": "Chunk Size (characters)", "y": "Count"},
        title="Chunk Size Distribution",
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, width="stretch")

def render_embedding_viz(embeddings: np.ndarray, texts: List[str], sources: List[str],
                         method: str = "t-SNE", dimensions: int = 2):
    """Render embedding visualization."""
    if len(embeddings) < 3:
        st.warning("Need at least 3 embeddings")
        return

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
        else:
            reducer = PCA(n_components=dimensions)
            coords = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "text": [t[:100] + "..." if len(t) > 100 else t for t in texts],
        "source": sources,
    })
    if dimensions == 3:
        df["z"] = coords[:, 2]

    if dimensions == 2:
        fig = px.scatter(df, x="x", y="y", color="source", hover_data=["text"],
                        title=f"{method} Projection ({len(embeddings)} chunks)")
    else:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="source", hover_data=["text"],
                           title=f"{method} 3D Projection ({len(embeddings)} chunks)")

    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

# =============================================================================
# Pages
# =============================================================================

def page_quick_start():
    """Quick Start page with presets."""
    st.header("🚀 Quick Start")
    st.caption("Index and query in one click using presets")

    # Mode selector at the top
    st.subheader("Pipeline Mode")
    mode = st.radio(
        "Choose operation mode:",
        ["Full Pipeline", "Index Only", "Query Only"],
        horizontal=True,
        help="Choose what operations to perform",
        key="qs_mode"
    )

    if mode == "Index Only":
        st.info("✅ Will index documents without querying")
    elif mode == "Query Only":
        st.info("✅ Will use existing index (no re-indexing)")
    else:
        st.info("✅ Will index documents then run test query")

    st.divider()

    # Info banner
    st.info("""
    💡 **Quick Start Guide:**
    1. Choose a quality preset (Fast/Balanced/Quality)
    2. Select your document or folder
    3. Click "Index Now" - uses safe, reliable settings
    4. Go to Query tab to search your indexed data

    ✅ **Safe mode enabled:** Pure vector search, no advanced features
    """)

    # Initialize session state for preset selection
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = None

    # Step 1: Select preset
    st.subheader("1. Choose Quality Preset")

    # Show current selection with option to change
    if st.session_state.selected_preset:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"✓ Selected: **{st.session_state.selected_preset}**")
        with col2:
            if st.button("Change Preset", key="change_preset"):
                st.session_state.selected_preset = None
                st.rerun()

    # Only show preset cards if nothing is selected
    if not st.session_state.selected_preset:
        # Create columns dynamically based on number of presets
        num_presets = len(PRESETS)
        cols = st.columns(num_presets)

        for idx, (preset_name, preset_config) in enumerate(PRESETS.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="preset-card">
                    <h3>{preset_name}</h3>
                    <p>{preset_config['description']}</p>
                    <ul style="font-size: 0.9em; color: #666;">
                        <li>Chunk: {preset_config['chunk_size']}/{preset_config['chunk_overlap']}</li>
                        <li>Model: {preset_config['embed_model'].split('/')[-1]}</li>
                        <li>TOP_K: {preset_config['top_k']}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Use {preset_name}", key=f"preset_{idx}", width="stretch"):
                    st.session_state.selected_preset = preset_name
                    st.rerun()

    # Show document selection if preset is selected
    if st.session_state.selected_preset:
        preset = PRESETS[st.session_state.selected_preset]

        # Step 2: Select document
        st.subheader("2. Select Document")

        files, folders = list_documents()
        options = []
        option_paths = {}

        for folder, count in folders:
            label = f"📁 {folder.name}/ ({count} files)"
            options.append(label)
            option_paths[label] = folder

        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            label = f"📄 {f.name} ({size_mb:.1f} MB)"
            options.append(label)
            option_paths[label] = f

        options.append("📝 Custom path")

        selected = st.selectbox("Document:", options, key="qs_doc")

        if selected == "📝 Custom path":
            doc_path = st.text_input("Path:", value=str(DATA_DIR))
            doc_path = Path(doc_path) if doc_path else None
        else:
            doc_path = option_paths.get(selected)

        if doc_path and doc_path.exists():
            st.success(f"✓ Document: `{doc_path}`")

            # Generate table name
            table_name = generate_table_name(
                doc_path,
                preset['chunk_size'],
                preset['chunk_overlap'],
                preset['embed_model']
            )

            table_name = st.text_input("Index name:", value=table_name, key="qs_table",
                                       help="⚠️ Use the suggested name to avoid creating empty duplicate tables")

            # Warning about table naming
            if table_name != generate_table_name(doc_path, preset['chunk_size'], preset['chunk_overlap'], preset['embed_model']):
                st.warning("⚠️ **Table name changed!** If you mistype this during querying, an empty duplicate table will be created.")

            # Step 3: Configuration
            st.subheader("3. Configuration")

            with st.expander("⚙️ Advanced Options (Optional)", expanded=False):
                st.caption("These are automatically set by the preset, but you can customize:")

                col1, col2 = st.columns(2)
                with col1:
                    custom_chunk_size = st.number_input("Chunk Size", 100, 3000, preset['chunk_size'], 50, key="qs_cs")
                    custom_embed_batch = st.number_input("Embed Batch", 16, 256, preset['embed_batch'], 16, key="qs_eb")
                with col2:
                    custom_chunk_overlap = st.number_input("Chunk Overlap", 0, 500, preset['chunk_overlap'], 10, key="qs_co")
                    custom_top_k = st.number_input("TOP_K", 1, 10, preset['top_k'], 1, key="qs_tk")

                st.info(f"💡 Preset defaults: cs={preset['chunk_size']}, ov={preset['chunk_overlap']}, batch={preset['embed_batch']}, top_k={preset['top_k']}")

            # Step 4: Start
            st.subheader("4. Start Indexing")

            col1, col2 = st.columns(2)
            with col1:
                reset_table = st.checkbox("Reset if exists", value=True, key="qs_reset")
            with col2:
                st.caption("✅ Safe mode: Pure vector search")

            # Info box
            st.info("""
            ✅ **Safe Configuration Enabled:**
            - Pure vector search (no hybrid BM25)
            - No metadata filtering
            - No query expansion
            - Fast and reliable results
            """)

            if st.button("🚀 Index Now", type="primary", width="stretch", key="qs_index"):
                # Use custom values if changed, otherwise preset defaults
                run_indexing(
                    doc_path=doc_path,
                    table_name=table_name,
                    chunk_size=custom_chunk_size if 'qs_cs' in st.session_state else preset['chunk_size'],
                    chunk_overlap=custom_chunk_overlap if 'qs_co' in st.session_state else preset['chunk_overlap'],
                    embed_model_name=preset['embed_model'],
                    embed_dim=preset['embed_dim'],
                    embed_batch=custom_embed_batch if 'qs_eb' in st.session_state else preset['embed_batch'],
                    reset_table=reset_table,
                    mode=mode,
                )

def page_advanced_index():
    """Advanced indexing with all parameters."""
    st.header("⚙️ Advanced Indexing")
    st.caption("Full control over all indexing parameters")

    # Mode selector at the top
    st.subheader("Pipeline Mode")
    mode = st.radio(
        "Choose operation mode:",
        ["Full Pipeline", "Index Only", "Query Only"],
        horizontal=True,
        help="Choose what operations to perform",
        key="adv_mode"
    )

    if mode == "Index Only":
        st.info("✅ Will index documents without querying")
    elif mode == "Query Only":
        st.info("✅ Will use existing index (no re-indexing)")
    else:
        st.info("✅ Will index documents then run test query")

    st.divider()

    # Info banner
    st.info("""
    📌 **Indexing vs Querying:**
    - **This page:** Configure how documents are indexed (chunking, embedding model)
    - **Query page:** Configure search behavior (hybrid search, filters, reranking)

    💡 After indexing here, go to the **Query** page to search your data.
    """)

    # Document selection
    st.subheader("1. Document Selection")

    files, folders = list_documents()
    options = []
    option_paths = {}

    for folder, count in folders:
        label = f"📁 {folder.name}/ ({count} files)"
        options.append(label)
        option_paths[label] = folder

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        label = f"📄 {f.name} ({size_mb:.1f} MB)"
        options.append(label)
        option_paths[label] = f

    options.append("📝 Custom path")

    selected = st.selectbox("Document:", options, key="adv_doc")

    if selected == "📝 Custom path":
        doc_path = st.text_input("Path:", value=str(DATA_DIR))
        doc_path = Path(doc_path) if doc_path else None
    else:
        doc_path = option_paths.get(selected)

    if not doc_path or not doc_path.exists():
        st.warning("Please select a valid document")
        return

    st.success(f"✓ Selected: `{doc_path}`")

    # Parameters
    st.subheader("2. Indexing Parameters")

    # Chunking
    with st.expander("📄 Chunking Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size", 100, 3000, 700, 50,
                                   help="Characters per chunk (100-3000)")
        with col2:
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 150, 10,
                                      help="Overlap between chunks")

        overlap_pct = 100 * chunk_overlap / chunk_size
        st.caption(f"ℹ️ Overlap: {overlap_pct:.0f}% (recommended: 15-25%)")

    # Embedding
    with st.expander("🔢 Embedding Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            embed_choice = st.selectbox("Model:", list(EMBED_MODELS.keys()), index=1)
            embed_model_name, embed_dim = EMBED_MODELS[embed_choice]

        with col2:
            embed_batch = st.number_input("Batch Size", 8, 256, 64, 8,
                                         help="Batch size for embedding")

        with col3:
            embed_backend = st.selectbox("Backend:", EMBED_BACKENDS, index=0,
                                        help="huggingface or mlx (Apple Silicon)")

        st.caption(f"ℹ️ Model: `{embed_model_name}` | Dimensions: {embed_dim}")

    # Table
    st.subheader("3. Index Configuration")

    suggested_name = generate_table_name(doc_path, chunk_size, chunk_overlap, embed_model_name)
    table_name = st.text_input("Table name:", value=suggested_name)
    reset_table = st.checkbox("Reset table if exists", value=True)

    # Start indexing
    st.subheader("4. Start Indexing")

    if st.button("🚀 Start Indexing", type="primary", width="stretch"):
        run_indexing(
            doc_path=doc_path,
            table_name=table_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model_name=embed_model_name,
            embed_dim=embed_dim,
            embed_batch=embed_batch,
            reset_table=reset_table,
            mode=mode,
        )

def run_indexing(doc_path: Path, table_name: str, chunk_size: int, chunk_overlap: int,
                 embed_model_name: str, embed_dim: int, embed_batch: int, reset_table: bool,
                 mode: str = "Full Pipeline"):
    """Run indexing pipeline with configurable mode.

    Args:
        doc_path: Path to document or folder to index
        table_name: Name of the PostgreSQL table
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embed_model_name: Name of embedding model
        embed_dim: Embedding dimension
        embed_batch: Batch size for embedding
        reset_table: Whether to drop existing table
        mode: Pipeline mode - "Full Pipeline", "Index Only", or "Query Only"
    """
    import time

    # Update settings
    import rag_low_level_m1_16gb_verbose as rag
    rag.S.pdf_path = str(doc_path)
    rag.S.table = table_name
    rag.S.chunk_size = chunk_size
    rag.S.chunk_overlap = chunk_overlap
    rag.S.embed_model_name = embed_model_name
    rag.S.embed_dim = embed_dim
    rag.S.embed_batch = embed_batch
    rag.S.reset_table = reset_table

    # Force safe defaults for indexing (no advanced features during index creation)
    rag.S.enable_query_expansion = False
    rag.S.enable_reranking = False
    rag.S.hybrid_alpha = 1.0  # Pure vector search
    rag.S.enable_filters = False
    rag.S.mmr_threshold = 0.0  # Disable MMR diversity

    # Handle Query Only mode - skip indexing
    if mode == "Query Only":
        st.info("🔍 Query Only mode selected - skipping indexing")
        st.warning("This mode is primarily for the Query page. Use the Query page to search existing indexes.")
        st.info(f"To query the index `{table_name}`, go to the Query page and select it from the dropdown.")
        return

    status_label = "Indexing Pipeline" if mode != "Index Only" else "Indexing Pipeline (Index Only)"
    status = st.status(status_label, expanded=True)
    pipeline_start = time.time()

    # Track timing for each stage
    timings = {}

    with status:
        # Load
        st.write("**Step 1: Loading documents...**")
        progress = st.progress(0)
        step_start = time.time()

        try:
            docs = load_documents(str(doc_path))
            step_time = time.time() - step_start
            progress.progress(20)

            st.success(f"✓ Loaded {len(docs)} document(s) in {step_time:.2f}s")

            total_chars = sum(len(d.text) for d in docs)
            total_words = sum(len(d.text.split()) for d in docs)

            # Show detailed stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", f"{len(docs):,}")
            with col2:
                st.metric("Characters", f"{total_chars:,}")
            with col3:
                st.metric("Words", f"{total_words:,}")

            st.caption(f"⚡ Loading speed: {len(docs)/step_time:.1f} docs/sec | Avg doc size: {total_chars//len(docs):,} chars")
            timings['load'] = step_time
        except Exception as e:
            st.error(f"Error loading: {e}")
            status.update(label="❌ Failed", state="error")
            return

        # Chunk
        st.write("**Step 2: Chunking...**")
        step_start = time.time()

        try:
            chunks, doc_idxs = chunk_documents(docs)
            step_time = time.time() - step_start
            progress.progress(40)

            st.success(f"✓ Created {len(chunks):,} chunks in {step_time:.2f}s")

            st.session_state.last_chunks = chunks
            sizes = [len(c) for c in chunks]
            avg_size = sum(sizes) / len(sizes)

            # Show detailed stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chunks", f"{len(chunks):,}")
            with col2:
                st.metric("Avg Size", f"{avg_size:.0f} chars")
            with col3:
                st.metric("Range", f"{min(sizes)}-{max(sizes)}")
            with col4:
                st.metric("Per Doc", f"{len(chunks)/len(docs):.1f}")

            overlap_pct = 100 * chunk_overlap / chunk_size
            st.caption(f"⚡ Chunking speed: {len(docs)/step_time:.1f} docs/sec, {len(chunks)/step_time:.0f} chunks/sec | Overlap: {overlap_pct:.0f}%")
            timings['chunk'] = step_time
        except Exception as e:
            st.error(f"Error chunking: {e}")
            status.update(label="❌ Failed", state="error")
            return

        # Embed
        st.write("**Step 3: Computing embeddings...**")
        embed_stage_start = time.time()

        # Warn if too many chunks
        if len(chunks) > 10000:
            est_time_min = len(chunks) // 1000
            st.warning(f"⚠️ Large dataset: {len(chunks):,} chunks will take ~{est_time_min} minutes to embed")

        try:
            # Build nodes
            node_build_start = time.time()
            nodes = build_nodes(docs, chunks, doc_idxs)
            node_build_time = time.time() - node_build_start
            st.caption(f"Built {len(nodes):,} nodes in {node_build_time:.2f}s")

            # Load embedding model
            model_load_start = time.time()
            embed_model = get_embed_model(embed_model_name)
            model_load_time = time.time() - model_load_start
            st.caption(f"Loaded embedding model `{embed_model_name.split('/')[-1]}` in {model_load_time:.2f}s")

            total_batches = (len(nodes) + embed_batch - 1) // embed_batch
            embeddings_list = []

            st.caption(f"Embedding {len(nodes):,} chunks in {total_batches:,} batches (batch size: {embed_batch})...")
            embed_progress_bar = st.progress(0)
            embed_status_text = st.empty()
            embed_metrics = st.empty()

            embed_start = time.time()

            for i, batch in enumerate(chunked(nodes, embed_batch)):
                batch_start = time.time()
                texts = [n.get_content() for n in batch]

                try:
                    batch_embeddings = embed_model.get_text_embedding_batch(texts)

                    # Verify batch size matches
                    if len(batch_embeddings) != len(batch):
                        st.error(f"Batch {i+1}: Embedding count mismatch! Expected {len(batch)}, got {len(batch_embeddings)}")
                        st.warning("Falling back to one-by-one embedding for this batch...")

                        # Fallback: embed one by one
                        batch_embeddings = []
                        for text in texts:
                            try:
                                emb = embed_model.get_text_embedding(text)
                                batch_embeddings.append(emb)
                            except Exception as e:
                                st.error(f"Failed to embed text (length {len(text)}): {str(e)[:100]}")
                                # Use zero vector as fallback
                                batch_embeddings.append([0.0] * embed_dim)

                    for node, emb in zip(batch, batch_embeddings):
                        node.embedding = emb
                        embeddings_list.append(emb)

                except Exception as e:
                    st.error(f"Error in batch {i+1}/{total_batches}: {e}")
                    st.warning("Attempting fallback to individual embedding...")

                    # Fallback: try embedding each text individually
                    batch_embeddings = []
                    for j, text in enumerate(texts):
                        try:
                            emb = embed_model.get_text_embedding(text)
                            batch_embeddings.append(emb)
                        except Exception as e2:
                            st.error(f"Failed to embed chunk {j+1} in batch: {str(e2)[:100]}")
                            # Use zero vector as fallback
                            batch_embeddings.append([0.0] * embed_dim)

                    for node, emb in zip(batch, batch_embeddings):
                        node.embedding = emb
                        embeddings_list.append(emb)

                # Calculate progress and metrics
                progress_pct = (i + 1) / total_batches
                chunks_done = min((i + 1) * embed_batch, len(nodes))
                elapsed = time.time() - embed_start
                rate = chunks_done / elapsed if elapsed > 0 else 0
                eta_seconds = (len(nodes) - chunks_done) / rate if rate > 0 else 0

                # Update progress bar
                embed_progress_bar.progress(progress_pct)

                # Update status with detailed info
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = f"{eta_seconds/60:.1f}m"

                embed_status_text.text(
                    f"Embedded {chunks_done:,} / {len(nodes):,} chunks ({progress_pct*100:.1f}%) | "
                    f"Rate: {rate:.0f} chunks/sec | ETA: {eta_str}"
                )

                # Show live metrics every 10 batches
                if i % 10 == 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        embed_metrics.metric("Progress", f"{chunks_done:,} / {len(nodes):,}")
                    with col2:
                        embed_metrics.metric("Speed", f"{rate:.0f} chunks/sec")
                    with col3:
                        embed_metrics.metric("ETA", eta_str)

                progress.progress(40 + int(40 * progress_pct))

            embed_time = time.time() - embed_start
            embed_status_text.empty()

            st.success(f"✓ Embedded {len(nodes):,} chunks in {embed_time:.2f}s")

            # Final embedding stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embedding Dim", f"{embed_dim}")
            with col2:
                st.metric("Avg Speed", f"{len(nodes)/embed_time:.0f} chunks/sec")
            with col3:
                st.metric("Total Time", f"{embed_time:.1f}s")

            st.session_state.last_indexed_nodes = nodes
            st.session_state.last_embeddings = np.array(embeddings_list)
            timings['embed'] = time.time() - embed_stage_start
        except Exception as e:
            st.error(f"Error embedding: {e}")
            status.update(label="❌ Failed", state="error")
            return

        # Store
        st.write("**Step 4: Storing in database...**")
        store_stage_start = time.time()

        try:
            ensure_db_exists()
            ensure_pgvector_extension()

            if reset_table:
                st.caption("Dropping existing table...")
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE').format(sql.Identifier(table_name)))
                    conn.commit()
                    cur.close()
                    conn.close()
                    st.caption(f"✓ Dropped table `{table_name}`")
                except:
                    pass

            st.caption(f"Connecting to database: `{st.session_state.db_name}` @ {st.session_state.db_host}:{st.session_state.db_port}")
            vector_store = make_vector_store()

            batch_size = 250
            total_batches = (len(nodes) + batch_size - 1) // batch_size

            st.caption(f"Inserting {len(nodes):,} chunks in {total_batches:,} batches (batch size: {batch_size})...")
            store_progress = st.progress(0)
            store_status = st.empty()

            store_start = time.time()

            for i, batch in enumerate(chunked(nodes, batch_size)):
                vector_store.add(list(batch))

                progress_pct = (i + 1) / total_batches
                chunks_stored = min((i + 1) * batch_size, len(nodes))
                elapsed = time.time() - store_start
                rate = chunks_stored / elapsed if elapsed > 0 else 0

                store_progress.progress(progress_pct)
                store_status.text(f"Stored {chunks_stored:,} / {len(nodes):,} chunks ({progress_pct*100:.0f}%) | Rate: {rate:.0f} chunks/sec")
                progress.progress(80 + int(20 * progress_pct))

            store_time = time.time() - store_start
            store_status.empty()

            st.success(f"✓ Stored {len(nodes):,} chunks in `{table_name}` in {store_time:.2f}s")

            # Storage stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Table", table_name)
            with col2:
                st.metric("Insert Speed", f"{len(nodes)/store_time:.0f} chunks/sec")

            timings['store'] = time.time() - store_stage_start
        except Exception as e:
            st.error(f"Error storing: {e}")
            status.update(label="❌ Failed", state="error")
            return

        # If Index Only mode, stop here
        if mode == "Index Only":
            pipeline_time = time.time() - pipeline_start
            status.update(label=f"✅ Indexing Complete! ({pipeline_time:.1f}s total)", state="complete")

            st.success(f"🎉 Index Only mode complete! Documents indexed to `{table_name}`")
            st.info("💡 To query this index, go to the Query page and select it from the dropdown.")

            # Show summary for Index Only mode
            st.divider()
            st.subheader("📊 Indexing Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", f"{len(docs):,}")
            with col2:
                st.metric("Total Chunks", f"{len(chunks):,}")
            with col3:
                st.metric("Chunks/Doc", f"{len(chunks)/len(docs):.1f}")
            with col4:
                st.metric("Total Time", f"{pipeline_time:.1f}s")

            st.caption(f"⚡ Overall throughput: {len(chunks)/pipeline_time:.0f} chunks/sec | {len(docs)/pipeline_time:.1f} docs/sec")

            # Time breakdown
            with st.expander("⏱️ Time Breakdown by Stage"):
                timing_data = {
                    "Stage": ["Load Documents", "Chunking", "Embedding", "Database Storage"],
                    "Time (s)": [
                        timings.get('load', 0),
                        timings.get('chunk', 0),
                        timings.get('embed', 0),
                        timings.get('store', 0)
                    ],
                    "% of Total": [
                        100 * timings.get('load', 0) / pipeline_time,
                        100 * timings.get('chunk', 0) / pipeline_time,
                        100 * timings.get('embed', 0) / pipeline_time,
                        100 * timings.get('store', 0) / pipeline_time,
                    ]
                }

                df_timing = pd.DataFrame(timing_data)
                df_timing['Time (s)'] = df_timing['Time (s)'].round(2)
                df_timing['% of Total'] = df_timing['% of Total'].round(1)
                st.dataframe(df_timing, width="stretch", hide_index=True)

            # Track performance metrics
            st.session_state.perf_chunks_indexed += len(chunks)
            st.session_state.perf_chunks_indexed_this_session += len(chunks)
            st.session_state.perf_indexing_metrics.append({
                "timestamp": pd.Timestamp.now(),
                "table_name": table_name,
                "num_docs": len(docs),
                "num_chunks": len(chunks),
                "total_time": pipeline_time,
                "load_time": timings.get('load', 0),
                "chunk_time": timings.get('chunk', 0),
                "embed_time": timings.get('embed', 0),
                "store_time": timings.get('store', 0),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embed_model": embed_model_name,
            })

            return

    # Pipeline complete - show summary
    pipeline_time = time.time() - pipeline_start
    status.update(label=f"✅ Indexing Complete! ({pipeline_time:.1f}s total)", state="complete")

    # Show pipeline summary
    st.divider()
    st.subheader("📊 Pipeline Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", f"{len(docs):,}")
    with col2:
        st.metric("Total Chunks", f"{len(chunks):,}")
    with col3:
        st.metric("Chunks/Doc", f"{len(chunks)/len(docs):.1f}")
    with col4:
        st.metric("Total Time", f"{pipeline_time:.1f}s")

    st.caption(f"⚡ Overall throughput: {len(chunks)/pipeline_time:.0f} chunks/sec | {len(docs)/pipeline_time:.1f} docs/sec")

    # Time breakdown
    with st.expander("⏱️ Time Breakdown by Stage"):
        timing_data = {
            "Stage": ["Load Documents", "Chunking", "Embedding", "Database Storage"],
            "Time (s)": [
                timings.get('load', 0),
                timings.get('chunk', 0),
                timings.get('embed', 0),
                timings.get('store', 0)
            ],
            "% of Total": [
                100 * timings.get('load', 0) / pipeline_time,
                100 * timings.get('chunk', 0) / pipeline_time,
                100 * timings.get('embed', 0) / pipeline_time,
                100 * timings.get('store', 0) / pipeline_time,
            ]
        }

        df_timing = pd.DataFrame(timing_data)
        df_timing['Time (s)'] = df_timing['Time (s)'].round(2)
        df_timing['% of Total'] = df_timing['% of Total'].round(1)

        # Create bar chart
        fig = px.bar(
            df_timing,
            x='Stage',
            y='Time (s)',
            text='% of Total',
            title='Time Spent per Stage',
            labels={'Time (s)': 'Time (seconds)'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(df_timing, width="stretch", hide_index=True)

    # Visualizations
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Chunk Distribution")
        if st.session_state.last_chunks:
            render_chunk_distribution(st.session_state.last_chunks)

    with col2:
        st.subheader("Sample Chunks")
        if st.session_state.last_chunks:
            for i, chunk in enumerate(st.session_state.last_chunks[:3]):
                with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)

    # Embedding viz
    if st.session_state.last_embeddings is not None and len(st.session_state.last_embeddings) > 2:
        st.subheader("Embedding Visualization")

        col1, col2 = st.columns(2)
        with col1:
            viz_method = st.selectbox("Method:", ["t-SNE", "PCA", "UMAP"], key="index_viz_method")
        with col2:
            viz_dims = st.radio("Dimensions:", [2, 3], horizontal=True, key="index_viz_dims")

        texts = [n.get_content() for n in st.session_state.last_indexed_nodes]
        sources = [Path(n.metadata.get("source", "unknown")).name for n in st.session_state.last_indexed_nodes]

        limit = min(500, len(st.session_state.last_embeddings))
        render_embedding_viz(
            st.session_state.last_embeddings[:limit],
            texts[:limit],
            sources[:limit],
            method=viz_method,
            dimensions=viz_dims
        )

    # Track performance metrics (also for full pipeline)
    st.session_state.perf_chunks_indexed += len(chunks)
    st.session_state.perf_chunks_indexed_this_session += len(chunks)
    st.session_state.perf_indexing_metrics.append({
        "timestamp": pd.Timestamp.now(),
        "table_name": table_name,
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "total_time": pipeline_time,
        "load_time": timings.get('load', 0),
        "chunk_time": timings.get('chunk', 0),
        "embed_time": timings.get('embed', 0),
        "store_time": timings.get('store', 0),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model": embed_model_name,
    })

def search_chunks(table_name: str, query: str, top_k: int,
                  hybrid_alpha: float = 1.0, enable_filters: bool = False, mmr_threshold: float = 0.0,
                  enable_query_expansion: bool = False, query_expansion_method: str = "llm",
                  query_expansion_count: int = 2, enable_reranking: bool = False,
                  rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                  rerank_candidates: int = 12, rerank_top_k: int = 4,
                  enable_semantic_cache: bool = False, semantic_cache_threshold: float = 0.92,
                  semantic_cache_max_size: int = 1000, semantic_cache_ttl: int = 86400,
                  enable_hyde: bool = False, num_hypotheses: int = 1,
                  hypothesis_length: int = 100, fusion_method: str = "rrf"):
    """Search and display chunks only (no LLM generation). Uses all retrieval parameters."""

    import rag_low_level_m1_16gb_verbose as rag

    # Update settings to match query parameters
    rag.S.table = table_name
    rag.S.top_k = top_k
    rag.S.hybrid_alpha = hybrid_alpha
    rag.S.enable_filters = enable_filters
    rag.S.mmr_threshold = mmr_threshold
    rag.S.enable_query_expansion = enable_query_expansion
    rag.S.query_expansion_method = query_expansion_method
    rag.S.query_expansion_count = query_expansion_count
    rag.S.enable_reranking = enable_reranking
    rag.S.rerank_model = rerank_model
    rag.S.rerank_candidates = rerank_candidates
    rag.S.rerank_top_k = rerank_top_k
    rag.S.enable_semantic_cache = enable_semantic_cache
    rag.S.semantic_cache_threshold = semantic_cache_threshold
    rag.S.semantic_cache_max_size = semantic_cache_max_size
    rag.S.semantic_cache_ttl = semantic_cache_ttl
    rag.S.enable_hyde = enable_hyde
    rag.S.num_hypotheses = num_hypotheses
    rag.S.hypothesis_length = hypothesis_length
    rag.S.fusion_method = fusion_method

    st.subheader(f"🔍 Top {top_k} Chunks for: \"{query}\"")

    # Show active search features
    features = []
    if hybrid_alpha == 1.0:
        features.append("Pure Vector")
    elif hybrid_alpha == 0.0:
        features.append("Pure BM25")
    else:
        features.append(f"Hybrid (α={hybrid_alpha:.1f})")

    if enable_filters:
        features.append("Metadata Filtering")
    if mmr_threshold > 0:
        features.append(f"MMR Diversity ({mmr_threshold:.1f})")
    if enable_query_expansion:
        features.append("Query Expansion")
    if enable_reranking:
        features.append("Reranking")
    if enable_semantic_cache:
        features.append("Semantic Cache")
    if enable_hyde:
        features.append("HyDE")

    st.caption(f"🔍 Active features: {', '.join(features) if features else 'Standard search'}")

    with st.spinner("Retrieving chunks..."):
        try:
            # Build retriever based on settings (same logic as run_query)
            embed_model = build_embed_model()
            vector_store = make_vector_store()

            # Use HybridRetriever if advanced features enabled, otherwise simple VectorDBRetriever
            if hybrid_alpha < 1.0 or enable_filters or mmr_threshold > 0:
                from rag_low_level_m1_16gb_verbose import HybridRetriever
                retriever = HybridRetriever(
                    vector_store,
                    embed_model,
                    similarity_top_k=top_k,
                    alpha=hybrid_alpha,
                    enable_metadata_filter=enable_filters,
                    mmr_threshold=mmr_threshold
                )
            else:
                retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=top_k)

            # Retrieve chunks
            from llama_index.core import QueryBundle
            try:
                results = retriever._retrieve(QueryBundle(query_str=query))
            except ZeroDivisionError as e:
                st.error("❌ BM25 initialization failed - corpus is empty")
                st.warning("💡 **Solutions:**")
                st.markdown("""
                1. Set **Hybrid Search to 1.0** (Pure Vector) - recommended
                2. Check embedding model dimension matches table
                3. Verify table is not empty
                """)
                st.info("ℹ️ BM25 requires loading all chunks into memory. If this fails, use pure vector search (Hybrid=1.0).")

                # Show debug info
                with st.expander("🔍 Debug Info"):
                    st.code(f"Table: {table_name}\nHybrid Alpha: {hybrid_alpha}\nError: {str(e)}")
                return
            except Exception as e:
                st.error(f"❌ Retrieval error: {str(e)}")
                st.warning("💡 Try setting Hybrid Search to 1.0 (Pure Vector)")
                with st.expander("🔍 Full Error"):
                    import traceback
                    st.code(traceback.format_exc())
                return

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

            # Display each chunk with rich metadata
            for i, result in enumerate(results):
                score = result.score
                meta = result.node.metadata or {}

                # Color coding
                if score > 0.7:
                    badge = "🟢 Excellent"
                elif score > 0.5:
                    badge = "🟡 Good"
                elif score > 0.3:
                    badge = "🟠 Fair"
                else:
                    badge = "🔴 Low"

                with st.expander(f"**Chunk {i+1}:** {badge} (Score: {score:.4f})", expanded=(i < 3)):
                    # Two-column layout
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown("**📊 Metadata:**")
                        st.metric("Similarity", f"{score:.4f}")

                        # Participants (new field!)
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
                            if len(participants) > 5:
                                st.caption(f"... and {len(participants)-5} more")

                        # Conversation type
                        conv_type = meta.get('_conversation_type')
                        if conv_type:
                            icon = "👥" if conv_type == "group_chat" else "💬"
                            st.caption(f"{icon} {conv_type.replace('_', ' ').title()}")

                        # Dates
                        primary_date = meta.get('_primary_date')
                        if primary_date:
                            st.markdown(f"**📅 Date:** {primary_date}")

                        dates = meta.get('_dates', [])
                        if isinstance(dates, str):
                            import json
                            try:
                                dates = json.loads(dates)
                            except:
                                dates = []
                        if dates and len(dates) > 1:
                            st.caption(f"Date range: {len(dates)} dates")

                        # Content type
                        content_type = meta.get('_content_type')
                        if content_type:
                            st.caption(f"Type: {content_type}")

                        # Chunk config
                        chunk_size = meta.get('_chunk_size')
                        chunk_overlap = meta.get('_chunk_overlap')
                        if chunk_size:
                            st.caption(f"Config: {chunk_size}/{chunk_overlap}")

                        # Word stats
                        word_count = meta.get('_word_count')
                        if word_count:
                            st.caption(f"Words: {word_count}")

                    with col2:
                        st.markdown("**📝 Content:**")
                        text = result.node.get_content()

                        # Truncate very long chunks
                        if len(text) > 1500:
                            st.text_area("", value=text, height=400, disabled=True, label_visibility="collapsed")
                        else:
                            st.text(text)

                        # Source file
                        source = meta.get("file_path", meta.get("source", "Unknown"))
                        if source:
                            from pathlib import Path
                            st.caption(f"📁 {Path(source).name}")

        except Exception as e:
            st.error(f"Search error: {e}")
            import traceback
            st.code(traceback.format_exc())


def page_query():
    """Query page with LLM parameter tuning."""
    st.header("🔍 Query Index")

    # Compact tip
    st.caption("💡 **Quick tip:** For best results, keep advanced features collapsed (disabled). Expand only if needed.")

    # Get indexes
    all_indexes = list_vector_tables()

    if not all_indexes:
        st.warning("No indexes found. Please index documents first.")
        return

    # Filter out empty tables
    indexes = [idx for idx in all_indexes if idx["rows"] > 0]

    if not indexes:
        st.warning("All indexes are empty. Please index some documents first.")
        st.info(f"Found {len(all_indexes)} empty table(s). Go to View Indexes → Clean Empty to remove them.")
        return

    # Compact Row 1: Index selection + metrics
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        index_options = [f"{idx['name']} ({idx['rows']} chunks, cs={idx['chunk_size']}, model={idx['embed_model']})"
                         for idx in indexes]
        selected_idx = st.selectbox("**Index**", index_options, label_visibility="visible")
        table_name = indexes[index_options.index(selected_idx)]["name"]
        selected_index = indexes[index_options.index(selected_idx)]

        # Strip 'data_' prefix if present (PGVectorStore adds it automatically)
        if table_name.startswith("data_"):
            table_name = table_name[5:]  # Remove 'data_' prefix

    with col2:
        st.metric("Chunks", f"{selected_index['rows']:,}")
    with col3:
        st.metric("Chunk Size", selected_index['chunk_size'])
    with col4:
        st.metric("Model", selected_index['embed_model'].split('/')[-1][:8] if '/' in selected_index['embed_model'] else selected_index['embed_model'][:8])

    # Compact Row 2: Basic retrieval settings
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        top_k = st.number_input("**TOP_K**", 1, 20, 4, help="Chunks to retrieve")

    with col2:
        st.write("**Display**")
        show_sources = st.checkbox("Sources", value=True)

    with col3:
        st.write("**‎**")  # Align with col2
        show_scores = st.checkbox("Scores", value=True)

    with col4:
        st.metric("Status", "✅ Ready")

    # Collapsible LLM parameters (less frequently changed)
    with st.expander("🤖 LLM Parameters (optional)"):
        col1, col2, col3 = st.columns(3)

        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, help="Higher = more creative")

        with col2:
            max_tokens = st.number_input("Max Tokens", 64, 1024, 256, 32, help="Max generation length")

        with col3:
            context_window = st.number_input("Context Window", 1024, 8192, 3072, 256, help="LLM context size")

    # Advanced features - collapsed by default
    with st.expander("🔧 Advanced Features (optional)"):
        st.warning("⚠️ **Getting 0 results?** Make sure these settings are correct:")

        col1, col2 = st.columns(2)

        with col1:
            hybrid_alpha = st.slider("Hybrid Search (BM25 + Vector)", 0.0, 1.0, 1.0, 0.1,
                                    help="1.0=pure vector (recommended), 0.5=balanced, 0.0=pure BM25")
            if hybrid_alpha < 1.0:
                st.warning(f"⚠️ Hybrid mode {hybrid_alpha:.1f} may cause 0 results. Try 1.0 first.")
            else:
                st.success("✅ Pure vector search (recommended)")

        with col2:
            enable_filters = st.checkbox("Enable Metadata Filtering", value=False,
                                        help="Enable filtering by participant, date, etc.")
            if enable_filters:
                st.warning("⚠️ Filtering may exclude all results. Try disabling first.")
            else:
                st.success("✅ No filtering (recommended)")

        st.divider()

        # MMR Diversity
        st.subheader("📊 MMR Diversity")
        mmr_threshold = st.slider(
            "MMR Diversity",
            0.0, 1.0, 0.0, 0.1,
            help="0.0=most relevant (disable diversity), 1.0=most diverse. Prevents repetitive chunks."
        )
        if mmr_threshold > 0:
            st.success(f"✅ Diversity enabled: {mmr_threshold:.1f}")
        else:
            st.info("⏹️ Diversity disabled")

        st.divider()

        # Query Expansion
        st.subheader("🔍 Query Expansion")
        enable_query_expansion = st.checkbox(
            "Enable Query Expansion",
            value=False,
            help="Expand query with LLM for better results (slower, downloads 4.4GB)"
        )

        if enable_query_expansion:
            col1, col2 = st.columns(2)
            with col1:
                query_expansion_method = st.selectbox(
                    "Expansion Method",
                    ["llm", "keyword", "multi"],
                    help="llm=best quality (slow), keyword=fast, multi=combine both"
                )
            with col2:
                query_expansion_count = st.slider(
                    "Expansion Count",
                    1, 5, 2,
                    help="Number of expansion queries to generate"
                )

            # Estimate latency based on method
            if query_expansion_method in ["llm", "multi"]:
                est_time = "1-3s"
            else:
                est_time = "<0.1s"

            st.caption(f"⏱️ Added latency: {est_time}")
            st.success(f"✅ Query expansion: {query_expansion_method} (count: {query_expansion_count})")
        else:
            # Set defaults when disabled
            query_expansion_method = "llm"
            query_expansion_count = 2
            st.info("⏹️ Query expansion disabled")

        st.divider()

        # Reranking
        st.subheader("🎯 Reranking")
        enable_reranking = st.checkbox(
            "Enable Reranking",
            value=False,
            help="Rerank results for better relevance (slower)"
        )

        if enable_reranking:
            col1, col2, col3 = st.columns(3)
            with col1:
                rerank_candidates = st.number_input(
                    "Candidates",
                    5, 50, 12,
                    help="Initial candidates to retrieve before reranking"
                )
            with col2:
                rerank_top_k = st.number_input(
                    "Rerank TOP_K",
                    1, 20, 4,
                    help="Final number of results after reranking"
                )
            with col3:
                rerank_model = st.selectbox(
                    "Model",
                    ["cross-encoder/ms-marco-MiniLM-L-6-v2",
                     "cross-encoder/ms-marco-MiniLM-L-12-v2",
                     "BAAI/bge-reranker-base"],
                    help="Reranking model (L-12 is more accurate but slower)"
                )

            # Estimate latency based on model and candidates
            model_size = "L-12" if "L-12" in rerank_model else "L-6"
            if model_size == "L-12":
                est_time = f"{rerank_candidates * 15}ms"  # ~15ms per candidate
            else:
                est_time = f"{rerank_candidates * 8}ms"   # ~8ms per candidate

            st.caption(f"⏱️ Added latency: ~{est_time}")
            st.success(f"✅ Reranking: {rerank_model.split('/')[-1]} (candidates: {rerank_candidates} → top_k: {rerank_top_k})")
        else:
            # Set defaults when disabled
            rerank_candidates = 12
            rerank_top_k = 4
            rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            st.info("⏹️ Reranking disabled")

        st.divider()

        # Semantic Cache
        st.subheader("💾 Semantic Cache")
        enable_semantic_cache = st.checkbox(
            "Enable Semantic Cache",
            value=False,
            help="Cache similar queries to speed up repeated searches"
        )

        if enable_semantic_cache:
            col1, col2, col3 = st.columns(3)
            with col1:
                semantic_cache_threshold = st.slider(
                    "Similarity Threshold",
                    0.80, 0.99, 0.92, 0.01,
                    help="Minimum similarity to consider a cache hit (higher = stricter)"
                )
            with col2:
                semantic_cache_max_size = st.number_input(
                    "Max Cached Queries",
                    100, 10000, 1000, 100,
                    help="Maximum number of cached queries"
                )
            with col3:
                semantic_cache_ttl_hours = st.number_input(
                    "TTL (hours)",
                    0, 72, 24,
                    help="Cache expiration time (0=no expiration)"
                )
                semantic_cache_ttl = semantic_cache_ttl_hours * 3600  # Convert to seconds

            # Show cache statistics if available
            if 'cache_stats' in st.session_state:
                st.caption("📊 Cache Statistics:")
                stats = st.session_state.cache_stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Hits", stats.get('hits', 0))
                col2.metric("Misses", stats.get('misses', 0))
                total = stats.get('hits', 0) + stats.get('misses', 0)
                hit_rate = (stats.get('hits', 0) / total * 100) if total > 0 else 0
                col3.metric("Hit Rate", f"{hit_rate:.1f}%")

            st.success(f"✅ Semantic cache: threshold={semantic_cache_threshold:.2f}, max={semantic_cache_max_size}, TTL={semantic_cache_ttl_hours}h")
        else:
            # Set defaults when disabled
            semantic_cache_threshold = 0.92
            semantic_cache_max_size = 1000
            semantic_cache_ttl = 86400  # 24 hours in seconds
            st.info("⏹️ Semantic cache disabled")

        st.divider()

        # HyDE (Hypothetical Document Embeddings)
        st.subheader("🔬 HyDE (Hypothetical Document Embeddings)")
        st.caption("Advanced: Generates hypothetical answers before retrieval (+10-20% quality, +100-400ms latency)")

        enable_hyde = st.checkbox("Enable HyDE", value=False,
                                  help="Uses LLM to generate hypothetical answers, then retrieves documents similar to those answers")

        if enable_hyde:
            col1, col2, col3 = st.columns(3)

            with col1:
                num_hypotheses = st.slider("Hypotheses", 1, 3, 1,
                                          help="Number of hypothetical answers to generate (more = slower but better coverage)")
                latency_impact = num_hypotheses * 100
                st.caption(f"Est. latency: +{latency_impact}ms")

            with col2:
                hypothesis_length = st.slider("Length (tokens)", 50, 200, 100, 10,
                                             help="Target length of each hypothetical answer")

            with col3:
                fusion_method = st.selectbox("Fusion", ["rrf", "avg", "max"],
                                            help="rrf=Reciprocal Rank Fusion (recommended), avg=Average scores, max=Maximum score")

            st.info("💡 **Best for:** Technical queries, complex questions, domain-specific content")
            st.caption("📖 See `docs/HYDE_GUIDE.md` for detailed documentation and examples")
            st.success(f"✅ HyDE: {num_hypotheses} hypothesis/hypotheses, length={hypothesis_length}, fusion={fusion_method}")
        else:
            # Set defaults when disabled
            num_hypotheses = 1
            hypothesis_length = 100
            fusion_method = "rrf"
            st.info("⏹️ HyDE disabled")

        st.divider()

        # Summary status
        if not enable_query_expansion and not enable_reranking and hybrid_alpha == 1.0 and not enable_filters and mmr_threshold == 0.0 and not enable_semantic_cache and not enable_hyde:
            st.success("✅ **Safe Configuration:** All advanced features disabled. Queries should work reliably.")

    # Query input (compact)
    query = st.text_area("**Your Question:**", height=70,
                        placeholder="What is the main topic?",
                        key="query_input")

    # Search buttons side by side
    col1, col2 = st.columns(2)

    with col1:
        search_and_answer = st.button("🔍 Search & Answer", type="primary", use_container_width=True)

    with col2:
        search_chunks_only = st.button("📄 Search Chunks Only", use_container_width=True)

    if search_and_answer or search_chunks_only:
        if not query.strip():
            st.warning("Please enter a question")
            return

        if search_chunks_only:
            # Chunks-only search (no LLM) - pass all retrieval parameters
            search_chunks(table_name, query, top_k,
                         hybrid_alpha, enable_filters, mmr_threshold,
                         enable_query_expansion, query_expansion_method, query_expansion_count,
                         enable_reranking, rerank_model, rerank_candidates, rerank_top_k,
                         enable_semantic_cache, semantic_cache_threshold, semantic_cache_max_size, semantic_cache_ttl,
                         enable_hyde, num_hypotheses, hypothesis_length, fusion_method)
        else:
            # Full search with LLM answer generation
            run_query(table_name, query, top_k, show_sources, show_scores,
                     temperature, max_tokens, context_window, enable_query_expansion, enable_reranking,
                     hybrid_alpha, enable_filters, mmr_threshold, query_expansion_method, query_expansion_count,
                     rerank_model, rerank_candidates, rerank_top_k, enable_semantic_cache,
                     semantic_cache_threshold, semantic_cache_max_size, semantic_cache_ttl,
                     enable_hyde, num_hypotheses, hypothesis_length, fusion_method)

    # History
    if st.session_state.query_history:
        st.subheader("Query History")
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Q: {item['query'][:60]}..."):
                st.write(f"**A:** {item['answer']}")
                st.caption(f"Score: {item['top_score']:.4f} | Chunks: {item['chunks']}")

    # Troubleshooting section
    st.divider()
    with st.expander("❓ Troubleshooting: Getting 0 Results?"):
        st.markdown("""
        **If queries return 0 chunks, check:**

        1. **Embedding Model Match** ✅ Auto-detected from index
           - The UI shows which model is being used
           - Must match the model used during indexing

        2. **Advanced Features** (disable these first)
           - ☑️ Hybrid Search: Set to **1.0** (pure vector)
           - ☑️ Metadata Filtering: **Unchecked**
           - ☑️ Query Expansion: **Unchecked**
           - ☑️ Reranking: **Unchecked**

        3. **TOP_K Value**
           - Try increasing to 10
           - Some indexes may need higher k

        4. **Database Connection**
           - Go to Settings → Test Connection
           - Ensure PostgreSQL is running

        5. **Table Has Data**
           - Go to View Indexes
           - Check the table has chunks > 0
           - Empty tables won't return results

        **Quick Fix:** Try a different index or reindex your documents with default settings.
        """)

        if st.button("🔄 Reload Page"):
            st.rerun()

def get_index_embedding_model(table_name: str) -> Optional[str]:
    """Get the embedding model used for this index from metadata."""
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor()
        cur.execute(
            sql.SQL("""
                SELECT metadata_->>'_embed_model' as model
                FROM {}
                WHERE metadata_->>'_embed_model' IS NOT NULL
                LIMIT 1
            """).format(sql.Identifier(table_name))
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None

def run_query(table_name: str, query: str, top_k: int, show_sources: bool,
              show_scores: bool, temperature: float, max_tokens: int, context_window: int,
              enable_query_expansion: bool = False, enable_reranking: bool = False,
              hybrid_alpha: float = 1.0, enable_filters: bool = False,
              mmr_threshold: float = 0.0, query_expansion_method: str = "llm",
              query_expansion_count: int = 2, rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
              rerank_candidates: int = 12, rerank_top_k: int = 4,
              enable_semantic_cache: bool = False, semantic_cache_threshold: float = 0.92,
              semantic_cache_max_size: int = 1000, semantic_cache_ttl: int = 86400,
              enable_hyde: bool = False, num_hypotheses: int = 1,
              hypothesis_length: int = 100, fusion_method: str = "rrf"):
    """Run query with custom parameters."""

    import rag_low_level_m1_16gb_verbose as rag
    rag.S.table = table_name
    rag.S.top_k = top_k
    rag.S.temperature = temperature
    rag.S.max_new_tokens = max_tokens
    rag.S.context_window = context_window

    # Advanced features - Basic
    rag.S.enable_query_expansion = enable_query_expansion
    rag.S.enable_reranking = enable_reranking
    rag.S.hybrid_alpha = hybrid_alpha
    rag.S.enable_filters = enable_filters
    rag.S.mmr_threshold = mmr_threshold

    # Advanced features - Query Expansion
    if enable_query_expansion:
        rag.S.query_expansion_method = query_expansion_method
        rag.S.query_expansion_count = query_expansion_count

    # Advanced features - Reranking
    if enable_reranking:
        rag.S.rerank_model = rerank_model
        rag.S.rerank_candidates = rerank_candidates
        rag.S.rerank_top_k = rerank_top_k

    # Advanced features - Semantic Cache
    rag.S.enable_semantic_cache = enable_semantic_cache
    if enable_semantic_cache:
        rag.S.semantic_cache_threshold = semantic_cache_threshold
        rag.S.semantic_cache_max_size = semantic_cache_max_size
        rag.S.semantic_cache_ttl = semantic_cache_ttl

    # Advanced features - HyDE
    if enable_hyde:
        rag.S.enable_hyde = enable_hyde
        rag.S.num_hypotheses = num_hypotheses
        rag.S.hypothesis_length = hypothesis_length
        rag.S.fusion_method = fusion_method

    # Show config in UI
    active_features = []
    if hybrid_alpha < 1.0:
        active_features.append(f"Hybrid α={hybrid_alpha:.1f}")
    if enable_filters:
        active_features.append("Filters=ON")
    if mmr_threshold > 0:
        active_features.append(f"MMR={mmr_threshold:.2f}")
    if enable_query_expansion:
        active_features.append(f"QE={query_expansion_method}")
    if enable_reranking:
        active_features.append(f"Rerank={rerank_model.split('/')[-1]}")
    if enable_semantic_cache:
        active_features.append(f"Cache={semantic_cache_threshold:.2f}")
    if enable_hyde:
        active_features.append(f"HyDE={num_hypotheses}h")

    if active_features:
        st.info(f"🔧 Active Features: {' | '.join(active_features)}")

    # Check if table exists before querying
    try:
        conn = get_db_connection(autocommit=True)
        cur = conn.cursor()
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))
        )
        row_count = cur.fetchone()[0]
        cur.close()
        conn.close()

        if row_count == 0:
            st.error(f"❌ Table `{table_name}` is empty (0 chunks). Please index some documents first.")
            return
    except Exception as e:
        st.error(f"❌ Table `{table_name}` does not exist or cannot be accessed: {e}")
        st.info("💡 Go to 'View Indexes' to see available tables, or use 'Quick Start' to create a new index.")
        return

    # CRITICAL FIX: PGVectorStore auto-prepends "data_" prefix!
    # If table name already starts with "data_", strip it to avoid double prefix
    query_table_name = table_name
    if table_name.startswith("data_"):
        query_table_name = table_name[5:]  # Remove "data_" prefix
        st.caption(f"🔧 Table name adjusted: `{table_name}` → `{query_table_name}` (PGVectorStore adds 'data_' prefix)")

    # Update settings with corrected table name
    rag.S.table = query_table_name

    # Auto-detect embedding model from index metadata
    index_model = get_index_embedding_model(table_name)  # Use original name for metadata lookup
    if index_model:
        st.info(f"ℹ️ Using embedding model from index: `{index_model}`")
        query_embed_model = index_model
    else:
        st.warning(f"⚠️ Could not detect embedding model from index, using default: `{rag.S.embed_model_name}`")
        query_embed_model = rag.S.embed_model_name

    import time
    query_start = time.time()
    retrieval_time = 0
    generation_time = 0
    cache_hit = False

    # Get metrics instance
    metrics = st.session_state.get("metrics")

    with st.spinner("Searching..."):
        try:
            retrieval_start = time.time()
            embed_model = get_embed_model(query_embed_model)
            vector_store = make_vector_store()  # Now uses corrected table name
            retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=top_k)
            results = retriever._retrieve(QueryBundle(query_str=query))
            retrieval_time = time.time() - retrieval_start

            # Record retrieval metrics
            if metrics:
                scores = [node.score for node in results]
                metrics.record_retrieval(num_documents=len(results), scores=scores)

        except Exception as e:
            # Record query error
            if metrics:
                metrics.record_query_error(type(e).__name__)

            st.error(f"Retrieval error: {e}")
            st.info("💡 Try: Increase TOP_K, disable Advanced Features, or check Settings → Configuration")
            return

    # Show sources
    if show_sources and results:
        st.subheader("Retrieved Chunks")

        for i, result in enumerate(results):
            score = result.score

            if score > 0.7:
                badge = "🟢 Excellent"
            elif score > 0.5:
                badge = "🟡 Good"
            elif score > 0.3:
                badge = "🟠 Fair"
            else:
                badge = "🔴 Low"

            with st.expander(f"Chunk {i+1}: {badge} (Score: {score:.4f})", expanded=(i == 0)):
                if show_scores:
                    st.metric("Similarity", f"{score:.4f}")

                st.text(result.node.get_content())

    # Generate answer
    st.subheader("Generated Answer")

    # If no results, show helpful message
    if not results or len(results) == 0:
        st.warning(f"⚠️ No chunks found in table `{table_name}`. The index might be empty or using a different embedding model.")
        st.info(f"**Debug Info:**\n- Table: `{table_name}`\n- Query: `{query}`\n- TOP_K: {top_k}\n- Retrieval returned: {len(results)} chunks")

        # Show available tables
        with st.expander("🔍 Available tables in database"):
            all_tables = list_vector_tables()
            for tbl in all_tables:
                st.text(f"- {tbl['name']}: {tbl['rows']} chunks")
        return

    with st.spinner("Generating..."):
        try:
            generation_start = time.time()
            llm = get_llm(
                context_window=context_window,
                max_tokens=max_tokens,
                temperature=temperature
            )
            query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
            response = query_engine.query(query)
            generation_time = time.time() - generation_start

            st.success(str(response))

            # Calculate total query time
            total_query_time = time.time() - query_start

            # Record successful query metrics
            if metrics:
                metrics.record_query_duration(total_query_time)
                metrics.record_query_success()

                # Export metrics periodically (every 10 queries)
                if st.session_state.perf_total_queries % 10 == 0:
                    try:
                        metrics.export()
                    except Exception:
                        pass  # Silently fail metrics export

            # Save history
            st.session_state.query_history.append({
                "query": query,
                "answer": str(response),
                "chunks": len(results),
                "top_score": results[0].score if results else 0,
            })

            # Track performance metrics
            st.session_state.perf_total_queries += 1
            st.session_state.perf_queries_this_session += 1

            # Keep only last 50 queries for performance
            query_perf_entry = {
                "timestamp": pd.Timestamp.now(),
                "query": query[:100],  # Truncate long queries
                "table_name": table_name,
                "total_time": total_query_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "top_k": top_k,
                "num_results": len(results),
                "top_score": results[0].score if results else 0,
                "cache_hit": cache_hit,
            }
            st.session_state.perf_query_performance.append(query_perf_entry)
            if len(st.session_state.perf_query_performance) > 50:
                st.session_state.perf_query_performance.pop(0)

            # Update cache stats if semantic cache was used
            if enable_semantic_cache:
                if cache_hit:
                    st.session_state.perf_cache_stats["hits"] += 1
                    if metrics:
                        metrics.record_cache_hit()
                else:
                    st.session_state.perf_cache_stats["misses"] += 1
                    if metrics:
                        metrics.record_cache_miss()
                st.session_state.perf_cache_stats["total"] += 1

        except Exception as e:
            # Record query error
            if metrics:
                metrics.record_query_error(type(e).__name__)

            st.error(f"Generation error: {e}")

def page_view_indexes():
    """View and manage indexes."""
    st.header("📊 View Indexes")

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        if st.button("🧹 Clean Empty", help="Delete all empty tables"):
            delete_empty_tables()
            st.rerun()

    indexes = list_vector_tables()

    if not indexes:
        st.info("No indexes found.")
        return

    # Filter out empty tables for main display, but show them separately
    non_empty = [idx for idx in indexes if idx["rows"] > 0]
    empty = [idx for idx in indexes if idx["rows"] == 0]

    # Stats (only non-empty)
    total_chunks = sum(idx["rows"] for idx in non_empty)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Indexes", len(non_empty), delta=f"-{len(empty)} empty" if empty else None, delta_color="off")
    with col2:
        st.metric("Total Chunks", f"{total_chunks:,}")
    with col3:
        st.metric("Database", st.session_state.db_name)

    # Show warning for empty tables
    if empty:
        st.warning(f"⚠️ Found {len(empty)} empty table(s). Click '🧹 Clean Empty' to remove them.")

    st.divider()

    # Main table (non-empty only)
    st.subheader("Active Indexes")
    if non_empty:
        df = pd.DataFrame(non_empty)
        df.columns = ["Name", "Chunks", "Chunk Size", "Overlap", "Embed Model"]
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("No active indexes found.")

    # Show empty tables in collapsible section
    if empty:
        with st.expander(f"⚠️ Empty Tables ({len(empty)})"):
            df_empty = pd.DataFrame(empty)
            df_empty.columns = ["Name", "Chunks", "Chunk Size", "Overlap", "Embed Model"]
            st.dataframe(df_empty, width="stretch", hide_index=True)
            st.caption("These tables have 0 chunks and can be safely deleted.")

    # Actions (only for non-empty tables)
    if non_empty:
        st.subheader("Index Actions")

        selected_table = st.selectbox("Select index to manage:", [idx["name"] for idx in non_empty])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Delete Index", type="secondary"):
                st.session_state["confirm_delete"] = selected_table

        # Confirm delete
        if st.session_state.get("confirm_delete") == selected_table:
            st.warning(f"Delete `{selected_table}`?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes", type="primary"):
                    if delete_table(selected_table):
                        st.success(f"Deleted `{selected_table}`")
                        st.session_state["confirm_delete"] = None
                        st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state["confirm_delete"] = None
                    st.rerun()


# =============================================================================
# Configuration Management
# =============================================================================

def gather_current_config() -> Dict[str, Any]:
    """Gather all current RAG configuration settings."""
    import rag_low_level_m1_16gb_verbose as rag

    config = {
        # Document & Index
        "PDF_PATH": rag.S.pdf_path,
        "PGTABLE": rag.S.table,
        "RESET_TABLE": "1" if rag.S.reset_table else "0",

        # Chunking
        "CHUNK_SIZE": str(rag.S.chunk_size),
        "CHUNK_OVERLAP": str(rag.S.chunk_overlap),

        # Embedding
        "EMBED_MODEL": rag.S.embed_model_name,
        "EMBED_DIM": str(rag.S.embed_dim),
        "EMBED_BATCH": str(rag.S.embed_batch),

        # LLM
        "CTX": str(rag.S.context_window),
        "MAX_NEW_TOKENS": str(rag.S.max_new_tokens),
        "TEMP": str(rag.S.temperature),
        "N_GPU_LAYERS": str(rag.S.n_gpu_layers),
        "N_BATCH": str(rag.S.n_batch),

        # Retrieval
        "TOP_K": str(rag.S.top_k),

        # Database
        "PGHOST": st.session_state.db_host,
        "PGPORT": str(st.session_state.db_port),
        "PGUSER": st.session_state.db_user,
        "PGPASSWORD": st.session_state.db_password,
        "DB_NAME": st.session_state.db_name,

        # Advanced Features
        "HYBRID_ALPHA": str(rag.S.hybrid_alpha),
        "ENABLE_QUERY_EXPANSION": "1" if rag.S.enable_query_expansion else "0",
        "ENABLE_RERANKING": "1" if rag.S.enable_reranking else "0",
        "ENABLE_FILTERS": "1" if rag.S.enable_filters else "0",
        "ENABLE_SEMANTIC_CACHE": "1" if rag.S.enable_semantic_cache else "0",
    }

    return config

def export_config_to_env() -> str:
    """Export current configuration to .env format."""
    config = gather_current_config()

    lines = [
        "# RAG Pipeline Configuration",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "# ============================================",
        "# Document & Index Configuration",
        "# ============================================",
        f"PDF_PATH={config['PDF_PATH']}",
        f"PGTABLE={config['PGTABLE']}",
        f"RESET_TABLE={config['RESET_TABLE']}  # 0=keep existing, 1=drop and recreate",
        "",
        "# ============================================",
        "# Chunking Configuration",
        "# ============================================",
        f"CHUNK_SIZE={config['CHUNK_SIZE']}  # Characters per chunk (100-3000)",
        f"CHUNK_OVERLAP={config['CHUNK_OVERLAP']}  # Overlap between chunks",
        "",
        "# ============================================",
        "# Embedding Configuration",
        "# ============================================",
        f"EMBED_MODEL={config['EMBED_MODEL']}",
        f"EMBED_DIM={config['EMBED_DIM']}  # Vector dimensions",
        f"EMBED_BATCH={config['EMBED_BATCH']}  # Batch size for embedding",
        "",
        "# ============================================",
        "# LLM Configuration",
        "# ============================================",
        f"CTX={config['CTX']}  # Context window size",
        f"MAX_NEW_TOKENS={config['MAX_NEW_TOKENS']}  # Max generation length",
        f"TEMP={config['TEMP']}  # Temperature (0.0-1.0)",
        f"N_GPU_LAYERS={config['N_GPU_LAYERS']}  # Layers to offload to GPU",
        f"N_BATCH={config['N_BATCH']}  # LLM batch size",
        "",
        "# ============================================",
        "# Retrieval Configuration",
        "# ============================================",
        f"TOP_K={config['TOP_K']}  # Number of chunks to retrieve",
        "",
        "# ============================================",
        "# Database Configuration",
        "# ============================================",
        f"PGHOST={config['PGHOST']}",
        f"PGPORT={config['PGPORT']}",
        f"PGUSER={config['PGUSER']}",
        f"PGPASSWORD={config['PGPASSWORD']}",
        f"DB_NAME={config['DB_NAME']}",
        "",
        "# ============================================",
        "# Advanced Features",
        "# ============================================",
        f"HYBRID_ALPHA={config['HYBRID_ALPHA']}  # 1.0=pure vector, 0.0=pure BM25",
        f"ENABLE_QUERY_EXPANSION={config['ENABLE_QUERY_EXPANSION']}  # 0=off, 1=on",
        f"ENABLE_RERANKING={config['ENABLE_RERANKING']}  # 0=off, 1=on",
        f"ENABLE_FILTERS={config['ENABLE_FILTERS']}  # 0=off, 1=on",
        f"ENABLE_SEMANTIC_CACHE={config['ENABLE_SEMANTIC_CACHE']}  # 0=off, 1=on",
    ]

    return "\n".join(lines)

def import_config_from_env(content: str) -> Dict[str, str]:
    """Parse .env format and extract configuration."""
    config = {}

    for line in content.split('\n'):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Parse KEY=VALUE format
        if '=' in line:
            # Split only on first '=' to handle values with '='
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Remove inline comments
            if '#' in value:
                value = value.split('#')[0].strip()

            config[key] = value

    return config

def apply_imported_config(config: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Apply imported configuration to current settings."""
    import rag_low_level_m1_16gb_verbose as rag

    applied = []
    errors = []

    # Document & Index
    if "PDF_PATH" in config:
        rag.S.pdf_path = config["PDF_PATH"]
        applied.append("PDF_PATH")

    if "PGTABLE" in config:
        rag.S.table = config["PGTABLE"]
        applied.append("PGTABLE")

    if "RESET_TABLE" in config:
        rag.S.reset_table = config["RESET_TABLE"] == "1"
        applied.append("RESET_TABLE")

    # Chunking
    if "CHUNK_SIZE" in config:
        try:
            rag.S.chunk_size = int(config["CHUNK_SIZE"])
            applied.append("CHUNK_SIZE")
        except ValueError:
            errors.append(f"Invalid CHUNK_SIZE: {config['CHUNK_SIZE']}")

    if "CHUNK_OVERLAP" in config:
        try:
            rag.S.chunk_overlap = int(config["CHUNK_OVERLAP"])
            applied.append("CHUNK_OVERLAP")
        except ValueError:
            errors.append(f"Invalid CHUNK_OVERLAP: {config['CHUNK_OVERLAP']}")

    # Embedding
    if "EMBED_MODEL" in config:
        rag.S.embed_model_name = config["EMBED_MODEL"]
        applied.append("EMBED_MODEL")

    if "EMBED_DIM" in config:
        try:
            rag.S.embed_dim = int(config["EMBED_DIM"])
            applied.append("EMBED_DIM")
        except ValueError:
            errors.append(f"Invalid EMBED_DIM: {config['EMBED_DIM']}")

    if "EMBED_BATCH" in config:
        try:
            rag.S.embed_batch = int(config["EMBED_BATCH"])
            applied.append("EMBED_BATCH")
        except ValueError:
            errors.append(f"Invalid EMBED_BATCH: {config['EMBED_BATCH']}")

    # LLM
    if "CTX" in config:
        try:
            rag.S.context_window = int(config["CTX"])
            applied.append("CTX")
        except ValueError:
            errors.append(f"Invalid CTX: {config['CTX']}")

    if "MAX_NEW_TOKENS" in config:
        try:
            rag.S.max_new_tokens = int(config["MAX_NEW_TOKENS"])
            applied.append("MAX_NEW_TOKENS")
        except ValueError:
            errors.append(f"Invalid MAX_NEW_TOKENS: {config['MAX_NEW_TOKENS']}")

    if "TEMP" in config:
        try:
            rag.S.temperature = float(config["TEMP"])
            applied.append("TEMP")
        except ValueError:
            errors.append(f"Invalid TEMP: {config['TEMP']}")

    if "N_GPU_LAYERS" in config:
        try:
            rag.S.n_gpu_layers = int(config["N_GPU_LAYERS"])
            applied.append("N_GPU_LAYERS")
        except ValueError:
            errors.append(f"Invalid N_GPU_LAYERS: {config['N_GPU_LAYERS']}")

    if "N_BATCH" in config:
        try:
            rag.S.n_batch = int(config["N_BATCH"])
            applied.append("N_BATCH")
        except ValueError:
            errors.append(f"Invalid N_BATCH: {config['N_BATCH']}")

    # Retrieval
    if "TOP_K" in config:
        try:
            rag.S.top_k = int(config["TOP_K"])
            applied.append("TOP_K")
        except ValueError:
            errors.append(f"Invalid TOP_K: {config['TOP_K']}")

    # Database
    if "PGHOST" in config:
        st.session_state.db_host = config["PGHOST"]
        applied.append("PGHOST")

    if "PGPORT" in config:
        st.session_state.db_port = config["PGPORT"]
        applied.append("PGPORT")

    if "PGUSER" in config:
        st.session_state.db_user = config["PGUSER"]
        applied.append("PGUSER")

    if "PGPASSWORD" in config:
        st.session_state.db_password = config["PGPASSWORD"]
        applied.append("PGPASSWORD")

    if "DB_NAME" in config:
        st.session_state.db_name = config["DB_NAME"]
        applied.append("DB_NAME")

    # Advanced Features
    if "HYBRID_ALPHA" in config:
        try:
            rag.S.hybrid_alpha = float(config["HYBRID_ALPHA"])
            applied.append("HYBRID_ALPHA")
        except ValueError:
            errors.append(f"Invalid HYBRID_ALPHA: {config['HYBRID_ALPHA']}")

    if "ENABLE_QUERY_EXPANSION" in config:
        rag.S.enable_query_expansion = config["ENABLE_QUERY_EXPANSION"] == "1"
        applied.append("ENABLE_QUERY_EXPANSION")

    if "ENABLE_RERANKING" in config:
        rag.S.enable_reranking = config["ENABLE_RERANKING"] == "1"
        applied.append("ENABLE_RERANKING")

    if "ENABLE_FILTERS" in config:
        rag.S.enable_filters = config["ENABLE_FILTERS"] == "1"
        applied.append("ENABLE_FILTERS")

    if "ENABLE_SEMANTIC_CACHE" in config:
        rag.S.enable_semantic_cache = config["ENABLE_SEMANTIC_CACHE"] == "1"
        applied.append("ENABLE_SEMANTIC_CACHE")

    return applied, errors

def save_config_preset(name: str, description: str = "") -> bool:
    """Save current configuration as a preset."""
    try:
        config = gather_current_config()

        preset_data = {
            "name": name,
            "description": description,
            "created": datetime.now().isoformat(),
            "config": config
        }

        # Save to JSON file
        preset_file = CONFIG_DIR / f"{name}.json"
        with open(preset_file, 'w') as f:
            json.dump(preset_data, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error saving preset: {e}")
        return False

def load_config_presets() -> List[Dict[str, Any]]:
    """Load all saved configuration presets."""
    presets = []

    try:
        for preset_file in CONFIG_DIR.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    preset_data = json.load(f)
                    presets.append(preset_data)
            except Exception as e:
                st.warning(f"Could not load preset {preset_file.name}: {e}")

        # Sort by created date (newest first)
        presets.sort(key=lambda x: x.get("created", ""), reverse=True)
    except Exception as e:
        st.error(f"Error loading presets: {e}")

    return presets

def delete_config_preset(name: str) -> bool:
    """Delete a saved configuration preset."""
    try:
        preset_file = CONFIG_DIR / f"{name}.json"
        if preset_file.exists():
            preset_file.unlink()
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting preset: {e}")
        return False

def compare_configs(current: Dict[str, Any], saved: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Compare current config with saved config and return differences."""
    differences = {}

    all_keys = set(current.keys()) | set(saved.keys())

    for key in sorted(all_keys):
        current_val = current.get(key, "NOT SET")
        saved_val = saved.get(key, "NOT SET")

        if current_val != saved_val:
            differences[key] = {
                "current": current_val,
                "saved": saved_val
            }

    return differences


def page_settings():
    """Settings page."""
    st.header("⚙️ Settings")

    # Current configuration status
    st.subheader("📊 Current Configuration")

    import rag_low_level_m1_16gb_verbose as rag

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Embedding Model", rag.S.embed_model_name.split('/')[-1])
        st.metric("Embed Dim", rag.S.embed_dim)
    with col2:
        st.metric("Chunk Size", rag.S.chunk_size)
        st.metric("Chunk Overlap", rag.S.chunk_overlap)
    with col3:
        st.metric("TOP_K", rag.S.top_k)
        hybrid_status = "Pure Vector" if rag.S.hybrid_alpha >= 1.0 else f"Hybrid {rag.S.hybrid_alpha:.1f}"
        st.metric("Search Mode", hybrid_status)

    # Advanced features status
    with st.expander("🔧 Advanced Features Status"):
        features_status = {
            "Query Expansion": "✅ Enabled" if rag.S.enable_query_expansion else "⏹️ Disabled",
            "Reranking": "✅ Enabled" if rag.S.enable_reranking else "⏹️ Disabled",
            "Metadata Filtering": "✅ Enabled" if rag.S.enable_filters else "⏹️ Disabled",
            "Semantic Cache": "✅ Enabled" if rag.S.enable_semantic_cache else "⏹️ Disabled",
            "Hybrid Search Alpha": f"{rag.S.hybrid_alpha:.2f} (1.0=pure vector, 0.0=pure BM25)",
        }

        for feature, status in features_status.items():
            st.text(f"{feature}: {status}")

        if rag.S.hybrid_alpha < 1.0 or rag.S.enable_filters:
            st.warning("⚠️ Advanced features enabled may cause 0 results. Set HYBRID_ALPHA=1.0 and ENABLE_FILTERS=0 in .env for best results.")

    st.divider()

    # Database
    st.subheader("Database Connection")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.db_host = st.text_input("Host:", value=st.session_state.db_host)
        st.session_state.db_user = st.text_input("User:", value=st.session_state.db_user)
        st.session_state.db_name = st.text_input("Database:", value=st.session_state.db_name)

    with col2:
        st.session_state.db_port = st.text_input("Port:", value=st.session_state.db_port)
        st.session_state.db_password = st.text_input("Password:", value=st.session_state.db_password,
                                                     type="password")

    if st.button("Test Connection"):
        success, message = test_db_connection()
        if success:
            st.success(message)
        else:
            st.error(message)

    st.divider()

    # LLM info
    st.subheader("LLM Configuration")
    st.info("""
    LLM settings (most configurable via Query page):
    - Temperature, Max Tokens, Context Window (configurable in Query page)
    - Model URL, GPU Layers, Batch Size (via environment variables)

    Environment variables:
    - `MODEL_URL` - Hugging Face model URL
    - `N_GPU_LAYERS` - GPU layers for Metal (default: 24)
    - `N_BATCH` - Batch size (default: 256)
    """)

    st.divider()

    # Cache
    st.subheader("Cache Management")

    if st.button("Clear All Caches"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Caches cleared!")

def page_chat():
    """Chat Mode page with conversation memory."""
    st.header("💬 Chat Mode")
    st.caption("ChatGPT-style interface with conversation memory")

    # Quick guide
    st.info("""
    💡 **Chat Mode Guide:**
    1. Select an index (your knowledge base)
    2. Configure conversation settings in sidebar
    3. Ask questions - the assistant remembers context!
    4. Sources are shown for each response
    """)

    # Get indexes
    all_indexes = list_vector_tables()
    if not all_indexes:
        st.warning("No indexes found. Please index documents first.")
        return

    indexes = [idx for idx in all_indexes if idx["rows"] > 0]
    if not indexes:
        st.warning("All indexes are empty. Please index some documents first.")
        return

    # Sidebar configuration
    with st.sidebar:
        st.subheader("💬 Conversation Settings")

        # Enable/disable memory
        st.session_state.chat_enable_memory = st.checkbox(
            "Enable Memory",
            value=st.session_state.chat_enable_memory,
            help="Remember previous conversation turns"
        )

        # Max turns
        st.session_state.chat_max_turns = st.slider(
            "Max Turns",
            min_value=3,
            max_value=20,
            value=st.session_state.chat_max_turns,
            help="Maximum conversation turns to remember"
        )

        # Auto-summarize (placeholder for future feature)
        st.session_state.chat_auto_summarize = st.checkbox(
            "Auto Summarize",
            value=st.session_state.chat_auto_summarize,
            help="Summarize long conversations (coming soon)",
            disabled=True
        )

        st.divider()

        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # Stats
        st.subheader("Stats")
        st.metric("Current Turns", len(st.session_state.chat_history) // 2)
        if st.session_state.chat_history:
            st.metric("Last Update", st.session_state.chat_history[-1].get("timestamp", "N/A"))

    # Index selection
    st.subheader("1. Select Knowledge Base")

    index_options = [f"{idx['name']} ({idx['rows']} chunks, cs={idx['chunk_size']}, model={idx['embed_model']})"
                     for idx in indexes]
    selected_idx = st.selectbox("Index:", index_options, key="chat_index")
    table_name = indexes[index_options.index(selected_idx)]["name"]

    # Strip 'data_' prefix if present (PGVectorStore adds it automatically)
    if table_name.startswith("data_"):
        table_name = table_name[5:]  # Remove 'data_' prefix

    selected_index = indexes[index_options.index(selected_idx)]
    st.caption(f"📊 Using: `{table_name}` | Chunks: {selected_index['rows']} | Model: `{selected_index['embed_model']}`")

    # Query parameters (collapsible)
    with st.expander("⚙️ Query Settings", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            top_k = st.slider("TOP_K", 1, 10, 4, key="chat_top_k", help="Number of chunks to retrieve")

        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05, key="chat_temp", help="Higher = more creative")

        with col3:
            max_tokens = st.number_input("Max Tokens", 64, 1024, 256, 32, key="chat_tokens", help="Max generation length")

    st.divider()

    # Chat display area
    st.subheader("2. Conversation")

    # Display chat history
    for idx, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

            # Show sources for assistant messages
            if role == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        score = source.get("score", 0)
                        text = source.get("text", "")

                        if score > 0.7:
                            badge = "🟢"
                        elif score > 0.5:
                            badge = "🟡"
                        elif score > 0.3:
                            badge = "🟠"
                        else:
                            badge = "🔴"

                        st.markdown(f"**{badge} Source {i+1}** (Score: {score:.4f})")
                        st.text(text[:300] + "..." if len(text) > 300 else text)
                        if i < len(message["sources"]) - 1:
                            st.divider()

    # Chat input
    user_input = st.chat_input("Ask a question...", key="chat_input")

    if user_input:
        # Add user message
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.chat_history.append(user_message)

        # Generate assistant response
        try:
            # Build conversation context if memory is enabled
            context_query = user_input
            if st.session_state.chat_enable_memory and len(st.session_state.chat_history) > 1:
                # Get recent conversation turns
                recent_turns = st.session_state.chat_history[-(st.session_state.chat_max_turns * 2):-1]
                context_parts = []

                for msg in recent_turns[-6:]:  # Last 3 turns (user + assistant = 2 messages per turn)
                    if msg["role"] == "user":
                        context_parts.append(f"Previous question: {msg['content']}")
                    elif msg["role"] == "assistant":
                        context_parts.append(f"Previous answer: {msg['content'][:200]}")

                if context_parts:
                    context_query = "\n".join(context_parts) + f"\n\nCurrent question: {user_input}"

            # Query the index
            response_text, sources = run_chat_query(
                table_name=table_name,
                query=context_query,
                original_query=user_input,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "sources": sources,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.chat_history.append(assistant_message)

            # Trim history if needed
            max_messages = st.session_state.chat_max_turns * 2
            if len(st.session_state.chat_history) > max_messages:
                st.session_state.chat_history = st.session_state.chat_history[-max_messages:]

        except Exception as e:
            error_msg = f"Error: {str(e)}"

            # Add error to history
            assistant_message = {
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.chat_history.append(assistant_message)

        # Rerun to refresh display
        st.rerun()

    # Export conversation
    if st.session_state.chat_history:
        st.divider()
        col1, col2 = st.columns([3, 1])

        with col2:
            # Convert chat history to markdown
            export_text = "# Chat Conversation Export\n\n"
            export_text += f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            export_text += f"**Index:** {table_name}\n\n"
            export_text += "---\n\n"

            for msg in st.session_state.chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                export_text += f"## {role}\n\n"
                export_text += f"{msg['content']}\n\n"

                if msg["role"] == "assistant" and msg.get("sources"):
                    export_text += "### Sources\n\n"
                    for i, source in enumerate(msg["sources"]):
                        export_text += f"{i+1}. Score: {source.get('score', 0):.4f}\n"
                        export_text += f"   {source.get('text', '')[:200]}...\n\n"

                export_text += "---\n\n"

            st.download_button(
                label="💾 Export Chat",
                data=export_text,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def page_performance():
    """Performance Dashboard - track and visualize metrics."""
    st.header("📊 Performance Dashboard")

    st.markdown("""
    Track indexing performance, query latency, cache efficiency, and overall system metrics.
    """)

    # Calculate overview metrics
    indexes = list_vector_tables()
    total_chunks = sum(idx["rows"] for idx in indexes) if indexes else 0
    total_queries = st.session_state.perf_total_queries

    # Calculate average query time
    if st.session_state.perf_query_performance:
        avg_query_time = sum(q["total_time"] for q in st.session_state.perf_query_performance) / len(st.session_state.perf_query_performance)
    else:
        avg_query_time = 0

    # Calculate cache hit rate
    cache_stats = st.session_state.perf_cache_stats
    if cache_stats["total"] > 0:
        cache_hit_rate = 100 * cache_stats["hits"] / cache_stats["total"]
    else:
        cache_hit_rate = 0

    # Section 1: Overview Metrics
    st.subheader("📈 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Indexed", f"{total_chunks:,} chunks")
    with col2:
        st.metric("Queries Run", total_queries, delta=f"+{st.session_state.perf_queries_this_session} this session")
    with col3:
        st.metric("Avg Query Time", f"{avg_query_time:.2f}s" if avg_query_time > 0 else "N/A")
    with col4:
        st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%" if cache_stats["total"] > 0 else "N/A")

    st.divider()

    # Section 2: Query Performance Trend
    if st.session_state.perf_query_performance:
        st.subheader("⚡ Query Performance Trend")

        df_queries = pd.DataFrame(st.session_state.perf_query_performance)

        # Create line chart with query times
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(1, len(df_queries) + 1)),
            y=df_queries['total_time'],
            mode='lines+markers',
            name='Total Time',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Query %{x}</b><br>Time: %{y:.2f}s<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=list(range(1, len(df_queries) + 1)),
            y=df_queries['retrieval_time'],
            mode='lines+markers',
            name='Retrieval Time',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate='<b>Query %{x}</b><br>Retrieval: %{y:.2f}s<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=list(range(1, len(df_queries) + 1)),
            y=df_queries['generation_time'],
            mode='lines+markers',
            name='Generation Time',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            marker=dict(size=4),
            hovertemplate='<b>Query %{x}</b><br>Generation: %{y:.2f}s<extra></extra>'
        ))

        fig.update_layout(
            title='Query Latency Over Time (Last 50 Queries)',
            xaxis_title='Query Number',
            yaxis_title='Time (seconds)',
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Query statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Query Time", f"{df_queries['total_time'].min():.2f}s")
        with col2:
            st.metric("Max Query Time", f"{df_queries['total_time'].max():.2f}s")
        with col3:
            st.metric("Median Query Time", f"{df_queries['total_time'].median():.2f}s")
        with col4:
            st.metric("Std Dev", f"{df_queries['total_time'].std():.2f}s")

        # Show detailed query table
        with st.expander("📋 Detailed Query Log"):
            display_df = df_queries[['timestamp', 'query', 'table_name', 'total_time', 'retrieval_time', 'generation_time', 'num_results', 'top_score']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['total_time'] = display_df['total_time'].round(2)
            display_df['retrieval_time'] = display_df['retrieval_time'].round(2)
            display_df['generation_time'] = display_df['generation_time'].round(2)
            display_df['top_score'] = display_df['top_score'].round(4)

            st.dataframe(
                display_df,
                column_config={
                    "timestamp": "Time",
                    "query": "Query",
                    "table_name": "Index",
                    "total_time": "Total (s)",
                    "retrieval_time": "Retrieval (s)",
                    "generation_time": "Generation (s)",
                    "num_results": "Results",
                    "top_score": "Score"
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("No query data available yet. Run some queries to see performance metrics!")

    st.divider()

    # Section 3: Cache Statistics
    st.subheader("🗄️ Cache Statistics")

    if cache_stats["total"] > 0:
        col1, col2 = st.columns([1, 1])

        with col1:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Cache Hits', 'Cache Misses'],
                values=[cache_stats["hits"], cache_stats["misses"]],
                hole=0.4,
                marker=dict(colors=['#2ecc71', '#e74c3c']),
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])

            fig_pie.update_layout(
                title=f'Cache Efficiency ({cache_stats["total"]} total queries)',
                height=350,
                showlegend=True
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.metric("Total Cache Queries", cache_stats["total"])
            st.metric("Cache Hits", cache_stats["hits"], delta=f"{cache_hit_rate:.1f}% hit rate")
            st.metric("Cache Misses", cache_stats["misses"])

            if cache_hit_rate > 70:
                st.success("Excellent cache performance!")
            elif cache_hit_rate > 40:
                st.info("Good cache performance")
            else:
                st.warning("Consider tuning cache threshold for better performance")
    else:
        st.info("Cache statistics will appear here when semantic cache is enabled and used.")
        st.caption("Enable semantic cache in Query page → Advanced Settings to track cache performance")

    st.divider()

    # Section 4: Indexing Performance
    if st.session_state.perf_indexing_metrics:
        st.subheader("📦 Indexing Performance")

        df_indexing = pd.DataFrame(st.session_state.perf_indexing_metrics)

        # Show summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Indexing Runs", len(df_indexing))
        with col2:
            st.metric("Total Chunks Indexed", f"{df_indexing['num_chunks'].sum():,}")
        with col3:
            st.metric("Avg Indexing Time", f"{df_indexing['total_time'].mean():.1f}s")
        with col4:
            avg_throughput = (df_indexing['num_chunks'] / df_indexing['total_time']).mean()
            st.metric("Avg Throughput", f"{avg_throughput:.0f} chunks/s")

        # Time breakdown chart
        st.markdown("**Time Breakdown by Stage**")

        # Calculate average percentages
        avg_load = (df_indexing['load_time'] / df_indexing['total_time']).mean() * 100
        avg_chunk = (df_indexing['chunk_time'] / df_indexing['total_time']).mean() * 100
        avg_embed = (df_indexing['embed_time'] / df_indexing['total_time']).mean() * 100
        avg_store = (df_indexing['store_time'] / df_indexing['total_time']).mean() * 100

        fig_breakdown = go.Figure(data=[go.Bar(
            x=['Load', 'Chunk', 'Embed', 'Store'],
            y=[avg_load, avg_chunk, avg_embed, avg_store],
            text=[f'{avg_load:.1f}%', f'{avg_chunk:.1f}%', f'{avg_embed:.1f}%', f'{avg_store:.1f}%'],
            textposition='auto',
            marker=dict(color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'])
        )])

        fig_breakdown.update_layout(
            title='Average Time Distribution Across Stages',
            xaxis_title='Stage',
            yaxis_title='Percentage of Total Time (%)',
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig_breakdown, use_container_width=True)

        # Detailed indexing table
        with st.expander("📋 Detailed Indexing Log"):
            display_df = df_indexing[['timestamp', 'table_name', 'num_docs', 'num_chunks', 'total_time', 'load_time', 'chunk_time', 'embed_time', 'store_time']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['total_time'] = display_df['total_time'].round(2)
            display_df['load_time'] = display_df['load_time'].round(2)
            display_df['chunk_time'] = display_df['chunk_time'].round(2)
            display_df['embed_time'] = display_df['embed_time'].round(2)
            display_df['store_time'] = display_df['store_time'].round(2)

            st.dataframe(
                display_df,
                column_config={
                    "timestamp": "Time",
                    "table_name": "Index",
                    "num_docs": "Docs",
                    "num_chunks": "Chunks",
                    "total_time": "Total (s)",
                    "load_time": "Load (s)",
                    "chunk_time": "Chunk (s)",
                    "embed_time": "Embed (s)",
                    "store_time": "Store (s)"
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("Indexing performance data will appear here after you index documents.")

    st.divider()

    # Section 5: Export Options
    st.subheader("💾 Export Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Export Query Metrics", use_container_width=True):
            if st.session_state.perf_query_performance:
                df_export = pd.DataFrame(st.session_state.perf_query_performance)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"query_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No query data to export")

    with col2:
        if st.button("📥 Export Indexing Metrics", use_container_width=True):
            if st.session_state.perf_indexing_metrics:
                df_export = pd.DataFrame(st.session_state.perf_indexing_metrics)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"indexing_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No indexing data to export")

    with col3:
        if st.button("🗑️ Clear All Metrics", type="secondary", use_container_width=True):
            if st.button("⚠️ Confirm Clear", type="primary"):
                st.session_state.perf_query_performance = []
                st.session_state.perf_indexing_metrics = []
                st.session_state.perf_cache_stats = {"hits": 0, "misses": 0, "total": 0}
                st.session_state.perf_total_queries = 0
                st.session_state.perf_queries_this_session = 0
                st.session_state.perf_chunks_indexed = 0
                st.session_state.perf_chunks_indexed_this_session = 0
                st.success("All metrics cleared!")
                st.rerun()

    # Session info
    st.divider()
    st.caption(f"Session started: {st.session_state.perf_session_start.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.perf_session_start else 'Unknown'}")
    st.caption(f"Session queries: {st.session_state.perf_queries_this_session} | Session chunks indexed: {st.session_state.perf_chunks_indexed_this_session:,}")

def run_chat_query(table_name: str, query: str, original_query: str, top_k: int,
                   temperature: float, max_tokens: int) -> Tuple[str, List[Dict[str, Any]]]:
    """Run chat query and return response with sources."""
    import rag_low_level_m1_16gb_verbose as rag

    # Configure settings
    rag.S.table = table_name
    rag.S.top_k = top_k
    rag.S.temperature = temperature
    rag.S.max_new_tokens = max_tokens
    rag.S.context_window = 3072

    # Safe defaults for chat mode
    rag.S.enable_query_expansion = False
    rag.S.enable_reranking = False
    rag.S.hybrid_alpha = 1.0
    rag.S.enable_filters = False

    # Handle table name prefix (PGVectorStore auto-prepends "data_")
    query_table_name = table_name
    if table_name.startswith("data_"):
        query_table_name = table_name[5:]

    rag.S.table = query_table_name

    # Auto-detect embedding model
    index_model = get_index_embedding_model(table_name)
    query_embed_model = index_model if index_model else rag.S.embed_model_name

    # Retrieve relevant chunks
    embed_model = get_embed_model(query_embed_model)
    vector_store = make_vector_store()
    retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=top_k)
    results = retriever._retrieve(QueryBundle(query_str=query))

    # Format sources
    sources = []
    for result in results:
        sources.append({
            "score": result.score,
            "text": result.node.get_content()
        })

    # Generate response
    llm = get_llm(
        context_window=3072,  # Fixed for chat mode
        max_tokens=max_tokens,
        temperature=temperature
    )
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    response = query_engine.query(original_query)  # Use original query for generation

    return str(response), sources

# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application."""

    # Sidebar
    st.sidebar.title("🔍 RAG Pipeline")
    st.sidebar.caption("Enhanced UI with full control")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["🚀 Quick Start", "⚙️ Advanced Index", "🔍 Query", "💬 Chat Mode", "📊 View Indexes", "📈 Performance", "⚙️ Settings"],
        format_func=lambda x: x.split(" ", 1)[1] if " " in x else x
    )

    st.sidebar.divider()

    # Quick stats
    try:
        indexes = list_vector_tables()
        st.sidebar.metric("Indexes", len(indexes))
        if indexes:
            total_chunks = sum(idx["rows"] for idx in indexes)
            st.sidebar.metric("Total Chunks", f"{total_chunks:,}")
    except:
        st.sidebar.metric("Indexes", "?")

    st.sidebar.caption(f"DB: {st.session_state.db_name}")

    # Route
    if page == "🚀 Quick Start":
        page_quick_start()
    elif page == "⚙️ Advanced Index":
        page_advanced_index()
    elif page == "🔍 Query":
        page_query()
    elif page == "💬 Chat Mode":
        page_chat()
    elif page == "📊 View Indexes":
        page_view_indexes()
    elif page == "📈 Performance":
        page_performance()
    elif page == "⚙️ Settings":
        page_settings()

if __name__ == "__main__":
    main()
