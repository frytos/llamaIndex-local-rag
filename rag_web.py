#!/usr/bin/env python3
"""
Streamlit Web UI for RAG Pipeline

Launch with: streamlit run rag_web.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

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

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"

CHUNK_PRESETS = {
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
    defaults = {
        # Database settings
        "db_host": os.environ.get("PGHOST", "localhost"),
        "db_port": os.environ.get("PGPORT", "5432"),
        "db_user": os.environ.get("PGUSER"),
        "db_password": os.environ.get("PGPASSWORD"),
        "db_name": os.environ.get("DB_NAME", "vector_db"),

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
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

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
            info = {"name": table, "rows": 0, "chunk_size": "?", "chunk_overlap": "?"}

            # Get row count
            try:
                cur.execute(
                    sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(table))
                )
                info["rows"] = cur.fetchone()["count"]
            except Exception as e:
                conn.rollback()
                continue

            # Try to get metadata from first row
            try:
                cur.execute(
                    sql.SQL("""
                        SELECT metadata_->>'_chunk_size' as cs,
                               metadata_->>'_chunk_overlap' as co
                        FROM {}
                        WHERE metadata_->>'_chunk_size' IS NOT NULL
                        LIMIT 1
                    """).format(sql.Identifier(table))
                )
                row = cur.fetchone()
                if row:
                    info["chunk_size"] = row["cs"] or "?"
                    info["chunk_overlap"] = row["co"] or "?"
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
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)

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

    if selected == "📝 Enter custom path":
        doc_path = st.text_input("Enter path:", value=str(DATA_DIR))
        doc_path = Path(doc_path) if doc_path else None
    else:
        doc_path = option_paths.get(selected)

    if not doc_path or not doc_path.exists():
        st.warning("Please select a valid document or folder")
        return

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

    st.info(f"Model: `{embed_model_name}` | Dimensions: **{embed_dim}**")

    # Table Name
    st.subheader("4. Index Name")

    suggested_name = generate_table_name(doc_path, chunk_size, chunk_overlap, embed_model_name)
    table_name = st.text_input("Table name:", value=suggested_name)

    reset_table = st.checkbox("Reset table if exists", value=True)

    # Start Indexing
    st.subheader("5. Start Indexing")

    if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
        run_indexing(doc_path, table_name, chunk_size, chunk_overlap, embed_model_name, embed_dim, reset_table)

def run_indexing(doc_path: Path, table_name: str, chunk_size: int, chunk_overlap: int,
                 embed_model_name: str, embed_dim: int, reset_table: bool):
    """Run the indexing pipeline with progress visualization."""

    # Update settings
    import rag_low_level_m1_16gb_verbose as rag
    rag.S.pdf_path = str(doc_path)
    rag.S.table = table_name
    rag.S.chunk_size = chunk_size
    rag.S.chunk_overlap = chunk_overlap
    rag.S.embed_model_name = embed_model_name
    rag.S.embed_dim = embed_dim
    rag.S.reset_table = reset_table

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

            # Get or load embedding model
            embed_model = get_embed_model(embed_model_name)

            # Embed in batches with progress
            batch_size = 64
            total_batches = (len(nodes) + batch_size - 1) // batch_size

            embeddings_list = []
            for i, batch in enumerate(chunked(nodes, batch_size)):
                texts = [n.get_content() for n in batch]
                batch_embeddings = embed_model.get_text_embedding_batch(texts)

                for node, emb in zip(batch, batch_embeddings):
                    node.embedding = emb
                    embeddings_list.append(emb)

                progress.progress(40 + int(40 * (i + 1) / total_batches))

            st.success(f"✓ Embedded {len(nodes)} chunks")
            st.session_state.last_indexed_nodes = nodes
            st.session_state.last_embeddings = np.array(embeddings_list)
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

def page_query():
    """Query page."""
    st.header("Query Index")

    # Get available indexes
    indexes = list_vector_tables()

    if not indexes:
        st.warning("No indexes found. Please index some documents first.")
        return

    # Index selection
    st.subheader("1. Select Index")

    index_options = [f"{idx['name']} ({idx['rows']} chunks, cs={idx['chunk_size']})" for idx in indexes]
    selected_idx = st.selectbox("Index:", index_options)
    table_name = indexes[index_options.index(selected_idx)]["name"]

    # Query parameters
    st.subheader("2. Query Settings")

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("TOP_K (chunks to retrieve):", 1, 10, 4)
    with col2:
        show_sources = st.checkbox("Show source chunks", value=True)

    # Query input
    st.subheader("3. Ask a Question")

    query = st.text_area("Your question:", height=100, placeholder="What is the main topic of the document?")

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question")
            return

        run_query(table_name, query, top_k, show_sources)

    # Query history
    if st.session_state.query_history:
        st.subheader("Query History")
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Q: {item['query'][:50]}..."):
                st.write(f"**Answer:** {item['answer']}")
                st.caption(f"Top score: {item['top_score']:.4f} | Chunks: {item['chunks']}")

def run_query(table_name: str, query: str, top_k: int, show_sources: bool):
    """Run a query against the index."""

    import rag_low_level_m1_16gb_verbose as rag
    rag.S.table = table_name
    rag.S.top_k = top_k

    with st.spinner("Searching..."):
        try:
            # Build retriever
            embed_model = get_embed_model(rag.S.embed_model_name)
            vector_store = make_vector_store()
            retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=top_k)

            # Retrieve
            results = retriever._retrieve(QueryBundle(query_str=query))
        except Exception as e:
            st.error(f"Retrieval error: {e}")
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
    df.columns = ["Name", "Chunks", "Chunk Size", "Overlap"]

    st.dataframe(df, use_container_width=True, hide_index=True)

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
        ["Index Documents", "Query", "View Indexes", "Settings"],
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

if __name__ == "__main__":
    main()
