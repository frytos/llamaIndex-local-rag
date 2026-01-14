#!/usr/bin/env python3
"""
Example: Integrating Backend Management Classes with Streamlit RAG Web UI

This example demonstrates how to integrate the backend management classes
into the existing rag_web.py Streamlit application.

Classes demonstrated:
- PerformanceTracker: Track indexing/query performance
- ConfigurationManager: Save/load configurations
- CacheManager: Cache statistics
- ConversationManager: Multi-turn conversations

Usage:
    1. Copy the relevant sections into rag_web.py
    2. Initialize backend managers in session_state
    3. Use them in indexing/query functions
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import time
from typing import Dict, Any

# Import backend management classes
from rag_web_backend import (
    PerformanceTracker,
    ConfigurationManager,
    CacheManager,
    ConversationManager,
)


# =============================================================================
# Integration Pattern 1: Initialize in Session State
# =============================================================================

def init_backend_managers():
    """
    Initialize backend managers in Streamlit session state.

    Add this to your init_session_state() function in rag_web.py:
    """
    if "performance_tracker" not in st.session_state:
        st.session_state.performance_tracker = PerformanceTracker()

    if "config_manager" not in st.session_state:
        st.session_state.config_manager = ConfigurationManager()

    if "cache_manager" not in st.session_state:
        st.session_state.cache_manager = CacheManager()

    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager(max_turns=10)


# =============================================================================
# Integration Pattern 2: Track Indexing Performance
# =============================================================================

def run_indexing_with_tracking(doc_path, table_name, chunk_size, chunk_overlap,
                                embed_model_name, embed_dim, reset_table):
    """
    Enhanced indexing function with performance tracking.

    Wrap your existing run_indexing() function in rag_web.py:
    """
    tracker = st.session_state.performance_tracker

    # Start timing
    start_time = time.time()

    # Your existing indexing code here...
    # (load docs, chunk, embed, store)
    # For this example, we'll simulate:
    num_chunks = 1000  # This would be len(nodes) in real code

    # Simulate indexing
    with st.status("Indexing...", expanded=True):
        st.write("Loading documents...")
        time.sleep(0.5)

        st.write("Chunking...")
        time.sleep(0.5)

        st.write("Computing embeddings...")
        time.sleep(1)

        st.write("Storing in database...")
        time.sleep(0.5)

    # Record performance
    duration = time.time() - start_time

    tracker.record_indexing(
        duration=duration,
        chunks=num_chunks,
        config={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embed_model": embed_model_name,
            "table_name": table_name,
        }
    )

    st.success(f"✓ Indexed {num_chunks} chunks in {duration:.2f}s")


# =============================================================================
# Integration Pattern 3: Track Query Performance with Cache
# =============================================================================

def run_query_with_tracking(table_name, query, top_k):
    """
    Enhanced query function with performance tracking and cache management.

    Wrap your existing run_query() function in rag_web.py:
    """
    tracker = st.session_state.performance_tracker
    cache_mgr = st.session_state.cache_manager
    conv_mgr = st.session_state.conversation_manager

    # Get session ID (create one if not exists)
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

    session_id = st.session_state.session_id

    # Add user message to conversation
    conv_mgr.add_turn(session_id, "user", query)

    # Start timing
    start_time = time.time()

    # Check cache (simplified - real implementation uses semantic cache)
    cache_hit = False  # This would use cache_mgr.semantic_cache.get_semantic()

    # Your existing query code here...
    # (retrieve, generate answer)
    # For this example, we'll simulate:
    with st.spinner("Searching..."):
        time.sleep(0.5)

    results_count = top_k
    answer = "This is a simulated answer. In real code, this comes from the LLM."

    # Record performance
    duration = time.time() - start_time

    tracker.record_query(
        duration=duration,
        results_count=results_count,
        cache_hit=cache_hit,
        config={
            "table_name": table_name,
            "top_k": top_k,
        }
    )

    # Update cache stats
    if cache_hit:
        cache_mgr.record_hit()
    else:
        cache_mgr.record_miss()

    # Add assistant response to conversation
    conv_mgr.add_turn(session_id, "assistant", answer)

    # Display answer
    st.subheader("Answer")
    st.write(answer)

    # Show conversation context
    with st.expander("Conversation History"):
        formatted_context = conv_mgr.get_formatted_context(session_id)
        st.text(formatted_context)


# =============================================================================
# Integration Pattern 4: Configuration Management UI
# =============================================================================

def configuration_ui():
    """
    Add configuration save/load UI to your settings page.

    Add this to your page_settings() function:
    """
    st.subheader("Configuration Management")

    config_mgr = st.session_state.config_manager

    # Current configuration
    current_config = {
        "chunk_size": 700,
        "chunk_overlap": 150,
        "embed_model": "bge-small-en",
        "top_k": 4,
    }

    # Save configuration
    col1, col2 = st.columns(2)

    with col1:
        config_name = st.text_input("Configuration Name:", value="my_config")

    with col2:
        if st.button("💾 Save Configuration"):
            config_mgr.save(config_name, current_config)
            st.success(f"Saved configuration: {config_name}")

    # Load configuration
    st.divider()

    saved_configs = config_mgr.list_saved()

    if saved_configs:
        config_options = [c["name"] for c in saved_configs]
        selected_config = st.selectbox("Load Configuration:", config_options)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📂 Load"):
                loaded = config_mgr.load(selected_config)
                if loaded:
                    st.success(f"Loaded: {selected_config}")
                    st.json(loaded)

        with col2:
            if st.button("📋 Export .env"):
                loaded = config_mgr.load(selected_config)
                if loaded:
                    env_content = config_mgr.export_to_env(loaded)
                    st.code(env_content, language="bash")

        with col3:
            if st.button("🗑️ Delete"):
                if config_mgr.delete(selected_config):
                    st.success(f"Deleted: {selected_config}")
                    st.rerun()


# =============================================================================
# Integration Pattern 5: Performance Dashboard
# =============================================================================

def performance_dashboard():
    """
    Add performance dashboard to your web UI.

    Create a new page or add to existing page:
    """
    st.header("Performance Dashboard")

    tracker = st.session_state.performance_tracker
    cache_mgr = st.session_state.cache_manager

    # Summary statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Last 7 Days")
        summary_7d = tracker.get_summary(days=7)

        if summary_7d["indexing"]:
            st.metric(
                "Avg Indexing Speed",
                f"{summary_7d['indexing']['avg_throughput']:.1f} chunks/s"
            )

        if summary_7d["queries"]:
            st.metric(
                "Avg Query Time",
                f"{summary_7d['queries']['avg_duration']:.2f}s"
            )

    with col2:
        st.subheader("Cache Performance")
        cache_stats = cache_mgr.get_stats()

        st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
        st.metric("Total Requests", cache_stats['total'])

    # Export option
    st.divider()

    if st.button("📊 Export Performance Data"):
        tracker.export_to_csv("performance_export.csv")
        st.success("Exported to performance_export.csv")


# =============================================================================
# Demo Application
# =============================================================================

def main():
    """Demo Streamlit application showing integration patterns."""

    st.set_page_config(
        page_title="RAG Backend Integration Demo",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 RAG Backend Integration Demo")

    # Initialize backend managers
    init_backend_managers()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Demo:",
        [
            "Indexing with Tracking",
            "Query with Tracking",
            "Configuration Management",
            "Performance Dashboard",
        ]
    )

    st.sidebar.divider()

    # Show quick stats
    tracker = st.session_state.performance_tracker
    summary = tracker.get_summary()

    st.sidebar.metric("Indexing Operations", summary["indexing"].get("count", 0))
    st.sidebar.metric("Query Operations", summary["queries"].get("count", 0))

    cache_mgr = st.session_state.cache_manager
    cache_stats = cache_mgr.get_stats()
    st.sidebar.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1f}%")

    # Route to selected page
    if page == "Indexing with Tracking":
        st.header("Indexing with Performance Tracking")

        col1, col2 = st.columns(2)

        with col1:
            doc_path = st.text_input("Document Path:", value="data/document.pdf")
            chunk_size = st.slider("Chunk Size:", 100, 2000, 700)

        with col2:
            table_name = st.text_input("Table Name:", value="test_index")
            chunk_overlap = st.slider("Overlap:", 0, 500, 150)

        if st.button("🚀 Start Indexing", type="primary"):
            run_indexing_with_tracking(
                doc_path=doc_path,
                table_name=table_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embed_model_name="bge-small-en",
                embed_dim=384,
                reset_table=True,
            )

    elif page == "Query with Tracking":
        st.header("Query with Performance Tracking")

        table_name = st.text_input("Index Name:", value="test_index")
        top_k = st.slider("TOP_K:", 1, 10, 4)

        query = st.text_area(
            "Your Question:",
            height=100,
            placeholder="What is the main topic?"
        )

        if st.button("🔍 Search", type="primary"):
            if query.strip():
                run_query_with_tracking(table_name, query, top_k)
            else:
                st.warning("Please enter a question")

    elif page == "Configuration Management":
        configuration_ui()

    elif page == "Performance Dashboard":
        performance_dashboard()


if __name__ == "__main__":
    main()
