"""
Chat Mode additions for rag_web_enhanced.py
Copy and paste these functions into rag_web_enhanced.py before the Main App section
"""

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
    llm = get_llm()
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    response = query_engine.query(original_query)  # Use original query for generation

    return str(response), sources


# =============================================================================
# UPDATE THE NAVIGATION IN main() FUNCTION:
# =============================================================================
#
# Change this line:
#     page = st.sidebar.radio(
#         "Navigation",
#         ["🚀 Quick Start", "⚙️ Advanced Index", "🔍 Query", "📊 View Indexes", "⚙️ Settings"],
#         format_func=lambda x: x.split(" ", 1)[1] if " " in x else x
#     )
#
# To this:
#     page = st.sidebar.radio(
#         "Navigation",
#         ["🚀 Quick Start", "⚙️ Advanced Index", "🔍 Query", "💬 Chat", "📊 View Indexes", "⚙️ Settings"],
#         format_func=lambda x: x.split(" ", 1)[1] if " " in x else x
#     )
#
# And add this routing:
#     elif page == "💬 Chat":
#         page_chat()
