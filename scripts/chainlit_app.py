#!/usr/bin/env python3
"""
Chainlit RAG UI - Interactive visualization of your RAG system
Shows retrieved chunks, similarity scores, and sources inline
"""

import os
import chainlit as cl
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


# Configuration
DB_NAME = os.getenv("DB_NAME", "vector_db")
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGUSER = os.getenv("PGUSER", "fryt")
PGPASSWORD = os.getenv("PGPASSWORD", "frytos")
PGTABLE = os.getenv("PGTABLE", "llama2_paper")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
TOP_K = int(os.getenv("TOP_K", "4"))


@cl.on_chat_start
async def start():
    """Initialize RAG system when chat starts"""

    # Show loading message
    msg = cl.Message(content="🚀 Initializing RAG system...")
    await msg.send()

    # 1. Load embedding model
    await msg.stream_token("\n📊 Loading embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        cache_folder=os.path.expanduser("~/.cache/huggingface")
    )
    Settings.embed_model = embed_model

    # 2. Connect to vector store
    await msg.stream_token("\n🗄️  Connecting to PostgreSQL + pgvector...")
    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=PGHOST,
        port=PGPORT,
        user=PGUSER,
        password=PGPASSWORD,
        table_name=PGTABLE,
        embed_dim=EMBED_DIM,
    )

    # 3. Load LLM
    await msg.stream_token("\n🤖 Loading LLM (Mistral 7B)...")
    llm = LlamaCPP(
        model_url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3072,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 16},
        verbose=False,
    )
    Settings.llm = llm

    # 4. Create index and query engine
    await msg.stream_token("\n🔗 Creating query engine...")
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Create retriever with custom top_k
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
    )

    # Store in session
    cl.user_session.set("query_engine", query_engine)
    cl.user_session.set("retriever", retriever)

    await msg.stream_token("\n\n✅ Ready! Ask me anything about your documents.")
    await msg.update()

    # Show welcome message
    welcome = cl.Message(
        content=f"""
# 🎯 RAG System Ready!

**Configuration:**
- Database: `{PGTABLE}` table
- Embedding Model: `{EMBED_MODEL}`
- Top-K Retrieval: `{TOP_K}` chunks
- LLM: Mistral 7B Instruct

**How it works:**
1. Your question → converted to embedding vector
2. Find {TOP_K} most similar chunks from database
3. LLM generates answer based on retrieved context
4. See sources and similarity scores below each answer!

Try asking: *"What are the key findings?"* or *"Summarize the main points"*
"""
    )
    await welcome.send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""

    query_engine = cl.user_session.get("query_engine")
    retriever = cl.user_session.get("retriever")

    if not query_engine:
        await cl.Message(content="❌ Query engine not initialized. Please refresh.").send()
        return

    # Show "thinking" message
    thinking_msg = cl.Message(content="🤔 Searching documents...")
    await thinking_msg.send()

    try:
        # First, show retrieval step
        from llama_index.core import QueryBundle
        query_bundle = QueryBundle(query_str=message.content)

        # Retrieve nodes
        retrieved_nodes = retriever.retrieve(query_bundle.query_str)

        # Show retrieved chunks
        source_elements = []
        retrieval_info = "## 🔍 Retrieved Chunks:\n\n"

        for i, node in enumerate(retrieved_nodes):
            score = node.score if hasattr(node, 'score') else 0.0

            # Quality indicator
            if score > 0.8:
                quality = "🟢 Excellent"
                color = "#4caf50"
            elif score > 0.6:
                quality = "🟡 Good"
                color = "#ff9800"
            else:
                quality = "🔴 Fair"
                color = "#f44336"

            # Get source metadata
            source_file = node.metadata.get('file_name', 'Unknown')
            page = node.metadata.get('page_label', 'N/A')

            retrieval_info += f"**{quality} Chunk {i+1}** (Score: `{score:.4f}`) - Source: `{source_file}` Page {page}\n\n"

            # Create text element with the chunk content
            source_elements.append(
                cl.Text(
                    name=f"📄 Source {i+1} (score: {score:.3f})",
                    content=node.text,
                    display="side"
                )
            )

        await thinking_msg.stream_token("\n" + retrieval_info)
        await thinking_msg.update()

        # Now generate answer
        generation_msg = cl.Message(content="✨ Generating answer...")
        await generation_msg.send()

        # Query and get response
        response = query_engine.query(message.content)

        # Build final response with answer
        final_content = f"""
## ✅ Answer:

{response.response}

---

## 📚 Sources Used:

This answer is based on {len(retrieved_nodes)} retrieved chunks shown in the sidebar →

**Quality Metrics:**
- Best match score: `{max(n.score for n in retrieved_nodes):.4f}`
- Average score: `{sum(n.score for n in retrieved_nodes)/len(retrieved_nodes):.4f}`
"""

        # Send final message with sources
        await cl.Message(
            content=final_content,
            elements=source_elements
        ).send()

        # Clean up thinking messages
        await thinking_msg.remove()
        await generation_msg.remove()

    except Exception as e:
        await cl.Message(
            content=f"❌ Error: {str(e)}\n\nPlease check your database connection and settings."
        ).send()


@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates"""
    print(f"Settings updated: {settings}")


if __name__ == "__main__":
    # Run with: chainlit run chainlit_app.py -w
    pass
