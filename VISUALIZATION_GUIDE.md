# 🎨 RAG Visualization Tools Guide

Three powerful tools to visualize and understand your RAG system in action!

---

## 1️⃣ Chainlit - Interactive Chat UI

**Best for:** Testing your RAG with a beautiful interface, seeing retrieval in real-time

### Quick Start:
```bash
chainlit run chainlit_app.py -w
```

### What You'll See:
- 💬 Chat interface for asking questions
- 📊 Retrieval scores for each chunk
- 📄 Full source text in sidebar
- 🎯 Quality indicators (🟢 Excellent, 🟡 Good, 🔴 Fair)
- 📈 Metrics (best score, average, etc.)

### Features:
- Real-time streaming responses
- Click sources to expand
- See exactly what the LLM sees
- No coding required - just chat!

### Screenshot Preview:
```
┌──────────────────────────────────────┐
│ 🎯 RAG System Ready!                 │
│                                      │
│ Ask: "What are the main findings?"  │
└──────────────────────────────────────┘

User: What is Llama 2?

🔍 Retrieved Chunks:
🟢 Excellent Chunk 1 (Score: 0.8276)
   Source: llama2.pdf Page 1

✅ Answer:
Llama 2 is an open-source large language
model developed by Meta AI...

📚 Sources Used: [View in sidebar →]
```

---

## 2️⃣ TensorBoard Projector - 3D Embeddings

**Best for:** Understanding semantic relationships, finding clusters

### Step 1: Generate Embeddings Data
```bash
python tensorboard_embeddings.py
```

### Step 2: Launch TensorBoard
```bash
tensorboard --logdir=runs/rag_embeddings
```

### Step 3: Open Browser
```
http://localhost:6006
```

### What You'll See:
- 🌐 Interactive 3D scatter plot of all your document chunks
- 🔍 Hover to see chunk text
- 🎨 Color by metadata
- 📊 Multiple projection methods (PCA, t-SNE, UMAP)

### Controls:
- **Rotate:** Click + drag
- **Zoom:** Scroll wheel
- **Search:** Type text to highlight matching points
- **Nearest neighbors:** Click point to see similar chunks

### Use Cases:
- Find duplicate content (points very close together)
- Identify document clusters (groups of related content)
- Discover outliers (isolated points = unique content)
- Validate embedding quality (similar topics should cluster)

### Tips:
```python
# For best visualization, try different projections:
1. PCA - Fast, shows main variance
2. t-SNE - Better clusters, slower (2-3 min for 500 points)
3. UMAP - Balance between speed and quality
```

---

## 3️⃣ Atlas by Nomic - Beautiful Semantic Maps

**Best for:** Publishing/sharing embeddings, production monitoring

### Setup (First Time Only):
```bash
# 1. Install
pip install nomic

# 2. Create free account at https://atlas.nomic.ai

# 3. Login
nomic login
```

### Create Map:
```bash
python atlas_embeddings.py
```

### What You'll See:
- 🗺️ Gorgeous 2D map of your embeddings
- 🎨 Semantic clusters automatically colored
- 🔍 Search and filter capabilities
- 📊 Metadata overlays
- 🌐 Shareable web link

### Features:
- **Public/Private:** Choose who can view
- **Collaborative:** Share with team
- **Persistent:** Maps stay online
- **Fast:** Optimized for millions of points

### Example Use Cases:
```
1. Content Quality Audit
   → Color by similarity score
   → Find low-quality chunks (outliers)

2. Topic Discovery
   → See natural topic clusters
   → Identify gaps in documentation

3. Search Debugging
   → Plot query alongside docs
   → Visualize why certain chunks retrieved

4. Production Monitoring
   → Upload new embeddings daily
   → Track content drift over time
```

---

## 🆚 Comparison Matrix

| Feature | Chainlit | TensorBoard | Atlas |
|---------|----------|-------------|-------|
| **Best For** | Testing queries | Understanding embeddings | Production monitoring |
| **Setup Time** | 1 minute | 2 minutes | 5 minutes (account) |
| **Internet Required** | No | No | Yes |
| **Interactive** | ✅ Chat | ✅ 3D rotation | ✅ Web UI |
| **Shows Sources** | ✅ Yes | ❌ No | ⚠️ Limited |
| **Shows Retrieval** | ✅ Real-time | ❌ No | ❌ No |
| **Embedding Viz** | ❌ No | ✅ 3D | ✅ 2D map |
| **Shareable** | ❌ Local | ❌ Local | ✅ Web link |
| **Cost** | Free | Free | Free (5M points) |
| **Performance** | Fast | Medium | Fast |

---

## 🎯 Recommended Workflow

### During Development:
1. **Use Chainlit** for testing queries and debugging retrieval
   ```bash
   chainlit run chainlit_app.py -w
   ```

### For Analysis:
2. **Use TensorBoard** to understand your document structure
   ```bash
   python tensorboard_embeddings.py
   tensorboard --logdir=runs/rag_embeddings
   ```

### For Production:
3. **Use Atlas** to monitor embedding quality over time
   ```bash
   python atlas_embeddings.py
   ```

---

## 🔧 Configuration

All tools read from your environment variables:

```bash
export DB_NAME=vector_db
export PGHOST=localhost
export PGPORT=5432
export PGUSER=fryt
export PGPASSWORD=frytos
export PGTABLE=llama2_paper  # Change this for different documents
export TOP_K=4
```

---

## 🐛 Troubleshooting

### Chainlit shows "Query engine not initialized"
- Check PostgreSQL is running: `docker-compose ps`
- Verify table exists: `PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "\dt"`

### TensorBoard shows empty projector
- Make sure you ran `tensorboard_embeddings.py` first
- Check that `runs/rag_embeddings/` directory has data

### Atlas says "Not logged in"
- Run: `nomic login`
- Follow browser prompt
- Try again

### No embeddings found
- Ensure you've indexed documents first
- Run: `RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py`
- Check table name matches PGTABLE env var

---

## 📚 Advanced Usage

### Compare Multiple Documents in TensorBoard:
```bash
# Index document 1
export PGTABLE=llama2_paper
python tensorboard_embeddings.py

# Index document 2
export PGTABLE=mastering_rag_paper
python tensorboard_embeddings.py

# Both will appear in TensorBoard!
tensorboard --logdir=runs/rag_embeddings
```

### Query-specific Visualization:
```python
# Modify chainlit_app.py to color-code by query relevance
# Modify tensorboard_embeddings.py to highlight query result
```

---

## 🎓 Learning Resources

### Chainlit:
- Docs: https://docs.chainlit.io
- Examples: https://github.com/Chainlit/chainlit/tree/main/examples

### TensorBoard:
- Projector Guide: https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin
- Embedding Tutorial: https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin

### Atlas:
- Docs: https://docs.nomic.ai/
- Examples: https://github.com/nomic-ai/atlas-examples

---

## 💡 Pro Tips

1. **Chainlit + TensorBoard Together:**
   - Run Chainlit on port 8000 (default)
   - Run TensorBoard on port 6006
   - Keep both open in browser tabs
   - Query in Chainlit, analyze patterns in TensorBoard

2. **Use Atlas for Demos:**
   - Create beautiful maps
   - Share link with stakeholders
   - No need to install anything

3. **Debug Bad Retrieval:**
   - Use Chainlit to identify problematic queries
   - Use TensorBoard to see if embeddings are clustered correctly
   - Adjust CHUNK_SIZE if needed

---

## 🚀 Next Steps

1. Start with **Chainlit** - test your RAG interactively
2. Move to **TensorBoard** - understand your document structure
3. Deploy with **Atlas** - share with team or monitor production

**Ready to visualize?** Pick a tool and run it! 🎉
