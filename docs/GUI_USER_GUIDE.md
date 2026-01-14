# RAG Web UI - Complete User Guide

**Version**: 2.0.0 | **Last Updated**: January 2026

---

## Table of Contents

- [Part 1: Overview](#part-1-overview)
- [Part 2: Page-by-Page Guide](#part-2-page-by-page-guide)
- [Part 3: Feature Deep Dives](#part-3-feature-deep-dives)
- [Part 4: Troubleshooting](#part-4-troubleshooting)
- [Part 5: Advanced Workflows](#part-5-advanced-workflows)
- [Part 6: Quick Reference](#part-6-quick-reference)

---

# Part 1: Overview

## What is the RAG Web UI?

The RAG Web UI is a **Streamlit-based graphical interface** for the LlamaIndex Local RAG pipeline. It provides an intuitive way to:

- Index documents into PostgreSQL+pgvector
- Query your indexed documents with natural language
- Visualize embeddings and chunk distributions
- Manage multiple indexes
- Configure advanced RAG parameters

**Launch command:**
```bash
streamlit run rag_web.py
```

The UI will open in your browser at `http://localhost:8501`.

---

## Feature Comparison: CLI vs GUI

| Feature | CLI (rag_low_level_m1_16gb_verbose.py) | GUI (rag_web.py) |
|---------|----------------------------------------|-------------------|
| **Indexing** | Environment variables + command-line flags | Visual form with presets and sliders |
| **Querying** | Interactive REPL or single query | Web-based query interface |
| **Visualization** | None (logs only) | Embedding plots, chunk histograms |
| **Index Management** | Manual SQL commands | Browse, view, delete indexes |
| **Configuration** | `.env` file or exports | Settings page + per-query overrides |
| **Multi-document** | Manual table switching | Dropdown index selector |
| **Learning Curve** | Steeper (requires env var knowledge) | Gentler (guided workflow) |
| **Automation** | Excellent (scriptable) | Limited (manual workflows) |
| **Performance Tracking** | Built-in metrics | Visual dashboards |
| **Batch Operations** | Easy with shell scripts | One-at-a-time |

---

## When to Use GUI vs CLI

### Use the **GUI** when:
- Learning the system for the first time
- Experimenting with different chunk sizes
- Comparing embedding models visually
- One-off document indexing
- Exploring retrieval parameters
- Demonstrating RAG to stakeholders
- Debugging retrieval quality (chunk inspection)

### Use the **CLI** when:
- Indexing large document collections (>1000 files)
- Automating workflows (CI/CD, cron jobs)
- Performance benchmarking
- Scripting repetitive tasks
- Production deployments
- Using advanced features (HyDE, query expansion, reranking)
- Integration with other tools

---

## Quick Start Guide

**Prerequisites:**
- Python 3.11+ virtual environment activated
- PostgreSQL + pgvector running (`docker-compose up -d`)
- `.env` file configured with database credentials

**Step-by-step:**

1. **Install optional dependencies:**
   ```bash
   pip install -r requirements-optional.txt
   ```

2. **Launch the web UI:**
   ```bash
   streamlit run rag_web.py
   ```

3. **Place a document in `data/` folder:**
   ```bash
   cp ~/Downloads/mydocument.pdf data/
   ```

4. **Index your document:**
   - Go to "Index Documents" tab
   - Select your document
   - Choose "Balanced (700/150)" preset
   - Click "Start Indexing"

5. **Query your document:**
   - Go to "Query" tab
   - Select your index
   - Type your question
   - Click "Search"

**That's it!** You've indexed and queried your first document.

---

# Part 2: Page-by-Page Guide

## Index Documents Page

This page handles document ingestion and embedding.

### Step 1: Select Document or Folder

**What you see:**
- List of documents in `data/` folder
- Folders (with file counts)
- Option to enter a custom path

**Supported formats:**
- PDF (`.pdf`)
- Word documents (`.docx`)
- PowerPoint (`.pptx`)
- Text files (`.txt`, `.md`)
- HTML/HTM (`.html`, `.htm`)
- JSON, CSV, XML
- Code files (`.py`, `.js`, `.ts`, `.java`)

**Examples:**

| Display | Type | What it indexes |
|---------|------|-----------------|
| `📁 inbox_clean/ (1523 files)` | Folder | All supported files in folder |
| `📄 llama2.pdf (13.2 MB)` | File | Single PDF document |
| `📝 Enter custom path` | Custom | Any absolute path |

**Tips:**
- Folders are indexed recursively (includes subdirectories)
- Large folders (>1000 files) may take 30+ minutes
- Use custom path for files outside `data/` directory

---

### Step 2: Chunking Parameters

**What you see:**
- Preset selector with 5 options
- Optional custom chunk size/overlap sliders
- Overlap percentage display

**Available presets:**

| Preset | Chunk Size | Overlap | Overlap % | Best For |
|--------|------------|---------|-----------|----------|
| **Ultra-fine (100/20)** | 100 | 20 | 20% | Chat logs, tweets, short messages |
| **Fine-grained (300/60)** | 300 | 60 | 20% | Q&A, specific facts |
| **Balanced (700/150)** ⭐ | 700 | 150 | 21% | General documents (recommended) |
| **Contextual (1200/240)** | 1200 | 240 | 20% | Long-form content, summaries |
| **Large context (2000/400)** | 2000 | 400 | 20% | Essays, technical papers |

**What do these numbers mean?**

- **Chunk Size**: How many characters each chunk contains
  - Smaller = more precise retrieval, less context
  - Larger = more context, less precise

- **Overlap**: How many characters are shared between adjacent chunks
  - Prevents context from being split across boundaries
  - Typical range: 15-25% of chunk size

**Custom values:**
- Enable "Custom values" checkbox
- Adjust sliders (Chunk size: 50-3000, Overlap: 0-500)
- The UI shows overlap percentage automatically

**Example decision tree:**

```
What are you indexing?
│
├─ Chat logs, social media → Ultra-fine (100/20)
│
├─ Q&A documentation, FAQs → Fine-grained (300/60)
│
├─ Books, articles, reports → Balanced (700/150) ⭐
│
├─ Technical papers, long-form → Contextual (1200/240)
│
└─ Academic research, theses → Large context (2000/400)
```

---

### Step 3: Embedding Model

**What you see:**
- Model selector with 4 options
- Model name and dimensions display

**Available models:**

| Model | Dimensions | Speed | Quality | Memory | Best For |
|-------|------------|-------|---------|--------|----------|
| **all-MiniLM-L6-v2 (Fast)** | 384 | 🚀🚀🚀 | ⭐⭐⭐ | 90 MB | Quick prototyping |
| **bge-small-en (Recommended)** ⭐ | 384 | 🚀🚀 | ⭐⭐⭐⭐ | 130 MB | General use (best balance) |
| **bge-base-en (Better)** | 768 | 🚀 | ⭐⭐⭐⭐⭐ | 440 MB | Higher quality needs |
| **bge-large-en (Best)** | 1024 | 🐌 | ⭐⭐⭐⭐⭐+ | 1.3 GB | Maximum quality (slow) |

**Performance comparison (M1 Mac 16GB):**

| Model | Embedding Speed | 10K chunks | 50K chunks |
|-------|----------------|------------|------------|
| all-MiniLM-L6-v2 | ~80 chunks/s | 2 min | 10 min |
| bge-small-en | ~67 chunks/s | 2.5 min | 12 min |
| bge-base-en | ~40 chunks/s | 4 min | 20 min |
| bge-large-en | ~20 chunks/s | 8 min | 40 min |

**Which model should I choose?**

- **bge-small-en** (recommended): Best balance of speed and quality
- **all-MiniLM-L6-v2**: Choose if indexing >100K chunks
- **bge-base-en**: Choose if quality is more important than speed
- **bge-large-en**: Choose only for research-grade retrieval

**Important:** Once you index with a model, you must use the same model for queries. Different models produce incompatible embeddings.

---

### Step 4: Index Name

**What you see:**
- Auto-generated table name
- Editable text field
- "Reset table if exists" checkbox

**Auto-generated naming convention:**
```
{document}_{chunk_size}_{overlap}_{model_short}
```

**Examples:**
- `llama2_700_150_bge_small_en`
- `inbox_clean_300_60_minilm`
- `report_1200_240_bge_base`

**Why this matters:**
- Multiple indexes can coexist in the same database
- Descriptive names help track configurations
- Avoids accidental overwrites

**Reset table checkbox:**
- ✅ **Checked (recommended)**: Drops existing table before indexing
  - Use when: Re-indexing with different parameters
  - Prevents: Duplicate data, mixed configurations

- ❌ **Unchecked**: Appends to existing table
  - Use when: Adding new documents to existing index
  - Warning: Mixing chunk sizes in one table causes poor retrieval

**Pro tip:** Include your initials or date in the name for team projects:
```
inbox_john_2026_01_08_balanced
```

---

### Step 5: Start Indexing

**What happens when you click "Start Indexing":**

1. **Loading documents** (10-60s for large folders)
   - Reads all supported files
   - Extracts text content
   - Shows total character count

2. **Chunking documents** (5-30s)
   - Splits text into chunks using SentenceSplitter
   - Preserves sentence boundaries
   - Shows chunk count and size distribution

3. **Computing embeddings** (2-40 min depending on size)
   - Batch embedding with progress bar
   - Auto-detects Apple Silicon Metal acceleration
   - Shows chunks/second throughput

4. **Storing in database** (10-120s)
   - Creates pgvector table
   - Inserts nodes in batches of 250
   - Stores embeddings and metadata

**Progress indicators:**
- ✓ Green checkmarks for completed steps
- Progress bar (0-100%)
- Time estimates for each phase
- Error messages if something fails

**After indexing completes:**

You'll see three visualizations:

1. **Chunk Distribution Histogram**
   - X-axis: Chunk size in characters
   - Y-axis: Number of chunks
   - Shows if chunking worked as expected
   - Look for: Bell curve centered near your target chunk size

2. **Chunk Samples**
   - First 3 chunks from your document
   - Truncated to 500 characters
   - Shows actual content being indexed

3. **Embedding Visualization**
   - 2D or 3D scatter plot of embeddings
   - Colors = different source files
   - Interactive (hover to see text)
   - Choose method: t-SNE, PCA, or UMAP

**Troubleshooting indexing:**

| Issue | Cause | Solution |
|-------|-------|----------|
| "Error loading documents" | File format not supported | Convert to PDF or TXT |
| Chunks too small/large | Preset mismatch | Try different preset |
| Out of memory | Model too large | Use bge-small-en or MiniLM |
| Very slow embedding | CPU-only mode | Verify Metal/MPS detected |
| "Table already exists" | Reset unchecked | Enable "Reset table" |

---

## Query Page

This page handles question answering over your indexed documents.

### Step 1: Select Index

**What you see:**
- Dropdown with all available indexes
- Format: `{table_name} ({rows} chunks, cs={chunk_size})`

**Example:**
```
inbox_700_150_bge_small (47651 chunks, cs=700)
llama2_700_150_bge_small (2341 chunks, cs=700)
```

**What the metadata means:**
- **47651 chunks**: Number of searchable text chunks
- **cs=700**: Chunk size used during indexing

**No indexes found?**
- Go to "Index Documents" page first
- Index at least one document
- Return to this page

---

### Step 2: Query Settings

**What you see:**
- TOP_K slider (1-10)
- "Show source chunks" checkbox

**TOP_K parameter:**

Controls how many chunks are retrieved before generating an answer.

| TOP_K | Retrieval | Quality | Speed | Best For |
|-------|-----------|---------|-------|----------|
| 2-3 | Narrow | Focused | Fast | Simple factual queries |
| 4-6 | Balanced | Good | Medium | General questions (recommended) |
| 8-10 | Broad | Comprehensive | Slow | Complex multi-part questions |

**Trade-offs:**
- **Higher TOP_K** = More context for LLM, but:
  - Slower generation (more text to process)
  - Risk of context window overflow
  - May include irrelevant chunks (noise)

- **Lower TOP_K** = Faster responses, but:
  - May miss relevant information
  - Less comprehensive answers
  - Better for focused queries

**Show source chunks:**
- ✅ **Checked (recommended)**: Shows retrieved chunks with scores
  - See what the LLM used to generate the answer
  - Verify retrieval quality
  - Debug poor answers

- ❌ **Unchecked**: Only shows generated answer
  - Cleaner output
  - Faster page load
  - For production use

---

### Step 3: Ask a Question

**What you see:**
- Text area for your question
- "Search" button
- Query history (last 5 queries)

**Writing effective queries:**

**Good queries:**
- Specific: "What did Elena say about traveling to Morocco?"
- Focused: "What are the three main benefits of attention mechanisms?"
- Natural: "When did we discuss the Q3 roadmap?"

**Poor queries:**
- Too vague: "Tell me everything"
- Single words: "Morocco" (use hybrid search instead)
- Multiple unrelated topics: "What about X, Y, and Z?"

**Query patterns:**

| Pattern | Example | When to Use |
|---------|---------|-------------|
| **Factual lookup** | "What is the definition of RAG?" | Extracting specific information |
| **Summarization** | "Summarize the main points of the paper" | Overview of long content |
| **Comparison** | "How does method A differ from method B?" | Contrasting topics |
| **Temporal** | "What happened after June 2024?" | Time-based filtering |
| **Entity-focused** | "What did Alice say about the project?" | Person/place/thing queries |

---

### Retrieved Chunks Display

After clicking "Search", you'll see the retrieved chunks (if enabled):

**Chunk card format:**
```
Chunk 1: 🟢 Excellent (Score: 0.8234) [Expandable]
  Similarity: 0.8234
  Source: document.pdf
  [Full chunk text...]
```

**Score interpretation:**

| Badge | Score Range | Meaning | Action |
|-------|-------------|---------|--------|
| 🟢 Excellent | 0.70 - 1.00 | Highly relevant | Great! Use this |
| 🟡 Good | 0.50 - 0.70 | Relevant | Probably useful |
| 🟠 Fair | 0.30 - 0.50 | Somewhat relevant | May be noise |
| 🔴 Low | 0.00 - 0.30 | Likely irrelevant | Check your query |

**What to look for:**
- **High scores (>0.7)**: Good retrieval, trust the answer
- **Low scores (<0.5)**: Poor retrieval, rephrase query or adjust chunk size
- **Mixed scores**: Some good, some bad - increase TOP_K might help

**Debugging retrieval issues:**

| Observation | Problem | Solution |
|-------------|---------|----------|
| All scores <0.5 | Query doesn't match document style | Rephrase query, try hybrid search |
| Relevant chunks ranked low | Semantic mismatch | Use BM25 (HYBRID_ALPHA=0.3) |
| Irrelevant chunks ranked high | Poor chunking | Re-index with smaller chunks |
| Duplicate chunks | Overlap too high | Re-index with less overlap |

---

### Generated Answer

Below the retrieved chunks, you'll see the LLM-generated answer.

**What you see:**
- Green success box with answer text
- Answer quality depends on:
  - Retrieval quality (chunk scores)
  - LLM model (Mistral 7B by default)
  - Temperature setting (0.1 = factual)

**Evaluating answer quality:**

**Good answers have:**
- Directly addresses your question
- Cites information from retrieved chunks
- Coherent and well-structured
- Appropriate length (not too verbose)

**Poor answers may:**
- Contradict retrieved chunks
- Include hallucinated information
- Be too generic ("As an AI...")
- Miss key information

**If answer quality is poor:**

1. **Check retrieved chunks first**
   - Are they relevant? (Scores >0.5)
   - Do they contain the answer?
   - If no → retrieval problem (adjust chunk size, TOP_K, or query)
   - If yes → generation problem (adjust LLM temperature)

2. **Improve retrieval:**
   - Increase TOP_K (more context)
   - Use hybrid search (HYBRID_ALPHA=0.5 in CLI)
   - Re-index with different chunk size

3. **Improve generation:**
   - Rephrase question more clearly
   - Adjust temperature (CLI: TEMP=0.05 for factual)
   - Increase MAX_NEW_TOKENS (CLI: MAX_NEW_TOKENS=512)

---

### Query History

At the bottom of the Query page, you'll see your last 5 queries.

**What's stored:**
- Your question
- Generated answer
- Number of chunks retrieved
- Top similarity score

**Expanding a history item:**
- Click to view full question and answer
- Shows top score and chunk count
- Useful for comparing queries

**Note:** Query history is session-based (lost on refresh). For persistent logging, use CLI with `LOG_QUERIES=1`.

---

## View Indexes Page

This page shows all your indexed documents and provides management tools.

### Index Overview

**Summary statistics:**
- **Total Indexes**: Number of tables
- **Total Chunks**: Sum of all indexed chunks
- **Database**: Current database name

**Index table columns:**
- **Name**: Table name
- **Chunks**: Number of indexed text chunks
- **Chunk Size**: Characters per chunk
- **Overlap**: Character overlap between chunks

**Example:**

| Name | Chunks | Chunk Size | Overlap |
|------|--------|------------|---------|
| inbox_700_150_bge_small | 47,651 | 700 | 150 |
| llama2_700_150_bge_small | 2,341 | 700 | 150 |
| report_300_60_minilm | 8,453 | 300 | 60 |

---

### Index Actions

**Available actions:**

1. **👁️ View Embeddings**
   - Visualizes embeddings in 2D or 3D space
   - Shows up to 500 randomly sampled chunks
   - Interactive scatter plot (hover for chunk text)
   - Useful for: Understanding semantic clustering, debugging poor retrieval

2. **🗑️ Delete Index**
   - Permanently removes table and all data
   - Requires confirmation
   - Cannot be undone
   - Use when: Cleaning up old experiments, freeing disk space

**Embedding visualization:**

After clicking "View Embeddings":

1. **Choose projection method:**
   - **t-SNE** (default): Good for cluster visualization
   - **PCA**: Fast, linear dimensionality reduction
   - **UMAP**: Best clustering, requires installation (`pip install umap-learn`)

2. **Choose dimensions:**
   - **2D**: Easier to read, faster to compute
   - **3D**: More information, interactive rotation

3. **Interpretation:**
   - **Clusters**: Chunks about similar topics
   - **Colors**: Different source documents
   - **Outliers**: Chunks very different from others
   - **Overlap**: Multiple documents discuss same topic

**What good embeddings look like:**
- Clear clusters for different topics
- Smooth transitions between related concepts
- No extreme outliers (unless expected)
- Documents of similar type group together

**What poor embeddings look like:**
- Random scatter (no clusters)
- All chunks in one dense blob
- Extreme outliers everywhere
- May indicate: Wrong embedding model, poor chunking, or corrupted data

---

## Settings Page

Configure database connections and system settings.

### Database Connection

**What you see:**
- Host, Port, User, Database name fields
- Password field (hidden)
- "Test Connection" button

**Default values:**
```
Host: localhost
Port: 5432
User: fryt
Password: ••••••
Database: vector_db
```

**Testing connection:**
1. Click "Test Connection"
2. Success: Green message with PostgreSQL version
3. Failure: Red error message

**Common connection errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| "could not connect" | PostgreSQL not running | `docker-compose up -d` |
| "authentication failed" | Wrong password | Check `.env` file |
| "database does not exist" | DB not created | Run `python rag_low_level_m1_16gb_verbose.py` once |
| "no pg_hba.conf entry" | Firewall/config issue | Check `pg_hba.conf` |

**Changing database:**

To use a different database:
1. Update "Database" field
2. Click "Test Connection"
3. If database doesn't exist, it will be created automatically
4. Refresh the page to see indexes

---

### LLM Configuration

**What you see:**
- Information box (read-only)
- Environment variables listed

**LLM settings (CLI/ENV only):**

These must be configured via environment variables (not in GUI):

```bash
MODEL_URL=...           # URL to GGUF model
MODEL_PATH=...          # Local path to GGUF file
CTX=3072                # Context window
TEMP=0.1                # Temperature
MAX_NEW_TOKENS=256      # Max tokens to generate
N_GPU_LAYERS=16         # Metal GPU layers (Mac)
N_BATCH=128             # Batch size
```

**Why not in GUI?**
- LLM is loaded once at startup (expensive)
- Changing settings requires restart
- Use CLI for LLM experimentation

**To change LLM settings:**
1. Close Streamlit UI
2. Export environment variables
3. Restart: `streamlit run rag_web.py`

---

### Cache Management

**What you see:**
- "Clear All Caches" button

**What gets cleared:**
- Cached embedding models
- Cached LLM
- Streamlit page cache
- Session state

**When to clear caches:**
- Switching between embedding models
- After updating dependencies
- Memory usage seems high
- Seeing stale data

**Note:** Clearing caches does NOT delete indexed data (that's in PostgreSQL).

---

# Part 3: Feature Deep Dives

## Chunking Strategies

### What is Chunking?

Chunking splits documents into smaller pieces for retrieval. It's the most important parameter for RAG quality.

**Why chunk?**
- Embeddings work best on focused text (100-2000 chars)
- Retrieve precise context (not entire documents)
- Fit within LLM context windows
- Enable parallel processing

**The chunking dilemma:**

```
Small chunks (100-300 chars)
  ✅ Precise retrieval
  ✅ Fast search
  ❌ Less context
  ❌ May split important info

Large chunks (1000-2000 chars)
  ✅ More context
  ✅ Complete thoughts
  ❌ Less precise
  ❌ May include noise
```

---

### Choosing Chunk Size: Decision Matrix

| Document Type | Recommended | Chunk Size | Overlap | Reasoning |
|---------------|-------------|------------|---------|-----------|
| **Chat logs** | Ultra-fine | 100 | 20 | Each message is self-contained |
| **Twitter/Social** | Ultra-fine | 100 | 20 | Short posts, hashtags |
| **Q&A / FAQs** | Fine-grained | 300 | 60 | Question + answer fits in one chunk |
| **News articles** | Balanced | 700 | 150 | Paragraphs ~500-1000 chars |
| **Books/Blogs** | Balanced | 700 | 150 | General reading material |
| **Technical docs** | Contextual | 1200 | 240 | Code examples need context |
| **Research papers** | Contextual | 1200 | 240 | Complex concepts span paragraphs |
| **Legal documents** | Large | 2000 | 400 | Clauses must stay together |
| **Code repositories** | Fine-grained | 300 | 60 | Functions/classes as units |

---

### Overlap Strategy

**Why overlap?**

Prevents context from being cut off at chunk boundaries.

**Example without overlap:**

```
Chunk 1: "The transformer architecture revolutionized NLP. It uses"
Chunk 2: "self-attention to process sequences in parallel."
```
Query: "What does transformer use?" → ❌ Split between chunks

**Example with overlap:**

```
Chunk 1: "The transformer architecture revolutionized NLP. It uses self-attention"
Chunk 2: "It uses self-attention to process sequences in parallel."
```
Query: "What does transformer use?" → ✅ Found in both chunks

**Overlap guidelines:**

| Overlap % | Use Case | Trade-off |
|-----------|----------|-----------|
| 10-15% | Fast indexing | May miss context at boundaries |
| 15-25% | **Recommended** | Good balance |
| 25-30% | Maximum preservation | Slower indexing, more storage |
| >30% | Not recommended | Redundant, wastes resources |

---

### Chunk Size Testing Methodology

**Step 1: Index with multiple sizes**

```bash
# Via CLI (recommended for testing)
for SIZE in 300 700 1200; do
  OVERLAP=$((SIZE / 5))  # 20% overlap
  CHUNK_SIZE=$SIZE CHUNK_OVERLAP=$OVERLAP \
    PGTABLE=test_cs${SIZE} RESET_TABLE=1 \
    python rag_low_level_m1_16gb_verbose.py --index-only
done
```

**Step 2: Query all indexes**

Prepare test queries representing your use case:
```
query1: "What is the main topic?"
query2: "Who are the key people mentioned?"
query3: "What happened in June 2024?"
```

**Step 3: Compare results**

| Chunk Size | Query 1 Score | Query 2 Score | Query 3 Score | Winner |
|------------|---------------|---------------|---------------|--------|
| 300 | 0.68 | 0.82 | 0.71 | Query 2 |
| 700 | 0.79 | 0.75 | 0.80 | Query 1, 3 |
| 1200 | 0.72 | 0.69 | 0.74 | - |

**Decision:** Use 700 (best overall scores)

---

## Embedding Model Selection Guide

### Model Comparison Table

| Model | Dims | Speed | Quality | Memory | Storage/10K | Best For |
|-------|------|-------|---------|--------|-------------|----------|
| **all-MiniLM-L6-v2** | 384 | 🚀🚀🚀 80/s | ⭐⭐⭐ 82% | 90 MB | 15 MB | Prototyping, massive scale |
| **bge-small-en** ⭐ | 384 | 🚀🚀 67/s | ⭐⭐⭐⭐ 87% | 130 MB | 15 MB | General purpose |
| **bge-base-en** | 768 | 🚀 40/s | ⭐⭐⭐⭐⭐ 89% | 440 MB | 30 MB | High quality |
| **bge-large-en** | 1024 | 🐌 20/s | ⭐⭐⭐⭐⭐+ 91% | 1.3 GB | 40 MB | Research, critical apps |

**Quality** = MTEB benchmark average score
**Speed** = Chunks/second on M1 Mac 16GB
**Storage** = Disk space for 10,000 chunks

---

### Model Selection Decision Tree

```
How much data are you indexing?

├─ < 10,000 chunks → Go to (1)
├─ 10,000 - 100,000 chunks → Go to (2)
└─ > 100,000 chunks → Go to (3)

(1) Small dataset:
    ├─ Quality critical? → bge-base-en or bge-large-en
    └─ Speed important? → bge-small-en ⭐

(2) Medium dataset:
    ├─ Have 16GB+ RAM? → bge-small-en ⭐ or bge-base-en
    └─ Limited RAM (8GB)? → all-MiniLM-L6-v2

(3) Large dataset:
    ├─ Have days to index? → bge-base-en or bge-large-en
    ├─ Need it done today? → bge-small-en ⭐
    └─ Need it done in hours? → all-MiniLM-L6-v2
```

---

### Model Characteristics Deep Dive

**all-MiniLM-L6-v2:**
- ✅ Fastest (3x faster than bge-small)
- ✅ Smallest memory footprint
- ✅ Good for massive scale (1M+ chunks)
- ❌ Lower quality (~5% worse than bge-small)
- ❌ Not optimized for MLX (M1 Mac)

**bge-small-en (RECOMMENDED):**
- ✅ Best balance of speed and quality
- ✅ Excellent MLX acceleration (5-20x on M1)
- ✅ Same dimensions as MiniLM (384) = less storage
- ✅ Trained on diverse corpus
- 👍 Default choice for 90% of use cases

**bge-base-en:**
- ✅ Significantly better quality (+2% over small)
- ✅ Good for domain-specific retrieval
- ✅ Works well with MLX
- ❌ 2x slower than small
- ❌ 2x storage requirements (768d)
- 👍 Choose if quality > speed

**bge-large-en:**
- ✅ Best-in-class retrieval quality
- ✅ Excels at subtle semantic differences
- ❌ 4x slower than small
- ❌ 3x storage requirements (1024d)
- ❌ Poor MLX performance (large models)
- 👍 Only for research or critical applications

---

### Switching Models: What You Need to Know

**Important:** You cannot query an index with a different model than it was indexed with.

**Why?**
- Embeddings are model-specific
- Incompatible vector spaces
- Cosine similarity becomes meaningless

**If you want to try a different model:**

1. **Index into a new table:**
   ```bash
   # Index with bge-small
   EMBED_MODEL=BAAI/bge-small-en PGTABLE=doc_small

   # Index with bge-base
   EMBED_MODEL=BAAI/bge-base-en-v1.5 PGTABLE=doc_base
   ```

2. **Query each table separately:**
   ```bash
   # Query bge-small index
   PGTABLE=doc_small EMBED_MODEL=BAAI/bge-small-en

   # Query bge-base index
   PGTABLE=doc_base EMBED_MODEL=BAAI/bge-base-en-v1.5
   ```

3. **Compare results and keep the better one**

---

## Hybrid Search Explained

**What is hybrid search?**

Combines two retrieval methods:
1. **BM25** (keyword matching): Traditional search, like grep/Ctrl+F
2. **Vector search** (semantic): Meaning-based, like "similar concepts"

**Formula:**
```
hybrid_score = (alpha * vector_score) + ((1 - alpha) * bm25_score)
```

### HYBRID_ALPHA Values

| ALPHA | Mode | BM25 Weight | Vector Weight | Best For |
|-------|------|-------------|---------------|----------|
| 0.0 | Pure BM25 | 100% | 0% | Exact keyword matching |
| 0.3 | BM25-heavy | 70% | 30% | Names, technical terms |
| 0.5 | Balanced | 50% | 50% | General queries ⭐ |
| 0.7 | Vector-heavy | 30% | 70% | Conceptual queries |
| 1.0 | Pure vector | 0% | 100% | Semantic-only (default) |

---

### When to Use Each Mode

**Use BM25-heavy (alpha=0.3) for:**

Example queries that benefit from keywords:
- "Find messages from Elena about Morocco"
- "What did Bob say about PostgreSQL?"
- "Conversations mentioning 'attention mechanism'"
- "Code snippets containing 'async def'"

Why: Proper nouns (Elena, Bob), specific terms (PostgreSQL, async def) may not be captured well by semantic models.

---

**Use Balanced (alpha=0.5) for:**

Example queries:
- "What did we discuss about travel plans?"
- "How does the transformer model work?"
- "When did the team meet to talk about the roadmap?"

Why: Combines keyword matching ("travel", "transformer") with semantic understanding ("plans", "work", "talk about").

**Recommended as default for most use cases.**

---

**Use Vector-heavy (alpha=0.7) for:**

Example queries:
- "What is the main theme of the document?"
- "How are these two concepts related?"
- "What are the implications of this approach?"

Why: Conceptual queries where exact word matching is less important.

---

**Use Pure Vector (alpha=1.0) for:**

Example queries:
- "Summarize the key findings"
- "What are alternative approaches to this problem?"
- "Explain the relationship between X and Y"

Why: Semantic understanding is critical, keywords might mislead.

---

### Hybrid Search Examples

**Query:** "What did Elena say about traveling to Morocco?"

**Pure Vector (alpha=1.0):**
```
Retrieved:
1. "EB mentioned her trip to France was amazing" (0.72)
2. "We should plan a vacation to Spain" (0.68)
3. "Travel to Morocco sounds exciting" (0.65)
```
❌ Missed the exact match because "Elena" != "EB" in embedding space

**Balanced Hybrid (alpha=0.5):**
```
Retrieved:
1. "Elena: I loved traveling to Morocco!" (0.89)  ← BM25 boosted
2. "EB mentioned her trip to France was amazing" (0.70)
3. "Travel to Morocco sounds exciting" (0.67)
```
✅ Found the exact match by combining keyword ("Elena", "Morocco") + semantic ("traveling")

---

### Enabling Hybrid Search (CLI Only)

Hybrid search is not yet available in the GUI. Use CLI:

```bash
# Install BM25
pip install rank-bm25

# Enable hybrid search
HYBRID_ALPHA=0.5 python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

**Future GUI support:** Planned for version 2.1.

---

## Query Expansion Methods Comparison

**What is query expansion?**

Automatically generates additional queries to improve recall. Example:

Original: "What is machine learning?"
Expanded:
- "Definition of machine learning"
- "How does machine learning work"
- "Machine learning explanation"

Retrieval runs on all 3 queries, then deduplicates and reranks.

---

### Method Comparison

| Method | Speed | Quality | LLM Required | How It Works |
|--------|-------|---------|--------------|--------------|
| **keyword** | <0.1s | ⭐⭐⭐ | No | Synonym replacement, term expansion |
| **multi** | 1-3s | ⭐⭐⭐⭐ | Yes | Generates queries from different angles |
| **llm** | 1-3s | ⭐⭐⭐⭐⭐ | Yes | LLM generates natural variations |

---

### Method 1: Keyword Expansion

**How it works:**
1. Parse query for key terms
2. Add synonyms from WordNet
3. Add related technical terms
4. Generate expanded queries

**Example:**
```
Input: "ML model performance"

Expanded:
- "machine learning model performance"
- "ML model accuracy metrics"
- "model evaluation performance"
```

**Pros:**
- Very fast (<0.1s)
- No LLM required
- Works offline
- Predictable

**Cons:**
- Limited creativity
- Misses context
- Synonym quality varies

**When to use:**
- Simple queries
- Technical terms with known synonyms
- Latency-critical applications (<500ms)
- No LLM available

---

### Method 2: Multi-Query Generation

**How it works:**
1. LLM generates 2-5 related queries
2. Each query approaches topic from different angle
3. Retrieve with all queries
4. Deduplicate and rerank

**Example:**
```
Input: "What did Elena say about Morocco?"

Generated:
- "Elena's opinions on Morocco"
- "Morocco-related comments by Elena"
- "What did Elena mention regarding Moroccan travel?"
```

**Pros:**
- Multiple perspectives
- Better coverage
- Finds different relevant chunks

**Cons:**
- Slower (1-3s overhead)
- Requires LLM
- May generate redundant queries
- Overkill for simple queries

**When to use:**
- Complex queries with multiple aspects
- When you want comprehensive results
- Ambiguous queries

---

### Method 3: LLM Expansion (Recommended)

**How it works:**
1. LLM analyzes query intent
2. Generates natural variations
3. Preserves original meaning
4. Expands vocabulary

**Example:**
```
Input: "How does attention mechanism work?"

Generated:
- "Explain the attention mechanism in neural networks"
- "What is the attention mechanism and its function?"
```

**Pros:**
- Best quality
- Natural language
- Context-aware
- General-purpose

**Cons:**
- Slower (1-3s overhead)
- Requires LLM
- Costs tokens

**When to use:**
- General-purpose RAG
- Natural language queries
- When quality > speed
- Default choice for most cases ⭐

---

### Performance Impact

**Benchmark (M1 Mac 16GB):**

| Method | Overhead | Total Query Time | Recall Improvement |
|--------|----------|------------------|-------------------|
| No expansion | 0s | 5-8s | Baseline |
| Keyword | +0.1s | 5-8s | +10-15% |
| Multi | +1-3s | 6-11s | +20-30% |
| LLM | +1-3s | 6-11s | +15-25% |

**Recommendation:**
- Start with LLM method (best quality)
- Switch to keyword if latency is critical
- Use multi for complex research queries

---

## HyDE Retrieval Explained

**What is HyDE?**

HyDE (Hypothetical Document Embeddings) is an advanced retrieval technique that improves retrieval quality by 10-20%.

**The problem:** Queries and documents are written differently.

Example:
- Query: "What is attention mechanism?"
- Document: "The attention mechanism allows models to focus on relevant parts of the input by computing weighted representations..."

Query embeddings and document embeddings may not match well.

---

### How HyDE Works

**Traditional RAG:**
```
User query → Embed query → Search vectors → Retrieve chunks
```

**HyDE:**
```
User query → LLM generates hypothetical answer → Embed answer → Search vectors → Retrieve chunks
```

**Step-by-step:**

1. **User asks:** "What is attention mechanism?"

2. **LLM generates hypothetical answer:**
   ```
   "The attention mechanism is a key component in neural networks that allows
   the model to focus on different parts of the input when making predictions.
   It computes attention weights that determine how much each input token
   contributes to the output representation..."
   ```

3. **Embed the hypothetical answer** (not the query)

4. **Search for chunks similar to hypothetical answer**

5. **Why it works:** Documents are more similar to answers than to questions

---

### HyDE Configuration

**CLI only (not in GUI yet):**

```bash
# Enable HyDE with single hypothesis
ENABLE_HYDE=1 HYDE_NUM_HYPOTHESES=1

# Multiple hypotheses (better coverage)
ENABLE_HYDE=1 HYDE_NUM_HYPOTHESES=2 HYDE_FUSION_METHOD=rrf

# Longer hypotheses for complex topics
ENABLE_HYDE=1 HYDE_HYPOTHESIS_LENGTH=150
```

---

### HyDE Parameters Explained

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ENABLE_HYDE` | 0 | 0 or 1 | Enable/disable HyDE |
| `HYDE_NUM_HYPOTHESES` | 1 | 1-3 | Number of hypothetical answers to generate |
| `HYDE_HYPOTHESIS_LENGTH` | 100 | 50-200 | Target length in tokens |
| `HYDE_FUSION_METHOD` | rrf | rrf/avg/max | How to combine multiple hypotheses |

---

### Fusion Methods

When generating multiple hypotheses, how should we combine the results?

**Reciprocal Rank Fusion (RRF)** - Recommended:
```
score = sum(1 / (rank + k)) for each hypothesis
```
- Robust to outliers
- Balances all hypotheses
- Default k=60

**Average Fusion (avg):**
```
score = mean(similarity_scores)
```
- Simple and fast
- Sensitive to outliers

**Max Fusion (max):**
```
score = max(similarity_scores)
```
- Aggressive matching
- Prefers highest similarity

---

### HyDE Performance Impact

**Benchmarks (M1 Mac 16GB):**

| Configuration | Added Latency | Quality Improvement | Total Query Time |
|--------------|---------------|---------------------|------------------|
| No HyDE | 0s | Baseline | 5-8s |
| 1 hypothesis | +0.1-0.2s | +10-15% | 5-8s |
| 2 hypotheses | +0.2-0.3s | +15-20% | 5-8s |
| 3 hypotheses | +0.3-0.4s | +15-20% | 6-9s |

**Note:** HyDE latency is mostly LLM generation (one-time per query). Retrieval time is similar.

---

### When to Use HyDE

**✅ Use HyDE for:**
- Technical/domain-specific queries
- Complex multi-part questions
- When query vocabulary differs from document vocabulary
- Academic/research queries
- When quality is more important than speed

**Examples that benefit from HyDE:**
- "What is attention mechanism?" → Generates technical explanation
- "How does RAG work?" → Generates architecture description
- "What are the benefits of chunking?" → Generates comprehensive answer

**❌ Don't use HyDE for:**
- Simple factual queries ("Who wrote this?")
- Keyword searches ("Find messages from Alice")
- Time-sensitive queries (need <500ms response)
- When retrieval is already good (>0.8 scores)

**Examples where HyDE is overkill:**
- "What is the price?" (simple lookup)
- "Elena Morocco" (keyword search)
- "Define RAG" (dictionary lookup)

---

### HyDE Example Walkthrough

**Query:** "What are the key innovations in transformer models?"

**Step 1: LLM generates hypothesis**
```
"Transformer models introduced several key innovations in neural networks.
The most significant is the self-attention mechanism, which allows the model
to weigh the importance of different input tokens when making predictions.
Unlike RNNs, transformers process entire sequences in parallel using
multi-head attention. They also use positional encodings to maintain
sequence information and employ residual connections for training stability..."
```

**Step 2: Embed the hypothesis**
```
embedding = [0.023, -0.145, 0.089, ..., 0.167]  # 384 dimensions
```

**Step 3: Search for similar chunks**
```
Retrieved chunks:
1. "The transformer architecture revolutionized NLP through self-attention..." (0.89)
2. "Multi-head attention computes attention from multiple representation subspaces..." (0.85)
3. "Positional encodings provide sequence information to the model..." (0.82)
```

**Step 4: LLM generates final answer**
```
"The key innovations in transformer models include:
1. Self-attention mechanism - allows parallel processing
2. Multi-head attention - captures different aspects
3. Positional encodings - preserves sequence order
4. Residual connections - enables training stability..."
```

**Why it worked:**
- Hypothesis matched document style (technical explanation)
- Covered multiple aspects (attention, positional encodings)
- Retrieved highly relevant chunks (scores >0.8)

---

## Reranking Configuration Guide

**What is reranking?**

After initial retrieval, reranking re-scores candidates using a more powerful cross-encoder model to improve precision.

**Two-stage pipeline:**

```
Stage 1: Fast retrieval (bi-encoder)
  User query → Embed query → Vector search → Top 12 candidates

Stage 2: Precise reranking (cross-encoder)
  Query + Each candidate → Cross-encoder → Scores → Top 4 final results
```

---

### How Reranking Works

**Bi-encoder (Stage 1):**
- Encodes query and documents separately
- Compares embeddings with cosine similarity
- Fast (parallel, no interaction between query and docs)
- Less accurate (no query-document interaction)

**Cross-encoder (Stage 2):**
- Encodes query and document together
- Full attention between query and document
- Slow (must process each pair separately)
- More accurate (captures interaction)

**Example:**

Query: "What did Elena say about Morocco?"

**Stage 1 (Bi-encoder retrieval):**
```
1. "Elena mentioned loving Morocco" (0.68)
2. "EB talked about France trip" (0.64)
3. "Morocco is a beautiful country" (0.63)
4. "Travel plans to Spain" (0.61)
...
12. "Elena's favorite food" (0.52)
```

**Stage 2 (Cross-encoder reranking):**
```
1. "Elena mentioned loving Morocco" (0.94) ← Boosted!
2. "Elena's plans to visit Morocco" (0.88) ← Moved up
3. "EB talked about France trip" (0.42) ← Demoted
4. "Morocco is a beautiful country" (0.38) ← Demoted
```

Final Top 4 sent to LLM for generation.

---

### Reranking Parameters

**CLI only (not in GUI yet):**

```bash
# Enable basic reranking
ENABLE_RERANKING=1 RERANK_CANDIDATES=12 RERANK_TOP_K=4

# High-quality reranking
ENABLE_RERANKING=1 RERANK_CANDIDATES=20 RERANK_TOP_K=4

# Custom reranker model
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
```

---

### Parameter Guidelines

| Parameter | Default | Recommended | Range | Description |
|-----------|---------|-------------|-------|-------------|
| `ENABLE_RERANKING` | 0 | 1 | 0 or 1 | Enable/disable reranking |
| `RERANK_CANDIDATES` | 12 | 12-20 | 6-30 | Candidates to rerank (Stage 1 retrieval) |
| `RERANK_TOP_K` | 4 | 4 | 2-10 | Final results after reranking (sent to LLM) |
| `RERANK_MODEL` | ms-marco-MiniLM-L-6-v2 | (default) | - | Cross-encoder model |

**Choosing RERANK_CANDIDATES:**
- **Too low (6-8)**: May miss relevant chunks in Stage 1
- **Too high (30+)**: Slow reranking, diminishing returns
- **Recommended**: 12-20 (balances coverage and speed)

**Rule of thumb:**
```
RERANK_CANDIDATES = 2-3 × RERANK_TOP_K
```

---

### Available Reranker Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `ms-marco-MiniLM-L-6-v2` ⭐ | 90 MB | Fast | Good | General purpose (default) |
| `ms-marco-MiniLM-L-12-v2` | 130 MB | Medium | Better | Higher quality |
| `cross-encoder/ms-marco-electra-base` | 440 MB | Slow | Best | Research, critical apps |

**Installation:**
```bash
pip install sentence-transformers
```

Models download automatically on first use (~90-440 MB).

---

### Performance Trade-offs

**Benchmark (M1 Mac 16GB):**

| Configuration | Retrieval | Reranking | Total | Quality Improvement |
|--------------|-----------|-----------|-------|---------------------|
| No reranking | 0.3s | 0s | 0.3s | Baseline |
| RERANK_CANDIDATES=12 | 0.3s | +0.5s | 0.8s | +10-15% |
| RERANK_CANDIDATES=20 | 0.3s | +1.0s | 1.3s | +15-20% |
| RERANK_CANDIDATES=30 | 0.3s | +1.5s | 1.8s | +15-20% |

**Recommendation:**
- **Balanced**: `RERANK_CANDIDATES=12` (+0.5s, +10-15% quality)
- **High-quality**: `RERANK_CANDIDATES=20` (+1.0s, +15-20% quality)
- **Maximum**: `RERANK_CANDIDATES=30` (+1.5s, +15-20% quality)

**Diminishing returns:** Beyond 20 candidates, quality improvement plateaus.

---

### When to Use Reranking

**✅ Use reranking for:**
- Production RAG systems (where quality matters)
- When initial retrieval scores are mediocre (0.5-0.7)
- Ambiguous queries with multiple interpretations
- Domain-specific queries (technical, legal, medical)
- When you have 12+ relevant candidates

**❌ Don't use reranking for:**
- Extremely fast queries (<500ms requirement)
- When retrieval is already excellent (>0.8 scores)
- Simple keyword searches
- Very large candidate sets (>50) - too slow

---

## Semantic Cache Optimization

**What is semantic caching?**

Caches query results based on semantic similarity, not exact string matching.

**Traditional cache:**
```
Query 1: "What is RAG?"        → Execute, cache
Query 2: "What is RAG?"        → Cache hit ✅
Query 3: "What is RAG exactly?" → Cache miss ❌ (different string)
```

**Semantic cache:**
```
Query 1: "What is RAG?"        → Execute, cache
Query 2: "What is RAG?"        → Cache hit ✅
Query 3: "What is RAG exactly?" → Cache hit ✅ (98% similar)
Query 4: "Define RAG"          → Cache hit ✅ (95% similar)
Query 5: "How does chunking work?" → Cache miss (only 30% similar)
```

---

### How Semantic Cache Works

**Step-by-step:**

1. **User submits query:** "What is RAG?"

2. **Embed the query:**
   ```
   query_embedding = [0.12, -0.34, 0.56, ...]
   ```

3. **Check cache:**
   - Compare query embedding with all cached query embeddings
   - Compute cosine similarity
   - If similarity > threshold (default 0.92) → cache hit

4. **Cache hit:** Return cached answer (milliseconds)

5. **Cache miss:** Execute full RAG pipeline, cache result

---

### Cache Configuration

**CLI only (not in GUI yet):**

```bash
# Enable semantic cache (recommended)
ENABLE_SEMANTIC_CACHE=1

# Adjust similarity threshold
SEMANTIC_CACHE_THRESHOLD=0.92  # Strict (only very similar queries)

# Aggressive caching
SEMANTIC_CACHE_THRESHOLD=0.85  # More cache hits

# Increase cache size
SEMANTIC_CACHE_MAX_SIZE=2000  # Store up to 2000 queries

# Time-based expiry
SEMANTIC_CACHE_TTL=86400  # 24 hours (0 = infinite)
```

---

### Threshold Selection

| Threshold | Hit Rate | False Positives | Behavior |
|-----------|----------|-----------------|----------|
| 0.85 | ~60-70% | Possible | Aggressive - returns similar queries |
| 0.90 | ~40-50% | Very low | Balanced - mostly identical intent ⭐ |
| 0.92 | ~20-30% | None | Strict - only near-identical queries |
| 0.95 | ~10-15% | None | Ultra-strict - only paraphrases |

**Choosing threshold:**

**Use 0.85-0.87 (aggressive) when:**
- Queries are often similar (customer support, FAQs)
- Speed is critical (need <100ms responses)
- Slight inaccuracies are acceptable

**Use 0.90-0.92 (balanced) when:** ⭐
- General-purpose RAG
- Want good hit rate without false positives
- Default choice for most applications

**Use 0.95+ (strict) when:**
- Every query must be unique
- Cache only exact paraphrases
- Research or critical applications

---

### Cache Performance

**Latency comparison:**

| Operation | Time | Speedup |
|-----------|------|---------|
| Cache lookup | <10ms | - |
| Cache hit | 10-20ms | **200-500x faster** |
| Full RAG (no vLLM) | 5-15s | Baseline |
| Full RAG (vLLM) | 2-3s | 2-5x faster than baseline |

**Cache hit scenarios:**

| Scenario | Original Query | New Query | Similarity | Hit? |
|----------|----------------|-----------|------------|------|
| Exact match | "What is RAG?" | "What is RAG?" | 1.00 | ✅ |
| Paraphrase | "What is RAG?" | "Define RAG" | 0.94 | ✅ |
| Added words | "What is RAG?" | "What exactly is RAG?" | 0.96 | ✅ |
| Typo | "What is RAG?" | "What is RAG?" (with space) | 0.98 | ✅ |
| Different intent | "What is RAG?" | "How to implement RAG?" | 0.72 | ❌ |
| Different topic | "What is RAG?" | "What is chunking?" | 0.35 | ❌ |

---

### Cache Strategies

**Persistent cache (recommended):**
```bash
ENABLE_SEMANTIC_CACHE=1
SEMANTIC_CACHE_THRESHOLD=0.92
SEMANTIC_CACHE_TTL=0  # Never expire
SEMANTIC_CACHE_MAX_SIZE=1000
```
- Cache survives restarts
- Good for production
- Periodically clear stale entries manually

**Session cache:**
```bash
ENABLE_SEMANTIC_CACHE=1
SEMANTIC_CACHE_THRESHOLD=0.90
SEMANTIC_CACHE_TTL=3600  # 1 hour
SEMANTIC_CACHE_MAX_SIZE=500
```
- Auto-expires old entries
- Good for short sessions
- Lower memory usage

**Aggressive cache (high-traffic):**
```bash
ENABLE_SEMANTIC_CACHE=1
SEMANTIC_CACHE_THRESHOLD=0.85  # More hits
SEMANTIC_CACHE_MAX_SIZE=2000  # Larger cache
SEMANTIC_CACHE_TTL=86400  # 24 hours
```
- Maximizes hit rate
- Good for customer support, FAQs
- Accepts some false positives

---

### Monitoring Cache Performance

**View cache stats (in logs):**

```
=== Semantic Cache Statistics ===
Total queries: 150
Cache hits: 63 (42%)
Cache misses: 87 (58%)
Average hit latency: 12ms
Average miss latency: 7.2s
Total time saved: 6.3 minutes
```

**Interpreting hit rate:**

| Hit Rate | Interpretation | Action |
|----------|----------------|--------|
| < 20% | Low - queries too diverse | Lower threshold or increase cache size |
| 20-40% | Normal for diverse queries | Current settings OK |
| 40-60% | Good - many similar queries | Cache is working well ⭐ |
| > 60% | Excellent - repetitive queries | Consider higher threshold to save memory |

---

## Conversation Memory Best Practices

**What is conversation memory?**

Enables multi-turn conversations by tracking context across queries. Essential for chat-style interfaces.

**Without conversation memory:**
```
User: "What is RAG?"
Bot: "RAG stands for Retrieval-Augmented Generation..."

User: "How does it work?"
Bot: "I'm not sure what 'it' refers to. Can you clarify?"
```

**With conversation memory:**
```
User: "What is RAG?"
Bot: "RAG stands for Retrieval-Augmented Generation..."

User: "How does it work?"
Bot: (resolves "it" → "RAG") "RAG works by first retrieving relevant documents..."
```

---

### Key Features

**1. Reference Resolution**

Resolves pronouns and implicit references:

```
User: "What is chunking?"
Bot: "Chunking splits documents into smaller pieces..."

User: "Why is it important?"
Bot: (resolves "it" → "chunking") "Chunking is important because..."

User: "What about overlap?"
Bot: (infers context = "chunking overlap") "Chunk overlap preserves context at boundaries..."
```

**2. Query Reformulation**

Adds conversation context to queries:

```
Original query: "What about the performance?"
Reformulated: "Given previous discussion about chunking, what about chunking performance?"
```

**3. Entity Tracking**

Tracks mentioned entities across turns:

```
Turn 1: User mentions "LlamaIndex", "pgvector"
Turn 2: User asks "How do they work together?"
System knows: they = LlamaIndex + pgvector
```

**4. Auto-Summarization**

Compresses old turns to save memory:

```
After 5 turns:
  Keep: Last 3 turns (full context)
  Summarize: Turns 1-2 ("Discussed RAG definition and chunking")
```

---

### Configuration

**CLI only (not in GUI yet):**

```bash
# Enable conversation memory (recommended)
ENABLE_CONVERSATION_MEMORY=1
MAX_CONVERSATION_TURNS=10

# Aggressive summarization (long conversations)
AUTO_SUMMARIZE=1
SUMMARIZE_THRESHOLD=5  # Summarize after 5 turns

# Session timeout (30 minutes)
CONVERSATION_TIMEOUT=1800
```

---

### Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `ENABLE_CONVERSATION_MEMORY` | 1 | 1 | Enable/disable conversation tracking |
| `MAX_CONVERSATION_TURNS` | 10 | 10-15 | Maximum turns to keep in memory |
| `CONVERSATION_TIMEOUT` | 3600 | 1800-3600 | Session timeout (seconds) |
| `AUTO_SUMMARIZE` | 1 | 1 | Enable auto-summarization |
| `SUMMARIZE_THRESHOLD` | 5 | 3-7 | Turns before summarization |
| `CONVERSATION_CACHE_DIR` | `.cache/conversations` | (custom) | Storage directory |

---

### Usage Patterns

**Pattern 1: Interactive CLI**

```bash
# Enable conversation memory
ENABLE_CONVERSATION_MEMORY=1 python rag_low_level_m1_16gb_verbose.py --interactive

# Session automatically managed
> What is RAG?
[Answer]

> How does it work?  ← "it" resolves to "RAG"
[Answer]

> What about chunking?
[Answer]

> Why is that important?  ← "that" resolves to "chunking"
[Answer]
```

**Pattern 2: Multi-user Web Application**

```python
from utils.conversation_memory import session_manager

# Each user gets their own session
user_id = "user_123"
memory = session_manager.get_or_create(user_id)

# Resolve references
resolved_query = memory.resolve_references(user_query)

# Execute RAG
answer = run_rag_query(resolved_query)

# Store turn
memory.add_turn(user_query, answer)
```

---

### Best Practices

**1. Set appropriate timeout:**

```bash
# Short sessions (chat support)
CONVERSATION_TIMEOUT=1800  # 30 minutes

# Long sessions (research)
CONVERSATION_TIMEOUT=7200  # 2 hours

# Persistent (dev/debug)
CONVERSATION_TIMEOUT=86400  # 24 hours
```

**2. Enable summarization for long conversations:**

```bash
# Summarize after 3 turns (aggressive)
SUMMARIZE_THRESHOLD=3

# Summarize after 7 turns (conservative)
SUMMARIZE_THRESHOLD=7
```

**3. Clear old sessions periodically:**

```python
# Cleanup sessions older than 7 days
from utils.conversation_memory import session_manager
session_manager.cleanup_old_sessions(max_age_days=7)
```

**4. Monitor memory usage:**

```python
# Get session stats
stats = memory.get_stats()
print(f"Turns: {stats['turn_count']}")
print(f"Entities: {stats['entity_count']}")
print(f"Memory: {stats['memory_bytes'] / 1024:.1f} KB")
```

---

### Limitations and Workarounds

**Limitation 1: Context drift**

After 10+ turns, conversation may lose focus.

**Workaround:**
```bash
# Reduce max turns
MAX_CONVERSATION_TURNS=5

# Enable aggressive summarization
SUMMARIZE_THRESHOLD=3
```

**Limitation 2: Cross-session references**

Cannot reference previous sessions.

**Workaround:**
- Use longer timeouts
- Or provide explicit context: "In our last conversation about X..."

**Limitation 3: Ambiguous pronouns**

"It" may refer to multiple entities.

**Workaround:**
- Be explicit: "How does chunking work?" vs "How does it work?"
- System will ask for clarification if ambiguous

---

# Part 4: Troubleshooting

## Getting 0 Results: Step-by-Step Debug

### Symptom

Query returns 0 chunks or all scores < 0.3.

---

### Step 1: Verify Index Exists

**Check:**
```bash
# List all tables
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "\dt"

# Count rows in your table
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "SELECT COUNT(*) FROM your_table_name;"
```

**Expected:** Table exists and has > 0 rows

**If not:** Go to "Index Documents" page and index your document

---

### Step 2: Check Embedding Model Compatibility

**Issue:** Querying with different model than indexed with

**Check chunk_size metadata:**
```bash
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "
  SELECT metadata_->>'_chunk_size' as chunk_size,
         metadata_->>'_embed_model' as embed_model
  FROM your_table_name
  LIMIT 1;
"
```

**Expected:** Model matches your current setting

**If not:** Re-index with correct model or change query model

---

### Step 3: Inspect Retrieved Chunks

**Enable full logging (CLI):**
```bash
LOG_FULL_CHUNKS=1 COLORIZE_CHUNKS=1 python rag_low_level_m1_16gb_verbose.py --query-only --query "your query"
```

**Look for:**
- Are any chunks retrieved?
- What are the similarity scores?
- Does chunk content match your query?

---

### Step 4: Adjust Retrieval Parameters

**If scores are low (0.2-0.4):**

**Problem:** Semantic mismatch

**Solutions:**
1. Try hybrid search:
   ```bash
   HYBRID_ALPHA=0.5
   ```

2. Try different query phrasing:
   ```
   Original: "ML performance"
   Try: "What is the performance of machine learning models?"
   ```

3. Enable query expansion:
   ```bash
   ENABLE_QUERY_EXPANSION=1 QUERY_EXPANSION_METHOD=llm
   ```

---

**If no chunks retrieved:**

**Problem:** Wrong table or no matching data

**Solutions:**
1. Verify table name:
   ```bash
   # List tables
   PGTABLE=wrong_name  ← Check this
   ```

2. Check document content:
   ```bash
   # View first chunk
   PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "
     SELECT text FROM your_table_name LIMIT 1;
   "
   ```

3. Verify query matches document topic

---

### Step 5: Re-index with Different Chunk Size

**If chunks don't contain relevant info:**

**Problem:** Chunk size mismatch

**Solution:** Re-index with appropriate chunk size

```bash
# For chat logs (short messages)
CHUNK_SIZE=100 CHUNK_OVERLAP=20 RESET_TABLE=1

# For documents (longer text)
CHUNK_SIZE=700 CHUNK_OVERLAP=150 RESET_TABLE=1
```

---

### Step 6: Check for Empty/Corrupted Data

**Symptoms:**
- Chunks retrieved but text is empty
- Scores are 0.0
- Embeddings are null

**Check:**
```bash
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "
  SELECT
    LENGTH(text) as text_length,
    embedding IS NOT NULL as has_embedding
  FROM your_table_name
  LIMIT 10;
"
```

**Expected:** text_length > 0, has_embedding = true

**If not:** Data corruption - re-index from source

---

## Performance Issues

### Slow Indexing

**Symptom:** Indexing takes hours for 10K-50K chunks

---

**Cause 1: CPU-only embedding**

**Solution:** Enable MLX on Apple Silicon

```bash
pip install mlx mlx-embedding-models
EMBED_BACKEND=mlx EMBED_MODEL=BAAI/bge-small-en EMBED_BATCH=64
```

**Expected speedup:** 5-20x faster

---

**Cause 2: Small batch size**

**Solution:** Increase embedding batch size

```bash
# Default
EMBED_BATCH=32

# Faster (if you have 16GB+ RAM)
EMBED_BATCH=64  # or 128 for MLX
```

---

**Cause 3: Large embedding model**

**Solution:** Switch to smaller model

```bash
# From
EMBED_MODEL=BAAI/bge-large-en-v1.5  # Slow

# To
EMBED_MODEL=BAAI/bge-small-en  # 4x faster
```

---

### Slow Queries

**Symptom:** Queries take 15-30+ seconds

---

**Cause 1: No vLLM server**

**Solution:** Use vLLM for 3-4x speedup

```bash
# Terminal 1: Start vLLM
./scripts/start_vllm_server.sh

# Terminal 2: Query with vLLM
USE_VLLM=1 python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

**Expected:** 2-3s per query vs 8-15s

---

**Cause 2: Too many GPU layers offloaded**

**Solution:** Reduce N_GPU_LAYERS if swapping

```bash
# From
N_GPU_LAYERS=32  # May swap on 16GB Mac

# To
N_GPU_LAYERS=16  # More stable
```

---

**Cause 3: Large context window overflow**

**Solution:** Reduce chunk size or TOP_K

```bash
# Reduce chunks sent to LLM
TOP_K=3  # From 6+

# Or reduce chunk size
CHUNK_SIZE=500  # From 1000+

# Or increase context window
CTX=8192  # From 3072
```

---

## Memory Problems

### Out of Memory During Indexing

**Symptom:** Process killed or "MemoryError"

---

**Cause 1: Large embedding model**

**Solution:** Use smaller model

```bash
# From
EMBED_MODEL=BAAI/bge-large-en-v1.5  # 1.3 GB

# To
EMBED_MODEL=BAAI/bge-small-en  # 130 MB
```

---

**Cause 2: Large batch size**

**Solution:** Reduce embedding batch size

```bash
# From
EMBED_BATCH=128

# To
EMBED_BATCH=32  # or 16
```

---

**Cause 3: Embedding all chunks at once**

**Solution:** Already batched - check system memory

```bash
# Monitor memory
htop  # or Activity Monitor on Mac

# If using 90%+ RAM, reduce batch size
EMBED_BATCH=16
```

---

### Out of Memory During Query

**Symptom:** Crash or swap during LLM inference

---

**Cause 1: Too many GPU layers**

**Solution:** Reduce layers offloaded to Metal

```bash
# From
N_GPU_LAYERS=32

# To
N_GPU_LAYERS=8  # For 8GB RAM
N_GPU_LAYERS=16  # For 16GB RAM
```

---

**Cause 2: Large context window**

**Solution:** Reduce CTX

```bash
# From
CTX=8192

# To
CTX=3072  # Default
CTX=2048  # For low memory
```

---

**Cause 3: Large batch size**

**Solution:** Reduce N_BATCH

```bash
# From
N_BATCH=512

# To
N_BATCH=128  # Default
N_BATCH=64   # For low memory
```

---

## Database Connection Issues

### "Could not connect to server"

**Cause:** PostgreSQL not running

**Solution:**

```bash
# Start PostgreSQL
docker-compose -f config/docker-compose.yml up -d

# Wait 10 seconds
sleep 10

# Verify running
docker-compose -f config/docker-compose.yml ps
```

**Expected:** Status = "Up"

---

### "Authentication failed"

**Cause:** Wrong password

**Solution:**

1. Check `.env` file:
   ```bash
   cat .env | grep PGPASSWORD
   ```

2. Update if wrong:
   ```bash
   PGPASSWORD=correct_password
   ```

3. Restart Streamlit:
   ```bash
   streamlit run rag_web.py
   ```

---

### "Database does not exist"

**Cause:** Database not created

**Solution:**

```bash
# Create database (automatic)
python rag_low_level_m1_16gb_verbose.py --query-only --query "test"

# Or manually
PGPASSWORD=frytos psql -h localhost -U fryt -c "CREATE DATABASE vector_db;"
```

---

### "pgvector extension not found"

**Cause:** pgvector not installed

**Solution:**

```bash
# Enable pgvector extension
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

**Expected:** One row showing vector extension

---

## Empty Tables Appearing

### Symptom

"View Indexes" shows tables with 0 chunks or "?" for chunk_size.

---

**Cause 1: Table created but indexing failed**

**Solution:** Delete empty table and re-index

```bash
# In GUI: View Indexes → Select table → Delete

# Or CLI
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "DROP TABLE empty_table_name;"

# Re-index
RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
```

---

**Cause 2: Metadata not stored**

**Solution:** Re-index with current version (automatic)

---

**Cause 3: Mixed configurations**

**Check:**
```bash
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "
  SELECT DISTINCT
    metadata_->>'_chunk_size' as chunk_size,
    metadata_->>'_chunk_overlap' as chunk_overlap
  FROM your_table_name;
"
```

**If multiple values:** Table has mixed configurations (bad)

**Solution:** Re-index with RESET_TABLE=1

```bash
RESET_TABLE=1 PGTABLE=your_table_name python rag_low_level_m1_16gb_verbose.py
```

---

# Part 5: Advanced Workflows

## Batch Indexing Multiple Documents

### Workflow 1: Multiple Files into Separate Indexes

**Use case:** Index many documents, query each separately

**CLI approach (recommended):**

```bash
#!/bin/bash
# batch_index.sh

for file in data/*.pdf; do
  basename=$(basename "$file" .pdf)
  echo "Indexing $basename..."

  PDF_PATH="$file" \
  PGTABLE="${basename}_700_150" \
  RESET_TABLE=1 \
  CHUNK_SIZE=700 \
  CHUNK_OVERLAP=150 \
  EMBED_BACKEND=mlx \
  python rag_low_level_m1_16gb_verbose.py --index-only

  echo "Completed $basename"
done

echo "All documents indexed!"
```

**Run:**
```bash
chmod +x batch_index.sh
./batch_index.sh
```

---

### Workflow 2: Folder into Single Index

**Use case:** Index entire folder, query all documents together

**GUI approach:**

1. Go to "Index Documents"
2. Select folder (e.g., `📁 data/reports/ (47 files)`)
3. Choose preset
4. Name table: `reports_all_700_150`
5. Enable "Reset table if exists"
6. Click "Start Indexing"

**Result:** All 47 files in one searchable index

---

### Workflow 3: Incremental Indexing (Append Mode)

**Use case:** Add new documents to existing index without re-indexing old ones

**Important:** Only append if using same chunk size and model!

**CLI approach:**

```bash
# Index initial documents
PDF_PATH=data/batch1 PGTABLE=my_docs RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only

# Later: Append new documents
PDF_PATH=data/batch2 PGTABLE=my_docs RESET_TABLE=0 \
  python rag_low_level_m1_16gb_verbose.py --index-only

# All documents now in 'my_docs' table
```

**Warning:** Mixing chunk sizes causes poor retrieval. Always verify:

```bash
# Check for mixed configs
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "
  SELECT
    metadata_->>'_chunk_size' as cs,
    COUNT(*) as count
  FROM my_docs
  GROUP BY metadata_->>'_chunk_size';
"
```

**Expected:** One row (uniform config)

---

## A/B Testing Different Configurations

### Scenario

Test which configuration gives better retrieval for your use case.

---

### Step 1: Define Test Configurations

```bash
# Config A: Fine-grained
CHUNK_SIZE=300 CHUNK_OVERLAP=60 EMBED_MODEL=BAAI/bge-small-en

# Config B: Balanced
CHUNK_SIZE=700 CHUNK_OVERLAP=150 EMBED_MODEL=BAAI/bge-small-en

# Config C: Contextual
CHUNK_SIZE=1200 CHUNK_OVERLAP=240 EMBED_MODEL=BAAI/bge-base-en-v1.5
```

---

### Step 2: Index Each Configuration

```bash
# Config A
CHUNK_SIZE=300 CHUNK_OVERLAP=60 PGTABLE=test_config_a RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only

# Config B
CHUNK_SIZE=700 CHUNK_OVERLAP=150 PGTABLE=test_config_b RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only

# Config C
CHUNK_SIZE=1200 CHUNK_OVERLAP=240 EMBED_MODEL=BAAI/bge-base-en-v1.5 \
  PGTABLE=test_config_c RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only
```

---

### Step 3: Prepare Test Queries

Create `test_queries.txt`:
```
What is the main topic of the document?
Who are the key people mentioned?
What happened in June 2024?
Summarize the conclusions.
What are the three main benefits?
```

---

### Step 4: Query Each Configuration

```bash
#!/bin/bash
# ab_test.sh

QUERIES="test_queries.txt"
CONFIGS=("test_config_a" "test_config_b" "test_config_c")

for config in "${CONFIGS[@]}"; do
  echo "Testing $config..."

  while IFS= read -r query; do
    echo "Query: $query"

    PGTABLE="$config" LOG_QUERIES=1 \
      python rag_low_level_m1_16gb_verbose.py \
      --query-only --query "$query" 2>&1 | grep "Score:"

    echo "---"
  done < "$QUERIES"

  echo "Completed $config"
  echo "===================================="
done
```

---

### Step 5: Analyze Results

**Collect scores from logs:**

| Query | Config A Score | Config B Score | Config C Score | Winner |
|-------|---------------|---------------|---------------|--------|
| Query 1 | 0.68 | 0.79 | 0.72 | B |
| Query 2 | 0.82 | 0.75 | 0.81 | A |
| Query 3 | 0.71 | 0.80 | 0.74 | B |
| Query 4 | 0.64 | 0.77 | 0.79 | C |
| Query 5 | 0.79 | 0.83 | 0.76 | B |
| **Average** | **0.73** | **0.79** | **0.76** | **B wins** |

**Decision:** Use Config B (chunk_size=700, overlap=150)

---

## Optimizing for Speed vs Quality

### Speed-Optimized Configuration

**Goal:** Fastest indexing and querying, reasonable quality

```bash
# Indexing
EMBED_BACKEND=mlx                           # 5-20x faster
EMBED_MODEL=BAAI/bge-small-en              # Fast model
EMBED_DIM=384                               # Small vectors
EMBED_BATCH=128                             # Large batches
CHUNK_SIZE=700                              # Balanced
CHUNK_OVERLAP=150                           # Standard
ENABLE_SEMANTIC_CACHE=1                     # Cache queries
SEMANTIC_CACHE_THRESHOLD=0.90               # Aggressive caching

# LLM
USE_VLLM=1                                  # vLLM server (3-4x faster)
N_GPU_LAYERS=24                             # Max GPU (if 16GB RAM)
N_BATCH=256                                 # Large batches
CTX=3072                                    # Standard context
TOP_K=3                                     # Fewer chunks

# Query
ENABLE_RERANKING=0                          # Skip reranking
ENABLE_QUERY_EXPANSION=0                    # Skip expansion
ENABLE_HYDE=0                               # Skip HyDE
```

**Expected performance (M1 Mac 16GB):**
- Indexing: ~8-10 min for 50K chunks
- Query: ~2-3s with vLLM

---

### Quality-Optimized Configuration

**Goal:** Best possible retrieval and generation quality

```bash
# Indexing
EMBED_BACKEND=mlx                           # Fast embedding
EMBED_MODEL=BAAI/bge-large-en-v1.5         # Best model
EMBED_DIM=1024                              # Large vectors
EMBED_BATCH=32                              # Conservative
CHUNK_SIZE=700                              # Balanced
CHUNK_OVERLAP=200                           # High overlap
EXTRACT_ENHANCED_METADATA=1                 # Rich metadata

# Retrieval
TOP_K=8                                     # More candidates
HYBRID_ALPHA=0.5                            # Hybrid search
ENABLE_RERANKING=1                          # Rerank results
RERANK_CANDIDATES=20                        # Many candidates
RERANK_TOP_K=6                              # Final top-k
ENABLE_QUERY_EXPANSION=1                    # Expand queries
QUERY_EXPANSION_METHOD=llm                  # Best method
ENABLE_HYDE=1                               # HyDE retrieval
HYDE_NUM_HYPOTHESES=2                       # Multiple hypotheses

# LLM
USE_VLLM=1                                  # Still use vLLM
N_GPU_LAYERS=16                             # Stable
N_BATCH=128                                 # Standard
CTX=8192                                    # Large context
MAX_NEW_TOKENS=512                          # Longer answers
TEMP=0.05                                   # Very factual
```

**Expected performance (M1 Mac 16GB):**
- Indexing: ~40-45 min for 50K chunks
- Query: ~8-12s per query

---

### Balanced Configuration (Recommended)

**Goal:** Good speed and quality for production

```bash
# Indexing
EMBED_BACKEND=mlx
EMBED_MODEL=BAAI/bge-small-en
EMBED_DIM=384
EMBED_BATCH=64
CHUNK_SIZE=700
CHUNK_OVERLAP=150
EXTRACT_CHAT_METADATA=1

# Retrieval
TOP_K=4
HYBRID_ALPHA=0.5
ENABLE_RERANKING=1
RERANK_CANDIDATES=12
RERANK_TOP_K=4
ENABLE_SEMANTIC_CACHE=1
SEMANTIC_CACHE_THRESHOLD=0.92

# LLM
USE_VLLM=1
N_GPU_LAYERS=16
N_BATCH=128
CTX=3072
MAX_NEW_TOKENS=256
TEMP=0.1
```

**Expected performance (M1 Mac 16GB):**
- Indexing: ~8-12 min for 50K chunks
- Query: ~3-5s per query

---

## Multi-Language Support

### Supported Languages

The RAG pipeline supports multilingual documents through multilingual embedding models.

**English-optimized models (default):**
- `BAAI/bge-small-en` - English only
- `BAAI/bge-base-en-v1.5` - English only
- `BAAI/bge-large-en-v1.5` - English only

**Multilingual models:**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - 50+ languages
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - 50+ languages (better quality)

---

### Configuration for Multilingual

```bash
# Use multilingual model
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBED_DIM=384

# Index documents (any language)
PDF_PATH=data/documents_french.pdf PGTABLE=docs_fr RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only

# Query in any language
PGTABLE=docs_fr python rag_low_level_m1_16gb_verbose.py \
  --query-only --query "Quel est le sujet principal?"
```

---

### Limitations

**LLM generation:**
- Mistral 7B Instruct works best in English
- May struggle with non-English queries
- Consider using multilingual LLM (e.g., multilingual-mistral)

**Chunking:**
- SentenceSplitter uses English sentence boundaries
- May not work well for languages without spaces (Chinese, Japanese)
- Consider character-based chunking for Asian languages

---

## Large Dataset Handling (100K+ Chunks)

### Challenges

- Long indexing times (hours)
- Large storage requirements (GBs)
- Slower retrieval (more vectors to search)
- Memory constraints

---

### Optimization Strategies

**1. Use HNSW index for fast search**

```bash
# After indexing, create HNSW index
./scripts/apply_hnsw.sh your_table_name
```

**Speedup:** 3-10x faster retrieval

---

**2. Batch indexing with checkpoints**

```bash
#!/bin/bash
# Index in batches of 10K docs

BATCH_SIZE=10000
TOTAL=100000

for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
  START=$i
  END=$((i + BATCH_SIZE))

  echo "Indexing batch $START-$END..."

  # Index batch (append mode after first)
  RESET_TABLE=$([[ $i -eq 0 ]] && echo 1 || echo 0) \
  PDF_PATH=data/batch_${i} \
  PGTABLE=large_index \
  python rag_low_level_m1_16gb_verbose.py --index-only

  echo "Completed batch $START-$END"
done
```

---

**3. Use smallest embedding model**

```bash
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Fastest
EMBED_DIM=384                                        # Smallest storage
EMBED_BATCH=128                                      # Large batches
```

---

**4. Increase database batch size**

```bash
DB_INSERT_BATCH=500  # From default 250
```

---

**5. Monitor and adjust**

```bash
# Monitor PostgreSQL performance
docker stats

# If database is bottleneck, increase shared buffers
# Edit docker-compose.yml:
command: >
  postgres
  -c shared_buffers=2GB
  -c work_mem=50MB
  -c maintenance_work_mem=512MB
```

---

### Expected Performance (100K Chunks)

**Indexing (M1 Mac 16GB):**
- With HuggingFace: ~2-3 hours
- With MLX: ~20-30 minutes

**Storage:**
- 384d embeddings: ~150 MB per 10K chunks
- 100K chunks: ~1.5 GB

**Query:**
- Without HNSW: ~500-1000ms
- With HNSW: ~50-150ms

---

# Part 6: Quick Reference

## Parameter Quick Reference Table

| Parameter | Default | Range | Page | Description |
|-----------|---------|-------|------|-------------|
| **Chunking** |
| `CHUNK_SIZE` | 700 | 100-2000 | Index | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | 0-500 | Index | Overlap between chunks |
| **Embedding** |
| `EMBED_MODEL` | bge-small-en | - | Index | HuggingFace model name |
| `EMBED_DIM` | 384 | 384/768/1024 | Index | Vector dimensions |
| `EMBED_BATCH` | 32 | 8-256 | Index | Embedding batch size |
| `EMBED_BACKEND` | huggingface | hf/mlx | CLI | Embedding backend |
| **Retrieval** |
| `TOP_K` | 4 | 1-10 | Query | Chunks to retrieve |
| `HYBRID_ALPHA` | 1.0 | 0.0-1.0 | CLI | Hybrid search weight |
| `ENABLE_FILTERS` | 1 | 0/1 | CLI | Metadata filtering |
| `MMR_THRESHOLD` | 0.0 | 0.0-1.0 | CLI | Diversity threshold |
| **LLM** |
| `N_GPU_LAYERS` | 16 | 0-32 | CLI | Metal GPU layers |
| `N_BATCH` | 128 | 64-512 | CLI | LLM batch size |
| `CTX` | 3072 | 2048-8192 | CLI | Context window |
| `TEMP` | 0.1 | 0.0-1.0 | CLI | Temperature |
| `MAX_NEW_TOKENS` | 256 | 64-1024 | CLI | Max generation |
| **Advanced** |
| `ENABLE_RERANKING` | 0 | 0/1 | CLI | Query reranking |
| `RERANK_CANDIDATES` | 12 | 6-30 | CLI | Candidates to rerank |
| `ENABLE_SEMANTIC_CACHE` | 1 | 0/1 | CLI | Query caching |
| `SEMANTIC_CACHE_THRESHOLD` | 0.92 | 0.8-0.99 | CLI | Cache similarity |
| `ENABLE_QUERY_EXPANSION` | 0 | 0/1 | CLI | Query expansion |
| `ENABLE_HYDE` | 0 | 0/1 | CLI | HyDE retrieval |
| `ENABLE_CONVERSATION_MEMORY` | 1 | 0/1 | CLI | Conversation tracking |

---

## Preset Comparison Matrix

| Preset | Chunk Size | Overlap | Overlap % | Use Case | Indexing Speed | Query Speed | Quality |
|--------|------------|---------|-----------|----------|----------------|-------------|---------|
| **Ultra-fine** | 100 | 20 | 20% | Chat logs, tweets | 🚀🚀🚀 | 🚀🚀🚀 | ⭐⭐⭐ |
| **Fine-grained** | 300 | 60 | 20% | Q&A, facts | 🚀🚀 | 🚀🚀 | ⭐⭐⭐⭐ |
| **Balanced** ⭐ | 700 | 150 | 21% | General docs | 🚀 | 🚀 | ⭐⭐⭐⭐ |
| **Contextual** | 1200 | 240 | 20% | Technical docs | 🚀 | 🐌 | ⭐⭐⭐⭐⭐ |
| **Large** | 2000 | 400 | 20% | Academic papers | 🐌 | 🐌 | ⭐⭐⭐⭐⭐ |

---

## Embedding Model Comparison

| Model | Dims | Speed | Quality | Memory | Best For |
|-------|------|-------|---------|--------|----------|
| **all-MiniLM-L6-v2** | 384 | 🚀🚀🚀 | ⭐⭐⭐ | 90 MB | Prototyping, scale |
| **bge-small-en** ⭐ | 384 | 🚀🚀 | ⭐⭐⭐⭐ | 130 MB | General use |
| **bge-base-en** | 768 | 🚀 | ⭐⭐⭐⭐⭐ | 440 MB | High quality |
| **bge-large-en** | 1024 | 🐌 | ⭐⭐⭐⭐⭐+ | 1.3 GB | Research |
| **multilingual-MiniLM** | 384 | 🚀🚀 | ⭐⭐⭐ | 470 MB | 50+ languages |

---

## Keyboard Shortcuts

**Streamlit UI:**
- `R` - Rerun app
- `C` - Clear cache
- `Q` - Stop running script
- `?` - Show shortcuts

**Terminal (Interactive CLI):**
- `Ctrl+C` - Interrupt query
- `Ctrl+D` or `exit` - Exit interactive mode
- `Up/Down arrows` - Command history
- `Tab` - (No autocomplete yet)

---

## Common Commands

### Indexing

```bash
# Quick index (GUI equivalent)
streamlit run rag_web.py
# → Index Documents → Select file → Balanced preset → Start Indexing

# CLI: Fast MLX indexing
EMBED_BACKEND=mlx CHUNK_SIZE=700 CHUNK_OVERLAP=150 \
  PDF_PATH=data/doc.pdf PGTABLE=doc_700_150 RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only

# CLI: Quality indexing
EMBED_MODEL=BAAI/bge-large-en-v1.5 EMBED_DIM=1024 \
  CHUNK_SIZE=700 CHUNK_OVERLAP=200 \
  PDF_PATH=data/doc.pdf PGTABLE=doc_quality RESET_TABLE=1 \
  python rag_low_level_m1_16gb_verbose.py --index-only
```

---

### Querying

```bash
# GUI: Simple query
streamlit run rag_web.py
# → Query → Select index → Type question → Search

# CLI: Interactive mode
PGTABLE=doc_700_150 python rag_low_level_m1_16gb_verbose.py --interactive

# CLI: Single query
PGTABLE=doc_700_150 python rag_low_level_m1_16gb_verbose.py \
  --query-only --query "What is the main topic?"

# CLI: Hybrid search query
PGTABLE=doc_700_150 HYBRID_ALPHA=0.5 TOP_K=6 \
  python rag_low_level_m1_16gb_verbose.py --interactive
```

---

### Database Management

```bash
# List all tables
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "\dt"

# Count rows in table
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "SELECT COUNT(*) FROM table_name;"

# Delete table
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "DROP TABLE table_name;"

# View table metadata
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "SELECT metadata_->>'_chunk_size', metadata_->>'_embed_model'
   FROM table_name LIMIT 1;"
```

---

### Debugging

```bash
# Enable full logging
LOG_LEVEL=DEBUG LOG_FULL_CHUNKS=1 COLORIZE_CHUNKS=1 \
  python rag_low_level_m1_16gb_verbose.py --query-only --query "test"

# Save queries to log
LOG_QUERIES=1 python rag_low_level_m1_16gb_verbose.py --interactive
# Logs saved to: query_logs/

# Check embedding model
python -c "from rag_low_level_m1_16gb_verbose import build_embed_model; \
  m = build_embed_model(); print('Model loaded successfully')"

# Test database connection
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "SELECT version();"
```

---

### Performance

```bash
# Fast indexing with MLX
EMBED_BACKEND=mlx EMBED_BATCH=128 CHUNK_SIZE=700 \
  python rag_low_level_m1_16gb_verbose.py --index-only

# Fast query with vLLM
./scripts/start_vllm_server.sh  # Terminal 1
USE_VLLM=1 python rag_low_level_m1_16gb_verbose.py --interactive  # Terminal 2

# Enable semantic cache
ENABLE_SEMANTIC_CACHE=1 SEMANTIC_CACHE_THRESHOLD=0.92 \
  python rag_low_level_m1_16gb_verbose.py --interactive

# Create HNSW index (fast retrieval)
./scripts/apply_hnsw.sh table_name
```

---

## Troubleshooting Quick Checks

```bash
# ✓ PostgreSQL running?
docker-compose -f config/docker-compose.yml ps

# ✓ Can connect to database?
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "SELECT 1;"

# ✓ Table exists and has data?
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c \
  "SELECT COUNT(*) FROM table_name;"

# ✓ MLX installed? (M1 Mac only)
python -c "import mlx.core; print('MLX available')"

# ✓ vLLM server running?
curl http://localhost:8000/health

# ✓ Embedding model working?
python -c "from rag_low_level_m1_16gb_verbose import build_embed_model; \
  build_embed_model().get_text_embedding('test')"
```

---

## Getting Help

**Documentation:**
- Full guide: `/docs/GUI_USER_GUIDE.md` (this file)
- Developer guide: `/CLAUDE.md`
- Environment variables: `/docs/ENVIRONMENT_VARIABLES.md`
- Performance tuning: `/docs/PERFORMANCE.md`

**Common Issues:**
- Getting 0 results: See "Part 4: Troubleshooting"
- Slow performance: See "Performance Issues" section
- Out of memory: See "Memory Problems" section
- Database errors: See "Database Connection Issues" section

**Quick links:**
- Chunking strategies: Part 3 → "Chunking Strategies"
- Model selection: Part 3 → "Embedding Model Selection Guide"
- Hybrid search: Part 3 → "Hybrid Search Explained"
- Advanced features: Part 3 (HyDE, reranking, caching)

---

## Next Steps

**After reading this guide:**

1. **Try the Quick Start** (Part 1)
   - Launch GUI: `streamlit run rag_web.py`
   - Index a sample document
   - Ask a few questions

2. **Experiment with settings** (Part 2)
   - Try different chunk sizes
   - Compare embedding models
   - Visualize embeddings

3. **Optimize for your use case** (Part 3)
   - Choose chunk strategy for your documents
   - Select appropriate embedding model
   - Test hybrid search if needed

4. **Advanced workflows** (Part 5)
   - Batch index multiple documents
   - A/B test configurations
   - Optimize speed vs quality

5. **Learn CLI for production** (CLI Guide)
   - Read `CLAUDE.md` for CLI details
   - Use environment variables for automation
   - Script your workflows

---

**Enjoy using the RAG Web UI!**

For issues, questions, or feature requests, please open an issue on GitHub or consult the documentation in `/docs/`.
