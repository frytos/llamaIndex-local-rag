# Environment Variables Reference

Complete guide to all configurable environment variables for the RAG pipeline.

---

## Quick Start

```bash
# Minimal indexing (fast)
EMBED_BACKEND=mlx PDF_PATH=data/myfile.pdf PGTABLE=myindex python rag_low_level_m1_16gb_verbose.py --index-only

# Minimal querying
PGTABLE=myindex python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

---

## Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_NAME` | `vector_db` | PostgreSQL database name |
| `PGHOST` | `localhost` | Database host address |
| `PGPORT` | `5432` | Database port |
| `PGUSER` | `fryt` | Database username |
| `PGPASSWORD` | `frytos` | Database password |
| `PGTABLE` | `llama2_paper` | Table name for vector store |

**Example:**
```bash
PGHOST=localhost PGPORT=5432 PGUSER=fryt PGPASSWORD=frytos DB_NAME=vector_db
```

---

## Document & Index Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_PATH` | `data/llama2.pdf` | File or folder path to index |
| `RESET_TABLE` | `0` | Drop table before indexing (`0` or `1`) |
| `RESET_DB` | `0` | Drop entire database (`0` or `1`) ⚠️ DANGEROUS! |
| `EXTRACT_CHAT_METADATA` | `1` | Extract metadata from chat logs (`0` or `1`) |

**Examples:**
```bash
# Index a single file
PDF_PATH=data/document.pdf

# Index a folder
PDF_PATH=data/inbox_clean

# Reset table before indexing (recommended during development)
RESET_TABLE=1

# Extract chat metadata (participant, dates, message counts)
EXTRACT_CHAT_METADATA=1
```

---

## Chunking Configuration

**Critical for RAG quality!** Chunking determines how documents are split into searchable pieces.

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `700` | Characters per chunk (100-2000) |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks (typically 15-25% of chunk_size) |

### Chunking Presets

| Preset | CHUNK_SIZE | CHUNK_OVERLAP | Best For |
|--------|------------|---------------|----------|
| **Ultra-fine** | 100 | 20 | Chat logs, tweets, short messages |
| **Fine-grained** | 300 | 60 | Q&A with specific facts |
| **Balanced** ⭐ | 700 | 150 | General-purpose documents (recommended) |
| **Contextual** | 1200 | 240 | Summaries, complex topics |
| **Large context** | 2000 | 400 | Lengthy explanations, essays |

**Examples:**
```bash
# Ultra-fine (chat logs)
CHUNK_SIZE=100 CHUNK_OVERLAP=20

# Balanced (recommended)
CHUNK_SIZE=700 CHUNK_OVERLAP=150

# Large context (essays, long-form content)
CHUNK_SIZE=2000 CHUNK_OVERLAP=400
```

### Chunking Guidelines

**Overlap ratio:**
- 10-15%: Minimal, fast indexing
- 15-25%: Balanced (recommended)
- 25-30%: Maximum context preservation

**Trade-offs:**
- **Small chunks** (100-300): Precise retrieval but less context
- **Medium chunks** (500-800): Balanced (best for most cases)
- **Large chunks** (1000-2000): More context but less precise

---

## Embedding Configuration

Controls how text is converted to vector representations.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `BAAI/bge-small-en` | HuggingFace model name |
| `EMBED_DIM` | `384` | Vector dimensions (384/768/1024) |
| `EMBED_BATCH` | `32` | Batch size for embedding |
| `EMBED_BACKEND` | `huggingface` | Backend: `huggingface` \| `mlx` |

### Model Options

| Model | EMBED_DIM | Speed | Quality | MLX Support | Best For |
|-------|-----------|-------|---------|-------------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fastest | Good | ⚠️ Limited | Quick prototyping |
| `BAAI/bge-small-en` ⭐ | 384 | Fast | Very Good | ✅ Excellent | General use (recommended) |
| `BAAI/bge-base-en` | 768 | Medium | Excellent | ✅ Good | Higher quality needs |
| `BAAI/bge-large-en-v1.5` | 1024 | Slow | Best | ⚠️ Poor | Maximum quality |

### Backend Options

| Backend | Speed | Hardware | Best For |
|---------|-------|----------|----------|
| `huggingface` | Baseline | Any (CPU/GPU/MPS) | Compatibility |
| `mlx` ⚡ | **5-20x faster** | Apple Silicon only | M1/M2/M3 Macs |

### Recommended Configurations

**Fast indexing with MLX (recommended for M1/M2/M3):**
```bash
EMBED_BACKEND=mlx
EMBED_MODEL=BAAI/bge-small-en
EMBED_DIM=384
EMBED_BATCH=64
```

**Maximum quality:**
```bash
EMBED_BACKEND=mlx
EMBED_MODEL=BAAI/bge-large-en-v1.5
EMBED_DIM=1024
EMBED_BATCH=32
```

**Balanced (no MLX):**
```bash
EMBED_BACKEND=huggingface
EMBED_MODEL=BAAI/bge-small-en
EMBED_DIM=384
EMBED_BATCH=32
```

### Batch Size Guidelines

| Backend | Model Size | Recommended EMBED_BATCH | Memory |
|---------|------------|------------------------|--------|
| HuggingFace | Small (384d) | 32-64 | 8GB+ |
| HuggingFace | Large (1024d) | 16-32 | 16GB+ |
| MLX | Small (384d) | 128-256 | 8GB+ |
| MLX | Large (1024d) | 64-128 | 16GB+ |

---

## Retrieval Configuration

Controls how chunks are retrieved during queries.

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | `4` | Number of chunks to retrieve (2-10) |
| `HYBRID_ALPHA` | `1.0` | Search mode: `0.0`=BM25, `0.5`=hybrid, `1.0`=vector |
| `ENABLE_FILTERS` | `1` | Enable metadata filtering (`0` or `1`) |
| `MMR_THRESHOLD` | `0.0` | Diversity: `0`=disabled, `0.5`=balanced, `1.0`=max |
| `ENABLE_QUERY_EXPANSION` | `0` | Enable query expansion (`0` or `1`) |
| `QUERY_EXPANSION_METHOD` | `llm` | Expansion method: `llm`, `multi`, or `keyword` |
| `QUERY_EXPANSION_COUNT` | `2` | Number of expansions to generate (1-5) |
| `ENABLE_HYDE` | `0` | Enable HyDE retrieval (`0` or `1`) |
| `HYDE_NUM_HYPOTHESES` | `1` | Number of hypotheses to generate (1-3) |
| `HYDE_HYPOTHESIS_LENGTH` | `100` | Target hypothesis length in tokens (50-200) |
| `HYDE_FUSION_METHOD` | `rrf` | Fusion method: `rrf`, `avg`, or `max` |

### Search Mode (HYBRID_ALPHA)

| Value | Mode | Description |
|-------|------|-------------|
| `0.0` | Pure BM25 | Keyword-only search (like grep) |
| `0.3` | BM25-heavy | Mostly keywords with some semantics |
| `0.5` | Balanced hybrid | Best of both worlds ⭐ |
| `0.7` | Vector-heavy | Mostly semantic with some keywords |
| `1.0` | Pure vector | Semantic-only search (default) |

**Examples:**
```bash
# Pure vector search (default)
HYBRID_ALPHA=1.0

# Balanced hybrid (recommended)
HYBRID_ALPHA=0.5

# Keyword-only search
HYBRID_ALPHA=0.0
```

### Metadata Filtering

When `ENABLE_FILTERS=1`, you can filter by chat metadata in queries:

```bash
# Query syntax examples:
participant:Alice meeting agenda
after:2024-06-01 project updates
before:2024-12-31 quarterly review
participant:Bob after:2024-01-01 travel plans
```

### MMR (Maximum Marginal Relevance)

Controls diversity in retrieved results:

| Value | Behavior |
|-------|----------|
| `0.0` | Disabled (default) - returns most similar chunks |
| `0.5` | Balanced - mix of relevance and diversity |
| `1.0` | Maximum relevance - focus on query match |

**Example:**
```bash
# Enable diversity (avoid repetitive chunks)
MMR_THRESHOLD=0.5
```

### Query Expansion

Expands user queries with synonyms and related terms to improve recall by 15-30%. Useful for complex queries or when users phrase questions differently from document content.

| Method | Speed | Quality | LLM Required | Best For |
|--------|-------|---------|--------------|----------|
| `keyword` | <0.1s | Good | No | Simple queries, fast responses |
| `multi` | 1-3s | Better | Yes | Complex queries, multiple perspectives |
| `llm` | 1-3s | Best | Yes | General-purpose (recommended) |

**Examples:**
```bash
# Enable query expansion with LLM (best quality)
ENABLE_QUERY_EXPANSION=1 QUERY_EXPANSION_METHOD=llm QUERY_EXPANSION_COUNT=2

# Fast keyword-based expansion (no LLM)
ENABLE_QUERY_EXPANSION=1 QUERY_EXPANSION_METHOD=keyword QUERY_EXPANSION_COUNT=2

# Multi-query generation (different angles)
ENABLE_QUERY_EXPANSION=1 QUERY_EXPANSION_METHOD=multi QUERY_EXPANSION_COUNT=3
```

**How it works:**
1. Original query: "What did Elena say about Morocco?"
2. LLM generates 2 variations: "What did Elena mention regarding Morocco?", "Elena's comments about Morocco"
3. Retrieval runs on all 3 queries (original + 2 expansions)
4. Results are deduplicated and reranked
5. Final top-k results returned

**Performance impact:**
- Keyword method: ~0.1s overhead (negligible)
- LLM/Multi method: ~1-3s overhead (one-time per query)
- Improves recall for complex queries where vocabulary mismatch is common

---

### HyDE (Hypothetical Document Embeddings)

Advanced retrieval technique that generates hypothetical answers before embedding, improving retrieval quality by 10-20% for technical/complex queries.

**How it works:**
1. User query: "What is attention mechanism?"
2. LLM generates hypothetical answer: "Attention mechanism is a key component in neural networks..."
3. Embed the hypothetical answer (not the query)
4. Retrieve documents similar to the hypothetical answer
5. Documents match answer style better than question style

**Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_HYDE` | `0` | Enable HyDE retrieval (`0` or `1`) |
| `HYDE_NUM_HYPOTHESES` | `1` | Number of hypotheses to generate (1-3) |
| `HYDE_HYPOTHESIS_LENGTH` | `100` | Target hypothesis length in tokens |
| `HYDE_FUSION_METHOD` | `rrf` | How to fuse multiple hypotheses |

**Fusion Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| `rrf` | Reciprocal Rank Fusion (default) | Balanced, robust |
| `avg` | Average similarity scores | Simple, fast |
| `max` | Maximum similarity score | Aggressive matching |

**Examples:**
```bash
# Enable HyDE with single hypothesis (fastest)
ENABLE_HYDE=1 HYDE_NUM_HYPOTHESES=1

# Multiple hypotheses for better coverage (slower but better)
ENABLE_HYDE=1 HYDE_NUM_HYPOTHESES=2 HYDE_FUSION_METHOD=rrf

# Longer hypotheses for complex queries
ENABLE_HYDE=1 HYDE_HYPOTHESIS_LENGTH=150
```

**Performance Impact:**

| Configuration | Added Latency | Quality Improvement | Best For |
|--------------|---------------|---------------------|----------|
| 1 hypothesis | +100-200ms | +10-15% | Technical queries |
| 2 hypotheses | +200-300ms | +15-20% | Complex questions |
| 3 hypotheses | +300-400ms | +15-20% | Multi-faceted queries |

**When to use HyDE:**
- ✅ Technical/domain-specific queries
- ✅ Complex questions with multiple aspects
- ✅ When query style differs from document style
- ❌ Simple factual queries (overkill)
- ❌ When latency is critical (<500ms required)

**Example workflow:**
```
Original query: "What is attention mechanism?"

Generated hypothesis:
"The attention mechanism is a key component in neural networks,
particularly in transformer models. It allows the model to focus
on different parts of the input sequence when making predictions..."

Embedding: [0.023, -0.145, 0.089, ...] (hypothesis embedding)

Retrieved documents:
1. "Attention mechanisms compute weighted representations..."
2. "The transformer architecture uses self-attention..."
3. "In neural networks, attention allows selective focus..."
```

---

## RAG Improvements

Advanced features to enhance retrieval quality and query performance.

### Reranking

Fine-tunes retrieved results by re-scoring candidates with a cross-encoder model. Improves precision by 10-20% at the cost of additional latency.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_RERANKING` | `0` | Enable reranking of retrieved candidates (`0` or `1`) |
| `RERANK_CANDIDATES` | `12` | Number of candidates to rerank (before TOP_K selection) |
| `RERANK_TOP_K` | `4` | Final results after reranking |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model from HuggingFace |

**How it works:**
1. Retriever returns top-N candidates (RERANK_CANDIDATES)
2. Cross-encoder re-scores all candidates
3. Top-K results selected after reranking (RERANK_TOP_K)
4. Results sent to LLM for generation

**Performance trade-offs:**

| Setting | Speed | Quality | Best For |
|---------|-------|---------|----------|
| Disabled | Baseline | Good | Fast retrieval, reasonable quality |
| RERANK_CANDIDATES=12, TOP_K=4 | +0.5-1s | +10-15% | Production (balanced) ⭐ |
| RERANK_CANDIDATES=20, TOP_K=4 | +1-2s | +15-20% | High-quality results |

**Examples:**
```bash
# Enable basic reranking (recommended)
ENABLE_RERANKING=1 RERANK_CANDIDATES=12 RERANK_TOP_K=4

# High-quality reranking
ENABLE_RERANKING=1 RERANK_CANDIDATES=20 RERANK_TOP_K=4

# Disable reranking (fast)
ENABLE_RERANKING=0
```

**Memory requirements:**
- Base model: ~150MB VRAM
- Batch reranking: minimal overhead

---

### Semantic Caching

Caches query results based on semantic similarity. Reduces redundant computations by 30-50% for similar queries, speeding up response time from seconds to milliseconds.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SEMANTIC_CACHE` | `1` | Enable semantic cache (`0` or `1`) |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Similarity threshold (0.0-1.0) for cache hits |
| `SEMANTIC_CACHE_MAX_SIZE` | `1000` | Maximum cached queries |
| `SEMANTIC_CACHE_TTL` | `86400` | Cache entry lifetime in seconds (0 = infinite) |

**How it works:**
1. User submits query
2. Embed the query
3. Compare against cached queries using cosine similarity
4. If similarity > THRESHOLD, return cached result
5. Otherwise, execute full RAG pipeline and cache result

**Cache hit rates by threshold:**

| THRESHOLD | Hit Rate | False Positives | Best For |
|-----------|----------|-----------------|----------|
| 0.85 | ~60-70% | Possible | Aggressive caching, accept some variance |
| 0.90 | ~40-50% | Very low | Balanced (recommended) ⭐ |
| 0.92 | ~20-30% | None | Strict mode, only identical queries |
| 0.95 | ~10-15% | None | Ultra-strict |

**Performance impact:**

| Feature | Latency | Memory | Storage |
|---------|---------|--------|---------|
| Cache lookup | <10ms | ~5MB per 1000 queries | Redis or in-memory |
| Full RAG | 5-15s | Standard | N/A |

**Examples:**
```bash
# Enable semantic cache (recommended - balanced mode)
ENABLE_SEMANTIC_CACHE=1 SEMANTIC_CACHE_THRESHOLD=0.92 SEMANTIC_CACHE_MAX_SIZE=1000

# Aggressive caching (more hits, less strict)
ENABLE_SEMANTIC_CACHE=1 SEMANTIC_CACHE_THRESHOLD=0.85 SEMANTIC_CACHE_MAX_SIZE=2000

# Disable cache (always execute full pipeline)
ENABLE_SEMANTIC_CACHE=0

# Time-based cache expiry (24 hours)
SEMANTIC_CACHE_TTL=86400
```

**Cache strategies:**

- **Persistent cache**: Keep TTL=0, cache survives restarts
- **Session cache**: TTL=3600 (1 hour), clean up old entries
- **No cache**: ENABLE_SEMANTIC_CACHE=0

---

### Conversation Memory

Enables multi-turn dialogues with context awareness, reference resolution, and automatic query reformulation. Essential for natural conversational RAG applications.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CONVERSATION_MEMORY` | `1` | Enable conversation memory (`0` or `1`) |
| `MAX_CONVERSATION_TURNS` | `10` | Maximum turns to keep in memory |
| `CONVERSATION_TIMEOUT` | `3600` | Session timeout in seconds (1 hour) |
| `AUTO_SUMMARIZE` | `1` | Auto-summarize long conversations (`0` or `1`) |
| `SUMMARIZE_THRESHOLD` | `5` | Turns before summarization kicks in |
| `CONVERSATION_CACHE_DIR` | `.cache/conversations` | Storage directory for conversation history |

**How it works:**
1. User asks: "What is machine learning?"
2. System responds with definition
3. User asks: "How does it work?"
4. System resolves "it" → "machine learning"
5. Reformulates: "How does machine learning work given previous discussion about..."
6. Retrieves with enhanced context
7. Stores conversation for future turns

**Key features:**

| Feature | Description | Example |
|---------|-------------|---------|
| **Reference Resolution** | Resolve pronouns ("it", "that", "them") | "What is RAG?" → "How does it work?" → "How does RAG work?" |
| **Query Reformulation** | Add conversation context to queries | "What about overlap?" → "Given previous discussion about chunking, what about chunk overlap?" |
| **Entity Tracking** | Track mentioned entities across turns | Extracts "LlamaIndex", "pgvector", technical terms |
| **Auto-Summarization** | Compress old turns to save memory | After 5 turns, summarize oldest conversations |
| **Session Management** | Multiple concurrent user sessions | Each user has independent conversation history |

**Performance impact:**

| Operation | Latency | Memory per Turn | Disk per Session |
|-----------|---------|-----------------|------------------|
| Reference resolution | <1ms | ~2KB | ~5KB |
| Query reformulation | <5ms | Negligible | - |
| Entity extraction | <2ms | ~100 bytes | - |
| Session persistence | ~5ms | - | ~10KB total |

**Examples:**
```bash
# Enable conversation memory (recommended)
ENABLE_CONVERSATION_MEMORY=1 MAX_CONVERSATION_TURNS=10

# Aggressive summarization (save memory for long conversations)
AUTO_SUMMARIZE=1 SUMMARIZE_THRESHOLD=3 MAX_CONVERSATION_TURNS=15

# Disable for stateless queries
ENABLE_CONVERSATION_MEMORY=0

# Short-lived sessions (30 minutes)
CONVERSATION_TIMEOUT=1800

# Custom cache directory
CONVERSATION_CACHE_DIR=/persistent/path/conversations
```

**Use cases:**

- **Interactive chat**: Natural back-and-forth conversations
- **Follow-up questions**: "Tell me more", "What about X?"
- **Multi-turn debugging**: "That didn't work", "Try another way"
- **Context-aware search**: Previous queries inform current retrieval
- **Multi-user support**: Each user has independent conversation history

**Integration patterns:**
```python
# Pattern 1: Session-based conversations
from utils.conversation_memory import session_manager

memory = session_manager.get_or_create("user_123")
resolved = memory.resolve_references(query)
result = run_rag_query(resolved)
memory.add_turn(query, result)

# Pattern 2: Standalone conversation
from utils.conversation_memory import ConversationMemory

memory = ConversationMemory()
memory.add_turn(query, answer)
context = memory.get_conversation_context()
```

**See also:**
- `utils/conversation_memory.py` - Core implementation
- `utils/README_CONVERSATION_MEMORY.md` - Detailed documentation
- `examples/conversation_memory_demo.py` - Usage examples
- `examples/rag_with_conversation_memory.py` - RAG integration

---

### Query Expansion

Expands user queries with synonyms, related terms, and alternative phrasings. Improves recall by 15-30% for complex or poorly-matched queries.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_QUERY_EXPANSION` | `0` | Enable query expansion (`0` or `1`) |
| `QUERY_EXPANSION_METHOD` | `llm` | Expansion method: `keyword` \| `multi` \| `llm` |
| `QUERY_EXPANSION_COUNT` | `2` | Number of expanded queries to generate (1-5) |

**Expansion methods comparison:**

| Method | Speed | Quality | LLM Required | Best For |
|--------|-------|---------|--------------|----------|
| `keyword` | <0.1s | Good | No | Simple queries, tech terms, synonyms |
| `multi` | 1-3s | Better | Yes | Multiple query angles, complex topics |
| `llm` | 1-3s | Best | Yes | General-purpose (recommended) ⭐ |

**Examples:**
```bash
# Enable LLM-based expansion (best quality)
ENABLE_QUERY_EXPANSION=1 QUERY_EXPANSION_METHOD=llm QUERY_EXPANSION_COUNT=2

# Fast keyword expansion (no LLM)
ENABLE_QUERY_EXPANSION=1 QUERY_EXPANSION_METHOD=keyword QUERY_EXPANSION_COUNT=3

# Multi-query generation (different perspectives)
ENABLE_QUERY_EXPANSION=1 QUERY_EXPANSION_METHOD=multi QUERY_EXPANSION_COUNT=2

# Disable expansion
ENABLE_QUERY_EXPANSION=0
```

**How it works (LLM method):**
1. Original query: "What did the team discuss about Q3 roadmap?"
2. LLM generates 2 variations:
   - "Team's discussion on third quarter planning"
   - "Q3 roadmap conversation among team members"
3. Retrieve chunks matching any of the 3 queries
4. Deduplicate and rerank results
5. Return top-K final results

**Recall improvement by use case:**

| Use Case | Improvement |
|----------|-------------|
| Short queries (1-2 words) | +10-15% |
| Long queries (7+ words) | +5-10% |
| Synonym-heavy queries | +15-25% |
| Technical queries | +20-30% |

**Performance impact:**
- Keyword method: ~0.1s overhead (negligible)
- LLM/Multi method: ~1-3s overhead (one-time per query)
- Retrieval: Runs 3 searches instead of 1 (3x complexity)
- Total latency increase: ~1-3s for LLM methods

---

### Enhanced Metadata Extraction

Automatically extracts structured metadata during indexing: topics, entities, code blocks, and tables. Enables advanced filtering, entity-based search, and code snippet retrieval.

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTRACT_ENHANCED_METADATA` | `0` | Enable enhanced metadata extraction (`0` or `1`) |
| `EXTRACT_TOPICS` | `1` | Extract document topics via LLM (`0` or `1`) |
| `EXTRACT_ENTITIES` | `1` | Extract named entities (persons, places, orgs) (`0` or `1`) |
| `EXTRACT_CODE_BLOCKS` | `1` | Identify and tag code blocks (`0` or `1`) |
| `EXTRACT_TABLES` | `1` | Extract and structure table data (`0` or `1`) |

**Extracted metadata types:**

| Type | Example | Use Case |
|------|---------|----------|
| **Topics** | ["machine-learning", "data-pipeline", "performance"] | Topic-based filtering, faceted search |
| **Entities** | {"persons": ["Alice"], "orgs": ["Acme Corp"], "locations": ["NYC"]} | Person/location search, relationship mapping |
| **Code blocks** | `{"language": "python", "lines": 10, "content": "..."}` | Code search, syntax highlighting |
| **Tables** | `{"rows": 5, "columns": 3, "headers": [...]}` | Structured data retrieval |

**Examples:**
```bash
# Enable all metadata extraction
EXTRACT_ENHANCED_METADATA=1 EXTRACT_TOPICS=1 EXTRACT_ENTITIES=1 EXTRACT_CODE_BLOCKS=1 EXTRACT_TABLES=1

# Metadata for knowledge graphs (entities focus)
EXTRACT_ENHANCED_METADATA=1 EXTRACT_ENTITIES=1 EXTRACT_TOPICS=0

# Code-focused extraction
EXTRACT_ENHANCED_METADATA=1 EXTRACT_CODE_BLOCKS=1 EXTRACT_TOPICS=0 EXTRACT_ENTITIES=0

# Disable metadata
EXTRACT_ENHANCED_METADATA=0
```

**Indexing performance impact:**

| Config | Extraction Time | Storage Overhead | Best For |
|--------|-----------------|------------------|----------|
| No extraction | Baseline | Baseline | General RAG |
| Topics only | +2-5% | +5MB per 10K docs | Topic filtering |
| Entities only | +5-10% | +10MB per 10K docs | Knowledge graphs |
| All enabled | +15-25% | +30MB per 10K docs | Advanced search |

**How it works:**
1. Extract text from chunk
2. Run entity extraction (NER model or LLM)
3. Run topic extraction (LLM classification)
4. Identify code blocks (regex + language detection)
5. Find and parse tables (HTML/CSV parsers)
6. Store metadata in vector store alongside chunk

**Query with metadata filters:**
```bash
# Search within specific topics
entity:Alice topic:Q3-planning

# Code search
type:code language:python machine-learning

# Table search
type:table rows>10 revenue
```

**Memory requirements during indexing:**
- Entity extraction: ~500MB (NER model)
- Topic extraction: ~1GB (LLM if enabled)
- Code detection: Negligible
- Table parsing: Minimal

---

## LLM Configuration

Controls the local LLM (Mistral 7B via llama.cpp).

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_URL` | Mistral 7B Q4_K_M | URL to download GGUF model |
| `MODEL_PATH` | _(empty)_ | Local path to GGUF model (overrides URL) |
| `TEMP` | `0.1` | Temperature (0.0=deterministic, 1.0=creative) |
| `MAX_NEW_TOKENS` | `256` | Maximum tokens to generate |
| `CTX` | `3072` | Context window size |
| `N_GPU_LAYERS` | `16` | Layers to offload to Metal GPU (0-32) |
| `N_BATCH` | `128` | Batch size for prompt processing |

### Apple Silicon Tuning

| Hardware | N_GPU_LAYERS | N_BATCH | CTX | Memory Usage |
|----------|--------------|---------|-----|--------------|
| M1/M2 8GB | 8 | 64 | 2048 | Conservative |
| M1/M2 16GB | 16 | 128 | 3072 | Balanced ⭐ |
| M1/M2 16GB | 24 | 256 | 8192 | Aggressive |
| M1 Pro/Max 32GB | 32 | 512 | 8192 | Maximum |

**Examples:**
```bash
# Conservative (8GB Mac)
N_GPU_LAYERS=8 N_BATCH=64 CTX=2048

# Balanced (16GB Mac, recommended)
N_GPU_LAYERS=16 N_BATCH=128 CTX=3072

# Aggressive (16GB Mac, fast inference)
N_GPU_LAYERS=24 N_BATCH=256 CTX=8192
```

### Temperature Guidelines

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.0-0.2` | Deterministic | Factual Q&A, retrieval tasks |
| `0.3-0.5` | Slightly creative | Summaries, explanations |
| `0.6-0.8` | Creative | Brainstorming, creative writing |
| `0.9-1.0` | Very creative | Story generation, poetry |

---

## Query Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QUESTION` | _(default question)_ | Default query text |

**Example:**
```bash
QUESTION="What did Alice say about the Morocco trip?" python rag_low_level_m1_16gb_verbose.py --query-only
```

---

## Logging & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_FULL_CHUNKS` | `0` | Log full chunk content (`0` or `1`) |
| `COLORIZE_CHUNKS` | `0` | Colorize chunk output (`0` or `1`) |
| `LOG_QUERIES` | `0` | Save queries to JSON log (`0` or `1`) |
| `DB_INSERT_BATCH` | `250` | Database insert batch size |

**Examples:**
```bash
# Debug mode with full chunk logging
LOG_LEVEL=DEBUG LOG_FULL_CHUNKS=1

# Colorized output (better terminal readability)
COLORIZE_CHUNKS=1

# Save all queries to query_logs/
LOG_QUERIES=1
```

---

## Performance Optimizations (Async & Pooling)

Advanced performance features for async operations and connection pooling. These optimizations can provide **2-10x speedup** for batch operations and high-concurrency workloads.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_ASYNC` | boolean | `1` | Enable async operations (async/await for embeddings and database) |
| `CONNECTION_POOL_SIZE` | integer | `10` | Database connection pool size (persistent connections) |
| `MIN_POOL_SIZE` | integer | `5` | Minimum number of connections to keep alive |
| `MAX_POOL_SIZE` | integer | `20` | Maximum number of connections allowed |
| `BATCH_SIZE` | integer | `32` | Number of queries to process together in a batch |
| `BATCH_TIMEOUT` | float | `1.0` | Maximum time (seconds) to wait before processing incomplete batch |

### Guidelines

**Connection Pool Size:**
- **5-10**: Laptop/desktop (low concurrency)
- **10-20**: Server (moderate concurrency)
- **20-50**: High-traffic server (high concurrency)
- Reduces connection overhead by ~80%

**Batch Size:**
- **16-32**: Standard (balanced)
- **32-64**: High throughput
- **64-128**: Maximum throughput (more memory)
- Provides 2-3x speedup through batch processing

**Batch Timeout:**
- **0.5-1.0s**: Low latency (process quickly)
- **1.0-2.0s**: Balanced
- **2.0-5.0s**: High throughput (wait for full batches)

### Performance Benchmarks

Based on M1 Mac Mini 16GB:

| Operation | Sync | Async | Speedup |
|-----------|------|-------|---------|
| 10 embeddings | ~1.5s | ~0.5s | 3x |
| 10 queries (no pooling) | ~2.0s | - | - |
| 10 queries (with pooling) | - | ~0.4s | 5x |
| 3 table retrieval (sequential) | ~0.9s | - | - |
| 3 table retrieval (parallel) | - | ~0.3s | 3x |
| 10 single queries | ~15s | - | - |
| 10 batched queries | - | ~5s | 3x |

### Usage Example

```python
from utils.performance_optimizations import (
    AsyncEmbedding,
    DatabaseConnectionPool,
    ParallelRetriever,
    BatchProcessor,
    PerformanceMonitor
)

# Async embeddings
async_embed = AsyncEmbedding(model_name="BAAI/bge-small-en")
embeddings = await async_embed.embed_batch(["query1", "query2", "query3"])

# Connection pooling
pool = DatabaseConnectionPool(min_size=5, max_size=10)
await pool.initialize()
async with pool.acquire() as conn:
    results = await conn.fetch("SELECT * FROM table")

# Parallel retrieval
retriever = ParallelRetriever(
    pool=pool,
    embed_model=async_embed,
    tables=["table1", "table2", "table3"]
)
results = await retriever.retrieve_parallel(query, top_k=4)

# Performance monitoring
monitor = PerformanceMonitor()
with monitor.track("operation_name"):
    # Your code here
    pass
stats = monitor.get_stats()
print(f"p50: {stats['operation_name']['p50']:.3f}s")
print(f"p95: {stats['operation_name']['p95']:.3f}s")
```

**Example Configuration:**
```bash
# Enable async optimizations
ENABLE_ASYNC=1

# Connection pooling (10 connections)
CONNECTION_POOL_SIZE=10
MIN_POOL_SIZE=5
MAX_POOL_SIZE=20

# Batch processing
BATCH_SIZE=32
BATCH_TIMEOUT=1.0
```

---

## Performance Tracking & Regression Testing

**NEW**: Automated performance tracking with CI/CD integration, historical trends, and regression detection.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PERFORMANCE_RECORDING` | `0` | Enable recording to history database (`0` or `1`) |
| `PERFORMANCE_RUN_TYPE` | `manual` | Run type: `ci`, `nightly`, or `manual` |
| `PERFORMANCE_PLATFORM` | auto-detect | Override platform detection (e.g., `M1_Mac_16GB`) |
| `PERFORMANCE_DB_PATH` | `benchmarks/history/performance.db` | Path to SQLite database |
| `REGRESSION_THRESHOLD` | `0.20` | Regression tolerance (0.20 = 20%) |
| `BENCHMARK_MODE` | `standard` | Benchmark mode: `quick`, `standard`, or `comprehensive` |

### Platform Auto-Detection

The system automatically detects your hardware platform:

| Platform | Auto-Detected As | Use Case |
|----------|------------------|----------|
| M1 Mac Mini 16GB | `M1_Mac_16GB` | Local development |
| M2 MacBook Pro 32GB | `M2_Mac_32GB` | Local development |
| GitHub Actions macOS | `GitHub_Actions_macOS` | CI/CD |
| RTX 4090 Server | `RTX_4090_24GB` | Production |
| RunPod A100 | `RunPod_A100` | Cloud GPU |

**Override detection:**
```bash
PERFORMANCE_PLATFORM=Custom_Server pytest tests/test_performance_regression.py
```

### Usage Examples

**Run performance tests with recording:**
```bash
# Record to history database
ENABLE_PERFORMANCE_RECORDING=1 \
PERFORMANCE_RUN_TYPE=manual \
pytest tests/test_performance_regression.py -v
```

**Generate performance dashboard:**
```bash
# Create dashboard (last 30 days)
python scripts/generate_performance_dashboard.py --days 30

# Open in browser
open benchmarks/dashboard.html
```

**Update baselines after optimization:**
```bash
# Preview proposed updates
python scripts/update_baselines.py --dry-run

# Apply updates (interactive)
python scripts/update_baselines.py

# Auto-approve improvements (CI)
python scripts/update_baselines.py --auto-approve-improvements --min-runs 5
```

**Run comprehensive benchmark:**
```bash
# Quick mode (for testing)
BENCHMARK_MODE=quick python scripts/run_comprehensive_benchmark.py

# Comprehensive mode (for nightly)
BENCHMARK_MODE=comprehensive \
ENABLE_PERFORMANCE_RECORDING=1 \
PERFORMANCE_RUN_TYPE=nightly \
python scripts/run_comprehensive_benchmark.py --output benchmarks/nightly/$(date +%Y%m%d)
```

### Tracked Metrics

| Metric | Unit | Direction | Description |
|--------|------|-----------|-------------|
| `embedding_throughput` | chunks/sec | ↑ Higher is better | Document embedding speed |
| `vector_search_latency` | seconds | ↓ Lower is better | pgvector search time |
| `query_latency_no_vllm` | seconds | ↓ Lower is better | End-to-end query (llama.cpp) |
| `query_latency_vllm` | seconds | ↓ Lower is better | End-to-end query (vLLM) |
| `db_insertion_throughput` | nodes/sec | ↑ Higher is better | Database write speed |
| `peak_memory_gb` | GB | ↓ Lower is better | Maximum memory usage |
| `cache_hit_rate` | 0-1 | ↑ Higher is better | Query cache effectiveness |
| `tokens_per_second` | tokens/sec | ↑ Higher is better | LLM generation speed |
| `avg_mrr` | 0-1 | ↑ Higher is better | Mean Reciprocal Rank |
| `avg_ndcg` | 0-1 | ↑ Higher is better | Normalized DCG |

### CI/CD Integration

**Automated on every PR:**
- Performance regression tests run automatically
- Report posted as PR comment
- PR blocked if >20% regression detected

**Nightly (2 AM UTC):**
- Comprehensive benchmark suite
- Baseline auto-update on sustained improvements
- Performance dashboard generated
- GitHub issue created on regression

**See**: `docs/PERFORMANCE_TRACKING.md` for complete guide.

---

## Complete Usage Examples

### 1. Fast MLX Indexing (Recommended)

```bash
EMBED_BACKEND=mlx \
EMBED_MODEL=BAAI/bge-small-en \
EMBED_DIM=384 \
EMBED_BATCH=64 \
CHUNK_SIZE=700 \
CHUNK_OVERLAP=150 \
PDF_PATH=data/inbox_clean \
PGTABLE=inbox_mlx_fast \
RESET_TABLE=1 \
EXTRACT_CHAT_METADATA=1 \
python rag_low_level_m1_16gb_verbose.py --index-only
```

**Performance:** ~6-10 minutes for 47K chunks

---

### 2. High-Quality Indexing

```bash
EMBED_BACKEND=mlx \
EMBED_MODEL=BAAI/bge-large-en-v1.5 \
EMBED_DIM=1024 \
EMBED_BATCH=32 \
CHUNK_SIZE=700 \
CHUNK_OVERLAP=150 \
PDF_PATH=data/inbox_clean \
PGTABLE=inbox_mlx_quality \
RESET_TABLE=1 \
python rag_low_level_m1_16gb_verbose.py --index-only
```

**Performance:** ~40-45 minutes for 47K chunks (higher quality)

---

### 3. Ultra-Fine Chunking (Chat Logs)

```bash
EMBED_BACKEND=mlx \
EMBED_MODEL=BAAI/bge-small-en \
EMBED_DIM=384 \
EMBED_BATCH=128 \
CHUNK_SIZE=100 \
CHUNK_OVERLAP=20 \
EXTRACT_CHAT_METADATA=1 \
PDF_PATH=data/inbox_clean \
PGTABLE=inbox_ultrafine \
RESET_TABLE=1 \
python rag_low_level_m1_16gb_verbose.py --index-only
```

**Performance:** ~8-12 minutes for 47K chunks (many small chunks)

---

### 4. Query with Hybrid Search + Filters

```bash
PGTABLE=inbox_mlx_fast \
TOP_K=6 \
HYBRID_ALPHA=0.5 \
ENABLE_FILTERS=1 \
MMR_THRESHOLD=0.5 \
LOG_FULL_CHUNKS=1 \
COLORIZE_CHUNKS=1 \
python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

Then query with filters:
```
> participant:Alice Morocco trip
> after:2024-06-01 project updates
```

---

### 5. Production Query (Fast LLM)

```bash
PGTABLE=inbox_mlx_fast \
TOP_K=4 \
HYBRID_ALPHA=0.5 \
N_GPU_LAYERS=24 \
N_BATCH=256 \
CTX=8192 \
TEMP=0.1 \
MAX_NEW_TOKENS=512 \
python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

---

### 6. Debug Mode (Troubleshooting)

```bash
LOG_LEVEL=DEBUG \
LOG_FULL_CHUNKS=1 \
COLORIZE_CHUNKS=1 \
LOG_QUERIES=1 \
PGTABLE=inbox_mlx_fast \
python rag_low_level_m1_16gb_verbose.py --query-only --query "test query"
```

---

## Performance Optimization Cheatsheet

### For Fastest Indexing:
```bash
EMBED_BACKEND=mlx
EMBED_MODEL=BAAI/bge-small-en
EMBED_BATCH=128
CHUNK_SIZE=700
```

### For Best Quality:
```bash
EMBED_MODEL=BAAI/bge-large-en-v1.5
EMBED_DIM=1024
CHUNK_SIZE=700
TOP_K=6
HYBRID_ALPHA=0.5
```

### For Chat Logs:
```bash
CHUNK_SIZE=100
CHUNK_OVERLAP=20
EXTRACT_CHAT_METADATA=1
ENABLE_FILTERS=1
```

### For Fast LLM Inference:
```bash
N_GPU_LAYERS=24
N_BATCH=256
CTX=8192
TEMP=0.1
```

---

## Tips & Tricks

### 1. Table Naming Convention
Include config in table names for easy tracking:
```bash
PGTABLE=inbox_cs700_ov150_bge_small_mlx
```

### 2. Iterative Development
Use `RESET_TABLE=1` during development:
```bash
RESET_TABLE=1  # Avoids duplicate rows
```

### 3. Test Before Full Index
Index a small subset first:
```bash
PDF_PATH=data/inbox_small PGTABLE=test_small
```

### 4. Monitor Memory
For 16GB Macs, stay conservative with batch sizes:
```bash
EMBED_BATCH=64 N_BATCH=128
```

### 5. Save Queries for Analysis
```bash
LOG_QUERIES=1  # Saves to query_logs/ directory
```

---

## Troubleshooting

### Slow Embedding
```bash
# Check if MLX is being used
EMBED_BACKEND=mlx EMBED_MODEL=BAAI/bge-small-en

# Increase batch size
EMBED_BATCH=64  # or 128 for MLX
```

### Out of Memory
```bash
# Reduce batch sizes
EMBED_BATCH=32
N_BATCH=64

# Reduce GPU layers
N_GPU_LAYERS=8
```

### Poor Retrieval Quality
```bash
# Try smaller chunks
CHUNK_SIZE=300 CHUNK_OVERLAP=60

# Use hybrid search
HYBRID_ALPHA=0.5

# Increase TOP_K
TOP_K=8
```

### Context Window Overflow
```bash
# Reduce chunk size
CHUNK_SIZE=500

# Reduce TOP_K
TOP_K=3

# Increase context window
CTX=8192
```

---

## Quick Reference Table

| Task | Key Variables |
|------|---------------|
| Fast indexing | `EMBED_BACKEND=mlx EMBED_BATCH=64` |
| Quality indexing | `EMBED_MODEL=BAAI/bge-large-en-v1.5 CHUNK_SIZE=700` |
| Chat logs | `CHUNK_SIZE=100 EXTRACT_CHAT_METADATA=1` |
| Hybrid search | `HYBRID_ALPHA=0.5 ENABLE_FILTERS=1` |
| Fast LLM | `N_GPU_LAYERS=24 N_BATCH=256` |
| Debug | `LOG_LEVEL=DEBUG LOG_FULL_CHUNKS=1` |

---

## See Also

- `CLAUDE.md` - Full project documentation
- `INTERACTIVE_GUIDE.md` - Interactive CLI guide
- `FIXES_APPLIED.md` - Recent fixes and improvements
- `START_HERE.md` - Quick start guide
