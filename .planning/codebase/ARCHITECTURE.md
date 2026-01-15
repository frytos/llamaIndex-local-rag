# Architecture

**Analysis Date:** 2026-01-15

## Pattern Overview

**Overall:** Layered Monolith with Modular Utilities

**Key Characteristics:**
- Single-entry-point design with modular utility layers
- CLI, Web UI, and Script interfaces
- Clear separation: ingestion → storage → retrieval → generation
- Local-first architecture (100% private)

## Layers

**Presentation Layer:**
- Purpose: User interfaces for different interaction modes
- Contains: Web UI (Streamlit), CLI menu, batch scripts
- Location: `rag_web.py`, `rag_interactive.py`, `scripts/`
- Depends on: Application layer
- Used by: End users

**Application/Orchestration Layer:**
- Purpose: Main RAG pipeline orchestration
- Contains: Document loading, chunking, embedding, storage, retrieval, generation
- Location: `rag_low_level_m1_16gb_verbose.py` (3277 lines - core pipeline)
- Depends on: Domain layer, infrastructure layer, utility layer
- Used by: Presentation layer

**LLM Integration Layer:**
- Purpose: GPU/CPU inference backends
- Contains: llama.cpp wrapper, vLLM client, OpenAI-compatible API
- Location: `vllm_wrapper.py`, `vllm_client.py`
- Depends on: External LLM libraries
- Used by: Application layer

**Data Processing Layer:**
- Purpose: Document transformations and feature extraction
- Contains: PDF/HTML/DOCX parsing, chunking, metadata extraction
- Location: Embedded in main pipeline, `utils/metadata_extractor.py`
- Depends on: PyMuPDF, BeautifulSoup, NLTK
- Used by: Application layer

**Embedding & Vector Search Layer:**
- Purpose: Vector database operations
- Contains: HuggingFace embeddings, MLX backend, pgvector integration
- Location: `build_embed_model()`, `make_vector_store()` in main pipeline
- Depends on: sentence-transformers, pgvector, psycopg2
- Used by: Application layer

**RAG Enhancement Layer:**
- Purpose: Advanced retrieval techniques
- Contains: Query expansion, reranking, semantic caching, conversation memory
- Location: `utils/query_expansion.py`, `utils/reranker.py`, `utils/query_cache.py`, `utils/conversation_memory.py`
- Depends on: sentence-transformers, scikit-learn
- Used by: Application layer

**Configuration Layer:**
- Purpose: Centralized settings management
- Contains: Settings dataclass, frozen constants, environment loading
- Location: `core/config.py`, `config/constants.py`
- Depends on: python-dotenv
- Used by: All layers

**Infrastructure/Utility Layer:**
- Purpose: Supporting services
- Contains: Platform detection, performance monitoring, health checks, deployment
- Location: `utils/*.py` (22 modules)
- Depends on: Various system libraries
- Used by: All layers

## Data Flow

**Complete RAG Query Lifecycle:**

1. **User Request** (Web UI / CLI / Script)
2. **Configuration** - Parse CLI args, load .env, validate settings
3. **Health Checks** - Verify PostgreSQL, pgvector extension, GPU backend
4. **Document Indexing** (if not query-only):
   - Load Documents → Chunk → Build Nodes → Generate Embeddings → Store in PostgreSQL
   - HNSW indices created automatically (100x+ faster queries)
   - Mixed index detection prevents configuration conflicts
5. **Query Retrieval** (interactive or single query):
   - Query Input → Query Expansion (optional) → Embed Query → Vector Similarity Search
   - Retrieve TOP_K candidates → Query Reranking (optional) → Return nodes
6. **LLM Generation**:
   - Build Context (concatenate TOP_K chunks) → Generate Response (Mistral 7B)
   - Temperature: 0.1 (factual), Max tokens: 256, GPU layers: 24
7. **Post-Processing** - Save query log, display metrics, output to user

**State Management:**
- File-based: All state lives in PostgreSQL database
- No persistent in-memory state
- Each query execution is independent

## Key Abstractions

**VectorDBRetriever:**
- Purpose: Custom retriever with query expansion and reranking
- Location: `rag_low_level_m1_16gb_verbose.py` lines 1804-1983
- Pattern: Implements LlamaIndex BaseRetriever interface
- Example: Query → embed → search → rerank → return nodes

**HybridRetriever:**
- Purpose: Combines BM25 keyword search + vector search
- Location: `rag_low_level_m1_16gb_verbose.py` lines 1333-1803
- Pattern: Multi-stage retrieval with fusion

**Settings (Singleton):**
- Purpose: Centralized configuration with validation
- Location: `core/config.py`
- Pattern: Dataclass with __post_init__ validation

**Factory Pattern for LLM:**
- Purpose: Select LLM backend (llama.cpp vs vLLM)
- Location: `rag_low_level_m1_16gb_verbose.py` lines 2042-2113
- Pattern: Factory function returns appropriate LLM instance

## Entry Points

**CLI Entry:**
- Location: `rag_low_level_m1_16gb_verbose.py`
- Triggers: User runs `python rag_low_level_m1_16gb_verbose.py [OPTIONS]`
- Responsibilities: Parse args, validate config, run RAG pipeline
- Options: `--query TEXT`, `--interactive`, `--query-only`, `--index-only`

**Interactive Menu:**
- Location: `rag_interactive.py`
- Triggers: User runs `python rag_interactive.py`
- Responsibilities: Menu-driven workflow for indexing and querying

**Web UI:**
- Location: `rag_web.py`
- Triggers: User runs `streamlit run rag_web.py`
- Responsibilities: Real-time querying, visualization, parameter tuning

**Batch Scripts:**
- Location: `scripts/` (80+ utility scripts)
- Triggers: Deployment, benchmarking, data cleaning
- Responsibilities: Automated operations

## Error Handling

**Strategy:** Throw exceptions, catch at command level, log and exit

**Patterns:**
- Services throw Error with descriptive messages
- Command handlers catch, log error to stderr, exit(1)
- Validation errors shown before execution (fail fast)
- Optional dependencies with graceful fallback

## Cross-Cutting Concerns

**Logging:**
- Python logging module with configurable level
- Console output with color coding (via custom formatting)
- Query logging to disk (`query_logs/`)

**Validation:**
- Settings validation with bounds checking (`core/config.py`)
- Database connection validation before operations
- Mixed index detection to prevent configuration conflicts

**Performance Monitoring:**
- Prometheus metrics export (`utils/metrics.py`)
- Performance history tracking (`utils/performance_history.py`)
- Query timing and token usage logging

---

*Architecture analysis: 2026-01-15*
*Update when major patterns change*
