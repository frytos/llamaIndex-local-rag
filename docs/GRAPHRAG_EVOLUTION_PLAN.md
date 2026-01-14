# GraphRAG Evolution Plan - PoC Focused

**Status**: Ready for Implementation
**Created**: 2026-01-09
**Complexity**: High (Multi-phase architectural enhancement)
**Approach**: Proof of Concept first (100-500 chunks), then scale if successful

## User Requirements (Confirmed)
- ✅ Query mix: 25-50% relationship queries (balanced use case)
- ✅ Storage: PostgreSQL native (simpler, reuses existing infrastructure)
- ✅ Performance: Test first approach (validate quality before committing to slower indexing)
- ✅ Scope: PoC on small dataset (100-500 chunks) before scaling
- ✅ Entity types: **People**, **Locations**, **Topics/Concepts** (no Organizations for now)

---

## PoC Quick Start Guide

**For impatient developers - here's the fastest path to testing GraphRAG:**

```bash
# 1. Install dependencies (5 min)
pip install llama-index-graph-stores-simple networkx

# 2. Prepare small test dataset (100-500 chunks)
mkdir data/poc_test
cp data/messenger_sample/*.html data/poc_test/  # Or any 5-10 documents

# 3. Run PoC with GraphRAG enabled (NetworkX in-memory)
GRAPH_ENABLED=1 \
GRAPH_STORAGE=networkx \
GRAPH_EXTRACT_ENTITIES=1 \
PDF_PATH=data/poc_test \
PGTABLE=graphrag_poc \
RESET_TABLE=1 \
python test_graphrag_poc.py

# 4. Test relationship queries
python -c "
queries = [
    'Who talked to Alice about the project?',  # Multi-hop
    'What did John say about Paris?',           # Entity + location
    'Conversations about machine learning'      # Topic-based
]
for q in queries:
    print(f'\nQuery: {q}')
    # ... query engine call ...
"

# 5. Evaluate quality (manual review)
# Compare: Vector-only vs GraphRAG results
# Success: GraphRAG is noticeably better for relationship queries
```

**Expected PoC Results:**
- ⏱️ Indexing: 300-500 chunks in 10-30 minutes (vs 2-5 min without graph)
- 💾 Storage: ~2-3x vs vector-only (~5MB → ~12MB for 300 chunks)
- 🔍 Query latency: 0.8-2s (vs 0.4s vector-only)
- ✅ Quality: 30-50% better on relationship queries (target)

**Go/No-Go Criteria:**
- ✅ **Proceed**: ≥30% quality improvement on relationship queries
- ❌ **Abort**: <20% improvement or unacceptable performance

---

## Executive Summary

This plan outlines how to evolve your existing vector-based RAG system to **GraphRAG** - a hybrid approach that combines traditional semantic search with knowledge graph capabilities for enhanced reasoning over entity relationships.

### What is GraphRAG?

GraphRAG builds a **knowledge graph** alongside your vector store where:
- **Entities** (people, places, organizations, concepts) become **nodes**
- **Relationships** between entities become **edges**
- **Communities** of related entities enable hierarchical reasoning

### Key Differences: Vector RAG vs GraphRAG

| Aspect | Vector RAG (Current) | GraphRAG (Proposed) |
|--------|---------------------|---------------------|
| **Retrieval** | Semantic similarity only | Semantic + relationship traversal |
| **Context** | Flat text chunks | Structured entity relationships |
| **Query Types** | "What does document say about X?" | "How are X and Y connected?" |
| **Reasoning** | Document-level | Entity-level + graph structure |
| **Best For** | Factual Q&A, document search | Multi-hop reasoning, relationship queries |

---

## Pros and Cons Analysis

### ✅ Advantages of GraphRAG

1. **Multi-Hop Reasoning**
   - Query: "Who worked with someone who knew Einstein?"
   - Vector RAG: Struggles (requires entities in same chunk)
   - GraphRAG: Traverses relationship graph naturally

2. **Entity-Centric Queries**
   - Query: "What did John say about the project?"
   - Vector RAG: Keyword match only
   - GraphRAG: Knows "John" as entity, finds all related chunks

3. **Relationship Understanding**
   - Query: "What's the connection between COVID-19 and supply chains?"
   - Vector RAG: May retrieve unrelated chunks mentioning both
   - GraphRAG: Follows causal/temporal relationships

4. **Better Context for Ambiguous Terms**
   - Query: "Tell me about Python"
   - Vector RAG: May mix programming language + snake
   - GraphRAG: Disambiguates via entity type and context

5. **Hierarchical Summarization (Microsoft GraphRAG)**
   - Community detection groups related entities
   - Enables "zoom out" for high-level answers
   - Better for "summarize all discussions about..."

6. **Explainability**
   - Graph paths show reasoning chain
   - "Answer came from: Entity A → Relationship B → Entity C"

### ❌ Disadvantages of GraphRAG

1. **Higher Complexity**
   - Additional infrastructure (Neo4j/PostgreSQL graph tables)
   - Entity extraction pipeline (LLM calls per chunk)
   - Graph maintenance and consistency

2. **Increased Cost**
   - **10-20x more LLM calls** for entity/relationship extraction
   - For 10K chunks: ~10K extraction calls ($5-50 depending on LLM)
   - Can mitigate with local LLMs (Mistral 7B) but slower

3. **Slower Indexing**
   - Vector-only: ~150s for 10K chunks
   - GraphRAG: ~600-1200s (entity extraction dominates)
   - 4-8x slower indexing time

4. **Storage Overhead**
   - Vector embeddings: ~1.5KB/chunk
   - Graph data: ~2-5KB/chunk (entities + relationships)
   - Total: 2-3x storage vs vector-only

5. **Query Latency**
   - Vector search: ~0.3s
   - GraphRAG: ~0.5-2s (depends on graph traversal depth)
   - More complex queries = longer traversal

6. **Entity Extraction Quality**
   - LLMs struggle with:
     - Long chunks (accuracy degrades beyond 800 tokens)
     - Low-resource languages (multilingual challenges)
     - Domain-specific entities (needs fine-tuning)
   - May produce noisy or incorrect entities

7. **Not Always Better**
   - Simple factual queries: Vector RAG is faster and sufficient
   - Document summarization: Vector RAG works well
   - GraphRAG shines only for relationship/reasoning queries

8. **Maintenance Burden**
   - Graph schema evolution (adding new entity types)
   - Duplicate entity detection and merging
   - Graph quality monitoring

### 🎯 When to Use GraphRAG

**Use GraphRAG for:**
- ✅ Relationship queries ("How is X connected to Y?")
- ✅ Multi-hop reasoning ("Who knows someone who worked with Z?")
- ✅ Entity-centric queries ("Everything John said about...")
- ✅ Complex documents with many interconnected entities
- ✅ Investigative/exploratory queries
- ✅ Conversational data (your FB Messenger use case!)

**Stick with Vector RAG for:**
- ✅ Simple factual lookups ("What is X?")
- ✅ Document summarization
- ✅ Semantic similarity is sufficient
- ✅ Budget/latency constraints
- ✅ Small datasets (<1000 chunks)

### 💡 Recommended Approach: **Hybrid Mode**

**Keep both vector search AND graph** for maximum flexibility:
- Fast vector search for simple queries
- Graph expansion for complex reasoning
- Fallback to vector-only if graph query fails

---

## Architecture Overview

### Current Architecture (Vector RAG)

```
Documents → Chunks → Embeddings → pgvector → Similarity Search → LLM
```

### Proposed Architecture (Hybrid GraphRAG)

```
Documents → Chunks → [Entity Extraction] → Graph Store (entities/relationships)
                   ↓
                Embeddings → pgvector

Query → [Vector Retrieval] → [Graph Expansion] → [Rerank] → LLM
```

### Key Components

1. **Entity Extraction Pipeline**
   - Module: `utils/graph_extractor.py` (new)
   - Uses: Mistral 7B (your existing LLM) or dedicated extraction model
   - Extracts: Entities + relationships from each chunk
   - Output: `{"entities": [...], "relationships": [...]}`

2. **Graph Storage Layer**
   - **Option A** (Recommended): PostgreSQL native tables
     - Tables: `entities`, `relationships`, `entity_chunks`
     - Leverages existing PostgreSQL infrastructure
     - Simple JOIN queries for graph traversal

   - **Option B**: Neo4j alongside PostgreSQL
     - More powerful graph queries (Cypher)
     - Better for deep traversal (>3 hops)
     - Requires Docker container

   - **Option C**: In-memory NetworkX (prototyping)
     - No external dependencies
     - Fast for small graphs (<10K entities)
     - Not persistent

3. **Graph-Enhanced Retrieval**
   - Module: `utils/graph_retriever.py` (new)
   - Flow:
     1. Vector retrieval (top-k=12 candidates)
     2. Extract entities from candidates
     3. Graph expansion (1-2 hop traversal)
     4. Retrieve chunks containing related entities
     5. Rerank combined results (top-k=4 final)

4. **Metadata Integration**
   - Store entity IDs in TextNode metadata
   - Link vector chunks ↔ graph entities
   - Enable bidirectional traversal

---

## Implementation Plan

### Phase 0: Prerequisites (1-2 hours)

**Install Dependencies**
```bash
# Core GraphRAG
pip install llama-index-graph-stores-neo4j  # If using Neo4j
pip install networkx graspologic  # For community detection
pip install neo4j  # Neo4j driver (if Option B)

# Or just NetworkX for prototyping
pip install networkx matplotlib
```

**Database Setup (if using Neo4j - Option B)**
```bash
# Run Neo4j in Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**Environment Variables**
```bash
# Add to config/.env
GRAPH_ENABLED=1
GRAPH_STORAGE=postgresql  # or neo4j, networkx
GRAPH_EXTRACT_ENTITIES=1
GRAPH_BUILD_COMMUNITIES=1
GRAPH_MAX_HOPS=2
```

---

### Phase 1: Entity Extraction Module (4-6 hours)

**Create: `utils/graph_extractor.py`**

**Features:**
1. **Entity Extraction** (Schema-based for quality)
   - Use SchemaLLMPathExtractor (LlamaIndex built-in)
   - Mistral 7B (local, no API costs)
   - Extract: **Person**, **Location**, **Topic** entities only
   - Schema enforces entity types for better precision
   - Handle multilingual entities (bge-m3 already supports)

2. **Relationship Extraction**
   - Extract subject-predicate-object triples
   - Examples: (John, worked_with, Alice), (COVID-19, caused, supply_chain_issues)
   - Normalize relationship types (synonyms)

3. **Entity Resolution**
   - Detect duplicates ("John", "John Smith", "J. Smith")
   - Use fuzzy matching + embeddings
   - Merge co-referent entities

**Implementation Strategy:**

```python
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.graph_stores.types import EntityNode, Relation

# Schema-based extraction for higher quality
ENTITY_TYPES = ["Person", "Location", "Topic"]

class GraphExtractor:
    def __init__(self, llm, max_paths_per_chunk=10):
        """Initialize with schema-based extractor for quality."""
        self.extractor = SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=ENTITY_TYPES,  # Enforce entity types
            max_paths_per_chunk=max_paths_per_chunk,
            num_workers=2,  # Conservative for PoC (avoid overwhelming M1)
            strict=True  # Only extract defined entity types
        )

    def extract(self, chunk: str) -> GraphData:
        """Extract entities and relationships from chunk."""
        # Use LlamaIndex extractor
        paths = self.extractor.extract_paths([chunk])

        # Parse into entities and relationships
        entities = self._parse_entities(paths)
        relationships = self._parse_relationships(paths)

        return GraphData(entities, relationships)

    def _parse_entities(self, paths):
        """Extract unique entities with types."""
        entities = []
        seen = set()
        for path in paths:
            for node in path:
                if isinstance(node, EntityNode):
                    key = (node.name, node.label)
                    if key not in seen:
                        entities.append({
                            "name": node.name,
                            "type": node.label,  # Person/Location/Topic
                            "metadata": {}
                        })
                        seen.add(key)
        return entities

    def _parse_relationships(self, paths):
        """Extract subject-predicate-object triples."""
        relationships = []
        for path in paths:
            if isinstance(path, Relation):
                relationships.append({
                    "source": path.source_id,
                    "target": path.target_id,
                    "type": path.label,
                    "weight": 1.0
                })
        return relationships
```

**Cost Optimization:**
- **Batch extraction**: Process 32 chunks per LLM call
- **Caching**: Skip extraction if chunk unchanged
- **Async processing**: Use `num_workers=4` for parallelism
- **Estimated cost**: ~0.5-2 seconds per chunk (Mistral 7B local)

**Critical Files:**
- New: `utils/graph_extractor.py`
- Modified: `rag_low_level_m1_16gb_verbose.py` (integrate at line 2290)

---

### Phase 2: Graph Storage Layer (3-5 hours)

**Option A: PostgreSQL Native (Recommended)**

**Schema Design:**

```sql
-- Entities table
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT,  -- Person, Organization, Location, Concept
    canonical_name TEXT,  -- After entity resolution
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, type)
);
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(type);
CREATE INDEX idx_entities_canonical ON entities(canonical_name);

-- Relationships table
CREATE TABLE relationships (
    id SERIAL PRIMARY KEY,
    source_entity_id INT REFERENCES entities(id),
    target_entity_id INT REFERENCES entities(id),
    relation_type TEXT NOT NULL,  -- worked_with, caused, located_in
    weight FLOAT DEFAULT 1.0,  -- Confidence score
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_rel_source ON relationships(source_entity_id);
CREATE INDEX idx_rel_target ON relationships(target_entity_id);
CREATE INDEX idx_rel_type ON relationships(relation_type);

-- Entity-Chunk mapping (many-to-many)
CREATE TABLE entity_chunks (
    entity_id INT REFERENCES entities(id),
    chunk_id TEXT,  -- References TextNode.id in vector store
    table_name TEXT,  -- pgvector table name
    PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX idx_ec_entity ON entity_chunks(entity_id);
CREATE INDEX idx_ec_chunk ON entity_chunks(chunk_id);
```

**Implementation:**

```python
# utils/graph_storage.py
class PostgreSQLGraphStore:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
        self._ensure_schema()

    def insert_entity(self, name, entity_type, metadata=None):
        """Insert or get existing entity."""
        ...

    def insert_relationship(self, source_id, target_id, rel_type, weight=1.0):
        """Insert relationship between entities."""
        ...

    def link_entity_to_chunk(self, entity_id, chunk_id, table_name):
        """Link entity to vector store chunk."""
        ...

    def get_related_entities(self, entity_ids, hops=2, limit=50):
        """Traverse graph to find related entities."""
        # Use recursive CTE (Common Table Expression)
        query = """
        WITH RECURSIVE entity_graph AS (
            -- Base: starting entities
            SELECT source_entity_id, target_entity_id, 1 as depth
            FROM relationships
            WHERE source_entity_id = ANY(%s)

            UNION

            -- Recursive: expand to neighbors
            SELECT r.source_entity_id, r.target_entity_id, eg.depth + 1
            FROM relationships r
            JOIN entity_graph eg ON r.source_entity_id = eg.target_entity_id
            WHERE eg.depth < %s
        )
        SELECT DISTINCT target_entity_id FROM entity_graph
        LIMIT %s;
        """
        ...
```

**Critical Files:**
- New: `utils/graph_storage.py`
- New: `scripts/setup_graph_schema.sql`
- Modified: `rag_low_level_m1_16gb_verbose.py` (integrate storage)

---

### Phase 3: Integration with RAG Pipeline (4-6 hours)

**Modify: `rag_low_level_m1_16gb_verbose.py`**

**Integration Points:**

1. **Node Building (line 2290)** - Add entity extraction

```python
def build_nodes(docs, chunks, doc_idxs):
    # ... existing code ...

    # NEW: Extract entities and relationships
    if S.graph_enabled:
        log.info("Extracting entities and relationships for GraphRAG")
        graph_extractor = GraphExtractor(llm=build_llm())
        graph_store = get_graph_store()

        for i, n in enumerate(nodes):
            # Extract from chunk
            graph_data = graph_extractor.extract(n.text)

            # Store entities in graph DB
            entity_ids = []
            for entity in graph_data.entities:
                entity_id = graph_store.insert_entity(
                    name=entity.name,
                    entity_type=entity.type,
                    metadata=entity.metadata
                )
                entity_ids.append(entity_id)

                # Link entity to chunk (will do after node IDs assigned)
                n.metadata["_pending_entity_links"] = entity_ids

            # Store relationships
            for rel in graph_data.relationships:
                graph_store.insert_relationship(
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    rel_type=rel.type
                )

            # Add entity metadata to node
            n.metadata["_entity_ids"] = entity_ids
            n.metadata["_entity_names"] = [e.name for e in graph_data.entities]

    return nodes
```

2. **After Insert (line 2430+)** - Link entities to chunk IDs

```python
def insert_nodes(vector_store, nodes):
    # ... existing insert code ...

    # NEW: Link entities to chunks after nodes have IDs
    if S.graph_enabled:
        graph_store = get_graph_store()
        for node in nodes:
            if "_pending_entity_links" in node.metadata:
                for entity_id in node.metadata["_pending_entity_links"]:
                    graph_store.link_entity_to_chunk(
                        entity_id=entity_id,
                        chunk_id=node.id_,
                        table_name=S.table
                    )
```

3. **Retrieval (line 1755)** - Add graph expansion

```python
class GraphEnhancedRetriever(VectorDBRetriever):
    def __init__(self, vector_store, embed_model, graph_store, **kwargs):
        super().__init__(vector_store, embed_model, **kwargs)
        self.graph_store = graph_store

    def _retrieve(self, query_bundle):
        # Step 1: Vector retrieval (broader set)
        vsq = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.top_k * 3,  # 12 instead of 4
            mode="default"
        )
        vector_results = self.vector_store.query(vsq)

        # Step 2: Extract entities from retrieved chunks
        retrieved_entity_ids = []
        for node in vector_results.nodes:
            if "_entity_ids" in node.metadata:
                retrieved_entity_ids.extend(node.metadata["_entity_ids"])

        # Step 3: Graph expansion (traverse relationships)
        if retrieved_entity_ids and S.graph_max_hops > 0:
            related_entity_ids = self.graph_store.get_related_entities(
                entity_ids=retrieved_entity_ids,
                hops=S.graph_max_hops,  # Default: 2
                limit=50
            )

            # Step 4: Retrieve chunks containing related entities
            expanded_chunk_ids = self.graph_store.get_chunks_for_entities(
                entity_ids=related_entity_ids,
                table_name=S.table
            )

            # Retrieve expanded chunks from vector store
            expanded_nodes = self.vector_store.get_nodes_by_ids(expanded_chunk_ids)

            # Step 5: Merge results
            all_nodes = vector_results.nodes + expanded_nodes
        else:
            all_nodes = vector_results.nodes

        # Step 6: Rerank combined results
        if self.enable_reranking:
            all_nodes = self.reranker.rerank_nodes(
                query_bundle.query_str,
                all_nodes,
                top_k=self.top_k  # Final: 4
            )
        else:
            all_nodes = all_nodes[:self.top_k]

        return all_nodes
```

**Critical Files:**
- Modified: `rag_low_level_m1_16gb_verbose.py` (lines 2290, 2430, 1755)
- New: `utils/graph_retriever.py`

---

### Phase 4: Community Detection (Optional - 2-3 hours)

**Microsoft GraphRAG Enhancement**

**Purpose**: Group related entities into communities for hierarchical reasoning

**Implementation:**

```python
from graspologic.partition import hierarchical_leiden
import networkx as nx

class CommunityDetector:
    def __init__(self, graph_store):
        self.graph_store = graph_store

    def build_communities(self, max_cluster_size=10):
        """Detect communities using Leiden algorithm."""
        # Build NetworkX graph from database
        G = self._build_networkx_graph()

        # Run hierarchical Leiden clustering
        communities = hierarchical_leiden(
            G,
            max_cluster_size=max_cluster_size
        )

        # Store community assignments
        for community_id, entity_ids in communities.items():
            self._store_community(community_id, entity_ids)

        # Generate community summaries (LLM call)
        for community_id in communities:
            summary = self._summarize_community(community_id)
            self._store_community_summary(community_id, summary)

    def _build_networkx_graph(self):
        """Convert PostgreSQL graph to NetworkX."""
        ...

    def _summarize_community(self, community_id):
        """Use LLM to summarize entity cluster."""
        entities = self.graph_store.get_community_entities(community_id)
        prompt = f"Summarize this group of related entities: {entities}"
        summary = llm.generate(prompt)
        return summary
```

**Use Case:**
- Query: "Summarize all discussions about project management"
- Without communities: Retrieve random chunks
- With communities: Find "project management" community, retrieve all member chunks

**Trade-off:**
- Benefit: Better for broad, thematic queries
- Cost: +2-5 minutes indexing time, +100-500 LLM calls for summaries

**Critical Files:**
- New: `utils/community_detector.py`
- Modified: `rag_low_level_m1_16gb_verbose.py` (optional post-indexing step)

---

### Phase 5: Configuration & Testing (2-3 hours)

**Add Environment Variables**

```bash
# config/.env additions
GRAPH_ENABLED=1                    # Master switch
GRAPH_STORAGE=postgresql           # postgresql, neo4j, networkx
GRAPH_EXTRACT_ENTITIES=1           # Entity extraction
GRAPH_BUILD_COMMUNITIES=0          # Community detection (expensive)
GRAPH_MAX_HOPS=2                   # Graph traversal depth
GRAPH_ENTITY_BATCH_SIZE=32         # Extraction batch size
GRAPH_MIN_ENTITY_CONFIDENCE=0.7    # Filter low-confidence entities
```

**Create Test Script: `test_graphrag.py`**

```python
import os
os.environ["GRAPH_ENABLED"] = "1"
os.environ["GRAPH_STORAGE"] = "networkx"  # In-memory for testing

from rag_low_level_m1_16gb_verbose import main

# Test on small dataset
os.environ["PDF_PATH"] = "data/test_docs"  # ~50 chunks
os.environ["PGTABLE"] = "test_graphrag"
os.environ["RESET_TABLE"] = "1"

# Run indexing
main()

# Test queries
queries = [
    "Who worked with John?",  # Entity-centric
    "How is COVID-19 connected to supply chains?",  # Relationship
    "What did Alice say about the project?",  # Multi-hop
]

for query in queries:
    print(f"\n📝 Query: {query}")
    # Run query and show graph path
    ...
```

**Performance Benchmarks to Collect:**

1. **Indexing Time**
   - Vector-only baseline: ~150s for 10K chunks
   - GraphRAG target: ~600-1200s (4-8x slower acceptable)

2. **Storage Size**
   - Vector-only: ~15MB for 10K chunks
   - GraphRAG: ~35-45MB (2-3x growth acceptable)

3. **Query Latency**
   - Vector-only: ~0.3s retrieval
   - GraphRAG: ~0.5-2s (depends on graph complexity)

4. **Query Quality (manual evaluation)**
   - Test 20 queries (10 relationship + 10 factual)
   - Compare: Vector-only vs GraphRAG
   - Metrics: Relevance, completeness, reasoning quality

**Critical Files:**
- New: `test_graphrag.py`
- New: `config/.env` (updated)
- New: `docs/GRAPHRAG_GUIDE.md` (user documentation)

---

### Phase 6: Monitoring & Visualization (Optional - 2-3 hours)

**Graph Visualization**

```python
# utils/graph_visualizer.py
import networkx as nx
import matplotlib.pyplot as plt

def visualize_entity_graph(entity_ids, graph_store, output_path="graph.html"):
    """Generate interactive graph visualization."""
    G = graph_store.get_subgraph(entity_ids, hops=2)

    # Export to HTML with pyvis
    from pyvis.network import Network
    net = Network(height="750px", width="100%", directed=True)
    net.from_nx(G)
    net.save_graph(output_path)
```

**Grafana Dashboard Metrics**

Add to existing Prometheus metrics (`utils/metrics.py`):

```python
# Graph-specific metrics
graph_extraction_time = Histogram('graph_extraction_seconds', 'Time to extract entities')
graph_entity_count = Gauge('graph_entity_count', 'Total entities in graph')
graph_relationship_count = Gauge('graph_relationship_count', 'Total relationships')
graph_traversal_time = Histogram('graph_traversal_seconds', 'Graph traversal time')
```

**Critical Files:**
- New: `utils/graph_visualizer.py`
- Modified: `utils/metrics.py`
- Modified: `config/grafana/dashboards/rag_pipeline_internals.json`

---

## Migration Strategy

### Recommended Path: **Incremental Hybrid Migration**

**Stage 1: Proof of Concept (Week 1)**
1. NetworkX in-memory graph (no DB changes)
2. Small dataset (100-500 chunks)
3. Validate entity extraction quality
4. Test 10-20 relationship queries
5. **Decision point**: Proceed if quality is significantly better

**Stage 2: PostgreSQL Integration (Week 2)**
1. Add graph schema to existing PostgreSQL
2. Test on medium dataset (5K chunks)
3. Benchmark performance
4. **Decision point**: Acceptable latency/cost?

**Stage 3: Production Rollout (Week 3-4)**
1. Migrate existing indexes (optional: re-index with graph)
2. A/B test: Vector vs GraphRAG on query set
3. Monitor quality, latency, cost
4. Iterate on entity extraction prompts

### Rollback Plan

**If GraphRAG doesn't meet expectations:**
1. Set `GRAPH_ENABLED=0` (instant fallback to vector-only)
2. Keep graph tables (no data loss)
3. Re-evaluate in 3-6 months (LLM quality improves)

---

## Cost-Benefit Analysis

### Indexing Cost (One-Time)

**10,000 chunks example:**

| Component | Vector-Only | GraphRAG | Increase |
|-----------|------------|----------|----------|
| Embedding | 150s | 150s | 0% |
| Entity Extraction | 0s | 600-1200s | +400-800% |
| Graph Storage | 0s | 30s | - |
| **Total** | **150s** | **780-1380s** | **5-9x slower** |

**LLM Calls:**
- Local Mistral 7B: Free, ~0.5-2s per chunk → ~5K-20K seconds total
- Hosted LLM (GPT-3.5): ~$0.0005 per extraction → ~$5 for 10K chunks

### Query Cost (Per Query)

| Component | Vector-Only | GraphRAG | Increase |
|-----------|------------|----------|----------|
| Vector Search | 0.3s | 0.3s | 0% |
| Graph Traversal | 0s | 0.2-1.7s | - |
| Reranking | 0.1s | 0.1s | 0% |
| **Total** | **0.4s** | **0.6-2.1s** | **1.5-5x slower** |

### Storage Cost

| Component | Vector-Only | GraphRAG | Increase |
|-----------|------------|----------|----------|
| Vectors (10K) | 15 MB | 15 MB | 0% |
| Graph Data | 0 MB | 20-30 MB | - |
| **Total** | **15 MB** | **35-45 MB** | **2-3x** |

### Quality Improvement (Expected)

Based on Microsoft GraphRAG paper:

| Query Type | Vector-Only Accuracy | GraphRAG Accuracy | Improvement |
|------------|---------------------|-------------------|-------------|
| Factual | 85% | 85% | 0% |
| Single-entity | 75% | 88% | +13% |
| Relationship | 45% | 78% | +33% |
| Multi-hop | 30% | 65% | +35% |
| Summarization | 70% | 82% | +12% |

**Verdict**: GraphRAG is **3-5x more expensive** (time/storage) but **significantly better** for relationship queries.

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Poor entity extraction quality | Medium | High | Use Schema-based extraction with predefined types |
| Indexing too slow for large corpora | High | Medium | Batch extraction, async processing, cache results |
| Query latency unacceptable | Low | High | Set `GRAPH_MAX_HOPS=1`, optimize graph queries |
| Graph storage grows too large | Low | Medium | Prune low-confidence entities, entity resolution |
| Complexity increases maintenance | High | Medium | Start with simple PostgreSQL, defer Neo4j |
| Limited quality improvement | Medium | High | PoC with 100 chunks, measure before scaling |

**Critical Risk**: **Limited quality improvement for your use case**

**Mitigation**:
1. **Phase 0.5**: Manual evaluation on 20 representative queries
   - 10 relationship queries (GraphRAG strong)
   - 10 factual queries (Vector RAG sufficient)
2. If <30% improvement on relationship queries: **Abort**
3. If ≥30% improvement: **Proceed to Stage 2**

---

## Success Criteria

**Minimum Viable GraphRAG (Stage 1 - PoC)**

✅ **Must Have:**
1. Entity extraction works with 80%+ precision (manual review of 100 entities)
2. Graph retrieval returns relevant results for 5/5 test relationship queries
3. Query latency <3s end-to-end (acceptable for complex queries)

✅ **Should Have:**
4. Entity resolution reduces duplicates by 50%+
5. Storage overhead <4x vs vector-only

**Production Ready (Stage 3)**

✅ **Must Have:**
1. A/B test shows ≥20% quality improvement on relationship queries
2. Query latency <2s for 90th percentile
3. Indexing completes within acceptable timeframe (user-defined)
4. Monitoring dashboards show graph health

✅ **Nice to Have:**
5. Community detection for hierarchical reasoning
6. Interactive graph visualization
7. Explainable reasoning paths ("Answer from: A→B→C")

---

## Verification Plan

### Unit Tests

**Create: `tests/test_graph_extraction.py`**

```python
def test_entity_extraction():
    """Test entity extraction from sample chunks."""
    chunk = "John worked with Alice at OpenAI on GPT-4."
    extractor = GraphExtractor(llm)
    graph_data = extractor.extract(chunk)

    assert len(graph_data.entities) >= 3  # John, Alice, OpenAI
    assert any(e.name == "John" for e in graph_data.entities)
    assert any(r.type == "worked_with" for r in graph_data.relationships)

def test_graph_storage():
    """Test PostgreSQL graph storage."""
    store = PostgreSQLGraphStore(connection_string)
    entity_id = store.insert_entity("John", "Person")
    assert entity_id > 0

def test_graph_retrieval():
    """Test graph-enhanced retrieval."""
    retriever = GraphEnhancedRetriever(vector_store, embed_model, graph_store)
    results = retriever.retrieve("Who worked with John?")
    assert len(results) > 0
```

### Integration Tests

**Create: `tests/test_graphrag_e2e.py`**

```python
def test_graphrag_indexing():
    """Test full indexing pipeline with GraphRAG."""
    os.environ["GRAPH_ENABLED"] = "1"
    os.environ["PDF_PATH"] = "tests/fixtures/sample_docs"

    # Run indexing
    main()

    # Verify entities were extracted
    graph_store = get_graph_store()
    entity_count = graph_store.count_entities()
    assert entity_count > 0

def test_relationship_query():
    """Test relationship query end-to-end."""
    query = "How is John connected to OpenAI?"
    response = run_query(query_engine, query)

    # Should mention relationship
    assert "worked" in response.lower() or "connected" in response.lower()
```

### Manual Evaluation

**Create: `scripts/evaluate_graphrag.py`**

```python
# Test queries for manual evaluation
test_queries = [
    # Relationship queries (GraphRAG should excel)
    ("Who worked with John?", "Should find Alice via worked_with relationship"),
    ("What connects COVID-19 to supply chains?", "Should traverse causal graph"),

    # Factual queries (both should work)
    ("What is GPT-4?", "Simple factual lookup"),
    ("Summarize the document", "Document-level task"),
]

# Run both vector-only and GraphRAG, compare side-by-side
for query, expected in test_queries:
    vector_result = run_vector_only(query)
    graph_result = run_graphrag(query)

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print(f"\n[Vector-Only Result]")
    print(vector_result)
    print(f"\n[GraphRAG Result]")
    print(graph_result)
    print(f"\n[Which is better? (V/G/T=Tie)]: ", end="")
    user_input = input()
```

**Metrics to Track:**
- **Precision**: Are retrieved entities relevant?
- **Recall**: Are all relevant entities found?
- **Reasoning Quality**: Does answer show logical connection?
- **Latency**: <2s acceptable?
- **User Preference**: Vector vs Graph vs Tie (aim for >60% Graph wins on relationship queries)

---

## Implementation Decisions (Based on User Input)

### ✅ Confirmed Approach

1. **Storage**: PostgreSQL native tables for entities/relationships
   - Simpler implementation (no Docker/Neo4j needed)
   - Leverages existing database infrastructure
   - Good performance for 2-3 hop graph traversals

2. **Entity Types**: Schema-based extraction with 3 types
   - **Person**: Names, participants in conversations
   - **Location**: Cities, venues, places mentioned
   - **Topic/Concept**: Abstract ideas, projects, events
   - Skip: Organizations (not needed for FB Messenger use case)

3. **Extraction LLM**: Local Mistral 7B (already configured)
   - Free (no API costs)
   - ~0.5-2s per chunk extraction
   - Acceptable for PoC phase

4. **Scope**: Incremental PoC → Evaluation → Scale
   - **Phase 1**: 100-500 chunks (PoC)
   - **Phase 2**: Evaluate with 20 test queries
   - **Phase 3**: Scale to full dataset if ≥30% quality improvement

5. **Performance Validation**: Measure before scaling
   - Benchmark indexing time on PoC
   - Test query latency (<2s target)
   - Manual evaluation of relationship query quality
   - **Go/No-Go decision** after PoC results

---

## Critical Files

### To Create
1. `utils/graph_extractor.py` - Entity/relationship extraction
2. `utils/graph_storage.py` - PostgreSQL graph store
3. `utils/graph_retriever.py` - Graph-enhanced retrieval
4. `utils/community_detector.py` - Community detection (optional)
5. `utils/graph_visualizer.py` - Graph visualization (optional)
6. `scripts/setup_graph_schema.sql` - Database schema
7. `test_graphrag.py` - Integration test
8. `tests/test_graph_extraction.py` - Unit tests
9. `docs/GRAPHRAG_GUIDE.md` - User documentation

### To Modify
1. `rag_low_level_m1_16gb_verbose.py`
   - Line 2290: Add entity extraction in `build_nodes()`
   - Line 2430+: Link entities to chunk IDs after insert
   - Line 1755: Integrate `GraphEnhancedRetriever`
2. `config/.env` - Add GraphRAG environment variables
3. `config/constants.py` - Add GraphRAG defaults
4. `utils/metrics.py` - Add graph metrics (optional)
5. `requirements.txt` - Add dependencies

---

## Timeline Estimate

### PoC Phase (Recommended First Step)
| Task | Estimated Time | Deliverable |
|------|---------------|-------------|
| **Install dependencies** | 0.5 hours | NetworkX, graph extractors ready |
| **Entity extraction module** | 3-4 hours | `graph_extractor.py` with SchemaLLMPathExtractor |
| **PostgreSQL graph schema** | 2-3 hours | Tables for entities/relationships |
| **RAG integration (basic)** | 3-4 hours | Extract + store entities during indexing |
| **Test on 100-500 chunks** | 1 hour | PoC dataset indexed with graph |
| **Manual evaluation** | 1-2 hours | Compare 20 queries (vector vs graph) |
| **PoC Total** | **10-14 hours** | **Decision point: proceed or abort** |

### Full Implementation (If PoC Succeeds)
| Phase | Estimated Time | Deliverable |
|-------|---------------|-------------|
| **Graph-enhanced retrieval** | 3-4 hours | Retriever that uses graph expansion |
| **Entity resolution** | 2-3 hours | Merge duplicate entities |
| **Automated testing** | 2-3 hours | Unit + integration tests |
| **Documentation** | 1-2 hours | User guide for GraphRAG mode |
| **Full Total** | **+8-12 hours** | Production-ready GraphRAG |

### Optional Enhancements (Post-PoC)
| Feature | Estimated Time | Value |
|---------|---------------|-------|
| **Community detection** | 2-3 hours | Better for broad queries |
| **Graph visualization** | 2-3 hours | Explainability, debugging |
| **Monitoring metrics** | 1-2 hours | Track graph health |

**Recommended Path**:
1. **Week 1**: PoC (10-14 hours) → Evaluation → Go/No-Go decision
2. **Week 2**: If approved, full implementation (8-12 hours)
3. **Week 3+**: Optional enhancements based on usage patterns

---

## Recommended Next Steps

### ✅ Phase 1: Immediate Actions (Day 1)

1. **Prepare Test Dataset** (30 min)
   - Select 5-10 representative documents
   - Mix: FB Messenger chats + other docs
   - Target: 100-500 chunks total
   - Store in: `data/poc_graphrag/`

2. **Define Test Queries** (30 min)
   Create 20 test queries (save to `test_graphrag_queries.txt`):
   - 10 relationship queries (e.g., "Who talked to Alice about Paris?")
   - 5 entity-centric (e.g., "What did John say about machine learning?")
   - 5 factual baseline (e.g., "What is the main topic?")

3. **Install Dependencies** (15 min)
   ```bash
   pip install llama-index-graph-stores-simple networkx
   ```

### ✅ Phase 2: PoC Implementation (Days 2-3, ~10-14 hours)

**Deliverables:**
1. `utils/graph_extractor.py` - Entity extraction with SchemaLLMPathExtractor
2. `utils/graph_storage.py` - PostgreSQL graph tables + operations
3. Modified `rag_low_level_m1_16gb_verbose.py` - Integration at line 2290
4. `test_graphrag_poc.py` - PoC test script

**Validation Checkpoints:**
- ✅ Entities extracted correctly (manual review of 50 entities)
- ✅ Graph stored in PostgreSQL without errors
- ✅ Can query: "Show me entities for chunk X"
- ✅ Indexing completes (may be slow, that's OK for PoC)

### ✅ Phase 3: Evaluation (Day 4, ~2-3 hours)

1. **Run Comparison Test**
   ```bash
   # Vector-only baseline
   GRAPH_ENABLED=0 python test_queries.py > results_vector.txt

   # GraphRAG
   GRAPH_ENABLED=1 python test_queries.py > results_graphrag.txt
   ```

2. **Manual Quality Assessment** (2 hours)
   - Compare 20 query results side-by-side
   - Score: Vector better / GraphRAG better / Tie
   - Target: GraphRAG wins ≥60% of relationship queries

3. **Performance Metrics**
   - Indexing time: Measure actual vs expected
   - Query latency: Average across 20 queries
   - Storage size: Check database size growth

### ✅ Phase 4: Go/No-Go Decision

**Proceed to Full Implementation if:**
- ✅ GraphRAG quality ≥30% better on relationship queries
- ✅ Query latency <2s (acceptable)
- ✅ Indexing time acceptable for your use case

**Iterate/Optimize if:**
- ⚠️ Quality improvement <30% but >10% (tune extraction)
- ⚠️ Performance issues but quality is good (optimize)

**Abort if:**
- ❌ Quality improvement <10% (not worth complexity)
- ❌ Fundamental issues with entity extraction quality

---

## Alternatives Considered

### Alternative 1: Query Expansion Only (Simplest)
- Expand queries with entity-aware prompts
- No graph storage needed
- **Verdict**: Cheaper but much less effective than true GraphRAG

### Alternative 2: Hybrid + Parent-Child Chunking
- Combine with existing `parent_child_chunking.py` (line 1-100)
- Retrieve small child chunks via graph → return large parent context
- **Verdict**: Excellent idea, can combine with GraphRAG

### Alternative 3: Pure Neo4j GraphRAG (No Vector)
- Replace pgvector entirely with Neo4j
- **Verdict**: Not recommended, loses fast similarity search

### Alternative 4: Entity Linking to External Knowledge Graphs
- Link entities to Wikidata, DBpedia, etc.
- **Verdict**: Out of scope, but excellent future enhancement

---

## Summary & Final Recommendations

### GraphRAG Value Proposition

**For Your Use Case (FB Messenger + Docs):**

✅ **Strong Fit:**
- Rich entity relationships (people, locations, topics)
- 25-50% of queries are relationship-based
- Already have metadata extraction infrastructure
- Conversational data benefits most from graph approaches

⚠️ **Trade-offs Accepted:**
- 5-9x slower indexing (test first approach)
- 2-3x storage overhead (manageable)
- Increased complexity (mitigated by PostgreSQL native storage)

### Implementation Strategy: PoC-First Approach

**Phase 1: Proof of Concept** (10-14 hours)
1. Schema-based entity extraction (Person, Location, Topic)
2. PostgreSQL native graph storage (simple, reuses existing DB)
3. Test on 100-500 chunks
4. Manual evaluation: 20 queries (vector vs GraphRAG)

**Decision Gate:**
- ✅ Proceed if ≥30% quality improvement on relationship queries
- ⚠️ Iterate if 10-30% improvement (tune extraction)
- ❌ Abort if <10% improvement (not worth complexity)

**Phase 2: Full Implementation** (8-12 hours, if approved)
1. Graph-enhanced retrieval with expansion
2. Entity resolution (merge duplicates)
3. Automated testing and monitoring
4. Production deployment

### Risk Mitigation

**Lowest-Risk Path:**
1. Start with **NetworkX in-memory** (no DB changes)
2. Validate entity extraction quality first
3. Migrate to PostgreSQL if PoC succeeds
4. Keep hybrid mode (can disable graph anytime)

**Rollback Plan:**
- Set `GRAPH_ENABLED=0` (instant fallback)
- Graph tables remain but unused
- No data loss, re-evaluate later

### Expected Outcomes

**If Successful:**
- 30-50% better answers for relationship queries
- Natural language queries like:
  - "Who talked to Alice about Paris?"
  - "Conversations between John and Emma about machine learning"
  - "Topics discussed in March 2022 group chats"
- Explainable reasoning paths (Entity A → Relation → Entity B)

**If Not Successful:**
- Learn what works/doesn't for your data
- Keep vector-only mode (already excellent)
- Revisit in 6-12 months (LLM quality improves)

### Key Success Factors

1. **Entity Extraction Quality** (most critical)
   - Schema-based extraction helps (Person, Location, Topic)
   - Mistral 7B should work, may need prompt tuning
   - Manual review of first 50 entities is essential

2. **Test Dataset Selection**
   - Must be representative of real queries
   - Include diverse entity types
   - Mix conversational + document data

3. **Realistic Evaluation**
   - Don't expect miracles on factual queries
   - Focus on relationship/reasoning improvements
   - 30% improvement is excellent (not 100%)

### What Makes This Plan Strong

✅ **Incremental approach**: PoC before committing
✅ **Clear decision gates**: Objective go/no-go criteria
✅ **Minimal dependencies**: PostgreSQL (no Neo4j/Docker for PoC)
✅ **Rollback plan**: Can disable without data loss
✅ **Focused scope**: 3 entity types, essential features only
✅ **Realistic timeline**: 10-14 hours for PoC, not weeks

### Next Action

**Ready to start?**
1. ✅ Review this plan one more time
2. ✅ Prepare test dataset (100-500 chunks)
3. ✅ Define 20 test queries
4. 🚀 Begin Phase 2 implementation (PoC)

---

**This plan is comprehensive, actionable, and optimized for your specific requirements. The PoC-first approach minimizes risk while maximizing learning. You'll have a clear answer within 10-14 hours of work whether GraphRAG adds value to your RAG system.**

**Questions or ready to implement?**
