---
description: Auto-generate comprehensive feature documentation
---

# Document Feature

Generate comprehensive documentation for a feature, including overview, architecture, usage examples, API documentation, and troubleshooting guide.

## Benefits
- **Complete Coverage**: All aspects of feature documented
- **Consistent Format**: Follows documentation standards
- **Time Saving**: Automated generation from code
- **Up-to-Date**: Generated from current implementation
- **Examples Included**: Real usage examples extracted from code

## When to Use
- After completing a new feature
- When documentation is missing or outdated
- Before feature release
- For onboarding new team members
- As part of PR requirements

## Usage
```bash
/document-feature FeatureName
/document-feature retrieval
/document-feature embedding-pipeline
```

## Generated Documentation Structure

### 1. Feature Overview
```markdown
# Feature Name

## Overview
Clear description of what the feature does and why it exists.

## Key Benefits
- Benefit 1: Description
- Benefit 2: Description

## Status
- **Version**: 1.0.0
- **Stability**: Stable | Beta | Experimental
- **Last Updated**: YYYY-MM-DD

## Quick Start
```python
# Basic usage example
```
```

### 2. Architecture
```markdown
## Architecture

### Components
- **Component A**: Responsibility and description
- **Component B**: Responsibility and description

### Data Flow
```
Document -> Chunking -> Embedding -> Storage -> Retrieval -> LLM -> Answer
```

### Dependencies
- **External**: LlamaIndex, pgvector, llama.cpp
- **Internal**: List of internal modules used

### File Structure
```
src/
├── rag_low_level_m1_16gb_verbose.py  # Main pipeline
├── rag_web.py                         # Web UI
└── rag_interactive.py                 # CLI interface
```
```

### 3. Setup & Configuration
```markdown
## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector
- llama.cpp compatible model

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
```python
# Environment variables
CHUNK_SIZE=700
CHUNK_OVERLAP=150
TOP_K=4
```

### Environment Variables
```env
PGHOST=localhost
PGPORT=5432
EMBED_MODEL=BAAI/bge-small-en
```
```

### 4. Usage Guide
```markdown
## Usage

### Basic Usage
```python
from rag_low_level_m1_16gb_verbose import (
    load_documents,
    chunk_documents,
    build_nodes,
    embed_nodes,
)

docs = load_documents("data/document.pdf")
chunks, doc_idxs = chunk_documents(docs)
nodes = build_nodes(docs, chunks, doc_idxs)
embed_nodes(embed_model, nodes)
```

### Advanced Usage
```python
# Custom configuration
S.chunk_size = 500
S.chunk_overlap = 100
S.top_k = 5
```

### Common Patterns
1. **Pattern 1**: Description and example
2. **Pattern 2**: Description and example
```

### 5. API Documentation
```markdown
## API Reference

### Functions

#### `load_documents(doc_path: str) -> List[Document]`
Load documents from file or folder.

**Parameters:**
- `doc_path`: Path to document or folder

**Returns:**
List of LlamaIndex Document objects

**Example:**
```python
docs = load_documents("data/llama2.pdf")
```

#### `chunk_documents(docs: List[Document]) -> Tuple[List[str], List[int]]`
Split documents into chunks.

**Parameters:**
- `docs`: List of documents to chunk

**Returns:**
Tuple of (chunks, doc_indices)

**Example:**
```python
chunks, doc_idxs = chunk_documents(docs)
```

### Classes

#### `VectorDBRetriever`
Custom retriever for pgvector queries.

**Constructor:**
```python
VectorDBRetriever(
    vector_store: PGVectorStore,
    embed_model: HuggingFaceEmbedding,
    similarity_top_k: int
)
```

**Methods:**
- `_retrieve(query_bundle)`: Retrieve relevant nodes
```

### 6. Troubleshooting
```markdown
## Troubleshooting

### Common Issues

#### Issue: "Connection refused" error
**Symptoms:**
- Cannot connect to PostgreSQL
- Database operations fail

**Possible Causes:**
1. PostgreSQL not running
2. Wrong connection parameters
3. Firewall blocking connection

**Solution:**
```bash
# Check PostgreSQL is running
docker-compose ps

# Verify connection
PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "SELECT 1;"
```

#### Issue: Context window overflow
**Symptoms:**
- LLM generation fails
- Error about token limit

**Solution:**
Reduce chunk_size or TOP_K:
```bash
CTX=4096 TOP_K=3 CHUNK_SIZE=500 python rag_low_level_m1_16gb_verbose.py
```

### Debug Mode
Enable verbose logging:
```bash
LOG_LEVEL=DEBUG python rag_low_level_m1_16gb_verbose.py
```

### Getting Help
- Check console for error messages
- Review this documentation
- Check CLAUDE.md for project guidelines
```

### 7. Performance Considerations
```markdown
## Performance

### Optimization Tips
1. **GPU Offload**: Use N_GPU_LAYERS=24 for M1 Mac
2. **Batching**: Increase EMBED_BATCH for faster embedding
3. **Indexing**: Create HNSW indexes for faster retrieval

### Benchmarks (M1 Mac Mini 16GB)
- Embedding: ~67 chunks/second
- Retrieval: ~0.3 seconds
- Generation: ~10 tokens/second

### Best Practices
- Use appropriate chunk_size for your content type
- Create HNSW indexes for tables >1000 rows
- Monitor memory usage with psutil
```

## Example Output

```markdown
# RAG Retrieval Feature

## Overview
The retrieval feature finds relevant document chunks for a given query using semantic similarity search with pgvector.

## Key Benefits
- Semantic search (not just keyword matching)
- Fast retrieval with vector indexes
- Configurable similarity thresholds
- Source attribution for answers

## Quick Start
```python
retriever = VectorDBRetriever(
    vector_store=store,
    embed_model=model,
    similarity_top_k=4
)
results = retriever._retrieve(QueryBundle("What is RAG?"))
```

[... continues with full documentation ...]
```

## Interactive Generation

```
User: /document-feature retrieval

Claude: I'll document the retrieval feature. Let me analyze the code...

Found:
- Classes: VectorDBRetriever
- Functions: _retrieve, make_vector_store
- Files: rag_low_level_m1_16gb_verbose.py

Generating documentation...

- Feature Overview
- Architecture
- Setup Guide
- Usage Examples (8 examples)
- API Reference (1 class, 3 functions)
- Troubleshooting (4 common issues)
- Performance Notes

Documentation generated: docs/features/retrieval.md

Would you like me to:
1. Add more examples?
2. Include diagrams?
3. Add to CLAUDE.md?
```

## Best Practices

### Before Generating
1. Ensure feature is complete and tested
2. Add docstrings to functions
3. Write meaningful variable names
4. Include usage examples in comments

### After Generating
1. Review generated documentation
2. Add project-specific context
3. Include diagrams if helpful
4. Review with team

### Maintenance
1. Regenerate after major changes
2. Keep examples up-to-date
3. Update troubleshooting section
4. Version documentation with releases
