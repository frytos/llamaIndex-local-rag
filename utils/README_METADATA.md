# Enhanced Metadata Extraction Module

## Quick Start

```python
from utils.metadata_extractor import DocumentMetadataExtractor, enhance_node_metadata

# Option 1: Direct extraction
extractor = DocumentMetadataExtractor()
metadata = extractor.extract_all_metadata(text, doc_format="pdf")
print(metadata.structure["doc_type"])    # "research_paper"
print(metadata.semantic["keywords"])      # ["python", "ml", "data"]
print(metadata.technical["has_code"])     # True
print(metadata.quality["reading_level"])  # "moderate"

# Option 2: Enhance existing node metadata
base_metadata = {"source": "doc.pdf", "format": "pdf"}
enhanced = enhance_node_metadata(text, base_metadata)
# Returns: base_metadata + all extracted metadata with prefixes
```

## What It Extracts

### 1. Structure Metadata (prefix: `struct_`)
- **Headings**: H1-H6 from Markdown/HTML
- **Document type**: research_paper, manual, tutorial, api_doc, code, blog_post
- **Section titles**: Extracted from first heading
- **Chunk position**: Position within document (e.g., "3/10")

### 2. Semantic Metadata (prefix: `sem_`)
- **Keywords**: Top 10 frequent important words (stopwords filtered)
- **Topics**: TF-IDF based topic extraction (top 5)
- **Named entities**: Technical terms (Python, PyTorch, AWS, etc.) and proper nouns

### 3. Technical Metadata (prefix: `tech_`)
- **Code blocks**: Detects markdown/HTML code blocks
- **Tables**: Markdown, HTML, space-separated
- **Equations**: LaTeX math (inline & display)
- **Code analysis** (for .py, .js, .ts, etc.):
  - Function/class names
  - Import statements
  - Programming language

### 4. Quality Signals (prefix: `qual_`)
- **Word count**: Total words
- **Sentence count**: Total sentences
- **Average sentence length**: Words per sentence
- **Reading level**: very_easy, easy, moderate, difficult, very_difficult
- **Character count**: Total characters

## Configuration

Control via environment variables (see `config/.env.example`):

```bash
# Enable/disable all enhanced extraction
EXTRACT_ENHANCED_METADATA=1  # Default: 1

# Fine-grained control
EXTRACT_TOPICS=1             # TF-IDF topics (expensive: ~10ms/chunk)
EXTRACT_ENTITIES=1           # Named entity detection
EXTRACT_CODE_BLOCKS=1        # Code/function/class extraction
EXTRACT_TABLES=1             # Table detection
```

### Performance Tuning

**Default (all features)**: ~15-20ms per chunk

**Fast mode** (disable expensive features):
```bash
EXTRACT_TOPICS=0             # Saves ~10ms
EXTRACT_ENTITIES=0           # Saves ~2ms
# Total: ~5-8ms per chunk
```

## Integration with build_nodes()

### Method 1: Direct Integration (Recommended)

```python
from utils.metadata_extractor import DocumentMetadataExtractor

def build_nodes(docs: List[Any], chunks: List[str], doc_idxs: List[int]) -> List[TextNode]:
    """Create TextNode objects with enhanced metadata."""
    # Initialize once (not per chunk!)
    extractor = DocumentMetadataExtractor()

    nodes: List[TextNode] = []
    for i, chunk in enumerate(chunks):
        n = TextNode(text=chunk)

        # Get base metadata from source document
        src_doc = docs[doc_idxs[i]]
        n.metadata = src_doc.metadata.copy() if src_doc.metadata else {}

        # Add enhanced metadata
        if extractor.enabled:
            doc_format = n.metadata.get("format", "txt")
            doc_metadata = extractor.extract_all_metadata(
                chunk,
                doc_format=doc_format,
                chunk_position=(i+1, len(chunks))
            )
            n.metadata.update(doc_metadata.to_dict())

        # Add chunking parameters
        n.metadata["_chunk_size"] = S.chunk_size
        n.metadata["_chunk_overlap"] = S.chunk_overlap

        nodes.append(n)

    return nodes
```

### Method 2: Using Convenience Function

```python
from utils.metadata_extractor import enhance_node_metadata

def build_nodes(docs: List[Any], chunks: List[str], doc_idxs: List[int]) -> List[TextNode]:
    """Create TextNode objects with enhanced metadata."""
    nodes: List[TextNode] = []

    for i, chunk in enumerate(chunks):
        n = TextNode(text=chunk)

        # Get base metadata
        src_doc = docs[doc_idxs[i]]
        base_metadata = src_doc.metadata.copy() if src_doc.metadata else {}

        # Enhance with extracted metadata
        n.metadata = enhance_node_metadata(chunk, base_metadata)

        # Add chunking parameters
        n.metadata["_chunk_size"] = S.chunk_size
        n.metadata["_chunk_overlap"] = S.chunk_overlap

        nodes.append(n)

    return nodes
```

## Use Cases

### 1. Improved Retrieval Quality
Rich metadata enables better filtering and ranking:
```python
# Filter by document type
results = retriever.retrieve("python tutorial", filters={"struct_doc_type": "tutorial"})

# Filter by technical content
results = retriever.retrieve("ML code", filters={"tech_has_code": True})

# Filter by reading level
results = retriever.retrieve("simple explanation", filters={"qual_reading_level": "easy"})
```

### 2. Better Reranking
Use metadata for smarter reranking:
```python
def rerank_with_metadata(results, query):
    for result in results:
        score = result.score

        # Boost code results for "how to" queries
        if "how to" in query.lower() and result.metadata.get("tech_has_code"):
            score *= 1.2

        # Boost tutorials for learning queries
        if "learn" in query.lower() and result.metadata.get("struct_doc_type") == "tutorial":
            score *= 1.3

        result.score = score

    return sorted(results, key=lambda x: x.score, reverse=True)
```

### 3. Document Classification
Automatic document organization:
```python
# Group chunks by document type
from collections import defaultdict

chunks_by_type = defaultdict(list)
for node in nodes:
    doc_type = node.metadata.get("struct_doc_type", "unknown")
    chunks_by_type[doc_type].append(node)

print(f"Tutorials: {len(chunks_by_type['tutorial'])}")
print(f"Code: {len(chunks_by_type['code'])}")
print(f"Research papers: {len(chunks_by_type['research_paper'])}")
```

### 4. Quality-Based Filtering
Filter low-quality chunks:
```python
# Remove very short chunks (likely noise)
quality_nodes = [
    n for n in nodes
    if n.metadata.get("qual_word_count", 0) >= 20
]

# Keep only chunks with moderate-to-difficult reading level for technical docs
technical_nodes = [
    n for n in nodes
    if n.metadata.get("qual_reading_level") in ["moderate", "difficult", "very_difficult"]
]
```

## Examples

See comprehensive examples in:
- **Demo script**: `examples/metadata_extraction_demo.py`
- **Unit tests**: `tests/test_metadata_extractor.py`
- **Full documentation**: `docs/METADATA_EXTRACTOR.md`

Run demo:
```bash
python examples/metadata_extraction_demo.py
```

Run tests:
```bash
python -m pytest tests/test_metadata_extractor.py -v
```

## Troubleshooting

### Issue: NLTK data not found
**Solution**: Auto-downloads on first use. If it fails:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Issue: Extraction too slow
**Solution**: Disable expensive features:
```bash
export EXTRACT_TOPICS=0        # Most expensive
export EXTRACT_ENTITIES=0
```

### Issue: Metadata not appearing
**Solution**: Check environment variable:
```bash
export EXTRACT_ENHANCED_METADATA=1
```

And verify metadata keys:
```python
print(node.metadata.keys())  # Should see struct_*, sem_*, tech_*, qual_* keys
```

## Architecture

```
DocumentMetadataExtractor
├── extract_all_metadata()           # Main entry point
│   ├── extract_structure_metadata()  # Headings, doc type
│   ├── extract_semantic_metadata()   # Keywords, topics, entities
│   ├── extract_technical_metadata()  # Code, tables, equations
│   └── extract_quality_signals()     # Word count, reading level
├── _classify_document_type()         # Pattern-based classification
├── _extract_keywords()               # Frequency-based keywords
├── _extract_topics_tfidf()          # TF-IDF topic extraction
├── _extract_entities()               # Technical entity detection
├── _detect_code_blocks()             # Code block patterns
├── _extract_function_names()         # Language-specific parsing
├── _extract_class_names()            # Class extraction
├── _extract_imports()                # Import statement parsing
├── _count_tables()                   # Table detection
└── _count_equations()                # LaTeX equation detection

DocumentMetadata (dataclass)
├── structure: Dict                   # Structure metadata
├── semantic: Dict                    # Semantic metadata
├── technical: Dict                   # Technical metadata
├── quality: Dict                     # Quality signals
└── to_dict()                         # Flatten with prefixes

MetadataExtractor (legacy)
└── extract()                         # Basic metadata (backward compat)
```

## Dependencies

All dependencies already in `requirements.txt`:
- `nltk`: Tokenization, stopwords (auto-downloaded)
- `scikit-learn`: TF-IDF topic extraction
- Standard library: `re`, `collections`, `dataclasses`

## Performance

Benchmark on M1 Mac with 1000 chunks:

| Operation | Time/Chunk | Notes |
|-----------|------------|-------|
| Structure | ~2ms | Fast (regex) |
| Semantic (no topics) | ~5ms | Keywords + entities |
| Semantic (with TF-IDF) | ~15ms | +10ms for TF-IDF |
| Technical | ~3ms | Fast (regex) |
| Quality | ~1ms | Very fast |
| **Total (all features)** | **~15-20ms** | Acceptable |
| **Total (fast mode)** | **~5-8ms** | Topics disabled |

## See Also

- **Full Documentation**: `docs/METADATA_EXTRACTOR.md`
- **Demo Script**: `examples/metadata_extraction_demo.py`
- **Unit Tests**: `tests/test_metadata_extractor.py`
- **Environment Config**: `config/.env.example`
