# Enhanced Metadata Extraction Module

**Location**: `/Users/frytos/code/llamaIndex-local-rag/utils/metadata_extractor.py`

## Overview

The metadata extraction module provides rich metadata extraction from documents to improve RAG (Retrieval-Augmented Generation) retrieval quality. It extracts:

- **Structure metadata**: Sections, headings, document type classification
- **Semantic metadata**: Topics (TF-IDF), keywords, named entities
- **Technical metadata**: Code blocks, tables, equations, function/class names
- **Quality signals**: Word count, sentence count, reading level

## Features

### 1. DocumentMetadataExtractor (Enhanced Extractor)

The main class for comprehensive metadata extraction:

```python
from utils.metadata_extractor import DocumentMetadataExtractor

extractor = DocumentMetadataExtractor()
metadata = extractor.extract_all_metadata(text, doc_format="pdf")
```

#### Structure Metadata

Detects document structure and classifies document types:

- **Headings**: Extracts H1, H2, H3 from Markdown, HTML, or generic text
- **Document types**: research_paper, manual, tutorial, api_doc, code, blog_post, general
- **Chunk position**: Tracks position within larger documents (e.g., "2/5")
- **Section titles**: Identifies the section the chunk belongs to

**Example output**:
```python
{
    "format": "md",
    "has_headings": True,
    "heading_count": 3,
    "doc_type": "tutorial",
    "section_title": "Getting Started",
    "headings": [
        {"level": 1, "title": "Getting Started"},
        {"level": 2, "title": "Installation"},
        {"level": 2, "title": "Configuration"}
    ]
}
```

#### Semantic Metadata

Extracts semantic information for better retrieval:

- **Keywords**: Top 10 most frequent important words (stopwords filtered)
- **Topics**: TF-IDF based topic extraction (top 5 topics)
- **Named entities**: Detects technical tools, technologies, and proper nouns

**Example output**:
```python
{
    "keywords": ["python", "django", "flask", "tutorial", "programming"],
    "topics": ["python programming", "web framework", "django tutorial"],
    "entities": ["python", "django", "flask", "postgresql", "docker"],
    "entity_count": 5
}
```

#### Technical Metadata

Detects technical content:

- **Code blocks**: Markdown, HTML, or indented code blocks
- **Tables**: Markdown, HTML, or space-separated tables
- **Equations**: LaTeX math equations (inline and display)
- **Code analysis** (for .py, .js, .ts, .java, .cpp, .go, .rs files):
  - Function names
  - Class names
  - Import statements
  - Programming language

**Example output (code file)**:
```python
{
    "has_code": True,
    "code_block_count": 5,
    "programming_language": "py",
    "functions": ["main", "train_model", "evaluate", "preprocess"],
    "function_count": 4,
    "classes": ["DataLoader", "ModelTrainer"],
    "class_count": 2,
    "imports": ["numpy", "torch", "sklearn"],
    "import_count": 3,
    "has_tables": True,
    "table_count": 2,
    "has_equations": True,
    "equation_count": 3
}
```

#### Quality Signals

Computes text quality metrics:

- **Word count**: Total words in text
- **Sentence count**: Total sentences
- **Average sentence length**: Words per sentence
- **Reading level**: very_easy, easy, moderate, difficult, very_difficult
- **Character count**: Total characters

**Example output**:
```python
{
    "word_count": 125,
    "sentence_count": 8,
    "avg_sentence_length": 15.6,
    "reading_level": "moderate",
    "char_count": 782
}
```

### 2. MetadataExtractor (Basic Extractor)

Legacy extractor for backward compatibility, focused on chat logs and basic metadata:

```python
from utils.metadata_extractor import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract(text)
```

Extracts:
- Dates (various formats)
- Email addresses
- URLs
- Participants (for chat logs)
- Content type classification
- Basic text statistics

### 3. Convenience Function: enhance_node_metadata()

For easy integration with `build_nodes()`:

```python
from utils.metadata_extractor import enhance_node_metadata

metadata = {"source": "doc.pdf", "format": "pdf"}
enhanced = enhance_node_metadata(text, metadata)
```

## Configuration

Control extraction behavior via environment variables:

```bash
# Enable/disable enhanced metadata extraction
EXTRACT_ENHANCED_METADATA=1  # Default: 1 (enabled)

# Enable/disable specific features
EXTRACT_TOPICS=1              # TF-IDF topic extraction (default: 1)
EXTRACT_ENTITIES=1            # Named entity recognition (default: 1)
EXTRACT_CODE_BLOCKS=1         # Code block detection (default: 1)
EXTRACT_TABLES=1              # Table detection (default: 1)
```

### Environment Variable Examples

```bash
# Disable all enhanced metadata (use basic extractor only)
export EXTRACT_ENHANCED_METADATA=0

# Enable only structure and quality, disable semantic analysis
export EXTRACT_TOPICS=0
export EXTRACT_ENTITIES=0

# Optimize for speed (disable expensive operations)
export EXTRACT_TOPICS=0
export EXTRACT_ENTITIES=0
export EXTRACT_TABLES=0
```

## Integration with RAG Pipeline

### Option 1: Direct Integration in build_nodes()

Modify `rag_low_level_m1_16gb_verbose.py`:

```python
from utils.metadata_extractor import DocumentMetadataExtractor

def build_nodes(docs: List[Any], chunks: List[str], doc_idxs: List[int]) -> List[TextNode]:
    """Create TextNode objects with enhanced metadata."""
    log.info("Building TextNode objects (text + metadata)")

    # Initialize extractor (only once)
    extractor = DocumentMetadataExtractor() if os.getenv("EXTRACT_ENHANCED_METADATA", "1") == "1" else None

    nodes: List[TextNode] = []

    for i, chunk in enumerate(chunks):
        n = TextNode(text=chunk)

        # Start with source document metadata
        src_doc = docs[doc_idxs[i]]
        n.metadata = src_doc.metadata.copy() if src_doc.metadata else {}

        # Add enhanced metadata (if enabled)
        if extractor:
            doc_format = n.metadata.get("format", "txt")
            doc_metadata = extractor.extract_all_metadata(
                chunk,
                doc_format=doc_format,
                chunk_position=(i+1, len(chunks))
            )
            n.metadata.update(doc_metadata.to_dict())

        # Add existing chunking parameters
        n.metadata["_chunk_size"] = S.chunk_size
        n.metadata["_chunk_overlap"] = S.chunk_overlap
        n.metadata["_embed_model"] = S.embed_model_name

        nodes.append(n)

    return nodes
```

### Option 2: Using enhance_node_metadata()

```python
from utils.metadata_extractor import enhance_node_metadata

def build_nodes(docs: List[Any], chunks: List[str], doc_idxs: List[int]) -> List[TextNode]:
    """Create TextNode objects with enhanced metadata."""
    nodes: List[TextNode] = []

    for i, chunk in enumerate(chunks):
        n = TextNode(text=chunk)

        # Start with source document metadata
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

## Usage Examples

### Example 1: Extract metadata from Markdown tutorial

```python
from utils.metadata_extractor import DocumentMetadataExtractor

text = """
# Python Tutorial: Getting Started

## Introduction

Welcome to this Python tutorial using Django and Flask.

## Installation

First, install Python:

```python
pip install django flask
```
"""

extractor = DocumentMetadataExtractor()
metadata = extractor.extract_all_metadata(text, doc_format="md")

print(metadata.structure["doc_type"])  # "tutorial"
print(metadata.semantic["keywords"])    # ["python", "django", "flask", ...]
print(metadata.technical["has_code"])   # True
print(metadata.quality["reading_level"])  # "easy"
```

### Example 2: Extract metadata from Python code

```python
code = """
import numpy as np
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def train(self, X, y):
        pass
"""

metadata = extractor.extract_all_metadata(code, doc_format="py")

print(metadata.structure["doc_type"])  # "code"
print(metadata.technical["functions"])  # ["__init__", "train"]
print(metadata.technical["classes"])    # ["NeuralNetwork"]
print(metadata.technical["imports"])    # ["numpy", "sklearn.metrics"]
```

### Example 3: Extract metadata from research paper

```python
paper = """
Abstract: This paper presents a novel approach to machine learning.

1. Introduction

Recent advances in deep learning using PyTorch and TensorFlow...

$$\\alpha = \\beta^2 + \\gamma$$

| Model | Accuracy |
|-------|----------|
| CNN   | 95.2%    |
"""

metadata = extractor.extract_all_metadata(paper, doc_format="pdf")

print(metadata.structure["doc_type"])       # "research_paper"
print(metadata.semantic["entities"])        # ["pytorch", "tensorflow"]
print(metadata.technical["has_equations"])  # True
print(metadata.technical["has_tables"])     # True
```

### Example 4: Batch processing with logging

```python
import logging

logging.basicConfig(level=logging.INFO)

extractor = DocumentMetadataExtractor()
documents = [
    ("tutorial.md", tutorial_text, "md"),
    ("code.py", code_text, "py"),
    ("paper.pdf", paper_text, "pdf"),
]

for filename, text, doc_format in documents:
    metadata = extractor.extract_all_metadata(text, doc_format=doc_format)

    print(f"\n{filename}:")
    print(f"  Type: {metadata.structure['doc_type']}")
    print(f"  Keywords: {', '.join(metadata.semantic.get('keywords', []))}")
    print(f"  Quality: {metadata.quality['word_count']} words, "
          f"{metadata.quality['reading_level']} level")
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/test_metadata_extractor.py -v

# Run specific test
python -m pytest tests/test_metadata_extractor.py::TestDocumentMetadataExtractor::test_extract_structure_metadata_markdown -v

# Run with coverage
python -m pytest tests/test_metadata_extractor.py --cov=utils.metadata_extractor
```

Run interactive demo:

```bash
# Test basic extractor
python utils/metadata_extractor.py

# Test enhanced extractor
python utils/metadata_extractor.py --enhanced
```

## Dependencies

### Required (already in requirements.txt)
- `nltk`: Tokenization and stopwords (auto-downloaded)
- `scikit-learn`: TF-IDF topic extraction

### Optional
- `spacy`: For advanced NER (not currently used, but can be added)

## Performance

### Benchmark (M1 Mac, 1000 chunks)

| Operation | Time per Chunk | Notes |
|-----------|----------------|-------|
| Structure extraction | ~2ms | Fast (regex-based) |
| Semantic extraction | ~5ms | Keywords only |
| Semantic + Topics (TF-IDF) | ~15ms | Slower with TF-IDF |
| Technical extraction | ~3ms | Fast (regex-based) |
| Quality signals | ~1ms | Very fast |
| **Total (all features)** | ~15-20ms | Acceptable for most use cases |

### Performance Tips

1. **Disable expensive features** if not needed:
   ```bash
   export EXTRACT_TOPICS=0        # Saves ~10ms per chunk
   export EXTRACT_ENTITIES=0      # Saves ~2ms per chunk
   ```

2. **Batch initialize extractor** once, not per chunk:
   ```python
   # Good (fast)
   extractor = DocumentMetadataExtractor()
   for chunk in chunks:
       metadata = extractor.extract_all_metadata(chunk)

   # Bad (slow)
   for chunk in chunks:
       extractor = DocumentMetadataExtractor()  # Downloads NLTK data each time!
       metadata = extractor.extract_all_metadata(chunk)
   ```

3. **Profile your pipeline**:
   ```python
   import time

   start = time.time()
   metadata = extractor.extract_all_metadata(text)
   print(f"Extraction took {(time.time() - start)*1000:.1f}ms")
   ```

## Troubleshooting

### Issue: "NLTK data not found"

**Solution**: NLTK data is auto-downloaded on first use. If it fails:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Issue: "TF-IDF topic extraction failed"

**Solution**: Ensure scikit-learn is installed:

```bash
pip install scikit-learn
```

Or disable topic extraction:

```bash
export EXTRACT_TOPICS=0
```

### Issue: Metadata not appearing in nodes

**Solution**: Check that environment variable is enabled:

```bash
export EXTRACT_ENHANCED_METADATA=1
```

And that metadata is being added to nodes:

```python
print(node.metadata.keys())  # Should see struct_*, sem_*, tech_*, qual_* keys
```

### Issue: Extraction is too slow

**Solution**: Profile and disable expensive features:

```bash
# Disable TF-IDF (most expensive)
export EXTRACT_TOPICS=0

# Disable entity extraction
export EXTRACT_ENTITIES=0
```

## Future Enhancements

Potential improvements for future versions:

1. **Advanced NER**: Use spaCy for better entity extraction
2. **Multilingual support**: Add support for non-English documents
3. **Custom entity dictionaries**: Allow users to add domain-specific entities
4. **Caching**: Cache extracted metadata for repeated chunks
5. **Async processing**: Parallelize extraction for large document sets
6. **More document types**: Add specialized extractors for PDFs, Word docs, etc.
7. **Metadata validation**: Validate and sanitize extracted metadata
8. **Metadata aggregation**: Aggregate metadata across all chunks in a document

## References

- **LlamaIndex Documentation**: https://docs.llamaindex.ai/
- **NLTK Documentation**: https://www.nltk.org/
- **scikit-learn TF-IDF**: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

## License

This module is part of the llamaIndex-local-rag project. See project LICENSE for details.
