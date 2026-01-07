# Enhanced Metadata Extraction - Quick Reference

## One-Line Usage

```python
from utils.metadata_extractor import DocumentMetadataExtractor, enhance_node_metadata

# Option 1: Direct extraction
extractor = DocumentMetadataExtractor()
metadata = extractor.extract_all_metadata(text, doc_format="pdf")

# Option 2: Enhance existing metadata
enhanced = enhance_node_metadata(text, {"source": "doc.pdf", "format": "pdf"})
```

## Environment Variables

```bash
# Quick configs
export EXTRACT_ENHANCED_METADATA=1   # 0=disable all, 1=enable (default)
export EXTRACT_TOPICS=0              # Disable TF-IDF (saves ~10ms)
export EXTRACT_ENTITIES=0            # Disable NER (saves ~2ms)
```

## Metadata Output (4 Categories)

```python
# Structure (prefix: struct_)
metadata.structure = {
    "format": "md",
    "doc_type": "tutorial",        # research_paper, manual, api_doc, code, blog_post
    "has_headings": True,
    "heading_count": 3,
    "section_title": "Introduction",
    "chunk_position": "1/10"
}

# Semantic (prefix: sem_)
metadata.semantic = {
    "keywords": ["python", "ml"],
    "topics": ["machine learning"],  # TF-IDF
    "entities": ["pytorch", "aws"],  # 40+ tech entities
    "entity_count": 2
}

# Technical (prefix: tech_)
metadata.technical = {
    "has_code": True,
    "has_tables": True,
    "has_equations": False,
    "programming_language": "py",
    "functions": ["train", "eval"],
    "classes": ["Model"],
    "imports": ["torch", "numpy"]
}

# Quality (prefix: qual_)
metadata.quality = {
    "word_count": 125,
    "sentence_count": 8,
    "avg_sentence_length": 15.6,
    "reading_level": "moderate",  # very_easy, easy, moderate, difficult, very_difficult
    "char_count": 782
}
```

## Integration with build_nodes()

```python
from utils.metadata_extractor import DocumentMetadataExtractor

def build_nodes(docs, chunks, doc_idxs):
    extractor = DocumentMetadataExtractor()  # Initialize ONCE
    nodes = []

    for i, chunk in enumerate(chunks):
        n = TextNode(text=chunk)
        n.metadata = docs[doc_idxs[i]].metadata.copy()

        # Add enhanced metadata
        doc_metadata = extractor.extract_all_metadata(
            chunk,
            doc_format=n.metadata.get("format", "txt"),
            chunk_position=(i+1, len(chunks))
        )
        n.metadata.update(doc_metadata.to_dict())
        nodes.append(n)

    return nodes
```

## Use Cases

```python
# Filter by document type
results = retriever.retrieve("query", filters={"struct_doc_type": "tutorial"})

# Filter by technical content
results = retriever.retrieve("query", filters={"tech_has_code": True})

# Filter by reading level
results = retriever.retrieve("query", filters={"qual_reading_level": "easy"})

# Smart reranking
if "how to" in query and result.metadata.get("tech_has_code"):
    score *= 1.2  # Boost code examples
```

## Document Type Classification

| Type | Detected Patterns |
|------|-------------------|
| research_paper | abstract, conclusion, references, citations |
| manual | installation, configuration, troubleshooting |
| tutorial | step N, example, walkthrough, guide |
| api_doc | endpoint, API, parameters, request/response |
| code | def, class, function, import |
| blog_post | posted, author, comments, share |

## Supported Code Languages

Python, JavaScript, TypeScript, Java, C++, C, Rust, Go

Extracts: functions, classes, imports

## Performance

| Configuration | Time/Chunk | Use Case |
|---------------|------------|----------|
| All features | ~15-20ms | Full metadata |
| No TF-IDF | ~5-8ms | Fast mode |
| Disabled | ~0ms | Baseline |

## Testing

```bash
# Unit tests
pytest tests/test_metadata_extractor.py -v

# Interactive demo
python utils/metadata_extractor.py --enhanced

# Comprehensive demo
python examples/metadata_extraction_demo.py

# Integration example
python examples/integrate_metadata_extraction.py
```

## Files

- **Module**: `utils/metadata_extractor.py`
- **Docs**: `docs/METADATA_EXTRACTOR.md`
- **Quick Guide**: `utils/README_METADATA.md`
- **Tests**: `tests/test_metadata_extractor.py`
- **Demo**: `examples/metadata_extraction_demo.py`
- **Integration**: `examples/integrate_metadata_extraction.py`
- **Config**: `config/.env.example` (updated)

## Troubleshooting

```python
# Issue: NLTK data not found
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Issue: Too slow
export EXTRACT_TOPICS=0        # Disable TF-IDF (~10ms savings)

# Issue: Metadata not appearing
export EXTRACT_ENHANCED_METADATA=1
print(node.metadata.keys())    # Check for struct_*, sem_*, tech_*, qual_*
```

## Dependencies

All in `requirements.txt`:
- `nltk` (3.9.2): Tokenization, stopwords
- `scikit-learn` (1.8.0): TF-IDF

## See Full Documentation

- **Complete Guide**: `docs/METADATA_EXTRACTOR.md`
- **Implementation Summary**: `METADATA_EXTRACTION_SUMMARY.md`
