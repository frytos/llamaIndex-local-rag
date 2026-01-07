"""Example integration of DocumentMetadataExtractor with build_nodes().

This script demonstrates how to integrate the enhanced metadata extractor
into the existing RAG pipeline's build_nodes() function.

This is a reference implementation showing the recommended integration pattern.
"""

import os
import sys
from pathlib import Path
from typing import Any, List
from llama_index.core.schema import TextNode

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metadata_extractor import DocumentMetadataExtractor


def build_nodes_enhanced(
    docs: List[Any],
    chunks: List[str],
    doc_idxs: List[int],
    chunk_size: int,
    chunk_overlap: int,
    embed_model_name: str,
) -> List[TextNode]:
    """Create TextNode objects with enhanced metadata.

    This is an enhanced version of build_nodes() that adds rich metadata
    extraction for improved RAG retrieval quality.

    Args:
        docs: List of Document objects
        chunks: List of text chunks
        doc_idxs: Document index for each chunk
        chunk_size: Chunk size parameter
        chunk_overlap: Chunk overlap parameter
        embed_model_name: Embedding model name

    Returns:
        List of TextNode objects with enhanced metadata
    """
    print(f"Building {len(chunks)} nodes with enhanced metadata...")

    # Initialize extractor once (not per chunk!)
    extractor = None
    if os.getenv("EXTRACT_ENHANCED_METADATA", "1") == "1":
        extractor = DocumentMetadataExtractor()
        print(f"  Enhanced metadata extraction: ENABLED")
        print(f"    - Topics (TF-IDF): {extractor.extract_topics}")
        print(f"    - Named entities: {extractor.extract_entities}")
        print(f"    - Code blocks: {extractor.extract_code}")
        print(f"    - Tables: {extractor.extract_tables}")
    else:
        print(f"  Enhanced metadata extraction: DISABLED")

    # Create index signature for tracking
    index_signature = f"cs{chunk_size}_ov{chunk_overlap}_{embed_model_name.replace('/', '_')}"

    nodes: List[TextNode] = []

    for i, chunk in enumerate(chunks):
        n = TextNode(text=chunk)

        # Start with source document metadata
        src_doc = docs[doc_idxs[i]]
        n.metadata = src_doc.metadata.copy() if src_doc.metadata else {}

        # Add enhanced metadata (if enabled)
        if extractor:
            doc_format = n.metadata.get("format", "txt")
            chunk_position = (i + 1, len(chunks))

            # Extract all metadata
            doc_metadata = extractor.extract_all_metadata(
                chunk,
                doc_format=doc_format,
                chunk_position=chunk_position,
            )

            # Merge with existing metadata (flattened with prefixes)
            n.metadata.update(doc_metadata.to_dict())

        # Add chunking parameters (for mixed-index detection)
        n.metadata["_chunk_size"] = chunk_size
        n.metadata["_chunk_overlap"] = chunk_overlap
        n.metadata["_embed_model"] = embed_model_name
        n.metadata["_index_signature"] = index_signature

        nodes.append(n)

        # Progress logging
        if (i + 1) % 500 == 0:
            print(f"  Built {i+1}/{len(chunks)} nodes")

    print(f"Built {len(nodes)} nodes with enhanced metadata")
    print(f"Index signature: {index_signature}")

    # Summary of extracted metadata
    if extractor:
        _print_metadata_summary(nodes)

    return nodes


def _print_metadata_summary(nodes: List[TextNode]) -> None:
    """Print summary of extracted metadata across all nodes."""
    from collections import Counter

    print("\nMetadata Summary:")

    # Document types
    doc_types = [n.metadata.get("struct_doc_type") for n in nodes if "struct_doc_type" in n.metadata]
    if doc_types:
        type_counts = Counter(doc_types)
        print(f"  Document types:")
        for doc_type, count in type_counts.most_common():
            print(f"    - {doc_type}: {count} chunks")

    # Code chunks
    code_chunks = sum(1 for n in nodes if n.metadata.get("tech_has_code"))
    if code_chunks:
        print(f"  Code chunks: {code_chunks} ({code_chunks/len(nodes)*100:.1f}%)")

    # Table chunks
    table_chunks = sum(1 for n in nodes if n.metadata.get("tech_has_tables"))
    if table_chunks:
        print(f"  Table chunks: {table_chunks} ({table_chunks/len(nodes)*100:.1f}%)")

    # Equation chunks
    equation_chunks = sum(1 for n in nodes if n.metadata.get("tech_has_equations"))
    if equation_chunks:
        print(f"  Equation chunks: {equation_chunks} ({equation_chunks/len(nodes)*100:.1f}%)")

    # Reading levels
    reading_levels = [n.metadata.get("qual_reading_level") for n in nodes if "qual_reading_level" in n.metadata]
    if reading_levels:
        level_counts = Counter(reading_levels)
        print(f"  Reading levels:")
        for level, count in level_counts.most_common():
            print(f"    - {level}: {count} chunks")

    # Top entities across all chunks
    all_entities = []
    for n in nodes:
        entities = n.metadata.get("sem_entities", [])
        if isinstance(entities, list):
            all_entities.extend(entities)

    if all_entities:
        entity_counts = Counter(all_entities)
        print(f"  Top entities:")
        for entity, count in entity_counts.most_common(10):
            print(f"    - {entity}: {count} occurrences")


def demonstrate_integration():
    """Demonstrate the integration with sample documents."""
    from llama_index.core.schema import Document

    print("="*70)
    print("Enhanced Metadata Integration Demo")
    print("="*70)

    # Sample documents
    doc1 = Document(
        text="""
# Python Tutorial: Machine Learning

## Introduction

Learn Python for machine learning using PyTorch and TensorFlow.

## Getting Started

First, install the required libraries:

```python
pip install torch tensorflow numpy
```

## Training a Model

Here's a simple example:

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
```
""",
        metadata={"source": "ml_tutorial.md", "format": "md"}
    )

    doc2 = Document(
        text="""
Abstract: This paper presents a novel approach to neural architecture search.

1. Introduction

Recent advances in deep learning have shown promising results. We propose
a method using reinforcement learning.

2. Methodology

The reward function is defined as:

$$R = \\sum_{i=1}^{n} r_i$$

3. Results

| Dataset  | Accuracy | Time  |
|----------|----------|-------|
| CIFAR-10 | 95.2%    | 2.5h  |
| ImageNet | 82.1%    | 12h   |
""",
        metadata={"source": "research_paper.pdf", "format": "pdf"}
    )

    # Simulate chunking
    docs = [doc1, doc2]
    chunks = [
        doc1.text[:200],
        doc1.text[200:400],
        doc1.text[400:],
        doc2.text[:200],
        doc2.text[200:],
    ]
    doc_idxs = [0, 0, 0, 1, 1]

    # Build nodes with enhanced metadata
    nodes = build_nodes_enhanced(
        docs=docs,
        chunks=chunks,
        doc_idxs=doc_idxs,
        chunk_size=700,
        chunk_overlap=150,
        embed_model_name="BAAI/bge-small-en",
    )

    # Show sample metadata
    print("\n" + "="*70)
    print("Sample Node Metadata (First Node)")
    print("="*70)

    sample_node = nodes[0]
    print(f"\nText preview: {sample_node.text[:100]}...\n")

    print("Metadata fields:")
    for key in sorted(sample_node.metadata.keys()):
        value = sample_node.metadata[key]
        if isinstance(value, list):
            print(f"  {key}: {', '.join(str(v) for v in value[:3])}...")
        elif isinstance(value, dict):
            print(f"  {key}: <dict with {len(value)} items>")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("Integration complete!")
    print("="*70)


def demonstrate_filtering():
    """Demonstrate filtering and retrieval with metadata."""
    print("\n" + "="*70)
    print("Metadata-Based Filtering Demo")
    print("="*70)

    # Simulate nodes with metadata
    nodes = [
        TextNode(
            text="Python tutorial content",
            metadata={
                "struct_doc_type": "tutorial",
                "tech_has_code": True,
                "qual_reading_level": "easy",
                "sem_entities": ["python", "programming"]
            }
        ),
        TextNode(
            text="Research paper abstract",
            metadata={
                "struct_doc_type": "research_paper",
                "tech_has_equations": True,
                "tech_has_tables": True,
                "qual_reading_level": "difficult",
                "sem_entities": ["machine learning", "neural networks"]
            }
        ),
        TextNode(
            text="Code example with PyTorch",
            metadata={
                "struct_doc_type": "code",
                "tech_has_code": True,
                "tech_programming_language": "py",
                "tech_functions": ["train", "evaluate"],
                "sem_entities": ["pytorch", "python"]
            }
        ),
    ]

    # Example queries with metadata filters
    queries = [
        ("Tutorial with code examples", {"struct_doc_type": "tutorial", "tech_has_code": True}),
        ("Research papers", {"struct_doc_type": "research_paper"}),
        ("Python code", {"tech_has_code": True, "tech_programming_language": "py"}),
        ("Easy to understand", {"qual_reading_level": "easy"}),
    ]

    print("\nFiltering examples:\n")

    for query, filters in queries:
        print(f"Query: '{query}'")
        print(f"Filters: {filters}")

        # Simple filter matching
        matches = []
        for node in nodes:
            match = all(
                node.metadata.get(k) == v
                for k, v in filters.items()
            )
            if match:
                matches.append(node)

        print(f"Matches: {len(matches)}")
        for match in matches:
            print(f"  - {match.text}")
        print()


if __name__ == "__main__":
    # Set environment variables for demo
    os.environ["EXTRACT_ENHANCED_METADATA"] = "1"
    os.environ["EXTRACT_TOPICS"] = "1"
    os.environ["EXTRACT_ENTITIES"] = "1"

    # Run demonstrations
    demonstrate_integration()
    demonstrate_filtering()

    print("\n" + "="*70)
    print("See utils/README_METADATA.md for integration instructions")
    print("="*70)
