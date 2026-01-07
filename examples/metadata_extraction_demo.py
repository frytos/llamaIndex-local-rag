"""Demonstration of enhanced metadata extraction for RAG pipeline.

This script demonstrates how to use the DocumentMetadataExtractor to extract
rich metadata from various document types to improve retrieval quality.

Usage:
    python examples/metadata_extraction_demo.py
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metadata_extractor import DocumentMetadataExtractor, enhance_node_metadata


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def demo_structure_extraction():
    """Demonstrate structure metadata extraction."""
    print("\n" + "="*70)
    print("Demo 1: Structure Metadata Extraction")
    print("="*70 + "\n")

    markdown_doc = """
# Machine Learning Best Practices

## Introduction

This guide covers essential ML best practices.

## Data Preparation

### 1. Data Cleaning

Remove duplicates and handle missing values.

### 2. Feature Engineering

Create meaningful features from raw data.

## Model Training

Use cross-validation to avoid overfitting.
"""

    extractor = DocumentMetadataExtractor()
    metadata = extractor.extract_structure_metadata(markdown_doc, doc_format="md")

    print("Input: Markdown document with hierarchical headings")
    print("\nExtracted Structure Metadata:")
    for key, value in metadata.items():
        if key != "headings":
            print(f"  {key}: {value}")

    if "headings" in metadata:
        print(f"\n  Headings hierarchy:")
        for heading in metadata["headings"]:
            indent = "  " * heading["level"]
            print(f"    {indent}H{heading['level']}: {heading['title']}")


def demo_semantic_extraction():
    """Demonstrate semantic metadata extraction."""
    print("\n" + "="*70)
    print("Demo 2: Semantic Metadata Extraction")
    print("="*70 + "\n")

    text = """
Deep learning with PyTorch and TensorFlow has revolutionized natural language processing.
Modern NLP models like BERT and GPT use transformer architectures. These frameworks provide
powerful tools for building neural networks. PyTorch offers dynamic computation graphs while
TensorFlow excels in production deployment. Both frameworks integrate well with CUDA for GPU
acceleration and support distributed training across multiple nodes.
"""

    extractor = DocumentMetadataExtractor()
    metadata = extractor.extract_semantic_metadata(text)

    print("Input: Technical text about deep learning frameworks")
    print("\nExtracted Semantic Metadata:")
    print(f"  Keywords: {', '.join(metadata.get('keywords', []))}")
    print(f"  Entities detected: {', '.join(metadata.get('entities', []))}")
    print(f"  Total entities: {metadata.get('entity_count', 0)}")

    if "topics" in metadata:
        print(f"  Topics (TF-IDF): {', '.join(metadata['topics'])}")


def demo_technical_extraction():
    """Demonstrate technical metadata extraction."""
    print("\n" + "="*70)
    print("Demo 3: Technical Metadata Extraction")
    print("="*70 + "\n")

    python_code = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total
"""

    extractor = DocumentMetadataExtractor()
    metadata = extractor.extract_technical_metadata(python_code, doc_format="py")

    print("Input: Python code with PyTorch neural network")
    print("\nExtracted Technical Metadata:")
    print(f"  Programming language: {metadata.get('programming_language', 'N/A')}")
    print(f"  Has code: {metadata.get('has_code', False)}")
    print(f"  Functions: {', '.join(metadata.get('functions', []))}")
    print(f"  Classes: {', '.join(metadata.get('classes', []))}")
    print(f"  Imports: {', '.join(metadata.get('imports', []))}")


def demo_quality_signals():
    """Demonstrate quality signal extraction."""
    print("\n" + "="*70)
    print("Demo 4: Quality Signal Extraction")
    print("="*70 + "\n")

    texts = {
        "Simple": "AI is cool. It works well. Use it.",
        "Moderate": """
            Machine learning algorithms learn patterns from data. They can make
            predictions on new, unseen data. This process involves training and testing.
        """,
        "Complex": """
            The paradigmatic shift towards deep learning architectures has fundamentally
            revolutionized our approach to solving complex computational problems in
            artificial intelligence, particularly in domains such as computer vision
            and natural language processing where traditional algorithmic approaches
            had previously demonstrated significant limitations.
        """
    }

    extractor = DocumentMetadataExtractor()

    for name, text in texts.items():
        metadata = extractor.extract_quality_signals(text)
        print(f"{name} text:")
        print(f"  Word count: {metadata['word_count']}")
        print(f"  Sentence count: {metadata['sentence_count']}")
        print(f"  Avg sentence length: {metadata['avg_sentence_length']}")
        print(f"  Reading level: {metadata['reading_level']}")
        print()


def demo_complete_extraction():
    """Demonstrate complete metadata extraction."""
    print("\n" + "="*70)
    print("Demo 5: Complete Metadata Extraction")
    print("="*70 + "\n")

    research_paper = """
Abstract: This paper presents a novel approach to neural architecture search using
reinforcement learning. We achieve state-of-the-art results on ImageNet with 15% fewer
parameters than previous methods.

1. Introduction

Deep learning has achieved remarkable success in computer vision tasks. However, designing
optimal neural network architectures remains challenging. We propose using PyTorch to implement
an automated search algorithm.

2. Methodology

Our approach uses the following equation for reward calculation:

$$R_t = \\sum_{i=1}^{n} \\alpha_i \\cdot a_i(t)$$

where $\\alpha_i$ are learnable parameters and $a_i(t)$ represents the accuracy at step t.

3. Results

| Dataset  | Accuracy | Parameters |
|----------|----------|------------|
| CIFAR-10 | 97.2%    | 3.2M       |
| ImageNet | 82.1%    | 11.5M      |

4. Conclusion

We demonstrate that automated neural architecture search can discover efficient models
using TensorFlow or PyTorch implementations.

References:
[1] Smith et al., "Neural Architecture Search", NeurIPS 2023
"""

    extractor = DocumentMetadataExtractor()
    doc_metadata = extractor.extract_all_metadata(research_paper, doc_format="pdf")

    print("Input: Research paper with abstract, equations, and tables")
    print("\n--- Structure Metadata ---")
    for key, value in doc_metadata.structure.items():
        if key != "headings":
            print(f"  {key}: {value}")

    print("\n--- Semantic Metadata ---")
    for key, value in doc_metadata.semantic.items():
        if isinstance(value, list):
            print(f"  {key}: {', '.join(str(v) for v in value[:5])}")
        else:
            print(f"  {key}: {value}")

    print("\n--- Technical Metadata ---")
    for key, value in doc_metadata.technical.items():
        print(f"  {key}: {value}")

    print("\n--- Quality Metadata ---")
    for key, value in doc_metadata.quality.items():
        print(f"  {key}: {value}")


def demo_integration_with_nodes():
    """Demonstrate integration with TextNode metadata."""
    print("\n" + "="*70)
    print("Demo 6: Integration with TextNode Metadata")
    print("="*70 + "\n")

    # Simulate a chunk from a larger document
    chunk_text = """
## Data Preprocessing

Before training, preprocess your data using standard normalization techniques.
Use PyTorch's transforms module:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
```
"""

    # Simulate existing metadata from document loader
    base_metadata = {
        "source": "ml_tutorial.md",
        "format": "md",
        "page": 3,
        "file_size": 15234,
    }

    # Enhance with extracted metadata
    enhanced = enhance_node_metadata(chunk_text, base_metadata)

    print("Base metadata:")
    for key, value in base_metadata.items():
        print(f"  {key}: {value}")

    print("\nEnhanced metadata (showing new fields only):")
    new_keys = set(enhanced.keys()) - set(base_metadata.keys())
    for key in sorted(new_keys):
        value = enhanced[key]
        if isinstance(value, list):
            print(f"  {key}: {', '.join(str(v) for v in value[:3])}...")
        else:
            print(f"  {key}: {value}")


def demo_environment_variables():
    """Demonstrate environment variable configuration."""
    print("\n" + "="*70)
    print("Demo 7: Environment Variable Configuration")
    print("="*70 + "\n")

    text = "Python is great for machine learning with PyTorch and TensorFlow."

    print("Testing with different configurations:\n")

    # Test 1: All features enabled (default)
    os.environ["EXTRACT_ENHANCED_METADATA"] = "1"
    os.environ["EXTRACT_TOPICS"] = "1"
    os.environ["EXTRACT_ENTITIES"] = "1"

    extractor1 = DocumentMetadataExtractor()
    metadata1 = extractor1.extract_semantic_metadata(text)

    print("1. All features enabled:")
    print(f"   Keywords: {len(metadata1.get('keywords', []))}")
    print(f"   Entities: {len(metadata1.get('entities', []))}")
    print(f"   Topics: {len(metadata1.get('topics', []))}")

    # Test 2: Disable topic extraction
    os.environ["EXTRACT_TOPICS"] = "0"

    extractor2 = DocumentMetadataExtractor()
    metadata2 = extractor2.extract_semantic_metadata(text)

    print("\n2. Topics disabled (faster):")
    print(f"   Keywords: {len(metadata2.get('keywords', []))}")
    print(f"   Entities: {len(metadata2.get('entities', []))}")
    print(f"   Topics: {'topics' in metadata2}")

    # Test 3: Disable all enhanced metadata
    os.environ["EXTRACT_ENHANCED_METADATA"] = "0"

    extractor3 = DocumentMetadataExtractor()
    metadata3 = extractor3.extract_semantic_metadata(text)

    print("\n3. Enhanced metadata disabled:")
    print(f"   Metadata extracted: {bool(metadata3)}")

    # Reset to defaults
    os.environ["EXTRACT_ENHANCED_METADATA"] = "1"
    os.environ["EXTRACT_TOPICS"] = "1"
    os.environ["EXTRACT_ENTITIES"] = "1"


def main():
    """Run all demonstrations."""
    setup_logging()

    print("\n" + "="*70)
    print("Enhanced Metadata Extraction - Comprehensive Demo")
    print("="*70)

    # Run all demos
    demo_structure_extraction()
    demo_semantic_extraction()
    demo_technical_extraction()
    demo_quality_signals()
    demo_complete_extraction()
    demo_integration_with_nodes()
    demo_environment_variables()

    print("\n" + "="*70)
    print("Demo complete! See docs/METADATA_EXTRACTOR.md for more details.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
