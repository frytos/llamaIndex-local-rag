"""Unit tests for metadata extractor module.

Tests both DocumentMetadataExtractor and MetadataExtractor classes.
"""

import unittest
from utils.metadata_extractor import (
    DocumentMetadataExtractor,
    DocumentMetadata,
    MetadataExtractor,
    enhance_node_metadata,
)


class TestDocumentMetadataExtractor(unittest.TestCase):
    """Test DocumentMetadataExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = DocumentMetadataExtractor()

    def test_extract_structure_metadata_markdown(self):
        """Test structure metadata extraction from Markdown."""
        text = """
# Main Title

## Section 1

This is a tutorial about Python.

## Section 2

More content here.
"""
        metadata = self.extractor.extract_structure_metadata(text, doc_format="md")

        self.assertEqual(metadata["format"], "md")
        self.assertTrue(metadata["has_headings"])
        self.assertEqual(metadata["heading_count"], 3)
        self.assertIn("tutorial", metadata["doc_type"])
        self.assertEqual(metadata["section_title"], "Main Title")

    def test_extract_semantic_metadata(self):
        """Test semantic metadata extraction."""
        text = """
        Python is a programming language. Django and Flask are Python frameworks.
        Python is widely used in data science with libraries like PyTorch and TensorFlow.
        """
        metadata = self.extractor.extract_semantic_metadata(text)

        self.assertIn("keywords", metadata)
        self.assertIn("entities", metadata)
        # Should detect Python, Django, Flask, PyTorch, TensorFlow
        self.assertGreater(metadata["entity_count"], 3)

    def test_extract_technical_metadata_code(self):
        """Test technical metadata extraction from code."""
        code = """
import numpy as np
from sklearn.metrics import accuracy_score

class DataProcessor:
    def __init__(self):
        pass

    def process(self, data):
        return data

def main():
    processor = DataProcessor()
    print("Done")
"""
        metadata = self.extractor.extract_technical_metadata(code, doc_format="py")

        self.assertTrue(metadata["has_code"])
        self.assertEqual(metadata["programming_language"], "py")
        self.assertIn("functions", metadata)
        self.assertIn("classes", metadata)
        self.assertIn("imports", metadata)
        self.assertEqual(metadata["function_count"], 3)  # __init__, process, main
        self.assertEqual(metadata["class_count"], 1)  # DataProcessor

    def test_extract_technical_metadata_tables(self):
        """Test table detection."""
        text = """
# Results

| Dataset | Accuracy |
|---------|----------|
| CIFAR10 | 95.2%    |
| ImageNet| 87.3%    |
"""
        metadata = self.extractor.extract_technical_metadata(text, doc_format="md")

        self.assertTrue(metadata["has_tables"])
        self.assertGreater(metadata["table_count"], 0)

    def test_extract_technical_metadata_equations(self):
        """Test equation detection."""
        text = """
The learning rate is computed as:

$$\\alpha_t = \\alpha_0 \\cdot \\sqrt{1 - \\beta_2^t}$$

And inline math: $x = y + z$
"""
        metadata = self.extractor.extract_technical_metadata(text, doc_format="md")

        self.assertTrue(metadata["has_equations"])
        self.assertGreaterEqual(metadata["equation_count"], 2)

    def test_extract_quality_signals(self):
        """Test quality signal extraction."""
        text = """
        This is a simple sentence. This is another sentence.
        Here is a third sentence with more words in it.
        """
        metadata = self.extractor.extract_quality_signals(text)

        self.assertIn("word_count", metadata)
        self.assertIn("sentence_count", metadata)
        self.assertIn("avg_sentence_length", metadata)
        self.assertIn("reading_level", metadata)
        self.assertIn("char_count", metadata)
        self.assertGreater(metadata["word_count"], 0)
        self.assertGreater(metadata["sentence_count"], 0)

    def test_extract_all_metadata(self):
        """Test complete metadata extraction."""
        text = """
# Tutorial: Machine Learning

## Introduction

This tutorial covers Python machine learning using PyTorch and TensorFlow.

```python
import torch

def train_model():
    pass
```
"""
        doc_metadata = self.extractor.extract_all_metadata(text, doc_format="md")

        # Check all metadata types are present
        self.assertIsInstance(doc_metadata, DocumentMetadata)
        self.assertIn("format", doc_metadata.structure)
        self.assertIn("keywords", doc_metadata.semantic)
        self.assertIn("has_code", doc_metadata.technical)
        self.assertIn("word_count", doc_metadata.quality)

    def test_to_dict_conversion(self):
        """Test DocumentMetadata to dict conversion."""
        doc_metadata = DocumentMetadata(
            structure={"format": "md", "doc_type": "tutorial"},
            semantic={"keywords": ["python", "ml"]},
            technical={"has_code": True},
            quality={"word_count": 100}
        )

        flat_dict = doc_metadata.to_dict()

        # Check prefixes are applied
        self.assertEqual(flat_dict["struct_format"], "md")
        self.assertEqual(flat_dict["struct_doc_type"], "tutorial")
        self.assertEqual(flat_dict["sem_keywords"], ["python", "ml"])
        self.assertEqual(flat_dict["tech_has_code"], True)
        self.assertEqual(flat_dict["qual_word_count"], 100)


class TestMetadataExtractor(unittest.TestCase):
    """Test MetadataExtractor class (basic extractor)."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()

    def test_extract_dates(self):
        """Test date extraction."""
        text = "The meeting is on 2024-01-15 and January 20, 2024."
        metadata = self.extractor.extract(text)

        self.assertIn("_dates", metadata)
        self.assertIn("_date_count", metadata)
        self.assertGreater(metadata["_date_count"], 0)

    def test_extract_emails(self):
        """Test email extraction."""
        text = "Contact me at john@example.com or jane@test.org"
        metadata = self.extractor.extract(text)

        self.assertIn("_emails", metadata)
        self.assertIn("_email_count", metadata)
        self.assertEqual(metadata["_email_count"], 2)

    def test_extract_urls(self):
        """Test URL extraction."""
        text = "Visit https://example.com and http://test.org for more info."
        metadata = self.extractor.extract(text)

        self.assertIn("_urls", metadata)
        self.assertIn("_url_count", metadata)
        self.assertEqual(metadata["_url_count"], 2)

    def test_classify_content_code(self):
        """Test code content classification."""
        text = "def hello():\n    print('hello')"
        metadata = self.extractor.extract(text)

        self.assertEqual(metadata["_content_type"], "code")
        self.assertTrue(metadata["_has_code"])

    def test_classify_content_documentation(self):
        """Test documentation content classification."""
        text = "# Title\n\n## Section\n\nSome content here."
        metadata = self.extractor.extract(text)

        self.assertEqual(metadata["_content_type"], "documentation")


class TestEnhanceNodeMetadata(unittest.TestCase):
    """Test enhance_node_metadata convenience function."""

    def test_enhance_node_metadata(self):
        """Test enhancing node metadata."""
        text = """
# Python Tutorial

This is a tutorial about Python programming.

```python
def hello():
    print("Hello")
```
"""
        base_metadata = {
            "source": "tutorial.md",
            "format": "md",
        }

        enhanced = enhance_node_metadata(text, base_metadata)

        # Check original metadata is preserved
        self.assertEqual(enhanced["source"], "tutorial.md")
        self.assertEqual(enhanced["format"], "md")

        # Check enhanced metadata is added
        self.assertIn("struct_doc_type", enhanced)
        self.assertIn("sem_keywords", enhanced)
        self.assertIn("tech_has_code", enhanced)
        self.assertIn("qual_word_count", enhanced)

    def test_enhance_node_metadata_disabled(self):
        """Test enhance_node_metadata when disabled."""
        import os
        os.environ["EXTRACT_ENHANCED_METADATA"] = "0"

        text = "# Test"
        base_metadata = {"source": "test.md"}

        enhanced = enhance_node_metadata(text, base_metadata)

        # Should return unchanged metadata
        self.assertEqual(enhanced, base_metadata)

        # Cleanup
        os.environ["EXTRACT_ENHANCED_METADATA"] = "1"


if __name__ == "__main__":
    unittest.main()
