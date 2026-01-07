"""Tests for naming utility functions."""

import pytest
from pathlib import Path
from utils.naming import sanitize_table_name, extract_model_short_name, generate_table_name


class TestSanitizeTableName:
    """Test table name sanitization."""

    def test_basic_sanitization(self):
        """Test basic character replacement."""
        assert sanitize_table_name("my-document") == "my_document"
        assert sanitize_table_name("my document") == "my_document"
        assert sanitize_table_name("my-document name") == "my_document_name"

    def test_special_characters(self):
        """Test special character removal."""
        # Consecutive special chars become multiple underscores (that's okay)
        result = sanitize_table_name("doc@#$%name")
        assert result.startswith("doc")
        assert result.endswith("name")
        assert "-" not in result  # No hyphens
        assert " " not in result  # No spaces

    def test_starts_with_number(self):
        """Test that names starting with numbers are prefixed."""
        assert sanitize_table_name("2024-report") == "t_2024_report"
        assert sanitize_table_name("123abc") == "t_123abc"

    def test_lowercase_conversion(self):
        """Test that output is lowercase."""
        assert sanitize_table_name("MyDocument") == "mydocument"
        assert sanitize_table_name("UPPERCASE") == "uppercase"


class TestExtractModelShortName:
    """Test model name extraction."""

    def test_bge_models(self):
        """Test BGE model name extraction."""
        assert extract_model_short_name("BAAI/bge-small-en") == "bge"
        assert extract_model_short_name("BAAI/bge-large-en-v1.5") == "bge"

    def test_minilm_models(self):
        """Test MiniLM model name extraction."""
        assert extract_model_short_name("sentence-transformers/all-MiniLM-L6-v2") == "minilm"
        assert extract_model_short_name("microsoft/MiniLM-L12-H384") == "minilm"

    def test_other_models(self):
        """Test other common model types."""
        assert extract_model_short_name("sentence-transformers/all-mpnet-base-v2") == "mpnet"
        assert extract_model_short_name("roberta-base") == "roberta"
        assert extract_model_short_name("bert-base-uncased") == "bert"

    def test_fallback(self):
        """Test fallback for unknown models."""
        result = extract_model_short_name("custom/my-model-v1")
        assert isinstance(result, str)
        assert len(result) <= 8


class TestGenerateTableName:
    """Test table name generation."""

    def test_basic_generation(self):
        """Test basic table name generation."""
        doc_path = Path("test_document.pdf")
        result = generate_table_name(doc_path, 700, 150, "BAAI/bge-small-en")

        # Check format components
        assert "_cs700_" in result
        assert "_ov150_" in result
        assert "_bge_" in result
        assert result.startswith("test_document")

    def test_date_suffix(self):
        """Test that date suffix is added."""
        from datetime import datetime

        doc_path = Path("report.pdf")
        result = generate_table_name(doc_path, 500, 100)

        # Check date format (YYMMDD)
        date_str = datetime.now().strftime("%y%m%d")
        assert result.endswith(date_str)

    def test_long_name_truncation(self):
        """Test that long names are truncated."""
        long_name = "a" * 50  # 50 characters
        doc_path = Path(f"{long_name}.pdf")
        result = generate_table_name(doc_path, 700, 150)

        # Name component should be <= 30 chars
        name_part = result.split("_cs")[0]
        assert len(name_part) <= 30

    def test_sanitization_applied(self):
        """Test that sanitization is applied to document name."""
        doc_path = Path("my-document name.pdf")
        result = generate_table_name(doc_path, 700, 150)

        # Should not contain hyphens or spaces
        assert "-" not in result.split("_cs")[0]
        assert " " not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
