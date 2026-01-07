"""Naming utilities for table and model name generation.

This module contains shared functions for generating consistent,
SQL-safe table names and extracting short model identifiers.
"""

from pathlib import Path
from datetime import datetime


def sanitize_table_name(name: str) -> str:
    """Sanitize table name by replacing invalid SQL characters.

    Args:
        name: Raw table name (may contain hyphens, spaces, etc.)

    Returns:
        Sanitized table name safe for SQL (underscores only)

    Examples:
        >>> sanitize_table_name("my-document name")
        't_my_document_name'
        >>> sanitize_table_name("2024-report")
        't_2024_report'
    """
    # Replace hyphens and spaces with underscores
    sanitized = name.replace("-", "_").replace(" ", "_")
    # Remove any other non-alphanumeric characters except underscores
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "t_" + sanitized
    return sanitized.lower()


def extract_model_short_name(model_name: str) -> str:
    """Extract a short, readable name from embedding model path.

    Args:
        model_name: Full model path (e.g., "BAAI/bge-small-en")

    Returns:
        Short model identifier (e.g., "bge")

    Examples:
        >>> extract_model_short_name("BAAI/bge-small-en")
        'bge'
        >>> extract_model_short_name("sentence-transformers/all-MiniLM-L6-v2")
        'minilm'
    """
    name = model_name.split("/")[-1]
    if "bge" in name.lower():
        return "bge"
    elif "minilm" in name.lower():
        return "minilm"
    elif "e5" in name.lower():
        return "e5"
    elif "mpnet" in name.lower():
        return "mpnet"
    elif "roberta" in name.lower():
        return "roberta"
    elif "bert" in name.lower():
        return "bert"
    else:
        parts = name.lower().replace("sentence-", "").replace("all-", "").split("-")
        return parts[0][:8]


def generate_table_name(
    doc_path: Path, chunk_size: int, chunk_overlap: int, embed_model: str = "BAAI/bge-small-en"
) -> str:
    """Generate a table name from document and config.

    Format: {doc}_cs{size}_ov{overlap}_{model}_{YYMMDD}

    Args:
        doc_path: Path to document
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap size in characters
        embed_model: Embedding model name

    Returns:
        Generated table name

    Examples:
        >>> from pathlib import Path
        >>> generate_table_name(Path("report.pdf"), 700, 150, "BAAI/bge-small-en")
        'report_cs700_ov150_bge_260107'
    """
    # Clean document name using sanitize function
    name = sanitize_table_name(doc_path.stem)

    # Limit length
    if len(name) > 30:
        name = name[:30]

    # Extract short model name
    model_short = extract_model_short_name(embed_model)

    # Get date in YYMMDD format
    date_str = datetime.now().strftime("%y%m%d")

    return f"{name}_cs{chunk_size}_ov{chunk_overlap}_{model_short}_{date_str}"
