#!/usr/bin/env python3
"""
rag_interactive.py

Interactive menu-driven launcher for the RAG pipeline.
Allows users to:
- Select documents to index
- Configure chunk parameters
- Query existing indexes
- Run full pipeline or individual steps
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import psycopg2

# Database connection settings (match main script defaults)
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", "5432"),
    "user": os.getenv("PGUSER", "fryt"),
    "password": os.getenv("PGPASSWORD", "frytos"),
    "dbname": os.getenv("DB_NAME", "vector_db"),
}

# Default paths
DATA_DIR = Path(__file__).parent / "data"


def sanitize_table_name(name: str) -> str:
    """Sanitize table name by replacing invalid SQL characters.

    Args:
        name: Raw table name (may contain hyphens, spaces, etc.)

    Returns:
        Sanitized table name safe for SQL (underscores only)
    """
    # Replace hyphens and spaces with underscores
    sanitized = name.replace("-", "_").replace(" ", "_")
    # Remove any other non-alphanumeric characters except underscores
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "t_" + sanitized
    return sanitized.lower()
MAIN_SCRIPT = Path(__file__).parent / "rag_low_level_m1_16gb_verbose.py"


def clear_screen():
    """Clear terminal screen."""
    os.system("clear" if os.name != "nt" else "cls")


def print_header(title: str):
    """Print a formatted header."""
    width = 60
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width + "\n")


def print_menu(options: List[str], title: str = "Options"):
    """Print a numbered menu."""
    print(f"\n{title}:")
    print("-" * 40)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("-" * 40)


def get_choice(max_val: int, prompt: str = "Enter choice", allow_zero: bool = False) -> int:
    """Get a numeric choice from user."""
    min_val = 0 if allow_zero else 1
    while True:
        try:
            choice = input(f"{prompt} [{min_val}-{max_val}]: ").strip()
            if not choice:
                continue
            val = int(choice)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("  Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            sys.exit(0)


def get_input(prompt: str, default: str = "") -> str:
    """Get string input with optional default."""
    try:
        default_str = f" [{default}]" if default else ""
        val = input(f"{prompt}{default_str}: ").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting...")
        sys.exit(0)


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input."""
    default_str = "Y/n" if default else "y/N"
    try:
        val = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not val:
            return default
        return val in ("y", "yes", "1", "true")
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting...")
        sys.exit(0)


# Supported text-based extensions
TEXT_EXTENSIONS = {
    ".txt", ".md", ".html", ".htm", ".json", ".csv", ".xml", ".xsl",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".cpp", ".c", ".h", ".hpp",
    ".java", ".rb", ".go", ".rs", ".php", ".m", ".swift", ".kt",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sql", ".r", ".scala", ".pl", ".lua", ".ex", ".exs", ".clj",
}

# All supported extensions (binary + text)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"} | TEXT_EXTENSIONS


def list_documents() -> Tuple[List[Path], List[Path]]:
    """List available documents and folders in the data directory."""
    if not DATA_DIR.exists():
        return [], []

    docs = []
    folders = []

    for path in DATA_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(path)
        elif path.is_dir() and not path.name.startswith("."):
            # Check if folder contains supported files
            file_count = sum(1 for f in path.rglob("*")
                           if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
            if file_count > 0:
                folders.append((path, file_count))

    # Sort by name
    docs.sort(key=lambda p: p.name.lower())
    folders.sort(key=lambda x: x[0].name.lower())
    return docs, folders


def get_table_info(table_name: str) -> Dict:
    """Get information about a vector store table."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Use SQL identifier to safely quote table name
        from psycopg2 import sql

        # Get row count
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))
        )
        row_count = cur.fetchone()[0]

        # Try to get chunk config from metadata
        # Note: PGVectorStore stores metadata in 'metadata_' column (with underscore)
        cur.execute(
            sql.SQL("""
                SELECT DISTINCT
                    metadata_->>'_chunk_size' as chunk_size,
                    metadata_->>'_chunk_overlap' as chunk_overlap,
                    metadata_->>'_index_signature' as signature
                FROM {}
                WHERE metadata_->>'_chunk_size' IS NOT NULL
                LIMIT 1
            """).format(sql.Identifier(table_name))
        )
        config = cur.fetchone()

        conn.close()

        return {
            "rows": row_count,
            "chunk_size": config[0] if config else "unknown",
            "chunk_overlap": config[1] if config else "unknown",
            "signature": config[2] if config else "legacy",
        }
    except Exception as e:
        return {"rows": 0, "error": str(e)}


def list_vector_tables() -> List[Tuple[str, Dict]]:
    """List all vector store tables with their info."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename LIKE 'data_%'
            ORDER BY tablename
        """)

        tables = []
        for (table_name,) in cur.fetchall():
            # Strip 'data_' prefix for display
            display_name = table_name[5:] if table_name.startswith("data_") else table_name
            info = get_table_info(table_name)
            tables.append((display_name, info))

        conn.close()
        return tables
    except Exception as e:
        print(f"  Warning: Could not connect to database: {e}")
        return []


def select_document() -> Optional[Path]:
    """Interactive document or folder selection."""
    print_header("Select Document or Folder to Index")

    docs, folders = list_documents()

    if not docs and not folders:
        print(f"  No documents or folders found in {DATA_DIR}")
        print(f"  Supported: .pdf, .docx, .pptx, .html, .json, .txt, .md, .py, .cpp, .java, etc.")
        custom = get_input("\n  Enter custom path (or press Enter to cancel)")
        if custom:
            path = Path(custom).expanduser()
            if path.exists():
                return path
            else:
                print(f"  Path not found: {path}")
        return None

    options = []

    # Add folders first (with folder icon)
    for folder, file_count in folders:
        options.append(f"[FOLDER] {folder.name}/ ({file_count} files)")

    # Add individual files
    for doc in docs:
        size_mb = doc.stat().st_size / (1024 * 1024)
        options.append(f"{doc.name} ({size_mb:.1f} MB)")

    options.append("Enter custom path")
    options.append("Cancel")

    print_menu(options, "Available Documents & Folders")
    choice = get_choice(len(options))

    if choice == len(options):  # Cancel
        return None
    elif choice == len(options) - 1:  # Custom path
        custom = get_input("  Enter document or folder path")
        if custom:
            path = Path(custom).expanduser()
            if path.exists():
                return path
            else:
                print(f"  Path not found: {path}")
                return None
        return None
    elif choice <= len(folders):  # Selected a folder
        return folders[choice - 1][0]
    else:  # Selected a file
        file_idx = choice - len(folders) - 1
        return docs[file_idx]


def configure_chunking() -> Tuple[int, int]:
    """Interactive chunking configuration."""
    print_header("Configure Chunking Parameters")

    print("Chunking controls how documents are split into searchable pieces.")
    print("\nGeneral guidelines:")
    print("  - Smaller chunks (200-500): More precise retrieval, may lose context")
    print("  - Medium chunks (500-1000): Balanced (recommended)")
    print("  - Larger chunks (1000-2000): More context, may include irrelevant text")
    print("  - Overlap: Usually 10-20% of chunk_size prevents splitting sentences")

    presets = [
        ("Ultra-fine", 100, 20, "Short messages, chat logs, tweets"),
        ("Fine-grained", 300, 60, "Good for Q&A with specific facts"),
        ("Balanced (Recommended)", 700, 150, "Good general-purpose setting"),
        ("Contextual", 1200, 240, "Good for summaries and complex topics"),
        ("Large context", 2000, 400, "Good for lengthy explanations"),
        ("Custom", 0, 0, "Enter your own values"),
    ]

    options = [f"{name} (cs={cs}, ov={ov}) - {desc}" if cs > 0 else f"{name} - {desc}"
               for name, cs, ov, desc in presets]

    print_menu(options, "Chunking Presets")
    choice = get_choice(len(presets))

    name, chunk_size, chunk_overlap, _ = presets[choice - 1]

    if name == "Custom":
        chunk_size = int(get_input("  Chunk size (characters)", "700"))
        chunk_overlap = int(get_input("  Chunk overlap (characters)", "150"))

    print(f"\n  Selected: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    return chunk_size, chunk_overlap


def check_mlx_available() -> bool:
    """Check if MLX is installed and available."""
    try:
        import mlx.core as mx
        from mlx_embedding_models.embedding import EmbeddingModel
        return True
    except ImportError:
        return False


def configure_embedding() -> Tuple[str, int, str, int]:
    """Configure embedding model and backend.

    Returns:
        Tuple of (model_name, dim, backend, batch_size)
    """
    print_header("Configure Embedding Model")

    print("Embedding models convert text to vectors for semantic search.")
    print("\nTrade-offs:")
    print("  - Smaller models: Faster indexing, less nuanced")
    print("  - Larger models: Slower indexing, better quality")

    presets = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", 384,
         "Fastest, good for general text"),
        ("bge-small-en (Recommended)", "BAAI/bge-small-en", 384,
         "Good balance of speed and quality"),
        ("bge-base-en", "BAAI/bge-base-en-v1.5", 768,
         "Better quality, 2x slower"),
        ("bge-large-en", "BAAI/bge-large-en-v1.5", 1024,
         "Best quality, 4x slower, needs more RAM"),
    ]

    options = [f"{name} ({dim}d) - {desc}" for name, _, dim, desc in presets]

    print_menu(options, "Embedding Models")
    choice = get_choice(len(presets))

    name, model_name, dim, _ = presets[choice - 1]
    print(f"\n  Selected: {name} ({dim} dimensions)")

    # Backend selection
    backend = "huggingface"
    batch_size = 32

    mlx_available = check_mlx_available()
    if mlx_available:
        print("\n" + "=" * 50)
        print("  🚀 MLX Backend Available (Apple Silicon)")
        print("=" * 50)
        print("\n  MLX provides 5-20x faster indexing on M1/M2/M3 Macs.")
        print("  Uses Metal GPU acceleration optimized for Apple Silicon.")
        print("\n  Performance comparison:")
        print("    • HuggingFace: ~20-40 chunks/sec")
        print("    • MLX:         ~150-400 chunks/sec")

        if get_yes_no("\n  Use MLX backend?", default=True):
            backend = "mlx"
            batch_size = 64  # MLX can handle larger batches
            print("  ✓ MLX backend selected")
            print(f"  ✓ Batch size optimized: {batch_size}")
        else:
            print("  Using HuggingFace backend (MPS)")
    else:
        print("\n  Note: MLX not installed. Using HuggingFace backend.")
        print("  Install MLX for 5-20x speedup: pip install mlx mlx-embedding-models")

    return model_name, dim, backend, batch_size


def configure_retrieval() -> int:
    """Configure retrieval parameters."""
    print_header("Configure Retrieval Parameters")

    print("TOP_K controls how many chunks are retrieved for each query.")
    print("\n  - Lower (2-3): Faster, more focused, may miss relevant info")
    print("  - Medium (4-6): Balanced (recommended)")
    print("  - Higher (8-10): More comprehensive, slower, may add noise")

    top_k = int(get_input("\n  TOP_K value", "6"))
    return top_k


def select_existing_table() -> Optional[str]:
    """Select an existing vector table to query."""
    print_header("Select Existing Index")

    tables = list_vector_tables()

    if not tables:
        print("  No existing indexes found in the database.")
        print("  You need to index a document first.")
        return None

    print(f"Found {len(tables)} existing index(es):\n")

    options = []
    for name, info in tables:
        if "error" in info:
            options.append(f"{name} (error: {info['error'][:30]}...)")
        else:
            config_str = f"cs={info['chunk_size']}, ov={info['chunk_overlap']}"
            options.append(f"{name} ({info['rows']} chunks, {config_str})")
    options.append("Cancel")

    print_menu(options, "Available Indexes")
    choice = get_choice(len(options))

    if choice == len(options):  # Cancel
        return None

    return tables[choice - 1][0]


def extract_model_short_name(model_name: str) -> str:
    """Extract a short, readable name from embedding model path."""
    name = model_name.split('/')[-1]
    if 'bge' in name.lower():
        return 'bge'
    elif 'minilm' in name.lower():
        return 'minilm'
    elif 'e5' in name.lower():
        return 'e5'
    elif 'mpnet' in name.lower():
        return 'mpnet'
    elif 'roberta' in name.lower():
        return 'roberta'
    elif 'bert' in name.lower():
        return 'bert'
    else:
        parts = name.lower().replace('sentence-', '').replace('all-', '').split('-')
        return parts[0][:8]


def generate_table_name(doc_path: Path, chunk_size: int, chunk_overlap: int,
                       embed_model: str = "BAAI/bge-small-en") -> str:
    """
    Generate a table name from document and config.

    Format: {doc}_cs{size}_ov{overlap}_{model}_{YYMMDD}
    """
    from datetime import datetime

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


def run_indexing(doc_path: Path, table_name: str, chunk_size: int, chunk_overlap: int,
                 embed_model: str = "BAAI/bge-small-en", embed_dim: int = 384,
                 backend: str = "huggingface", batch_size: int = 64, reset: bool = True):
    """Run the indexing pipeline."""
    print_header("Running Indexing Pipeline")

    env = os.environ.copy()
    env["PDF_PATH"] = str(doc_path)
    env["PGTABLE"] = table_name
    env["CHUNK_SIZE"] = str(chunk_size)
    env["CHUNK_OVERLAP"] = str(chunk_overlap)
    env["EMBED_MODEL"] = embed_model
    env["EMBED_DIM"] = str(embed_dim)
    env["EMBED_BACKEND"] = backend
    env["EMBED_BATCH"] = str(batch_size)
    env["RESET_TABLE"] = "1" if reset else "0"
    env["EXTRACT_CHAT_METADATA"] = "1"  # Enable metadata extraction

    # Optimized LLM settings (only set if not already in environment)
    env.setdefault("N_GPU_LAYERS", "30")
    env.setdefault("N_BATCH", "256")
    env.setdefault("CTX", "20000")

    # Advanced retrieval settings (only set if not already in environment)
    env.setdefault("HYBRID_ALPHA", "0.5")
    env.setdefault("MMR_THRESHOLD", "0.7")
    env.setdefault("ENABLE_FILTERS", "1")

    # Logging settings (only set if not already in environment)
    env.setdefault("LOG_FULL_CHUNKS", "1")
    env.setdefault("COLORIZE_CHUNKS", "1")
    env.setdefault("LOG_LEVEL", "INFO")

    print(f"  Document: {doc_path}")
    print(f"  Table: {table_name}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunk overlap: {chunk_overlap}")
    print(f"  Embedding: {embed_model} ({embed_dim}d)")
    print(f"  Backend: {backend.upper()}")
    print(f"  Batch size: {batch_size}")
    print(f"  Reset table: {reset}")

    if backend == "mlx":
        print(f"\n  ⚡ MLX acceleration enabled!")
        print(f"  Expected speedup: 5-20x vs standard backend")

    print("\n" + "-" * 50 + "\n")

    # Run indexing only (no query)
    cmd = [sys.executable, str(MAIN_SCRIPT), "--index-only"]

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n  Error: Indexing failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n  Indexing interrupted.")


def run_query_mode(table_name: str, top_k: int = 6, interactive: bool = True):
    """Run query mode on an existing index."""
    print_header("Query Mode")

    env = os.environ.copy()
    env["PGTABLE"] = table_name
    env["TOP_K"] = str(top_k)

    # Optimized LLM settings (only set if not already in environment)
    env.setdefault("N_GPU_LAYERS", "30")
    env.setdefault("N_BATCH", "256")
    env.setdefault("CTX", "20000")

    # Advanced retrieval settings (only set if not already in environment)
    env.setdefault("HYBRID_ALPHA", "0.5")
    env.setdefault("MMR_THRESHOLD", "0.7")
    env.setdefault("ENABLE_FILTERS", "1")

    # Embedding settings (only set if not already in environment)
    env.setdefault("EMBED_BACKEND", "mlx")
    env.setdefault("EMBED_BATCH", "64")

    # Logging settings (only set if not already in environment)
    env.setdefault("LOG_FULL_CHUNKS", "1")
    env.setdefault("COLORIZE_CHUNKS", "1")
    env.setdefault("LOG_LEVEL", "DEBUG")
    env.setdefault("EXTRACT_CHAT_METADATA", "1")

    print(f"  Table: {table_name}")
    print(f"  TOP_K: {top_k}")
    print("\n" + "-" * 50 + "\n")

    args = ["--query-only"]
    if interactive:
        args.append("--interactive")

    cmd = [sys.executable, str(MAIN_SCRIPT)] + args

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n  Error: Query failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n  Query session interrupted.")


def main_menu():
    """Main interactive menu."""
    while True:
        clear_screen()
        print_header("Local RAG Pipeline - Interactive Mode")

        # Show current database status
        tables = list_vector_tables()
        print(f"  Database: {DB_CONFIG['dbname']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        print(f"  Existing indexes: {len(tables)}")

        options = [
            "Index new document(s) (file or folder -> chunk -> embed -> store)",
            "Query an existing index (skip indexing, just ask questions)",
            "Index + Query (full pipeline then interactive query)",
            "View existing indexes (list all vector tables with details)",
            "Quick re-index / MLX optimization (re-index with new settings/backend)",
            "Exit",
        ]

        print_menu(options, "What would you like to do?")
        choice = get_choice(len(options))

        if choice == 1:  # Index new document
            doc = select_document()
            if doc:
                chunk_size, chunk_overlap = configure_chunking()
                embed_model, embed_dim, backend, batch_size = configure_embedding()
                table_name = generate_table_name(doc, chunk_size, chunk_overlap, embed_model)

                print(f"\n  Suggested table name: {table_name}")
                custom_name = get_input("  Press Enter to accept or type custom name", table_name)
                table_name = sanitize_table_name(custom_name) if custom_name else table_name

                reset = get_yes_no("  Reset table if exists?", default=True)

                if get_yes_no(f"\n  Ready to index '{doc.name}' into '{table_name}'?"):
                    run_indexing(doc, table_name, chunk_size, chunk_overlap,
                                embed_model, embed_dim, backend, batch_size, reset)
                    input("\n  Press Enter to continue...")

        elif choice == 2:  # Query existing
            table_name = select_existing_table()
            if table_name:
                top_k = configure_retrieval()
                run_query_mode(table_name, top_k, interactive=True)
                input("\n  Press Enter to continue...")

        elif choice == 3:  # Index + Query
            doc = select_document()
            if doc:
                chunk_size, chunk_overlap = configure_chunking()
                embed_model, embed_dim, backend, batch_size = configure_embedding()
                table_name = generate_table_name(doc, chunk_size, chunk_overlap, embed_model)

                print(f"\n  Suggested table name: {table_name}")
                custom_name = get_input("  Press Enter to accept or type custom name", table_name)
                table_name = sanitize_table_name(custom_name) if custom_name else table_name

                top_k = configure_retrieval()
                reset = get_yes_no("  Reset table if exists?", default=True)

                if get_yes_no(f"\n  Ready to index and query '{doc.name}'?"):
                    # First index
                    env = os.environ.copy()
                    env["PDF_PATH"] = str(doc)
                    env["PGTABLE"] = table_name
                    env["CHUNK_SIZE"] = str(chunk_size)
                    env["CHUNK_OVERLAP"] = str(chunk_overlap)
                    env["EMBED_MODEL"] = embed_model
                    env["EMBED_DIM"] = str(embed_dim)
                    env["EMBED_BACKEND"] = backend
                    env["EMBED_BATCH"] = str(batch_size)
                    env["RESET_TABLE"] = "1" if reset else "0"
                    env["TOP_K"] = str(top_k)
                    env["EXTRACT_CHAT_METADATA"] = "1"

                    # Optimized LLM settings (only set if not already in environment)
                    env.setdefault("N_GPU_LAYERS", "30")
                    env.setdefault("N_BATCH", "256")
                    env.setdefault("CTX", "20000")

                    # Advanced retrieval settings (only set if not already in environment)
                    env.setdefault("HYBRID_ALPHA", "0.5")
                    env.setdefault("MMR_THRESHOLD", "0.7")
                    env.setdefault("ENABLE_FILTERS", "1")

                    # Logging settings (only set if not already in environment)
                    env.setdefault("LOG_FULL_CHUNKS", "1")
                    env.setdefault("COLORIZE_CHUNKS", "1")
                    env.setdefault("LOG_LEVEL", "DEBUG")

                    print_header("Running Full Pipeline")
                    print(f"  Document: {doc}")
                    print(f"  Table: {table_name}")
                    print(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
                    print(f"  Embedding: {embed_model} ({embed_dim}d)")
                    print(f"  Backend: {backend.upper()}")
                    print(f"  Batch size: {batch_size}")
                    print(f"  TOP_K: {top_k}")

                    if backend == "mlx":
                        print(f"\n  ⚡ MLX acceleration enabled!")

                    print("\n" + "-" * 50 + "\n")

                    cmd = [sys.executable, str(MAIN_SCRIPT), "--interactive"]
                    try:
                        subprocess.run(cmd, env=env, check=True)
                    except (subprocess.CalledProcessError, KeyboardInterrupt):
                        pass
                    input("\n  Press Enter to continue...")

        elif choice == 4:  # View indexes
            print_header("Existing Vector Indexes")
            tables = list_vector_tables()

            if not tables:
                print("  No indexes found.")
            else:
                for name, info in tables:
                    print(f"\n  {name}:")
                    if "error" in info:
                        print(f"    Error: {info['error']}")
                    else:
                        print(f"    Chunks: {info['rows']}")
                        print(f"    Chunk size: {info['chunk_size']}")
                        print(f"    Chunk overlap: {info['chunk_overlap']}")
                        print(f"    Signature: {info['signature']}")

            input("\n  Press Enter to continue...")

        elif choice == 5:  # Quick re-index
            print_header("Quick Re-index / MLX Optimization")

            # Get list of unique source documents from existing tables
            print("  This allows you to re-index an existing document with different settings.")
            print("  Perfect for trying MLX backend or different chunking configurations.")

            doc = select_document()
            if doc:
                print(f"\n  Selected: {doc.name}")
                print("\n  Enter new parameters:")
                chunk_size, chunk_overlap = configure_chunking()
                embed_model, embed_dim, backend, batch_size = configure_embedding()

                table_name = generate_table_name(doc, chunk_size, chunk_overlap, embed_model)

                # Add backend suffix if MLX
                if backend == "mlx":
                    table_name = f"{table_name}_mlx"

                print(f"\n  New table name: {table_name}")

                # Check if table exists
                existing = [t[0] for t in tables]
                if table_name in existing:
                    print(f"  Warning: Table '{table_name}' already exists!")
                    if not get_yes_no("  Overwrite?"):
                        input("\n  Press Enter to continue...")
                        continue

                if get_yes_no(f"\n  Ready to index '{doc.name}' with new settings?"):
                    run_indexing(doc, table_name, chunk_size, chunk_overlap,
                                embed_model, embed_dim, backend, batch_size, reset=True)

                    if get_yes_no("\n  Start querying the new index?"):
                        top_k = configure_retrieval()
                        run_query_mode(table_name, top_k, interactive=True)

                    input("\n  Press Enter to continue...")

        elif choice == 6:  # Exit
            print("\nGoodbye!")
            sys.exit(0)


def main():
    """Entry point."""
    # Check if main script exists
    if not MAIN_SCRIPT.exists():
        print(f"Error: Main RAG script not found: {MAIN_SCRIPT}")
        sys.exit(1)

    # Check database connectivity
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
    except Exception as e:
        print(f"Warning: Could not connect to database: {e}")
        print("Make sure PostgreSQL is running (docker-compose up -d)")
        if not get_yes_no("Continue anyway?", default=False):
            sys.exit(1)

    main_menu()


if __name__ == "__main__":
    main()
