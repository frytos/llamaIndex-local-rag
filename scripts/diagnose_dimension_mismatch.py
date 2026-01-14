#!/usr/bin/env python3
"""
Diagnose embedding dimension mismatch between table and query model.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Database connection
def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        database=os.getenv("DB_NAME", "vector_db")
    )

# Table name
table_name = os.getenv("PGTABLE", "inbox_conversations_quality")
actual_table = f"data_{table_name}"  # PGVectorStore adds prefix

print("=" * 70)
print("EMBEDDING DIMENSION MISMATCH DIAGNOSIS")
print("=" * 70)

# 1. Check table's actual vector dimension
print(f"\n1. Checking table: {actual_table}")
try:
    conn = get_conn()
    with conn.cursor() as cur:
        # Get vector column dimension from PostgreSQL type
        cur.execute("""
            SELECT
                a.attname as column_name,
                pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
            JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
            WHERE c.relname = %s
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
        """, (actual_table,))

        result = cur.fetchone()
        if result:
            column_name, data_type = result
            # Extract dimension from "vector(N)" format
            if "vector(" in data_type:
                stored_dim = int(data_type.split("(")[1].split(")")[0])
                print(f"   ✓ Table column 'embedding': {data_type}")
                print(f"   ✓ Stored dimension: {stored_dim}")
            else:
                print(f"   ✗ Unexpected column type: {data_type}")
                stored_dim = None
        else:
            print(f"   ✗ Table or column not found")
            stored_dim = None

        # Sample metadata to see what model was used
        if stored_dim:
            cur.execute(f'SELECT metadata_::json FROM "{actual_table}" LIMIT 1')
            row = cur.fetchone()
            if row and row[0]:
                metadata = row[0]
                chunk_size = metadata.get('chunk_size', 'unknown')
                embed_model = metadata.get('embed_model', 'unknown')
                print(f"   ✓ Indexed with model: {embed_model}")
                print(f"   ✓ Chunk size: {chunk_size}")

    conn.close()
except Exception as e:
    print(f"   ✗ Error: {e}")
    stored_dim = None

# 2. Check current environment settings
print(f"\n2. Checking current environment settings:")
current_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
current_dim = int(os.getenv("EMBED_DIM", "384"))
print(f"   • EMBED_MODEL={current_model}")
print(f"   • EMBED_DIM={current_dim}")

# 3. Model dimension reference
print(f"\n3. Common embedding models and their dimensions:")
model_dims = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}

for model, dim in model_dims.items():
    marker = "←" if model == current_model else " "
    print(f"   {marker} {model}: {dim} dims")

# 4. Diagnosis
print(f"\n{'='*70}")
print("DIAGNOSIS:")
print("=" * 70)

if stored_dim and stored_dim != current_dim:
    print(f"❌ DIMENSION MISMATCH DETECTED!")
    print(f"   Table has: {stored_dim} dimensions")
    print(f"   Query using: {current_dim} dimensions")
    print(f"")
    print(f"FIX OPTIONS:")
    print(f"")
    print(f"Option 1: Use matching model for queries (fast, no re-indexing)")
    print(f"-" * 70)

    # Find models that match stored dimension
    matching_models = [m for m, d in model_dims.items() if d == stored_dim]
    if matching_models:
        print(f"   export EMBED_MODEL={matching_models[0]}")
        print(f"   export EMBED_DIM={stored_dim}")
        if stored_dim == 384:
            print(f"   # Or use MLX for Apple Silicon (9x faster):")
            print(f"   export EMBED_BACKEND=mlx")
    else:
        print(f"   (No common model found with {stored_dim} dimensions)")

    print(f"")
    print(f"Option 2: Re-index with current model (slower, better quality)")
    print(f"-" * 70)
    print(f"   export RESET_TABLE=1")
    print(f"   export EMBED_MODEL={current_model}")
    print(f"   export EMBED_DIM={current_dim}")
    print(f"   python rag_low_level_m1_16gb_verbose.py")

    print(f"")
    print(f"Option 3: Create new table with different model (safest)")
    print(f"-" * 70)
    print(f"   export PGTABLE={table_name}_new")
    print(f"   export EMBED_MODEL=BAAI/bge-m3  # Best multilingual")
    print(f"   export EMBED_DIM=1024")
    print(f"   export EMBED_BACKEND=mlx  # Fast on Apple Silicon")
    print(f"   python rag_low_level_m1_16gb_verbose.py")

elif stored_dim and stored_dim == current_dim:
    print(f"✓ Dimensions match: {stored_dim} == {current_dim}")
    print(f"  The dimension mismatch error might be intermittent.")
    print(f"  Check if you have multiple .env files or environment variables set.")
else:
    print(f"⚠️  Could not determine stored dimension from table")
    print(f"   Current query dimension: {current_dim}")

print("=" * 70)
