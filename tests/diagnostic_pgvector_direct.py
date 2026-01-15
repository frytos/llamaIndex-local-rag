#!/usr/bin/env python3
"""
Test PGVectorStore directly to isolate the issue.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Mock llama_index modules before importing
sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.vector_stores'] = MagicMock()
sys.modules['llama_index.core.vector_stores.types'] = MagicMock()
sys.modules['llama_index.vector_stores'] = MagicMock()
sys.modules['llama_index.vector_stores.postgres'] = MagicMock()

# Mock sentence_transformers
sys.modules['sentence_transformers'] = MagicMock()

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("🔬 Direct PGVectorStore Test")
print("=" * 70)
print()

# Configuration
TABLE_NAME = 'data_messages-text-slim_fast_1883_260108'
QUERY_TEXT = 'agathe'

print(f"📊 Configuration:")
print(f"   DB: {os.getenv('DB_NAME')}")
print(f"   Host: {os.getenv('PGHOST')}")
print(f"   Table: {TABLE_NAME}")
print(f"   Query: '{QUERY_TEXT}'")
print()

# Step 1: Load embedding model
print("1️⃣  Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"   ✅ Model loaded")
print()

# Step 2: Create embedding
print(f"2️⃣  Creating embedding for '{QUERY_TEXT}'...")
query_embedding = model.encode(QUERY_TEXT).tolist()
print(f"   ✅ Embedding created: {len(query_embedding)} dimensions")
print(f"   Sample values: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ...]")
print()

# Step 3: Connect to PGVectorStore
print("3️⃣  Connecting to PGVectorStore...")
store = PGVectorStore.from_params(
    database=os.getenv('DB_NAME', 'vector_db'),
    host=os.getenv('PGHOST', 'localhost'),
    port=os.getenv('PGPORT', '5432'),
    user=os.getenv('PGUSER'),
    password=os.getenv('PGPASSWORD'),
    table_name=TABLE_NAME,
    embed_dim=384,
)
print(f"   ✅ Connected")
print()

# Step 4: Query
print("4️⃣  Executing vector query...")
vsq = VectorStoreQuery(
    query_embedding=query_embedding,
    similarity_top_k=10,
    mode="default",
)

result = store.query(vsq)

print("=" * 70)
print(f"📊 RAW RESULTS from PGVectorStore.query():")
print("=" * 70)
print(f"   Nodes returned: {len(result.nodes)}")
print(f"   Similarities: {result.similarities[:5] if result.similarities else 'None'}")
print(f"   IDs: {result.ids[:5] if result.ids else 'None'}")
print()

if len(result.nodes) == 0:
    print("❌ PGVectorStore.query() returned ZERO nodes!")
    print()
    print("🔍 This is the root cause. Checking why...")
    print()

    # Direct SQL test
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv('PGHOST'),
        port=os.getenv('PGPORT'),
        user=os.getenv('PGUSER'),
        password=os.getenv('PGPASSWORD'),
        dbname=os.getenv('DB_NAME'),
    )
    cur = conn.cursor()

    # Test raw SQL
    print("Testing raw SQL vector search...")
    cur.execute(f"""
        SELECT
            substring(text, 1, 100) as chunk,
            1 - (embedding <=> %s::vector) as similarity
        FROM "{TABLE_NAME}"
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
    """, (query_embedding, query_embedding))

    rows = cur.fetchall()
    print(f"   SQL returned: {len(rows)} rows")

    for i, (chunk, sim) in enumerate(rows):
        print(f"   [{i+1}] Similarity: {sim:.4f} | Text: {chunk[:80]}...")

    cur.close()
    conn.close()
else:
    print("✅ SUCCESS! PGVectorStore is working!")
    for i, node in enumerate(result.nodes[:5]):
        sim = result.similarities[i] if result.similarities else 0
        print(f"[{i+1}] Similarity: {sim:.4f}")
        print(f"    Text: {node.get_content()[:150]}...")
        print()

print("=" * 70)
