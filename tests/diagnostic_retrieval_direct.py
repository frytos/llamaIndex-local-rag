#!/usr/bin/env python3
"""
Direct test of retrieval to diagnose 0 results issue.
Usage: python test_retrieval_direct.py
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock llama_index modules before any imports
sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.schema'] = MagicMock()
sys.modules['llama_index.core.node_parser'] = MagicMock()
sys.modules['llama_index.core.retrievers'] = MagicMock()
sys.modules['llama_index.core.query_engine'] = MagicMock()
sys.modules['llama_index.core.response_synthesizers'] = MagicMock()
sys.modules['llama_index.core.vector_stores'] = MagicMock()
sys.modules['llama_index.core.vector_stores.types'] = MagicMock()
sys.modules['llama_index.llms'] = MagicMock()
sys.modules['llama_index.llms.llama_cpp'] = MagicMock()
sys.modules['llama_index.llms.openai'] = MagicMock()
sys.modules['llama_index.embeddings'] = MagicMock()
sys.modules['llama_index.embeddings.huggingface'] = MagicMock()
sys.modules['llama_index.vector_stores'] = MagicMock()
sys.modules['llama_index.vector_stores.postgres'] = MagicMock()
sys.modules['llama_index.readers'] = MagicMock()
sys.modules['llama_index.readers.file'] = MagicMock()

# Create mock QueryBundle
class MockQueryBundle:
    def __init__(self, query_str="", embedding=None):
        self.query_str = query_str
        self.embedding = embedding or []

sys.modules['llama_index.core.schema'].QueryBundle = MockQueryBundle

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Force safe settings
os.environ['HYBRID_ALPHA'] = '1.0'
os.environ['ENABLE_FILTERS'] = '0'
os.environ['ENABLE_QUERY_EXPANSION'] = '0'
os.environ['ENABLE_RERANKING'] = '0'

print("=" * 70)
print("🧪 RAG Retrieval Diagnostic Test")
print("=" * 70)
print()

# Import after setting env vars
from rag_low_level_m1_16gb_verbose import (
    Settings, build_embed_model, make_vector_store, VectorDBRetriever
)
from llama_index.core.schema import QueryBundle

# Test configuration
TABLE_NAME = 'data_messages-text-slim_fast_1883_260108'
QUERY = 'agathe'
TOP_K = 5

print(f"📊 Test Configuration:")
print(f"   Table: {TABLE_NAME}")
print(f"   Query: '{QUERY}'")
print(f"   TOP_K: {TOP_K}")
print()

# Override settings
from rag_low_level_m1_16gb_verbose import S
S.table = TABLE_NAME
S.top_k = TOP_K
S.hybrid_alpha = 1.0
S.enable_filters = False
S.enable_query_expansion = False
S.enable_reranking = False

print(f"⚙️  Settings:")
print(f"   HYBRID_ALPHA: {S.hybrid_alpha}")
print(f"   ENABLE_FILTERS: {S.enable_filters}")
print(f"   EMBED_MODEL: {S.embed_model_name}")
print()

# Step 1: Build embedding model
print("1️⃣  Building embedding model...")
embed_model = build_embed_model()
print(f"   ✅ Model loaded: {S.embed_model_name}")
print()

# Step 2: Connect to vector store
print("2️⃣  Connecting to vector store...")
vector_store = make_vector_store()
print(f"   ✅ Connected to table: {TABLE_NAME}")
print()

# Step 3: Create retriever
print("3️⃣  Creating retriever...")
retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=TOP_K)
print(f"   ✅ Retriever created (VectorDBRetriever)")
print()

# Step 4: Test query
print(f"4️⃣  Executing query: '{QUERY}'...")
print()

try:
    results = retriever._retrieve(QueryBundle(query_str=QUERY))

    print("=" * 70)
    print(f"📊 RESULTS: {len(results)} chunks found")
    print("=" * 70)
    print()

    if len(results) == 0:
        print("❌ ZERO RESULTS - This is the problem!")
        print()
        print("🔍 Possible causes:")
        print("   1. Embedding model mismatch (but logs show correct model)")
        print("   2. Table empty (but we know it has 19,820 rows)")
        print("   3. PGVectorStore.query() returning empty")
        print("   4. Similarity threshold too high")
        print()
        print("💡 Next steps:")
        print("   - Check if PGVectorStore has a default similarity cutoff")
        print("   - Test with dummy embedding vector directly")
        print("   - Enable debug logging in llama-index")
    else:
        print("✅ SUCCESS! Retrieval is working!")
        print()
        for i, result in enumerate(results):
            print(f"[{i+1}] Score: {result.score:.4f}")
            print(f"    Text: {result.node.get_content()[:150]}...")
            print()

except Exception as e:
    print(f"❌ ERROR during retrieval: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("Test complete")
print("=" * 70)
