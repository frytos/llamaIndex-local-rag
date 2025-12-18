#!/usr/bin/env python3
"""
TensorBoard Multi-Document Visualizer
Compare embeddings from multiple documents in the same 3D space
"""

import os
import sys
import numpy as np
import psycopg2
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Configuration
DB_NAME = os.getenv("DB_NAME", "vector_db")
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGUSER = os.getenv("PGUSER", "fryt")
PGPASSWORD = os.getenv("PGPASSWORD", "frytos")


def fetch_embeddings_from_table(table_name):
    """Fetch embeddings from a specific table"""
    print(f"📊 Fetching from: {table_name}")

    conn = psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        database=DB_NAME,
        user=PGUSER,
        password=PGPASSWORD
    )

    actual_table = f"data_{table_name}"
    cursor = conn.cursor()

    query = f"""
        SELECT id, text, embedding
        FROM {actual_table}
        ORDER BY id
        LIMIT 500;
    """

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"  ⚠️  Table {actual_table} not found or error: {e}")
        cursor.close()
        conn.close()
        return None, None

    print(f"  ✓ Fetched {len(rows)} embeddings")

    embeddings = []
    labels = []

    for row in rows:
        id_val, text, embedding = row

        if isinstance(embedding, str):
            embedding = np.fromstring(embedding.strip('[]'), sep=',')
        else:
            embedding = np.array(embedding)

        embeddings.append(embedding)

        # Add document name to label
        label = f"[{table_name}] {text[:80]}" if text else f"[{table_name}] Chunk {id_val}"
        labels.append(label)

    cursor.close()
    conn.close()

    return np.array(embeddings, dtype=np.float32), labels


def create_combined_visualization(tables, log_dir="runs/rag_multi"):
    """Create TensorBoard visualization with multiple documents"""

    print(f"\n🎨 Creating combined visualization...")

    all_embeddings = []
    all_labels = []
    all_metadata = []

    for table in tables:
        embeddings, labels = fetch_embeddings_from_table(table)

        if embeddings is None:
            print(f"  ⚠️  Skipping {table}")
            continue

        all_embeddings.append(embeddings)
        all_labels.extend(labels)

        # Add metadata for coloring
        all_metadata.extend([{"document": table}] * len(embeddings))

    if len(all_embeddings) == 0:
        print("❌ No embeddings found!")
        return

    # Combine all embeddings
    combined_embeddings = np.vstack(all_embeddings)

    print(f"\n📊 Combined Statistics:")
    print(f"   • Total documents: {len(tables)}")
    print(f"   • Total embeddings: {len(combined_embeddings)}")
    print(f"   • Embedding dimension: {combined_embeddings.shape[1]}")

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{log_dir}/{timestamp}")

    # Convert to torch tensor
    embeddings_tensor = torch.FloatTensor(combined_embeddings)

    # Extract document names for metadata
    doc_names = [m["document"] for m in all_metadata]

    # Add embeddings with both text labels and document metadata
    writer.add_embedding(
        embeddings_tensor,
        metadata=all_labels,
        metadata_header=["text"],
        tag="multi_document_chunks"
    )

    writer.close()

    print(f"\n✅ Combined visualization created!")
    print(f"   Log directory: {log_dir}/{timestamp}")

    return f"{log_dir}/{timestamp}"


def main():
    """Main execution"""

    print("=" * 70)
    print("TensorBoard Multi-Document Visualizer")
    print("=" * 70)

    # Get tables to compare (from command line or default)
    if len(sys.argv) > 1:
        tables = sys.argv[1:]
    else:
        # Default: compare both documents
        tables = ["llama2_paper", "mastering-rag_paper"]

    print(f"\n📚 Documents to visualize:")
    for i, table in enumerate(tables, 1):
        print(f"   {i}. {table}")

    try:
        log_dir = create_combined_visualization(tables)

        print("\n" + "=" * 70)
        print("🚀 NEXT STEPS:")
        print("=" * 70)
        print("\n1. Start/Restart TensorBoard:")
        print(f"   tensorboard --logdir=runs/rag_multi")
        print("\n2. Open in browser:")
        print("   http://localhost:6006")
        print("\n3. In TensorBoard:")
        print("   • Click 'PROJECTOR' tab")
        print("   • See all documents in same 3D space!")
        print("   • Search '[llama2_paper]' to highlight one doc")
        print("   • Search '[mastering-rag_paper]' for the other")
        print("   • Compare how different topics cluster")
        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
