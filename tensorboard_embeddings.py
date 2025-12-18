#!/usr/bin/env python3
"""
TensorBoard Embedding Visualizer
Creates interactive 3D visualization of your document embeddings
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
PGTABLE = os.getenv("PGTABLE", "llama2_paper")


def fetch_embeddings_from_db():
    """Fetch all embeddings and text from PostgreSQL"""

    print(f"🔗 Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        database=DB_NAME,
        user=PGUSER,
        password=PGPASSWORD
    )

    # PGVectorStore uses "data_" prefix
    actual_table = f"data_{PGTABLE}"

    print(f"📊 Fetching embeddings from table: {actual_table}")

    cursor = conn.cursor()

    # Fetch embeddings and metadata
    # Use quotes to handle table names with hyphens
    query = f"""
        SELECT
            id,
            text,
            metadata_,
            embedding
        FROM "{actual_table}"
        ORDER BY id
        LIMIT 500;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    print(f"✓ Fetched {len(rows)} embeddings")

    embeddings = []
    labels = []
    metadata_list = []

    for row in rows:
        id_val, text, metadata, embedding = row

        # Convert embedding to numpy array
        if isinstance(embedding, str):
            # Parse string representation
            embedding = np.fromstring(embedding.strip('[]'), sep=',')
        else:
            embedding = np.array(embedding)

        embeddings.append(embedding)

        # Create label (first 100 chars of text)
        label = text[:100] if text else f"Chunk {id_val}"
        labels.append(label)
        metadata_list.append(metadata)

    cursor.close()
    conn.close()

    return np.array(embeddings), labels, metadata_list


def create_tensorboard_visualization(embeddings, labels, log_dir="runs/rag_embeddings"):
    """Create TensorBoard embedding visualization"""

    print(f"\n🎨 Creating TensorBoard visualization...")

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{log_dir}/{timestamp}")

    # Convert to torch tensor
    embeddings_tensor = torch.FloatTensor(embeddings)

    print(f"   Shape: {embeddings_tensor.shape}")
    print(f"   Labels: {len(labels)}")

    # Add embeddings to TensorBoard
    writer.add_embedding(
        embeddings_tensor,
        metadata=labels,
        tag="document_chunks"
    )

    writer.close()

    print(f"\n✅ TensorBoard visualization created!")
    print(f"   Log directory: {log_dir}/{timestamp}")

    return f"{log_dir}/{timestamp}"


def main():
    """Main execution"""

    print("=" * 70)
    print("TensorBoard Embedding Visualizer")
    print("=" * 70)

    try:
        # 1. Fetch embeddings from database
        embeddings, labels, metadata = fetch_embeddings_from_db()

        if len(embeddings) == 0:
            print("❌ No embeddings found in database!")
            print(f"   Check that table 'data_{PGTABLE}' exists and has data.")
            sys.exit(1)

        print(f"\n📊 Summary:")
        print(f"   • Total embeddings: {len(embeddings)}")
        print(f"   • Embedding dimension: {embeddings.shape[1]}")
        print(f"   • Total text chunks: {len(labels)}")

        # 2. Create visualization
        log_dir = create_tensorboard_visualization(embeddings, labels)

        # 3. Instructions
        print("\n" + "=" * 70)
        print("🚀 NEXT STEPS:")
        print("=" * 70)
        print("\n1. Start TensorBoard:")
        print(f"   tensorboard --logdir=runs/rag_embeddings")
        print("\n2. Open in browser:")
        print("   http://localhost:6006")
        print("\n3. In TensorBoard:")
        print("   • Click 'PROJECTOR' tab at top")
        print("   • Use mouse to rotate 3D view")
        print("   • Click points to see text")
        print("   • Use search to find specific chunks")
        print("   • Try different projections (PCA, t-SNE, UMAP)")
        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
