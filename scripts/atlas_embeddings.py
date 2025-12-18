#!/usr/bin/env python3
"""
Atlas by Nomic - Create beautiful interactive embedding maps
View your document embeddings as an explorable semantic map
"""

import os
import sys
import numpy as np
import psycopg2
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
    query = f"""
        SELECT
            id,
            text,
            metadata_,
            embedding
        FROM {actual_table}
        ORDER BY id
        LIMIT 1000;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    print(f"✓ Fetched {len(rows)} embeddings")

    embeddings = []
    data = []

    for row in rows:
        id_val, text, metadata, embedding = row

        # Convert embedding to numpy array
        if isinstance(embedding, str):
            # Parse string representation
            embedding = np.fromstring(embedding.strip('[]'), sep=',')
        else:
            embedding = np.array(embedding)

        embeddings.append(embedding)

        # Create data entry
        data.append({
            "id": str(id_val),
            "text": text[:500] if text else f"Chunk {id_val}",  # Limit text length
            "chunk_id": id_val,
            "text_length": len(text) if text else 0,
            "metadata": str(metadata) if metadata else ""
        })

    cursor.close()
    conn.close()

    return np.array(embeddings, dtype=np.float32), data


def create_atlas_map(embeddings, data, project_name=None):
    """Create Atlas map with embeddings"""

    try:
        from nomic import atlas
        import nomic
    except ImportError:
        print("\n❌ Nomic Atlas not installed!")
        print("\nInstall with:")
        print("  pip install nomic")
        print("\nThen login:")
        print("  nomic login")
        print("\n(You'll need a free Nomic account from https://atlas.nomic.ai)")
        sys.exit(1)

    print(f"\n🎨 Creating Atlas map...")

    if project_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"rag_{PGTABLE}_{timestamp}"

    print(f"   Project name: {project_name}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Data points: {len(data)}")

    try:
        # Create the map
        project = atlas.map_embeddings(
            embeddings=embeddings,
            data=data,
            id_field="id",
            name=project_name,
            description=f"RAG embeddings from {PGTABLE} table",
            colorable_fields=["text_length"],  # Color by text length
            is_public=False  # Keep private
        )

        print(f"\n✅ Atlas map created successfully!")
        print(f"\n🌐 View your map at:")
        print(f"   {project.map_link}")

        return project

    except Exception as e:
        print(f"\n❌ Error creating map: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: nomic login")
        print("2. Check your internet connection")
        print("3. Verify your Nomic account is active")
        raise


def main():
    """Main execution"""

    print("=" * 70)
    print("Atlas by Nomic - Embedding Map Creator")
    print("=" * 70)

    # Check if logged in
    try:
        from nomic import atlas
        # Try to get user info
        print("\n✓ Nomic library found")
    except ImportError:
        print("\n❌ Nomic not installed!")
        print("\nInstall:")
        print("  pip install nomic")
        print("\nThen:")
        print("  nomic login")
        sys.exit(1)

    try:
        # 1. Fetch embeddings from database
        embeddings, data = fetch_embeddings_from_db()

        if len(embeddings) == 0:
            print("❌ No embeddings found in database!")
            print(f"   Check that table 'data_{PGTABLE}' exists and has data.")
            sys.exit(1)

        print(f"\n📊 Summary:")
        print(f"   • Total embeddings: {len(embeddings)}")
        print(f"   • Embedding dimension: {embeddings.shape[1]}")
        print(f"   • Total text chunks: {len(data)}")

        # 2. Create Atlas map
        project = create_atlas_map(embeddings, data)

        # 3. Instructions
        print("\n" + "=" * 70)
        print("🎉 SUCCESS!")
        print("=" * 70)
        print("\n📍 Your embedding map is ready!")
        print("\n✨ Features:")
        print("   • Hover over points to see text")
        print("   • Click to select multiple points")
        print("   • Search for specific terms")
        print("   • Explore semantic clusters")
        print("   • Color by metadata fields")
        print("\n💡 Tip: Points close together = semantically similar content")
        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
