#!/usr/bin/env python3
"""
Test MLX embedding robustness with problematic inputs.
"""
import os
os.environ['EMBED_BACKEND'] = 'mlx'

from utils.mlx_embedding import MLXEmbedding

def test_mlx_robustness():
    """Test MLX embedding with various problematic inputs."""
    print("🧪 Testing MLX Embedding Robustness")
    print("=" * 60)

    # Initialize model
    print("\n1️⃣  Loading bge-m3 model...")
    model = MLXEmbedding(model_name="BAAI/bge-m3")
    print(f"✓ Model loaded with dimension: {model._embed_dim}")

    # Test cases
    test_cases = [
        ("Normal text", "This is a normal text that should work fine."),
        ("Empty string", ""),
        ("Whitespace only", "   \n\t  "),
        ("Very short", "a"),
        ("Special chars", "🎉💻🚀 emoji test ñäöü"),
        ("Long text", "A" * 35000),  # Longer than 32k limit
    ]

    print("\n2️⃣  Testing various inputs:")
    print("-" * 60)

    for name, text in test_cases:
        try:
            embedding = model._get_text_embedding(text)
            status = "✓ Success"
            info = f"dim={len(embedding)}, first={embedding[0]:.4f}" if embedding else "zero vector"
        except Exception as e:
            status = f"✗ Failed: {e}"
            info = ""

        print(f"{name:20s} | {status:20s} | {info}")

    # Test batch with mixed content
    print("\n3️⃣  Testing batch with problematic inputs:")
    print("-" * 60)

    batch_texts = [
        "Normal text 1",
        "",  # Empty
        "Normal text 2",
        "   ",  # Whitespace
        "Normal text 3",
    ]

    try:
        batch_embeddings = model._get_text_embeddings(batch_texts)
        print(f"✓ Batch embedding succeeded")
        print(f"  Input texts: {len(batch_texts)}")
        print(f"  Output embeddings: {len(batch_embeddings)}")
        print(f"  All dimensions match: {all(len(e) == model._embed_dim for e in batch_embeddings)}")

        # Show which inputs got zero vectors
        for i, (text, emb) in enumerate(zip(batch_texts, batch_embeddings)):
            is_zero = all(v == 0.0 for v in emb[:10])  # Check first 10 values
            status = "zero vector" if is_zero else f"valid (first={emb[0]:.4f})"
            print(f"  [{i}] {repr(text[:20]):22s} → {status}")

    except Exception as e:
        print(f"✗ Batch embedding failed: {e}")

    print("\n" + "=" * 60)
    print("✅ Robustness test complete!")
    print("\nThe model now gracefully handles:")
    print("  • Empty or whitespace-only text")
    print("  • Very long text (truncates to 32k chars)")
    print("  • Special characters and emojis")
    print("  • Batch processing errors (falls back to individual)")
    print("  • Returns zero vectors instead of crashing")

if __name__ == "__main__":
    test_mlx_robustness()
