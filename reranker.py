"""
Cross-encoder reranking for better retrieval quality.

Usage:
    python reranker.py --test

    # In your code:
    from reranker import Reranker
    reranker = Reranker()
    results = reranker.rerank(query, texts, top_n=3)
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple
import logging
import argparse

log = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved chunks with cross-encoder for better precision"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model
                       Default is optimized for semantic search (trained on MS MARCO)
        """
        log.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        texts: List[str],
        top_n: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Rerank texts by relevance to query.

        Args:
            query: Query string
            texts: List of text chunks to rerank
            top_n: Number of top results to return

        Returns:
            List of (index, score) tuples, sorted by relevance score (descending)

        Example:
            >>> reranker = Reranker()
            >>> texts = ["text1", "text2", "text3"]
            >>> results = reranker.rerank("query", texts, top_n=2)
            >>> [(idx, score), ...] = results
        """
        if not texts:
            return []

        # Create query-text pairs for cross-encoder
        pairs = [[query, text] for text in texts]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort by score (descending) and return top_n
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def rerank_with_texts(
        self,
        query: str,
        texts: List[str],
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Rerank and return texts with scores (instead of indices).

        Returns:
            List of (text, score) tuples
        """
        ranked_indices = self.rerank(query, texts, top_n)
        return [(texts[idx], score) for idx, score in ranked_indices]


def main():
    parser = argparse.ArgumentParser(description="Test cross-encoder reranking")
    parser.add_argument("--query", default="What did Elena say about Morocco?")
    parser.add_argument("--top-n", type=int, default=2)
    args = parser.parse_args()

    # Test data
    texts = [
        "Elena mentioned she wants to visit Morocco next summer for the food.",
        "The weather in Morocco is nice this time of year.",
        "I love Moroccan food, especially tagine and couscous.",
        "Elena and I discussed travel plans yesterday, including Morocco.",
        "Morocco has beautiful architecture and vibrant markets.",
    ]

    print("="*70)
    print("Cross-Encoder Reranking Test")
    print("="*70)
    print(f"\nQuery: {args.query}")
    print(f"\nCandidate texts ({len(texts)} total):")
    for i, text in enumerate(texts):
        print(f"  [{i}] {text}")

    # Initialize reranker
    print("\nInitializing reranker...")
    reranker = Reranker()

    # Rerank
    print(f"\nReranking to find top {args.top_n}...")
    results = reranker.rerank(args.query, texts, top_n=args.top_n)

    print(f"\nTop {args.top_n} results:")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   Index: {idx}")
        print(f"   Text: {texts[idx]}")

    print("\n" + "="*70)
    print("✓ Reranking test complete")
    print("="*70)


if __name__ == "__main__":
    main()
