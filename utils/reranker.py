"""
Cross-encoder reranking for better retrieval quality.

Provides 15-30% improvement in answer relevance by:
1. Retrieving more candidates (e.g., 12) with fast bi-encoder
2. Reranking with slower but more accurate cross-encoder
3. Returning top-k (e.g., 4) most relevant chunks

Usage:
    # Basic usage with texts
    from utils.reranker import Reranker
    reranker = Reranker()
    results = reranker.rerank(query, texts, top_n=3)

    # Advanced usage with NodeWithScore objects
    reranked_nodes = reranker.rerank_nodes(query, nodes, top_k=4)
"""

import os
import logging
import argparse
from typing import List, Tuple, Any, Optional

# Try to import sentence-transformers
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False

log = logging.getLogger(__name__)


class Reranker:
    """
    Rerank retrieved chunks with cross-encoder for better precision.

    Cross-encoders process query+document together (unlike bi-encoders that
    encode separately), achieving 10-20% better relevance at the cost of speed.
    """

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize reranker with cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model
                       Default: cross-encoder/ms-marco-MiniLM-L-6-v2
            device: Device to run on (cpu/cuda/mps). Auto-detects if None.
        """
        if not CROSSENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        # Use environment variable or default
        if model_name is None:
            model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Auto-detect device
        if device is None:
            device = self._detect_device()

        log.info(f"Loading reranker: {model_name}")
        log.info(f"  Device: {device}")

        try:
            self.model = CrossEncoder(model_name, device=device)
            self.model_name = model_name
            log.info(f"✓ Reranker loaded successfully")
        except Exception as e:
            log.error(f"Failed to load reranker: {e}")
            raise

    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

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

    def rerank_nodes(
        self,
        query: str,
        nodes: List[Any],  # List[NodeWithScore]
        top_k: Optional[int] = None,
        batch_size: int = 32
    ) -> List[Any]:
        """
        Rerank NodeWithScore objects from retrieval.

        This is the main method for RAG pipelines.

        Args:
            query: Query text
            nodes: List of NodeWithScore objects from retrieval
            top_k: Number of top results to return (None = return all)
            batch_size: Batch size for prediction

        Returns:
            Reranked list of NodeWithScore objects with updated scores

        Example:
            >>> reranker = Reranker()
            >>> # Get 12 candidates from retrieval
            >>> candidates = retriever.retrieve(query, top_k=12)
            >>> # Rerank to top 4
            >>> reranked = reranker.rerank_nodes(query, candidates, top_k=4)
        """
        if not nodes:
            return nodes

        # Extract texts from nodes
        texts = [node.node.get_content() for node in nodes]

        log.info(f"\n🎯 Reranking {len(nodes)} candidates with cross-encoder...")
        log.info(f"  Model: {self.model_name}")
        log.info(f"  Query: \"{query[:80]}...\"" if len(query) > 80 else f"  Query: \"{query}\"")

        # Create query-text pairs
        pairs = [[query, text] for text in texts]

        # Get relevance scores
        try:
            scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            log.error(f"Reranking failed: {e}, returning original nodes")
            return nodes

        # Create list of (node, original_rank, rerank_score)
        results = []
        for i, (node, score) in enumerate(zip(nodes, scores)):
            original_score = node.score if hasattr(node, 'score') else 0.0
            results.append({
                'node': node,
                'original_rank': i + 1,
                'original_score': original_score,
                'rerank_score': float(score)
            })

        # Sort by rerank score (descending)
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Update node scores with rerank scores
        for result in results:
            result['node'].score = result['rerank_score']

        # Log results
        best_score = results[0]['rerank_score']
        worst_score = results[-1]['rerank_score']
        log.info(f"\n📊 Reranking Results:")
        log.info(f"  • Best rerank score: {best_score:.4f}")
        log.info(f"  • Worst rerank score: {worst_score:.4f}")

        # Show significant rank changes in top 5
        rank_changes = []
        for i, result in enumerate(results[:5]):
            new_rank = i + 1
            old_rank = result['original_rank']
            if abs(new_rank - old_rank) >= 2:  # Changed by 2+ positions
                rank_changes.append((old_rank, new_rank, result['rerank_score']))

        if rank_changes:
            log.info(f"  • Significant rank changes:")
            for old, new, score in rank_changes:
                arrow = "↑" if new < old else "↓"
                log.info(f"    {arrow} Rank {old} → {new} (score: {score:.4f})")

        # Return top-k or all
        reranked_nodes = [r['node'] for r in results]
        if top_k:
            reranked_nodes = reranked_nodes[:top_k]
            log.info(f"  • Returning top {top_k} results after reranking")

        return reranked_nodes


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
