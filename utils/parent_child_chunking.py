"""
Parent-Child Chunking Strategy for RAG

This module implements parent-child chunking where:
- Small child chunks are used for precise retrieval
- Large parent chunks are returned to LLM for better context

Benefits:
- Retrieve with precision (small chunks match specific facts)
- Generate with context (large chunks provide full information)
- Best of both worlds approach

Usage:
    from utils.parent_child_chunking import ParentChildChunker
    
    chunker = ParentChildChunker(child_size=200, parent_size=800)
    parent_child_pairs = chunker.create_chunks(documents)
    
    # During retrieval: search child chunks
    # After retrieval: return parent chunks to LLM
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from llama_index.core.schema import TextNode

log = logging.getLogger(__name__)


@dataclass
class ChunkPair:
    """Represents a parent-child chunk relationship."""
    parent_id: str
    parent_text: str
    parent_start: int
    parent_end: int
    child_id: str
    child_text: str
    child_start: int
    child_end: int
    metadata: Dict = None


class ParentChildChunker:
    """
    Create parent-child chunk hierarchies for optimal RAG.
    
    Strategy:
    1. Create large parent chunks (e.g., 800 chars) with context
    2. Create small child chunks (e.g., 200 chars) for precision
    3. Link each child to its parent
    4. Store both in database with relationship metadata
    5. Retrieve using child chunks, return parent chunks
    
    Environment Variables:
        ENABLE_PARENT_CHILD=1 - Enable parent-child chunking (default: 0)
        PARENT_CHUNK_SIZE=800 - Size of parent chunks (default: 800)
        CHILD_CHUNK_SIZE=200 - Size of child chunks (default: 200)
        PARENT_OVERLAP=100 - Overlap between parent chunks (default: 100)
        CHILD_OVERLAP=50 - Overlap between child chunks (default: 50)
    """
    
    def __init__(
        self,
        parent_size: int = None,
        child_size: int = None,
        parent_overlap: int = None,
        child_overlap: int = None,
    ):
        """
        Initialize parent-child chunker.
        
        Args:
            parent_size: Size of parent chunks in characters
            child_size: Size of child chunks in characters
            parent_overlap: Overlap between parent chunks
            child_overlap: Overlap between child chunks
        """
        self.parent_size = int(parent_size or os.getenv("PARENT_CHUNK_SIZE", "800"))
        self.child_size = int(child_size or os.getenv("CHILD_CHUNK_SIZE", "200"))
        self.parent_overlap = int(parent_overlap or os.getenv("PARENT_OVERLAP", "100"))
        self.child_overlap = int(child_overlap or os.getenv("CHILD_OVERLAP", "50"))
        
        # Validation
        if self.child_size >= self.parent_size:
            raise ValueError("Child chunk size must be smaller than parent chunk size")
        
        log.info(f"ParentChildChunker initialized:")
        log.info(f"  Parent: {self.parent_size} chars, overlap {self.parent_overlap}")
        log.info(f"  Child: {self.child_size} chars, overlap {self.child_overlap}")
    
    def create_chunks(self, text: str, doc_id: str = "doc") -> List[ChunkPair]:
        """
        Create parent-child chunk pairs from text.
        
        Args:
            text: Text to chunk
            doc_id: Document identifier
            
        Returns:
            List of ChunkPair objects with parent-child relationships
        """
        parent_chunks = self._create_parent_chunks(text)
        
        all_pairs = []
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_id}_parent_{parent_idx}"
            parent_text, parent_start, parent_end = parent_chunk
            
            # Create child chunks within this parent
            child_chunks = self._create_child_chunks(
                parent_text,
                global_offset=parent_start
            )
            
            for child_idx, child_chunk in enumerate(child_chunks):
                child_id = f"{parent_id}_child_{child_idx}"
                child_text, child_start, child_end = child_chunk
                
                pair = ChunkPair(
                    parent_id=parent_id,
                    parent_text=parent_text,
                    parent_start=parent_start,
                    parent_end=parent_end,
                    child_id=child_id,
                    child_text=child_text,
                    child_start=child_start,
                    child_end=child_end,
                    metadata={
                        "doc_id": doc_id,
                        "parent_idx": parent_idx,
                        "child_idx": child_idx,
                        "is_parent_child": True,
                    }
                )
                all_pairs.append(pair)
        
        log.info(f"Created {len(parent_chunks)} parent chunks, {len(all_pairs)} child chunks")
        return all_pairs
    
    def _create_parent_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """Create parent chunks with overlap."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.parent_size, text_len)
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))
            
            if end >= text_len:
                break
                
            start += (self.parent_size - self.parent_overlap)
        
        return chunks
    
    def _create_child_chunks(
        self,
        parent_text: str,
        global_offset: int = 0
    ) -> List[Tuple[str, int, int]]:
        """Create child chunks within a parent chunk."""
        chunks = []
        start = 0
        text_len = len(parent_text)
        
        while start < text_len:
            end = min(start + self.child_size, text_len)
            chunk_text = parent_text[start:end]
            
            # Global offsets for tracking position in original document
            global_start = global_offset + start
            global_end = global_offset + end
            
            chunks.append((chunk_text, global_start, global_end))
            
            if end >= text_len:
                break
                
            start += (self.child_size - self.child_overlap)
        
        return chunks
    
    def create_text_nodes(
        self,
        chunk_pairs: List[ChunkPair],
        store_parent: bool = True
    ) -> Tuple[List[TextNode], List[TextNode]]:
        """
        Convert chunk pairs to TextNode objects for LlamaIndex.
        
        Args:
            chunk_pairs: List of ChunkPair objects
            store_parent: If True, return both child and parent nodes
            
        Returns:
            Tuple of (child_nodes, parent_nodes)
        """
        child_nodes = []
        parent_nodes = []
        parent_ids_seen = set()
        
        for pair in chunk_pairs:
            # Create child node (for retrieval)
            child_node = TextNode(
                text=pair.child_text,
                id_=pair.child_id,
                metadata={
                    **pair.metadata,
                    "parent_id": pair.parent_id,
                    "chunk_type": "child",
                }
            )
            child_nodes.append(child_node)
            
            # Create parent node (for LLM context) - only once per parent
            if store_parent and pair.parent_id not in parent_ids_seen:
                parent_node = TextNode(
                    text=pair.parent_text,
                    id_=pair.parent_id,
                    metadata={
                        **pair.metadata,
                        "chunk_type": "parent",
                    }
                )
                parent_nodes.append(parent_node)
                parent_ids_seen.add(pair.parent_id)
        
        log.info(f"Created {len(child_nodes)} child nodes, {len(parent_nodes)} parent nodes")
        return child_nodes, parent_nodes


def is_enabled() -> bool:
    """Check if parent-child chunking is enabled."""
    return os.getenv("ENABLE_PARENT_CHILD", "0") == "1"


if __name__ == "__main__":
    # Test parent-child chunking
    print("="*70)
    print("Parent-Child Chunking Test")
    print("="*70)
    
    # Sample text
    text = """
    Machine learning is a subset of artificial intelligence that focuses on 
    enabling systems to learn from data. Deep learning is a specialized area 
    within machine learning that uses neural networks with multiple layers. 
    Neural networks are inspired by biological neurons and consist of 
    interconnected nodes that process information. Training a neural network 
    involves adjusting weights through backpropagation to minimize error.
    The transformer architecture revolutionized natural language processing 
    by introducing self-attention mechanisms that allow models to process 
    entire sequences in parallel rather than sequentially.
    """.strip()
    
    print(f"\n1. Testing with sample text ({len(text)} chars)")
    print(f"   Text preview: {text[:100]}...")
    
    print("\n2. Creating parent-child chunks...")
    chunker = ParentChildChunker(
        parent_size=200,
        child_size=80,
        parent_overlap=50,
        child_overlap=20
    )
    
    pairs = chunker.create_chunks(text, doc_id="test_doc")
    
    print(f"\n3. Created {len(pairs)} child chunks")
    print(f"   Parent chunks: {len(set(p.parent_id for p in pairs))}")
    
    print("\n4. Sample parent-child relationships:")
    for i, pair in enumerate(pairs[:3]):  # Show first 3
        print(f"\n   Pair {i+1}:")
        print(f"   Parent ID: {pair.parent_id}")
        print(f"   Parent text: {pair.parent_text[:80]}...")
        print(f"   Child ID: {pair.child_id}")
        print(f"   Child text: {pair.child_text[:60]}...")
    
    print("\n5. Converting to TextNodes...")
    child_nodes, parent_nodes = chunker.create_text_nodes(pairs)
    
    print(f"   Child nodes: {len(child_nodes)}")
    print(f"   Parent nodes: {len(parent_nodes)}")
    
    print("\n6. Verifying parent-child links...")
    for child in child_nodes[:2]:
        parent_id = child.metadata.get("parent_id")
        print(f"   Child {child.id_} → Parent {parent_id}")
    
    print("\n" + "="*70)
    print("✓ Parent-child chunking test complete")
    print("="*70)
    print("\nUsage in RAG pipeline:")
    print("1. Store child nodes in vector DB (for retrieval)")
    print("2. Store parent nodes separately or in metadata")
    print("3. Retrieve using child nodes (precise matching)")
    print("4. Return parent nodes to LLM (full context)")
    print("5. Result: Best of both worlds!")
