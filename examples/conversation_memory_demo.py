"""
Conversation Memory Integration Examples

This script demonstrates how to integrate conversational memory
with the RAG pipeline for multi-turn dialogues.

Examples:
    1. Basic multi-turn conversation
    2. Reference resolution
    3. Query reformulation with context
    4. Session-based conversations (multi-user)
    5. Integration with RAG pipeline
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.conversation_memory import (
    ConversationMemory,
    SessionManager,
    get_conversation_memory,
)


def example_1_basic_conversation():
    """Example 1: Basic multi-turn conversation with memory"""
    print("=" * 70)
    print("Example 1: Basic Multi-Turn Conversation")
    print("=" * 70)

    memory = ConversationMemory()

    # Turn 1
    print("\nTurn 1:")
    query1 = "What is Retrieval-Augmented Generation?"
    answer1 = (
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with text generation. It retrieves relevant "
        "documents from a knowledge base and uses them to augment the prompt "
        "for a language model, resulting in more accurate and grounded responses."
    )
    memory.add_turn(query1, answer1)
    print(f"Q: {query1}")
    print(f"A: {answer1}")

    # Turn 2 (with reference)
    print("\nTurn 2:")
    query2 = "How does it work?"
    resolved_query2 = memory.resolve_references(query2)
    answer2 = (
        "RAG works in three steps: 1) Retrieve: Find relevant documents "
        "using semantic search, 2) Augment: Add retrieved context to the prompt, "
        "3) Generate: Use LLM to generate response with context."
    )
    memory.add_turn(query2, answer2, resolved_query=resolved_query2)
    print(f"Q: {query2}")
    print(f"Q (resolved): {resolved_query2}")
    print(f"A: {answer2}")

    # Turn 3 (follow-up)
    print("\nTurn 3:")
    query3 = "What are the benefits?"
    reformulated_query3 = memory.reformulate_query(query3, max_context_turns=2)
    answer3 = (
        "Benefits include: improved accuracy, reduced hallucinations, "
        "up-to-date information, citations/sources, and domain-specific knowledge."
    )
    memory.add_turn(query3, answer3, reformulated_query=reformulated_query3)
    print(f"Q: {query3}")
    print(f"Q (reformulated): {reformulated_query3[:150]}...")
    print(f"A: {answer3}")

    # Show conversation summary
    print("\n" + "=" * 70)
    print("Conversation Summary:")
    print("=" * 70)
    print(memory.summarize_conversation())


def example_2_reference_resolution():
    """Example 2: Reference resolution with entity tracking"""
    print("\n\n" + "=" * 70)
    print("Example 2: Reference Resolution")
    print("=" * 70)

    memory = ConversationMemory()

    # Turn 1: Introduce entity
    query1 = 'What is "LlamaIndex"?'
    answer1 = (
        "LlamaIndex is a data framework for LLM applications. "
        "It provides tools for ingesting, structuring, and accessing data."
    )
    memory.add_turn(query1, answer1)
    print(f"\nTurn 1:")
    print(f"Q: {query1}")
    print(f"A: {answer1}")
    print(f"Entities extracted: {memory.entities}")

    # Turn 2: Use pronoun reference
    query2 = "How do I install it?"
    resolved2 = memory.resolve_references(query2)
    answer2 = "You can install LlamaIndex with: pip install llama-index"
    memory.add_turn(query2, answer2, resolved_query=resolved2)
    print(f"\nTurn 2:")
    print(f"Q: {query2}")
    print(f"Q (resolved): {resolved2}")
    print(f"A: {answer2}")

    # Turn 3: Another reference
    query3 = "What are the key features of that?"
    resolved3 = memory.resolve_references(query3)
    answer3 = (
        "Key features include: document loaders, vector stores, "
        "query engines, and integration with many LLMs."
    )
    memory.add_turn(query3, answer3, resolved_query=resolved3)
    print(f"\nTurn 3:")
    print(f"Q: {query3}")
    print(f"Q (resolved): {resolved3}")
    print(f"A: {answer3}")


def example_3_session_management():
    """Example 3: Multi-user session management"""
    print("\n\n" + "=" * 70)
    print("Example 3: Multi-User Session Management")
    print("=" * 70)

    manager = SessionManager(max_sessions=10, session_timeout=3600)

    # User 1 session
    print("\nUser 1 (Alice):")
    alice_memory = manager.get_or_create("user_alice")
    alice_memory.add_turn(
        "What is machine learning?",
        "Machine learning is a subset of AI that learns from data."
    )
    alice_memory.add_turn(
        "What are the main types?",
        "The main types are supervised, unsupervised, and reinforcement learning."
    )
    print(f"  Total turns: {alice_memory.turn_count}")
    print(f"  Entities: {alice_memory.entities}")

    # User 2 session
    print("\nUser 2 (Bob):")
    bob_memory = manager.get_or_create("user_bob")
    bob_memory.add_turn(
        "How do I use pgvector?",
        "pgvector is a PostgreSQL extension for vector similarity search."
    )
    bob_memory.add_turn(
        "Can you show me an example?",
        "Sure! Here's how to create a vector column: CREATE TABLE items (embedding vector(384));"
    )
    print(f"  Total turns: {bob_memory.turn_count}")
    print(f"  Entities: {bob_memory.entities}")

    # User 3 session
    print("\nUser 3 (Carol):")
    carol_memory = manager.get_or_create("user_carol")
    carol_memory.add_turn(
        "What's the best embedding model?",
        "Popular choices include bge-small-en, all-MiniLM-L6-v2, and text-embedding-ada-002."
    )
    print(f"  Total turns: {carol_memory.turn_count}")

    # Manager statistics
    print("\n" + "=" * 70)
    print("Session Manager Statistics:")
    print("=" * 70)
    stats = manager.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # List active sessions
    print(f"\nActive sessions: {manager.list_active_sessions()}")


def example_4_rag_integration():
    """Example 4: Integration with RAG pipeline"""
    print("\n\n" + "=" * 70)
    print("Example 4: RAG Pipeline Integration")
    print("=" * 70)

    def mock_rag_query(query: str, context: str = "") -> str:
        """Mock RAG query function for demonstration"""
        # In real implementation, this would call the actual RAG pipeline
        return f"[RAG Response to: {query}]"

    def conversational_rag_query(
        query: str,
        session_id: str,
        manager: SessionManager
    ) -> dict:
        """
        RAG query with conversational memory.

        This function shows how to integrate conversation memory
        into your RAG pipeline.
        """
        # Get or create conversation memory for this session
        memory = manager.get_or_create(session_id)

        print(f"\n{'='*60}")
        print(f"Processing query for session: {session_id}")
        print(f"Query: {query}")

        # Step 1: Resolve references from previous turns
        resolved_query = memory.resolve_references(query)
        if resolved_query != query:
            print(f"Resolved: {resolved_query}")

        # Step 2: Get conversation context (optional - for LLM context)
        conversation_context = memory.get_conversation_context(max_turns=3)

        # Step 3: Reformulate query with context (optional - for better retrieval)
        reformulated_query = memory.reformulate_query(resolved_query, max_context_turns=2)
        if len(memory.turns) > 0:
            print(f"Reformulated: {reformulated_query[:100]}...")

        # Step 4: Run RAG query (use reformulated query for retrieval)
        # In real implementation:
        # - Use reformulated_query for vector search
        # - Add conversation_context to LLM prompt
        rag_answer = mock_rag_query(reformulated_query or resolved_query)

        print(f"Answer: {rag_answer}")

        # Step 5: Store turn in memory
        memory.add_turn(
            original_query=query,
            answer=rag_answer,
            resolved_query=resolved_query,
            reformulated_query=reformulated_query if len(memory.turns) > 0 else None,
            context={
                "conversation_context": conversation_context[:100] + "..." if conversation_context else "",
                "turn_count": memory.turn_count,
            }
        )

        return {
            "answer": rag_answer,
            "original_query": query,
            "resolved_query": resolved_query,
            "reformulated_query": reformulated_query,
            "turn_number": memory.turn_count - 1,
        }

    # Simulate conversation
    manager = SessionManager()
    session_id = "demo_user_001"

    # Clear any existing demo session
    if manager.get(session_id):
        manager.delete(session_id)

    # Turn 1
    result1 = conversational_rag_query(
        "What is the chunk size for indexing?",
        session_id,
        manager
    )

    # Turn 2 (with reference)
    result2 = conversational_rag_query(
        "What about the overlap?",
        session_id,
        manager
    )

    # Turn 3 (with pronoun)
    result3 = conversational_rag_query(
        "How do I change it?",
        session_id,
        manager
    )

    # Show final conversation state
    memory = manager.get(session_id)
    print("\n" + "=" * 60)
    print("Final Conversation State:")
    print("=" * 60)
    print(memory.summarize_conversation())


def example_5_auto_summarization():
    """Example 5: Automatic conversation summarization"""
    print("\n\n" + "=" * 70)
    print("Example 5: Automatic Summarization")
    print("=" * 70)

    memory = ConversationMemory(
        conversation_id="demo_long_conversation",
        auto_summarize=True,
        summarize_threshold=4,  # Summarize after 4 turns
        max_turns=10
    )

    # Clear any existing data
    memory.clear_conversation()

    print("\nAdding multiple turns...")

    # Simulate long conversation
    topics = [
        ("What is RAG?", "RAG is Retrieval-Augmented Generation..."),
        ("How does retrieval work?", "Retrieval uses vector similarity search..."),
        ("What about embedding models?", "Embedding models convert text to vectors..."),
        ("Which model is best?", "bge-small-en is a good choice for English..."),
        ("How do I use pgvector?", "pgvector is a Postgres extension..."),
        ("Can you show an example?", "Sure! Here's how to create a table..."),
        ("What about indexing?", "You can create an HNSW or IVFFlat index..."),
        ("Which index is faster?", "HNSW is generally faster but uses more memory..."),
    ]

    for i, (query, answer) in enumerate(topics):
        memory.add_turn(query, answer)
        print(f"  Turn {i+1}: {query[:40]}...")

        # Show summarization when it happens
        if i == 4:  # After threshold
            print(f"\n  >>> Auto-summarization triggered! <<<")
            print(f"      Active turns: {len(memory.turns)}")
            print(f"      Summaries: {len(memory.summaries)}")
            if memory.summaries:
                print(f"      Latest summary: {memory.summaries[-1].summary_text}")

    print(f"\nFinal state:")
    print(f"  Total turns: {memory.turn_count}")
    print(f"  Active turns in memory: {len(memory.turns)}")
    print(f"  Summaries created: {len(memory.summaries)}")

    # Show summaries
    if memory.summaries:
        print(f"\n{'='*60}")
        print("Conversation Summaries:")
        print("=" * 60)
        for i, summary in enumerate(memory.summaries):
            print(f"\nSummary {i+1}:")
            print(f"  Turns: {summary.turn_range[0]}-{summary.turn_range[1]}")
            print(f"  Text: {summary.summary_text}")
            print(f"  Topics: {', '.join(summary.topics[:5])}")


def example_6_environment_config():
    """Example 6: Environment variable configuration"""
    print("\n\n" + "=" * 70)
    print("Example 6: Environment Variable Configuration")
    print("=" * 70)

    print("\nAvailable environment variables:")
    print("-" * 60)

    env_vars = [
        ("ENABLE_CONVERSATION_MEMORY", "1", "Enable/disable conversation memory"),
        ("MAX_CONVERSATION_TURNS", "10", "Maximum turns to keep in memory"),
        ("CONVERSATION_TIMEOUT", "3600", "Session timeout in seconds (1 hour)"),
        ("AUTO_SUMMARIZE", "1", "Auto-summarize long conversations"),
        ("SUMMARIZE_THRESHOLD", "5", "Turns before summarization"),
        ("CONVERSATION_CACHE_DIR", ".cache/conversations", "Storage directory"),
    ]

    for var_name, default, description in env_vars:
        current_value = os.getenv(var_name, default)
        print(f"\n{var_name}={current_value}")
        print(f"  Default: {default}")
        print(f"  Description: {description}")

    print("\n" + "-" * 60)
    print("\nExample usage:")
    print("-" * 60)
    print("""
# Disable conversation memory
export ENABLE_CONVERSATION_MEMORY=0

# Increase turn limit
export MAX_CONVERSATION_TURNS=20

# Reduce summarization threshold
export SUMMARIZE_THRESHOLD=3

# Custom cache directory
export CONVERSATION_CACHE_DIR=/tmp/my_conversations

# Then run your RAG pipeline
python rag_low_level_m1_16gb_verbose.py --interactive
    """)


def main():
    """Run all examples"""
    print("\n")
    print("=" * 70)
    print("CONVERSATION MEMORY INTEGRATION EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_1_basic_conversation()
    example_2_reference_resolution()
    example_3_session_management()
    example_4_rag_integration()
    example_5_auto_summarization()
    example_6_environment_config()

    print("\n\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
