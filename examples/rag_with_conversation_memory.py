"""
RAG Pipeline with Conversational Memory Integration

This example shows how to integrate conversation memory with the
actual RAG pipeline for multi-turn dialogues.

Usage:
    # Single user conversation
    python examples/rag_with_conversation_memory.py --query-only --interactive

    # Multi-user sessions
    python examples/rag_with_conversation_memory.py --query-only --interactive --session user_123

    # Disable conversation memory
    ENABLE_CONVERSATION_MEMORY=0 python examples/rag_with_conversation_memory.py --query-only
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.conversation_memory import (
    ConversationMemory,
    SessionManager,
    session_manager,
)


def create_conversational_rag_wrapper(
    base_query_function,
    session_id: str = None,
    use_reformulation: bool = True,
    use_context_injection: bool = True,
    max_context_turns: int = 3,
):
    """
    Create a conversational wrapper around a RAG query function.

    This wrapper adds conversational capabilities to any RAG query function:
    - Reference resolution (pronouns, "it", "that", etc.)
    - Query reformulation with conversation context
    - Context injection into LLM prompts
    - Conversation history tracking

    Args:
        base_query_function: Original RAG query function (query_str -> response)
        session_id: Session ID for this conversation (None = standalone)
        use_reformulation: Enable query reformulation with context
        use_context_injection: Inject conversation context into prompts
        max_context_turns: Maximum previous turns to include in context

    Returns:
        Wrapped query function with conversational capabilities

    Example:
        def my_rag_query(query: str) -> dict:
            # Your RAG implementation
            return {"answer": "...", "sources": [...]}

        # Wrap it
        conversational_rag = create_conversational_rag_wrapper(
            my_rag_query,
            session_id="user_123"
        )

        # Use it
        result = conversational_rag("What is RAG?")
        result = conversational_rag("How does it work?")  # References resolved automatically
    """

    # Get or create conversation memory
    if session_id:
        memory = session_manager.get_or_create(session_id)
    else:
        memory = ConversationMemory()

    def conversational_query(query: str, **kwargs) -> dict:
        """
        Execute RAG query with conversational enhancements.

        Args:
            query: User query
            **kwargs: Additional arguments for base query function

        Returns:
            Dict with answer, sources, and conversation metadata
        """
        print(f"\n{'='*70}")
        print(f"Processing conversational query...")
        print(f"Session: {session_id or 'standalone'}")
        print(f"Original query: {query}")

        # Step 1: Resolve references
        resolved_query = memory.resolve_references(query)
        if resolved_query != query:
            print(f"Resolved query: {resolved_query}")
            query_for_rag = resolved_query
        else:
            query_for_rag = query

        # Step 2: Reformulate with context (optional)
        reformulated_query = None
        if use_reformulation and len(memory.turns) > 0:
            reformulated_query = memory.reformulate_query(
                query_for_rag,
                max_context_turns=max_context_turns
            )
            print(f"Reformulated query: {reformulated_query[:100]}...")
            query_for_rag = reformulated_query

        # Step 3: Get conversation context for LLM prompt (optional)
        conversation_context = ""
        if use_context_injection and len(memory.turns) > 0:
            conversation_context = memory.get_conversation_context(
                max_turns=max_context_turns,
                include_summaries=True
            )

        # Step 4: Execute base RAG query
        # Pass conversation context if the function supports it
        if use_context_injection and conversation_context:
            # Try to pass context to base function
            try:
                result = base_query_function(
                    query_for_rag,
                    conversation_context=conversation_context,
                    **kwargs
                )
            except TypeError:
                # Function doesn't accept conversation_context
                result = base_query_function(query_for_rag, **kwargs)
        else:
            result = base_query_function(query_for_rag, **kwargs)

        # Ensure result is a dict
        if isinstance(result, str):
            result = {"answer": result}

        # Step 5: Store turn in memory
        memory.add_turn(
            original_query=query,
            answer=result.get("answer", ""),
            resolved_query=resolved_query if resolved_query != query else None,
            reformulated_query=reformulated_query,
            context={
                "sources": result.get("sources", []),
                "retrieval_scores": result.get("scores", []),
                "turn_count": memory.turn_count,
            }
        )

        # Add conversation metadata to result
        result["conversation"] = {
            "turn_number": memory.turn_count - 1,
            "session_id": session_id,
            "resolved_query": resolved_query if resolved_query != query else None,
            "reformulated": bool(reformulated_query),
            "entities_tracked": len(memory.entities),
            "total_turns": memory.turn_count,
        }

        print(f"Turn {memory.turn_count - 1} completed")
        print(f"{'='*70}\n")

        return result

    # Attach memory to wrapper for external access
    conversational_query.memory = memory

    return conversational_query


def example_simple_integration():
    """Example: Simple RAG function integration"""
    print("="*70)
    print("EXAMPLE: Simple RAG Integration")
    print("="*70)

    # Mock RAG function
    def simple_rag_query(query: str) -> str:
        """Simplified RAG query that returns just an answer"""
        # In real implementation, this would:
        # 1. Compute query embedding
        # 2. Retrieve from vector store
        # 3. Generate answer with LLM
        return f"Mock answer to: {query}"

    # Wrap with conversational capabilities
    conversational_rag = create_conversational_rag_wrapper(
        simple_rag_query,
        session_id="demo_simple",
        use_reformulation=True,
    )

    # Use it
    print("\nTurn 1:")
    result1 = conversational_rag("What is the chunk size?")
    print(f"Answer: {result1['answer']}")

    print("\nTurn 2:")
    result2 = conversational_rag("What about the overlap?")
    print(f"Answer: {result2['answer']}")

    print("\nTurn 3:")
    result3 = conversational_rag("How do I change it?")
    print(f"Answer: {result3['answer']}")

    # Show conversation state
    print("\n" + "="*70)
    print("Conversation Summary:")
    print("="*70)
    print(conversational_rag.memory.summarize_conversation())


def example_full_integration():
    """Example: Full RAG function with sources and scores"""
    print("\n\n" + "="*70)
    print("EXAMPLE: Full RAG Integration with Sources")
    print("="*70)

    # Mock full RAG function
    def full_rag_query(query: str, conversation_context: str = "") -> dict:
        """Full RAG query returning answer, sources, and scores"""
        # In real implementation, this would:
        # 1. Compute query embedding
        # 2. Retrieve top-k chunks from vector store
        # 3. Optionally use conversation_context in LLM prompt
        # 4. Generate answer with citations
        return {
            "answer": f"Mock answer to: {query}",
            "sources": ["doc1.pdf", "doc2.pdf"],
            "scores": [0.92, 0.87],
            "retrieved_chunks": 5,
        }

    # Wrap with conversational capabilities
    conversational_rag = create_conversational_rag_wrapper(
        full_rag_query,
        session_id="demo_full",
        use_reformulation=True,
        use_context_injection=True,
        max_context_turns=3,
    )

    # Use it
    queries = [
        "What is RAG?",
        "How does it work?",
        "What are the benefits?",
        "Can you show me an example?",
    ]

    for i, query in enumerate(queries):
        print(f"\nTurn {i+1}:")
        result = conversational_rag(query)
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Conversation metadata: {result['conversation']}")


def example_interactive_session():
    """Example: Interactive conversational RAG session"""
    print("\n\n" + "="*70)
    print("EXAMPLE: Interactive Conversational Session")
    print("="*70)

    def mock_rag_query(query: str, conversation_context: str = "") -> dict:
        """Mock RAG for interactive demo"""
        # Simulate different responses based on query
        responses = {
            "chunk": "The default chunk size is 700 characters with 150 overlap.",
            "overlap": "Overlap helps maintain context between chunks. 15-25% is recommended.",
            "embedding": "The default embedding model is bge-small-en with 384 dimensions.",
            "database": "PostgreSQL with pgvector extension is used for vector storage.",
        }

        # Simple keyword matching
        answer = "I don't have specific information about that."
        for keyword, response in responses.items():
            if keyword in query.lower():
                answer = response
                break

        return {
            "answer": answer,
            "sources": ["docs/ENVIRONMENT_VARIABLES.md", "README.md"],
        }

    # Create conversational wrapper
    conversational_rag = create_conversational_rag_wrapper(
        mock_rag_query,
        session_id="demo_interactive",
        use_reformulation=True,
        use_context_injection=True,
    )

    # Predefined conversation for demo
    demo_queries = [
        "What is the chunk size?",
        "What about the overlap?",
        "Tell me about the embedding model",
        "Which database is used?",
    ]

    print("\nSimulated interactive conversation:")
    print("-"*70)

    for query in demo_queries:
        print(f"\nUser: {query}")
        result = conversational_rag(query)
        print(f"Assistant: {result['answer']}")

        # Show reference resolution if happened
        if result['conversation'].get('resolved_query'):
            print(f"  (Resolved: {result['conversation']['resolved_query']})")

    # Show final state
    print("\n" + "="*70)
    print("Final Conversation State:")
    print("="*70)
    stats = conversational_rag.memory.stats()
    print(f"Total turns: {stats['total_turns']}")
    print(f"Active turns: {stats['active_turns']}")
    print(f"Entities tracked: {stats['entities']}")
    print(f"Topics discussed: {stats['topics']}")


def example_real_rag_integration():
    """Example: Integration with actual RAG pipeline"""
    print("\n\n" + "="*70)
    print("EXAMPLE: Real RAG Pipeline Integration Pattern")
    print("="*70)

    print("""
This example shows the pattern for integrating conversation memory
with the actual rag_low_level_m1_16gb_verbose.py pipeline.

Pattern 1: Modify run_query() function
----------------------------------------------------------------------
# In rag_low_level_m1_16gb_verbose.py

from utils.conversation_memory import session_manager

def run_query_conversational(
    engine,
    question: str,
    session_id: str = None,
    enable_memory: bool = True,
):
    \"\"\"Run RAG query with conversational memory\"\"\"

    if not enable_memory:
        # Original behavior
        return run_query(engine, question)

    # Get conversation memory
    memory = session_manager.get_or_create(session_id or "default")

    # Resolve references
    resolved_query = memory.resolve_references(question)

    # Reformulate with context
    if len(memory.turns) > 0:
        reformulated_query = memory.reformulate_query(resolved_query)
    else:
        reformulated_query = resolved_query

    # Execute RAG with reformulated query
    response = engine.query(reformulated_query)

    # Store in memory
    memory.add_turn(
        original_query=question,
        answer=str(response),
        resolved_query=resolved_query,
        reformulated_query=reformulated_query,
    )

    return response


Pattern 2: Create wrapper class
----------------------------------------------------------------------
# In rag_low_level_m1_16gb_verbose.py

from utils.conversation_memory import ConversationMemory

class ConversationalQueryEngine:
    \"\"\"Wrapper around query engine with conversation memory\"\"\"

    def __init__(self, base_engine, session_id: str = None):
        self.base_engine = base_engine
        self.memory = ConversationMemory(conversation_id=session_id)

    def query(self, query_str: str):
        # Resolve + reformulate
        resolved = self.memory.resolve_references(query_str)
        reformulated = self.memory.reformulate_query(resolved)

        # Execute
        response = self.base_engine.query(reformulated)

        # Store
        self.memory.add_turn(query_str, str(response))

        return response


Pattern 3: Interactive CLI with sessions
----------------------------------------------------------------------
# In rag_interactive.py or rag_web.py

def interactive_rag_cli():
    \"\"\"Interactive CLI with conversation memory\"\"\"

    session_id = input("Enter session ID (or press Enter for new): ")
    if not session_id:
        session_id = None

    memory = session_manager.get_or_create(session_id or "default")

    print(f"Session: {memory.conversation_id}")
    print(f"Previous turns: {memory.turn_count}")

    while True:
        query = input("\\nQuery (or 'quit'): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break

        # Process with memory
        resolved = memory.resolve_references(query)
        result = run_rag_query(resolved)

        print(f"Answer: {result}")

        # Store
        memory.add_turn(query, result, resolved_query=resolved)


Environment Variables
----------------------------------------------------------------------
# Enable conversation memory
export ENABLE_CONVERSATION_MEMORY=1

# Configure behavior
export MAX_CONVERSATION_TURNS=10
export AUTO_SUMMARIZE=1
export SUMMARIZE_THRESHOLD=5

# Run with memory enabled
python rag_low_level_m1_16gb_verbose.py --query-only --interactive


Benefits
----------------------------------------------------------------------
1. Natural multi-turn conversations
2. Automatic reference resolution ("it", "that", etc.)
3. Context-aware retrieval (reformulated queries)
4. Session management (multiple users)
5. Automatic summarization (long conversations)
6. Disk persistence (recover sessions)
    """)


def main():
    """Run all examples"""
    print("\n")
    print("="*70)
    print("RAG WITH CONVERSATIONAL MEMORY - INTEGRATION EXAMPLES")
    print("="*70)

    # Run examples
    example_simple_integration()
    example_full_integration()
    example_interactive_session()
    example_real_rag_integration()

    print("\n\n" + "="*70)
    print("All integration examples completed!")
    print("="*70)

    print("\n\nNext Steps:")
    print("-"*70)
    print("1. Review the integration patterns above")
    print("2. Choose the pattern that fits your use case")
    print("3. Modify your RAG pipeline to use conversation memory")
    print("4. Test with multi-turn conversations")
    print("5. Monitor memory usage and tune configuration")
    print("\nFor more examples, see:")
    print("  - examples/conversation_memory_demo.py")
    print("  - utils/conversation_memory.py (tests at bottom)")
    print("="*70)


if __name__ == "__main__":
    main()
