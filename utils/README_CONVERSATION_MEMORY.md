# Conversation Memory Module

**Purpose**: Enable natural multi-turn dialogues with context awareness for RAG applications.

## Overview

The conversation memory module provides intelligent conversation tracking and context management for multi-turn RAG dialogues. It handles:

- **Reference Resolution**: Automatically resolve pronouns ("it", "that", "them") using entity tracking
- **Query Reformulation**: Enhance queries with conversation context for better retrieval
- **Context Management**: Track and compress conversation history to prevent context overflow
- **Session Management**: Support multiple concurrent user sessions with persistence
- **Auto-Summarization**: Automatically summarize long conversations to save memory

## Quick Start

### Basic Usage

```python
from utils.conversation_memory import ConversationMemory

# Create conversation memory
memory = ConversationMemory()

# Add turns
memory.add_turn("What is machine learning?", "Machine learning is...")
memory.add_turn("How does it work?", "It works by...")

# Resolve references
query = memory.resolve_references("What are the main types of it?")
# Result: "What are the main types of machine learning?"

# Get conversation context
context = memory.get_conversation_context(max_turns=3)
```

### Multi-User Sessions

```python
from utils.conversation_memory import session_manager

# Get or create session
user_memory = session_manager.get_or_create("user_123")

# Use it
user_memory.add_turn("What is RAG?", "RAG is...")
user_memory.add_turn("How does it work?", "It works by...")

# Automatic persistence and cleanup
session_manager.cleanup_expired()
```

### RAG Integration

```python
from utils.conversation_memory import session_manager

def conversational_rag_query(query: str, session_id: str):
    # Get conversation memory
    memory = session_manager.get_or_create(session_id)

    # Resolve references
    resolved_query = memory.resolve_references(query)

    # Reformulate with context
    if len(memory.turns) > 0:
        reformulated_query = memory.reformulate_query(resolved_query)
    else:
        reformulated_query = resolved_query

    # Run RAG with reformulated query
    result = run_rag_query(reformulated_query)

    # Store in memory
    memory.add_turn(
        original_query=query,
        answer=result,
        resolved_query=resolved_query,
        reformulated_query=reformulated_query
    )

    return result
```

## Key Features

### 1. Reference Resolution

Automatically resolves pronouns and references using entity tracking:

```python
memory.add_turn(
    'What is "LlamaIndex"?',
    "LlamaIndex is a data framework for LLM applications..."
)

# Later turn with reference
query = "How do I install it?"
resolved = memory.resolve_references(query)
# Result: "How do I install LlamaIndex?"
```

**Supported references:**
- `it` → most recent entity (technical term, quoted term, or proper noun)
- `that` → most recent quoted term or proper noun
- `they/them` → most recent proper noun
- `the previous one` → most recent entity
- `the last thing/one` → most recent entity

### 2. Query Reformulation

Enhance queries with conversation context for better retrieval:

```python
# Turn 1
memory.add_turn("What is RAG?", "RAG is...")

# Turn 2 - reformulate with context
query = "What are the benefits?"
reformulated = memory.reformulate_query(query, max_context_turns=2)
# Result includes previous Q&A pairs for context
```

### 3. Conversation Context

Get formatted conversation history for LLM prompts:

```python
context = memory.get_conversation_context(
    max_turns=3,
    include_summaries=True
)
# Returns formatted string with recent turns and summaries
```

### 4. Auto-Summarization

Automatically summarize old turns to prevent context overflow:

```python
memory = ConversationMemory(
    auto_summarize=True,
    summarize_threshold=5,  # Summarize after 5 turns
    max_turns=10
)

# After 6 turns, oldest turns are summarized
# Active turns + summary consume less memory than all turns
```

### 5. Session Management

Manage multiple concurrent conversations:

```python
manager = SessionManager(
    max_sessions=100,
    session_timeout=3600  # 1 hour
)

# Create sessions
alice = manager.get_or_create("user_alice")
bob = manager.get_or_create("user_bob")

# Auto-cleanup expired sessions
manager.cleanup_expired()

# List active sessions
active = manager.list_active_sessions()

# Save all sessions
manager.save_all()
```

## Configuration

### Environment Variables

```bash
# Enable/disable conversation memory
export ENABLE_CONVERSATION_MEMORY=1

# Maximum turns to keep in memory
export MAX_CONVERSATION_TURNS=10

# Session timeout (seconds)
export CONVERSATION_TIMEOUT=3600

# Auto-summarization
export AUTO_SUMMARIZE=1
export SUMMARIZE_THRESHOLD=5

# Storage directory
export CONVERSATION_CACHE_DIR=.cache/conversations
```

### Programmatic Configuration

```python
from utils.conversation_memory import ConversationMemory

memory = ConversationMemory(
    conversation_id="custom_id",
    max_turns=10,
    auto_summarize=True,
    summarize_threshold=5,
    enabled=True
)
```

## API Reference

### ConversationMemory

Main class for managing conversation history.

#### Methods

**`add_turn(original_query, answer, resolved_query=None, reformulated_query=None, context=None)`**
- Add a conversation turn to memory
- Automatically extracts entities and topics
- Triggers auto-summarization if needed

**`resolve_references(query)`**
- Resolve pronouns and references in query
- Returns query with references replaced by entities

**`reformulate_query(query, max_context_turns=3)`**
- Reformulate query with conversation context
- Adds recent turns as context for better retrieval

**`get_conversation_context(max_turns=None, include_summaries=True)`**
- Get formatted conversation history
- Useful for injecting into LLM prompts

**`summarize_conversation()`**
- Get human-readable conversation summary
- Shows turns, entities, topics, and summaries

**`clear_conversation()`**
- Clear all conversation history
- Deletes from disk

**`is_expired(timeout)`**
- Check if conversation has expired
- Based on last access time

**`stats()`**
- Get conversation statistics
- Returns dict with turn counts, entities, topics, etc.

### SessionManager

Manages multiple conversation sessions.

#### Methods

**`get_or_create(session_id)`**
- Get existing session or create new one
- Returns ConversationMemory instance

**`get(session_id)`**
- Get existing session without creating
- Returns None if not found

**`delete(session_id)`**
- Delete a session and its data

**`cleanup_expired()`**
- Remove expired sessions
- Returns number of sessions removed

**`list_active_sessions()`**
- Get list of active session IDs

**`save_all()`**
- Save all active sessions to disk

**`stats()`**
- Get manager statistics

## Integration Patterns

### Pattern 1: Modify Query Function

```python
from utils.conversation_memory import session_manager

def run_query_conversational(engine, question, session_id="default"):
    memory = session_manager.get_or_create(session_id)

    # Resolve and reformulate
    resolved = memory.resolve_references(question)
    reformulated = memory.reformulate_query(resolved)

    # Execute RAG
    response = engine.query(reformulated)

    # Store
    memory.add_turn(question, str(response))

    return response
```

### Pattern 2: Wrapper Class

```python
from utils.conversation_memory import ConversationMemory

class ConversationalQueryEngine:
    def __init__(self, base_engine, session_id=None):
        self.base_engine = base_engine
        self.memory = ConversationMemory(conversation_id=session_id)

    def query(self, query_str):
        resolved = self.memory.resolve_references(query_str)
        reformulated = self.memory.reformulate_query(resolved)

        response = self.base_engine.query(reformulated)

        self.memory.add_turn(query_str, str(response))

        return response
```

### Pattern 3: Decorator

```python
from functools import wraps
from utils.conversation_memory import session_manager

def with_conversation_memory(session_id_param="session_id"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            session_id = kwargs.get(session_id_param, "default")
            memory = session_manager.get_or_create(session_id)

            # Get query from args or kwargs
            query = args[0] if args else kwargs.get("query")

            # Process with memory
            resolved = memory.resolve_references(query)
            reformulated = memory.reformulate_query(resolved)

            # Call original function with reformulated query
            result = func(reformulated, *args[1:], **kwargs)

            # Store in memory
            memory.add_turn(query, result)

            return result
        return wrapper
    return decorator

@with_conversation_memory()
def rag_query(query: str, session_id: str = "default"):
    # Your RAG implementation
    return run_rag(query)
```

## Performance

### Memory Footprint

- **Per turn**: ~2KB (without embeddings)
- **Per entity**: ~100 bytes
- **Per summary**: ~500 bytes
- **Disk cache**: ~5-10KB per conversation

### Latency

- **Reference resolution**: <1ms (pattern matching)
- **Query reformulation**: <5ms (string concatenation)
- **Entity extraction**: <2ms (regex)
- **Disk save**: ~5ms per conversation
- **Session lookup**: <1ms (dict lookup)

### Scalability

- **Concurrent sessions**: Tested up to 1000 sessions
- **Turns per conversation**: Configurable (default 10, auto-summarization)
- **Disk usage**: ~10KB per active conversation
- **Memory usage**: ~2MB for 100 active conversations

## Examples

### Example 1: Basic Multi-Turn

```python
from utils.conversation_memory import ConversationMemory

memory = ConversationMemory()

# Turn 1
memory.add_turn(
    "What is machine learning?",
    "Machine learning is a subset of AI..."
)

# Turn 2
query2 = "What are the main types of it?"
resolved2 = memory.resolve_references(query2)
# resolved2 = "What are the main types of machine learning?"

memory.add_turn(query2, "The main types are...", resolved_query=resolved2)

# View summary
print(memory.summarize_conversation())
```

### Example 2: Multi-User Chat

```python
from utils.conversation_memory import session_manager

def handle_user_message(user_id: str, message: str):
    # Get user's conversation
    memory = session_manager.get_or_create(user_id)

    # Process query
    resolved = memory.resolve_references(message)
    response = run_rag_query(resolved)

    # Store
    memory.add_turn(message, response, resolved_query=resolved)

    return response

# Use it
response1 = handle_user_message("alice", "What is RAG?")
response2 = handle_user_message("alice", "How does it work?")
response3 = handle_user_message("bob", "Tell me about embeddings")
```

### Example 3: Streamlit Integration

```python
import streamlit as st
from utils.conversation_memory import session_manager

# Get session ID from Streamlit
session_id = st.session_state.get("session_id", "default")

# Get conversation memory
memory = session_manager.get_or_create(session_id)

# Show conversation history
st.sidebar.subheader("Conversation History")
for turn in memory.turns:
    st.sidebar.text(f"Q: {turn.original_query[:40]}...")
    st.sidebar.text(f"A: {turn.answer[:40]}...")

# Query input
query = st.text_input("Ask a question:")

if query:
    # Process with memory
    resolved = memory.resolve_references(query)
    result = run_rag_query(resolved)

    # Store
    memory.add_turn(query, result, resolved_query=resolved)

    # Display
    st.write(result)
```

## Testing

Run the comprehensive test suite:

```bash
# Test conversation memory module
python utils/conversation_memory.py

# Test integration examples
python examples/conversation_memory_demo.py

# Test RAG integration
python examples/rag_with_conversation_memory.py
```

## Troubleshooting

### Issue: References not being resolved

**Cause**: No entities extracted from previous turns

**Solution**: Ensure queries and answers contain proper nouns, quoted terms, or technical terms

```python
# Good - entity extraction works
memory.add_turn(
    'What is "LlamaIndex"?',
    "LlamaIndex is a framework..."
)

# Bad - no clear entities
memory.add_turn("What is it?", "It's a thing...")
```

### Issue: Conversation context too long

**Cause**: Too many turns without summarization

**Solution**: Enable auto-summarization or reduce max_turns

```python
memory = ConversationMemory(
    auto_summarize=True,
    summarize_threshold=5,
    max_turns=10
)
```

### Issue: Session not persisting

**Cause**: Not calling save or using temporary directory

**Solution**: Ensure cache directory is persistent

```bash
export CONVERSATION_CACHE_DIR=/persistent/path/conversations
```

### Issue: High memory usage

**Cause**: Too many active sessions

**Solution**: Reduce max_sessions or decrease session_timeout

```python
manager = SessionManager(
    max_sessions=50,
    session_timeout=1800  # 30 minutes
)

# Cleanup regularly
manager.cleanup_expired()
```

## Related Documentation

- **Main module**: `/Users/frytos/code/llamaIndex-local-rag/utils/conversation_memory.py`
- **Basic examples**: `/Users/frytos/code/llamaIndex-local-rag/examples/conversation_memory_demo.py`
- **RAG integration**: `/Users/frytos/code/llamaIndex-local-rag/examples/rag_with_conversation_memory.py`
- **Query cache**: `/Users/frytos/code/llamaIndex-local-rag/utils/query_cache.py`
- **Semantic cache**: `/Users/frytos/code/llamaIndex-local-rag/docs/SEMANTIC_CACHE_GUIDE.md`

## Contributing

When modifying the conversation memory module:

1. Run the test suite to ensure no regressions
2. Update examples if adding new features
3. Document new configuration options
4. Consider backward compatibility
5. Test with long conversations (10+ turns)

## License

Part of the llamaIndex-local-rag project.
