# Conversation Memory - Quick Start Guide

**5-minute guide to adding conversational capabilities to your RAG pipeline**

## Installation

No additional dependencies required! The module uses only standard library + numpy (already installed).

```bash
# Verify it's available
python -c "from utils.conversation_memory import session_manager; print('Ready!')"
```

## Minimal Example (3 lines)

```python
from utils.conversation_memory import session_manager

# Get session for user
memory = session_manager.get_or_create("user_123")

# Resolve references and add turn
query = memory.resolve_references("What is RAG?")
answer = run_rag_query(query)  # Your existing RAG function
memory.add_turn("What is RAG?", answer)
```

## Quick Integration (Copy-Paste)

### Option 1: Wrapper Function (Recommended)

```python
from utils.conversation_memory import session_manager

def conversational_rag(query: str, session_id: str = "default"):
    """RAG query with automatic conversation memory"""
    # Get conversation memory
    memory = session_manager.get_or_create(session_id)

    # Resolve references (it, that, them → actual entities)
    resolved = memory.resolve_references(query)

    # Add context from previous turns
    if len(memory.turns) > 0:
        reformulated = memory.reformulate_query(resolved)
    else:
        reformulated = resolved

    # Run your RAG query
    result = run_rag_query(reformulated)  # Your existing function

    # Store in memory
    memory.add_turn(query, result, resolved_query=resolved)

    return result

# Use it
conversational_rag("What is the chunk size?", "user_123")
conversational_rag("What about overlap?", "user_123")  # Auto-resolves context
conversational_rag("How do I change it?", "user_123")  # Resolves "it"
```

### Option 2: Direct Integration

```python
from utils.conversation_memory import session_manager

# In your existing RAG query function:
def run_query(question: str, session_id: str = None):
    # ... your existing code ...

    # Add conversation memory
    if session_id:
        memory = session_manager.get_or_create(session_id)
        question = memory.resolve_references(question)
        # ... rest of your code ...
        memory.add_turn(question, response)

    return response
```

### Option 3: Class Wrapper

```python
from utils.conversation_memory import ConversationMemory

class MyRAGSystem:
    def __init__(self, session_id=None):
        self.memory = ConversationMemory(conversation_id=session_id)
        # ... your existing initialization ...

    def query(self, question: str):
        # Resolve references
        question = self.memory.resolve_references(question)

        # ... your existing query code ...
        result = self.execute_rag(question)

        # Store in memory
        self.memory.add_turn(question, result)

        return result
```

## Configuration (Optional)

### Via Environment Variables

```bash
# Enable/disable
export ENABLE_CONVERSATION_MEMORY=1

# Tune behavior
export MAX_CONVERSATION_TURNS=10
export AUTO_SUMMARIZE=1
export SUMMARIZE_THRESHOLD=5

# Run your pipeline
python rag_low_level_m1_16gb_verbose.py --query-only --interactive
```

### Via Code

```python
from utils.conversation_memory import ConversationMemory

memory = ConversationMemory(
    conversation_id="user_123",
    max_turns=10,              # Keep last 10 turns
    auto_summarize=True,       # Auto-compress old turns
    summarize_threshold=5      # Summarize after 5 turns
)
```

## Common Patterns

### Pattern 1: Interactive CLI

```python
from utils.conversation_memory import session_manager

session_id = input("Enter session ID (or Enter for new): ") or None
memory = session_manager.get_or_create(session_id or "default")

print(f"Session: {memory.conversation_id}")
print(f"Previous turns: {memory.turn_count}")

while True:
    query = input("\nQuery: ")
    if query.lower() in ['quit', 'exit']:
        break

    # Process with memory
    resolved = memory.resolve_references(query)
    result = run_rag_query(resolved)

    print(f"Answer: {result}")
    memory.add_turn(query, result)
```

### Pattern 2: Web API (FastAPI)

```python
from fastapi import FastAPI
from utils.conversation_memory import session_manager

app = FastAPI()

@app.post("/query")
def query(question: str, session_id: str = "default"):
    memory = session_manager.get_or_create(session_id)
    resolved = memory.resolve_references(question)
    result = run_rag_query(resolved)
    memory.add_turn(question, result)

    return {
        "answer": result,
        "turn_number": memory.turn_count - 1,
        "resolved_query": resolved
    }
```

### Pattern 3: Streamlit Chat

```python
import streamlit as st
from utils.conversation_memory import session_manager

# Get or create session
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user"

memory = session_manager.get_or_create(st.session_state.session_id)

# Show history
for turn in memory.turns:
    st.chat_message("user").write(turn.original_query)
    st.chat_message("assistant").write(turn.answer)

# New query
if prompt := st.chat_input("Ask a question"):
    resolved = memory.resolve_references(prompt)
    result = run_rag_query(resolved)
    memory.add_turn(prompt, result)
    st.rerun()
```

## Features You Get

| Feature | What It Does | Example |
|---------|--------------|---------|
| **Reference Resolution** | Resolves pronouns automatically | "What is RAG?" → "How does it work?" → "How does RAG work?" |
| **Query Reformulation** | Adds context to queries | "What about overlap?" → "Given previous discussion about chunking, what about chunk overlap?" |
| **Entity Tracking** | Remembers mentioned terms | Extracts "LlamaIndex", "pgvector", technical terms |
| **Auto-Summarization** | Compresses old turns | After 5 turns, oldest conversations are summarized |
| **Session Management** | Multi-user support | Each user has independent conversation history |
| **Persistence** | Survives restarts | Conversations saved to disk automatically |

## Quick Reference

### Add a Turn
```python
memory.add_turn(query, answer)
```

### Resolve References
```python
resolved = memory.resolve_references("How does it work?")
# "it" → most recent entity
```

### Reformulate with Context
```python
reformulated = memory.reformulate_query(query)
# Adds previous turns as context
```

### Get Conversation Summary
```python
summary = memory.summarize_conversation()
print(summary)
```

### View Statistics
```python
stats = memory.stats()
print(f"Total turns: {stats['total_turns']}")
print(f"Entities: {stats['entities']}")
```

### Clear Conversation
```python
memory.clear_conversation()
```

### Session Management
```python
# Get or create
memory = session_manager.get_or_create("user_123")

# List active
active = session_manager.list_active_sessions()

# Cleanup expired
session_manager.cleanup_expired()
```

## Performance

- **Latency**: <5ms per operation
- **Memory**: ~2KB per turn
- **Storage**: ~10KB per conversation
- **Scalability**: 1000+ concurrent sessions tested

## Disable Temporarily

```python
# Via environment
export ENABLE_CONVERSATION_MEMORY=0

# Via code
memory = ConversationMemory(enabled=False)
```

## Examples & Documentation

- **Full docs**: `utils/README_CONVERSATION_MEMORY.md`
- **Basic examples**: `examples/conversation_memory_demo.py`
- **RAG integration**: `examples/rag_with_conversation_memory.py`
- **Environment vars**: `docs/ENVIRONMENT_VARIABLES.md`

## Troubleshooting

### References not resolving?
Ensure your answers contain clear entities:
```python
# Good
memory.add_turn('What is "RAG"?', "RAG is Retrieval-Augmented Generation...")

# Bad
memory.add_turn("What is it?", "It's a thing")
```

### Context too long?
Enable auto-summarization:
```python
memory = ConversationMemory(auto_summarize=True, summarize_threshold=5)
```

### Need separate user sessions?
Use session manager:
```python
alice = session_manager.get_or_create("alice")
bob = session_manager.get_or_create("bob")
```

## Next Steps

1. Copy one of the integration patterns above
2. Replace `run_rag_query()` with your actual RAG function
3. Test with multi-turn queries
4. Tune configuration if needed

**That's it!** You now have conversational RAG.

## Support

- Test the module: `python utils/conversation_memory.py`
- Run examples: `python examples/conversation_memory_demo.py`
- Check integration: `python examples/rag_with_conversation_memory.py`

For detailed API reference and advanced features, see:
- `utils/README_CONVERSATION_MEMORY.md`
