# Semantic Cache Quick Start

## 5-Minute Setup

### 1. Import the Cache

```python
from utils.query_cache import semantic_cache
```

### 2. Use in Your RAG Function

```python
def query_rag(query_text: str):
    # Get embedding
    query_embedding = embed_model.encode(query_text)
    
    # Try cache first
    cached = semantic_cache.get_semantic(query_text, query_embedding)
    if cached:
        return cached
    
    # Run RAG if cache miss
    result = run_full_rag(query_text)
    
    # Cache for next time
    semantic_cache.set_semantic(query_text, query_embedding, result)
    return result
```

### 3. Configure (Optional)

Add to your `.env` file:

```bash
ENABLE_SEMANTIC_CACHE=1
SEMANTIC_CACHE_THRESHOLD=0.92
SEMANTIC_CACHE_MAX_SIZE=1000
SEMANTIC_CACHE_TTL=86400
```

### 4. Monitor Performance

```python
stats = semantic_cache.stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

## That's It!

Your RAG queries will now be cached automatically. Similar queries (>92% similarity) will return cached responses in ~0.5ms instead of 5-15 seconds.

## Expected Results

- **First query**: Cache miss (~5-15s)
- **Similar queries**: Cache hit (~0.5ms)
- **Speedup**: 10,000x - 30,000x
- **Memory**: ~150KB per cached query

## Need Help?

See the complete guide: [SEMANTIC_CACHE_GUIDE.md](SEMANTIC_CACHE_GUIDE.md)

## Test It

```bash
# Run demo
python examples/semantic_cache_demo.py

# Run tests
python utils/query_cache.py
```
