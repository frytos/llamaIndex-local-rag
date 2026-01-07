# Answer Validation Module

**Location**: `/utils/answer_validator.py`
**Purpose**: Validate RAG-generated answers for quality, confidence, and hallucinations
**Performance**: ~200ms per validation (10-50x faster than LLM-based validation)

## Overview

The Answer Validation module provides quality assurance for RAG (Retrieval-Augmented Generation) responses. It uses embedding-based similarity (no additional LLM calls) to:

1. **Score confidence** (0-1): How confident should we be in this answer?
2. **Detect hallucinations**: Which claims are unsupported by retrieved context?
3. **Extract citations**: Which sources support each sentence?
4. **Generate warnings**: What quality issues were detected?

## Quick Start

### Basic Usage

```python
from utils.answer_validator import AnswerValidator

# Initialize with embedding model (reuse from RAG pipeline)
validator = AnswerValidator(embed_model)

# Validate answer
result = validator.validate_answer(
    answer="Mistral 7B uses sliding window attention...",
    query="What attention mechanism does Mistral use?",
    chunks=retrieved_chunks
)

# Check results
print(f"Confidence: {result['confidence_score']:.2f}")
print(f"Hallucinations: {result['hallucination_count']}")
print(f"Passed: {result['passed']}")

if result['warnings']:
    for warning in result['warnings']:
        print(f"Warning: {warning}")
```

### Integration with RAG Pipeline

Add validation after LLM response generation:

```python
# In rag_low_level_m1_16gb_verbose.py or similar

# 1. Import validator
from utils.answer_validator import AnswerValidator

# 2. Initialize validator (after building embed_model)
validator = AnswerValidator(embed_model)

# 3. Run query (existing code)
response = query_engine.query(question)

# 4. Validate answer
validation = validator.validate_answer(
    answer=str(response),
    query=question,
    chunks=response.source_nodes  # Retrieved chunks
)

# 5. Use validation results
if not validation['passed']:
    log.warning(f"Answer quality issues: {validation['warnings']}")

# 6. Include in response
final_response = {
    'answer': str(response),
    'confidence': validation['confidence_score'],
    'hallucinations': validation['hallucination_count'],
    'citations': validation['citations'],
    'warnings': validation['warnings'],
}
```

## Configuration

### Environment Variables

Add to `.env` file:

```bash
# Enable/disable validation (default: 1)
ENABLE_ANSWER_VALIDATION=1

# Confidence threshold - warn if below (default: 0.7)
CONFIDENCE_THRESHOLD=0.7

# Hallucination threshold - flag claims below this similarity (default: 0.5)
HALLUCINATION_THRESHOLD=0.5

# Relevance threshold - query-answer similarity (default: 0.6)
RELEVANCE_THRESHOLD=0.6

# Faithfulness threshold - answer-context grounding (default: 0.7)
FAITHFULNESS_THRESHOLD=0.7
```

### Programmatic Configuration

```python
from utils.answer_validator import AnswerValidator, ValidationConfig

# Custom configuration
config = ValidationConfig(
    enabled=True,
    confidence_threshold=0.85,  # Strict (85% confidence required)
    hallucination_threshold=0.6,
    relevance_threshold=0.7,
    faithfulness_threshold=0.8,
)

validator = AnswerValidator(embed_model, config)
```

## Core Components

### 1. Confidence Scoring

Scores answer confidence (0-1) based on:

- **Relevance**: Does answer address the query?
- **Faithfulness**: Is answer grounded in retrieved context?
- **Completeness**: Is answer substantial and complete?
- **Uncertainty**: Does answer contain uncertainty indicators?

```python
confidence = validator.score_confidence(answer, chunks, query)

print(f"Overall Score: {confidence['overall_score']:.2f}")
print(f"Relevance: {confidence['relevance']:.2f}")
print(f"Faithfulness: {confidence['faithfulness']:.2f}")
print(f"Completeness: {confidence['completeness']:.2f}")
print(f"Has Uncertainty: {confidence['has_uncertainty']}")
```

**Scoring Formula**:
```
overall_score = 0.35 * relevance +
                0.40 * faithfulness +
                0.25 * completeness -
                uncertainty_penalty
```

**Uncertainty Indicators**:
- "might", "possibly", "perhaps", "unclear"
- "may", "could", "probably", "likely"
- "i think", "i believe", "not sure"
- "don't know", "cannot determine"

### 2. Hallucination Detection

Detects unsupported claims by:

1. Splitting answer into sentences
2. Computing similarity between each sentence and retrieved chunks
3. Flagging sentences with similarity below threshold

```python
hallucinations = validator.detect_hallucinations(answer, chunks)

for claim, support_score, is_hallucination in hallucinations:
    if is_hallucination:
        print(f"Unsupported: {claim} (score: {support_score:.2f})")
```

**Example Output**:
```
✓ Supported (score: 0.82)
  Mistral 7B uses sliding window attention.

❌ UNSUPPORTED (score: 0.35)
  It was trained on 100 trillion tokens from Mars.

✓ Supported (score: 0.76)
  The model has 7 billion parameters.
```

### 3. Citation Extraction

Maps each answer sentence to source chunks:

```python
citations = validator.extract_citations(answer, chunks)

for citation in citations:
    print(f"Sentence: {citation['sentence'][:50]}...")
    for source in citation['sources']:
        print(f"  [Chunk {source['chunk_id']}] score={source['score']:.2f}")
        print(f"    {source['text'][:80]}...")
```

**Example Output**:
```
Sentence: Mistral uses sliding window attention...
  [Chunk 0] score=0.85
    Mistral 7B uses sliding window attention for efficient token processing...
  [Chunk 1] score=0.72
    Each token attends only to a fixed window of 4096 previous tokens...

Sentence: This reduces computational complexity...
  [Chunk 2] score=0.89
    This reduces computational complexity from O(n²) to O(n*w)...
```

### 4. Complete Validation Pipeline

Runs all checks in one call:

```python
result = validator.validate_answer(answer, query, chunks)
```

**Returns**:
```python
{
    'confidence_score': 0.85,
    'confidence_details': {
        'overall_score': 0.85,
        'relevance': 0.9,
        'faithfulness': 0.85,
        'completeness': 0.8,
        'has_uncertainty': False,
        'uncertainty_indicators': [],
    },
    'hallucinations': [
        {'claim': '...', 'support_score': 0.35}
    ],
    'hallucination_count': 1,
    'citations': [...],
    'warnings': [
        'Found 1 unsupported claim(s)',
    ],
    'passed': False
}
```

## Performance

### Benchmarks (M1 Mac 16GB, bge-small-en embeddings)

| Operation | Time | Notes |
|-----------|------|-------|
| Confidence scoring | ~50ms | Query + answer + context embeddings |
| Hallucination detection | ~100ms | Per-sentence similarity checks |
| Citation extraction | ~80ms | Sentence-to-chunk mapping |
| **Complete validation** | **~200ms** | All checks combined |

### Comparison with LLM-based Validation

| Method | Time | Accuracy | Cost |
|--------|------|----------|------|
| **Embedding-based** (this module) | ~200ms | Good | Free |
| LLM-based (GPT-3.5) | ~2-5s | Better | $0.001/query |
| LLM-based (local) | ~5-15s | Better | Free |

**Speedup**: 10-50x faster than LLM-based validation

### Memory Usage

- Reuses existing embedding model (no extra memory)
- Lightweight: ~1-2MB for validation logic
- No persistent storage (validation is stateless)

## Examples

### Example 1: Basic Validation

```python
validator = AnswerValidator(embed_model)

query = "What is sliding window attention?"
answer = "Sliding window attention limits each token's attention to a fixed window."
chunks = ["Mistral uses sliding window attention...", "Window size is 4096..."]

result = validator.validate_answer(answer, query, chunks)

if result['passed']:
    print("Answer validated successfully!")
else:
    print(f"Warnings: {result['warnings']}")
```

### Example 2: Custom Thresholds

```python
# Strict validation
strict_config = ValidationConfig(
    confidence_threshold=0.85,  # 85% confidence required
    hallucination_threshold=0.6,
)
strict_validator = AnswerValidator(embed_model, strict_config)

# Lenient validation
lenient_config = ValidationConfig(
    confidence_threshold=0.5,  # 50% acceptable
    hallucination_threshold=0.3,
)
lenient_validator = AnswerValidator(embed_model, lenient_config)
```

### Example 3: Integration with Logging

```python
import logging
log = logging.getLogger(__name__)

validator = AnswerValidator(embed_model)
result = validator.validate_answer(answer, query, chunks)

# Log validation results
log.info(f"Answer confidence: {result['confidence_score']:.2f}")

if result['hallucination_count'] > 0:
    log.warning(f"Found {result['hallucination_count']} hallucinations")
    for hall in result['hallucinations']:
        log.warning(f"  Unsupported: {hall['claim'][:80]}...")

if result['warnings']:
    for warning in result['warnings']:
        log.warning(f"Quality warning: {warning}")
```

### Example 4: Retry on Low Confidence

```python
def rag_query_with_retry(query: str, max_retries: int = 2):
    """Query with automatic retry on low confidence"""
    validator = AnswerValidator(embed_model)

    for attempt in range(max_retries):
        # Run RAG query
        response = query_engine.query(query)

        # Validate
        validation = validator.validate_answer(
            answer=str(response),
            query=query,
            chunks=response.source_nodes
        )

        # Check if passed
        if validation['passed']:
            return response, validation

        # Retry with different parameters
        log.warning(f"Attempt {attempt+1} failed validation, retrying...")
        # Could adjust TOP_K, temperature, etc.

    return response, validation  # Return last attempt
```

## Testing

### Run Tests

```bash
# With pytest
pytest tests/test_answer_validator.py -v

# Direct execution
python tests/test_answer_validator.py

# With coverage
pytest tests/test_answer_validator.py --cov=utils.answer_validator
```

### Run Examples

```bash
# All examples
python examples/answer_validation_example.py

# Individual example
python -c "from examples.answer_validation_example import example_1_basic_validation; example_1_basic_validation()"
```

## Integration Checklist

- [ ] Add `from utils.answer_validator import AnswerValidator` to your RAG script
- [ ] Initialize validator: `validator = AnswerValidator(embed_model)`
- [ ] Call validation after LLM response: `validator.validate_answer(...)`
- [ ] Configure thresholds in `.env` file
- [ ] Log validation warnings
- [ ] Include confidence scores in response
- [ ] (Optional) Implement retry logic for low confidence
- [ ] (Optional) Include citations in response
- [ ] Test with representative queries
- [ ] Monitor validation metrics

## Troubleshooting

### Issue: Validator gives low confidence for good answers

**Solution**: Lower thresholds in config

```python
config = ValidationConfig(
    confidence_threshold=0.6,  # Lower from 0.7
    faithfulness_threshold=0.6,  # Lower from 0.7
)
```

### Issue: Too many false hallucination warnings

**Solution**: Lower hallucination threshold

```bash
# In .env
HALLUCINATION_THRESHOLD=0.4  # Lower from 0.5
```

### Issue: Validation is too slow

**Causes**:
1. Embedding model is slow (CPU)
2. Many chunks to check
3. Long answers with many sentences

**Solutions**:
1. Use faster embedding model (MLX on M1, GPU on CUDA)
2. Reduce TOP_K to limit chunks
3. Disable validation for non-critical queries

### Issue: Citations are incorrect

**Note**: Citation extraction uses similarity-based matching, not semantic understanding. It provides *likely* sources, not guaranteed attributions.

**Improvement**: Use reranking before validation to ensure best chunks are retrieved.

## Advanced Usage

### Custom Uncertainty Indicators

```python
config = ValidationConfig()
config.uncertainty_words.extend(['approximately', 'roughly', 'around'])

validator = AnswerValidator(embed_model, config)
```

### Validation Metrics Logging

```python
import json
from datetime import datetime

def log_validation_metrics(query, answer, validation, output_file='validation_metrics.jsonl'):
    """Log validation results for analysis"""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': answer[:100],
        'confidence': validation['confidence_score'],
        'hallucination_count': validation['hallucination_count'],
        'passed': validation['passed'],
        'warnings': validation['warnings'],
    }

    with open(output_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')
```

### Confidence-Based Response Selection

```python
def get_best_answer(query: str, num_candidates: int = 3):
    """Generate multiple answers and return highest confidence"""
    validator = AnswerValidator(embed_model)
    candidates = []

    for i in range(num_candidates):
        # Generate answer with different temperature
        response = query_engine.query(query)
        validation = validator.validate_answer(
            answer=str(response),
            query=query,
            chunks=response.source_nodes
        )

        candidates.append({
            'answer': str(response),
            'confidence': validation['confidence_score'],
            'validation': validation,
        })

    # Return highest confidence answer
    best = max(candidates, key=lambda x: x['confidence'])
    return best
```

## Limitations

1. **Embedding-based**: Uses similarity, not semantic understanding
   - May miss subtle inconsistencies
   - Cannot detect logical contradictions
   - No deep reasoning about claims

2. **Context-dependent**: Quality depends on retrieved chunks
   - Poor retrieval → poor validation
   - Missing context → false hallucinations

3. **Language-specific**: Works best with English
   - Non-English requires appropriate embedding model
   - Mixed language may have lower accuracy

4. **Heuristic-based**: Uses simple heuristics for completeness
   - Not a substitute for human review
   - Best used as quality indicator, not absolute truth

## Future Enhancements

Potential improvements:

- [ ] Multi-language support
- [ ] Semantic entailment checking (vs pure similarity)
- [ ] Fine-tuned embedding models for validation
- [ ] Lightweight LLM-based validation option
- [ ] Validation result caching
- [ ] Answer quality scoring (grammar, coherence)
- [ ] Source reliability weighting
- [ ] Contradiction detection between chunks

## References

- **Embedding Models**: HuggingFace Sentence Transformers
- **Similarity Metrics**: Cosine similarity for semantic matching
- **RAG Evaluation**: [RAGAS framework](https://github.com/explodinggradients/ragas)
- **Hallucination Detection**: NLI-based approaches in literature

## Support

For issues, questions, or contributions:

1. Check examples in `examples/answer_validation_example.py`
2. Run tests to verify functionality
3. Review this documentation
4. Check project README.md

## Summary

The Answer Validation module provides:

- Fast, embedding-based validation (~200ms)
- Confidence scoring with detailed breakdown
- Hallucination detection via similarity matching
- Citation extraction for source attribution
- Configurable thresholds and warnings
- Easy integration with existing RAG pipeline

Use it to improve answer quality, detect issues early, and build trust in your RAG system.
