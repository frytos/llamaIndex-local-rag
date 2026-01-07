---
description: Compare different chunk configurations for RAG quality
---

# Compare Chunk Configurations

Test multiple chunk size configurations to find the optimal settings for your documents.

## Usage

```
/compare-chunks --doc <path> --query "<test question>"
```

## What It Does

1. **Index Document Multiple Times** with different chunk configs
2. **Run Same Query** against each configuration
3. **Compare Results**: similarity scores, answer quality, performance
4. **Recommend** best configuration for your use case

## Default Configurations Tested

| Config | Chunk Size | Overlap | Table Suffix |
|--------|------------|---------|--------------|
| Fine | 300 | 60 | _cs300 |
| Balanced | 700 | 150 | _cs700 |
| Large | 1200 | 240 | _cs1200 |

## Output Format

```markdown
# Chunk Configuration Comparison

Document: data/document.pdf
Test Query: "What is the main thesis?"

## Results

| Config | Chunks | Top Score | Avg Score | Answer Quality |
|--------|--------|-----------|-----------|----------------|
| Fine (300) | 245 | 0.78 | 0.62 | Precise |
| Balanced (700) | 112 | 0.71 | 0.58 | Good context |
| Large (1200) | 68 | 0.65 | 0.52 | Most context |

## Detailed Analysis

### Fine (cs=300, ov=60)
- **Pros**: Highest similarity, most precise retrieval
- **Cons**: May miss broader context
- **Best for**: Fact-based Q&A, specific lookups

### Balanced (cs=700, ov=150)
- **Pros**: Good balance of precision and context
- **Cons**: Middle ground, not optimized for either
- **Best for**: General use, mixed queries

### Large (cs=1200, ov=240)
- **Pros**: Maximum context per chunk
- **Cons**: Lower precision, risk of irrelevant content
- **Best for**: Summaries, explanations

## Recommendation

For your document and query style:
**Use Fine (300/60)** - Your queries are fact-seeking

To apply:
```bash
CHUNK_SIZE=300 CHUNK_OVERLAP=60 PGTABLE=myindex_cs300 \
  RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py
```
```

## Execution Steps

1. Create temporary tables for each config
2. Index document with each configuration
3. Run test query against each
4. Collect metrics (scores, timing, chunk counts)
5. Generate comparison report
6. Clean up temporary tables (optional)

## Custom Configurations

Override defaults with:
```
/compare-chunks --doc data/doc.pdf --query "question" \
  --configs "200:40,500:100,1000:200"
```

Format: `chunk_size:overlap,chunk_size:overlap,...`

## Tips

- Use a representative test query
- Run multiple queries for better comparison
- Consider your typical query patterns
- Factor in context window limits (CTX)

## Metrics Explained

| Metric | Meaning |
|--------|---------|
| Chunks | Total chunks created |
| Top Score | Best similarity score |
| Avg Score | Average of TOP_K scores |
| Answer Quality | Subjective assessment |

## Quality Indicators

- **Top Score > 0.7**: Excellent match
- **Top Score 0.5-0.7**: Good match
- **Top Score < 0.5**: Consider different config

## After Comparison

Use the recommended config for production:
```
/run-rag --doc data/doc.pdf --chunk-size <recommended> --overlap <recommended> --reset
```
