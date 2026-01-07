---
description: Analyze RAG pipeline and suggest optimizations
---

# Optimize RAG Pipeline

Analyze the current RAG setup and provide actionable optimization recommendations.

## Usage

```
/optimize-rag [--table <name>] [--focus <area>]
```

## Focus Areas

- `quality` - Improve retrieval relevance
- `speed` - Reduce latency
- `memory` - Reduce RAM usage
- `all` - Comprehensive analysis (default)

## Analysis Performed

### 1. Configuration Review
- Current chunk size and overlap
- Embedding model selection
- LLM parameters
- TOP_K setting

### 2. Performance Profiling
- Embedding throughput
- Database query time
- LLM generation speed
- Memory usage

### 3. Quality Assessment
- Similarity score distribution
- Answer relevance
- Context coverage

### 4. Resource Utilization
- GPU usage (Metal/MPS)
- CPU utilization
- Memory pressure

## Output Format

```markdown
# RAG Optimization Report

## Current Configuration
- Chunk: 700 chars, 150 overlap (21%)
- Embedding: bge-small-en (384d)
- LLM: Mistral 7B, 24 GPU layers
- Retrieval: TOP_K=4

## Performance Metrics
- Embedding: 67 chunks/s
- Retrieval: 0.12s
- Generation: 12 tokens/s
- Memory: 8.2 GB peak

## Issues Found

### 1. Suboptimal GPU Utilization
Current: 24 layers (75%)
Available: 8 more layers possible
**Impact**: 15% slower generation

### 2. Context Window Near Limit
Current: 2800 tokens typical
Limit: 3072 tokens
**Risk**: Overflow on complex queries

### 3. Low Similarity Scores
Average: 0.42
Target: > 0.50
**Impact**: Less relevant retrieval

## Recommendations

### Quick Wins (Low Effort, High Impact)
1. Increase N_GPU_LAYERS to 28
   ```bash
   N_GPU_LAYERS=28
   ```
   Expected: +15% generation speed

2. Increase context window
   ```bash
   CTX=4096
   ```
   Expected: Eliminate overflow errors

### Quality Improvements
3. Reduce chunk size for better precision
   ```bash
   CHUNK_SIZE=500 CHUNK_OVERLAP=100
   ```
   Expected: +20% similarity scores

### Advanced Optimizations
4. Use larger embedding model
   ```bash
   EMBED_MODEL=BAAI/bge-base-en-v1.5 EMBED_DIM=768
   ```
   Expected: +10% retrieval quality, -30% speed

## Recommended Configuration

```bash
# Optimized for M1 16GB
export CHUNK_SIZE=500
export CHUNK_OVERLAP=100
export TOP_K=4
export N_GPU_LAYERS=28
export N_BATCH=256
export CTX=4096
export EMBED_BATCH=64
```

## Expected Improvements
- Generation: 12 → 14 tokens/s (+17%)
- Similarity: 0.42 → 0.52 (+24%)
- Overflow Risk: High → Low
```

## Optimization Categories

### Speed Optimizations
| Setting | Change | Impact |
|---------|--------|--------|
| N_GPU_LAYERS | 24→28 | +15% gen speed |
| N_BATCH | 128→256 | +10% gen speed |
| EMBED_BATCH | 32→64 | +20% embed speed |

### Quality Optimizations
| Setting | Change | Impact |
|---------|--------|--------|
| CHUNK_SIZE | 700→500 | +15% precision |
| EMBED_MODEL | small→base | +10% relevance |
| TOP_K | 4→5 | +5% coverage |

### Memory Optimizations
| Setting | Change | Impact |
|---------|--------|--------|
| N_GPU_LAYERS | 28→20 | -1GB VRAM |
| CTX | 4096→3072 | -0.5GB RAM |
| TOP_K | 5→3 | Less context |

## Commands to Apply

After reviewing recommendations:

```bash
# Apply all recommended settings
/run-rag --doc data/doc.pdf --table optimized_index \
  --chunk-size 500 --overlap 100 --reset

# Or set environment and run manually
export CHUNK_SIZE=500 CHUNK_OVERLAP=100 N_GPU_LAYERS=28 CTX=4096
python rag_low_level_m1_16gb_verbose.py
```

## Monitoring After Optimization

```bash
# Check GPU utilization
sudo powermetrics --samplers gpu_power -i 1000 -n 5

# Check memory
python -c "import psutil; print(f'{psutil.virtual_memory().percent}% used')"

# Test query performance
time python rag_low_level_m1_16gb_verbose.py --query-only --query "test"
```
