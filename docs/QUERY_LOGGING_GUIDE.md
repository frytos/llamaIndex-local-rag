# Query Logging Guide

## Overview

The RAG system now includes comprehensive query logging functionality that automatically saves detailed information about each query, including parameters, retrieved chunks, and generated answers.

## Features

Query logs capture:
- **Query Details**: Question text, length, and word count
- **All Parameters**: chunk_size, chunk_overlap, top_k, temperature, model settings, etc.
- **Retrieved Chunks**: Full text, similarity scores, metadata, and ranking
- **Generated Answer**: Complete answer text with statistics
- **Performance Metrics**: Retrieval time, generation time, throughput

## How to Enable

Set the `LOG_QUERIES` environment variable to enable logging:

```bash
export LOG_QUERIES=1
```

Then run your RAG script normally:

```bash
# Single query
python rag_low_level_m1_16gb_verbose.py

# Interactive mode
python rag_low_level_m1_16gb_verbose.py --interactive

# With specific question
python rag_low_level_m1_16gb_verbose.py --query "What are the main findings?"
```

## Log File Structure

Logs are saved to: `query_logs/{table_name}/{timestamp}_query.json`

Example: `query_logs/llama2_paper/20251217_152430_123456_query.json`

### Log File Format

```json
{
  "metadata": {
    "timestamp": "2025-12-17T15:24:30.123456",
    "document_table": "llama2_paper",
    "pdf_path": "data/llama2.pdf",
    "log_file": "query_logs/llama2_paper/20251217_152430_123456_query.json"
  },
  "query": {
    "question": "What are the main safety features?",
    "question_length": 32,
    "question_words": 6
  },
  "parameters": {
    "chunk_size": 700,
    "chunk_overlap": 150,
    "top_k": 4,
    "temperature": 0.1,
    "max_new_tokens": 256,
    "context_window": 3072,
    "n_gpu_layers": 16,
    "n_batch": 128,
    "embed_model": "BAAI/bge-small-en",
    "embed_dim": 384,
    "embed_batch": 16
  },
  "retrieval": {
    "retrieved_chunks": [
      {
        "rank": 1,
        "similarity_score": 0.8542,
        "text": "Full chunk text here...",
        "text_length": 658,
        "metadata": {
          "page_label": "5",
          "file_name": "llama2.pdf"
        }
      },
      {
        "rank": 2,
        "similarity_score": 0.7891,
        "text": "Another chunk...",
        "text_length": 612,
        "metadata": {
          "page_label": "12",
          "file_name": "llama2.pdf"
        }
      }
    ],
    "num_chunks": 4,
    "retrieval_time_seconds": 1.234,
    "quality_metrics": {
      "best_score": 0.8542,
      "worst_score": 0.6234,
      "average_score": 0.7412,
      "score_range": 0.2308
    }
  },
  "generation": {
    "answer": "The main safety features include...",
    "answer_length": 623,
    "answer_words": 108,
    "generation_time_seconds": 81.340,
    "tokens_per_second": 1.33
  },
  "performance": {
    "total_time_seconds": 82.574,
    "retrieval_percentage": 1.5,
    "generation_percentage": 98.5
  }
}
```

## Use Cases

### 1. Query Analysis
Analyze which queries perform well:
```bash
# Find all queries with high retrieval scores
jq '.retrieval.quality_metrics.best_score' query_logs/llama2_paper/*.json
```

### 2. Performance Tuning
Track performance across different parameters:
```bash
# Compare generation times
jq '.generation.generation_time_seconds' query_logs/*/*.json
```

### 3. Quality Monitoring
Monitor retrieval quality over time:
```bash
# Check average similarity scores
jq '.retrieval.quality_metrics.average_score' query_logs/llama2_paper/*.json
```

### 4. Debugging
Examine exactly what chunks were retrieved for a query:
```bash
# View retrieved chunks for latest query
cat query_logs/llama2_paper/*.json | jq -r '.retrieval.retrieved_chunks[].text' | head -n 1
```

### 5. A/B Testing
Compare different parameter configurations:
```bash
# Run with different chunk sizes and compare logs
CHUNK_SIZE=500 LOG_QUERIES=1 python rag_low_level_m1_16gb_verbose.py --query "Test query"
CHUNK_SIZE=1000 LOG_QUERIES=1 python rag_low_level_m1_16gb_verbose.py --query "Test query"

# Compare quality metrics
jq '.retrieval.quality_metrics' query_logs/*/*.json
```

## Log Management

### View Recent Queries
```bash
ls -lt query_logs/llama2_paper/ | head
```

### Count Total Queries
```bash
find query_logs -name "*.json" | wc -l
```

### Clean Old Logs
```bash
# Delete logs older than 7 days
find query_logs -name "*.json" -mtime +7 -delete
```

### Archive Logs
```bash
# Archive logs by month
tar -czf query_logs_$(date +%Y%m).tar.gz query_logs/
```

## Integration with Analysis Tools

### Python Analysis
```python
import json
from pathlib import Path
import pandas as pd

# Load all logs for a document
logs = []
for log_file in Path("query_logs/llama2_paper").glob("*.json"):
    with open(log_file) as f:
        logs.append(json.load(f))

# Convert to DataFrame
df = pd.DataFrame([
    {
        "question": log["query"]["question"],
        "best_score": log["retrieval"]["quality_metrics"]["best_score"],
        "avg_score": log["retrieval"]["quality_metrics"]["average_score"],
        "gen_time": log["generation"]["generation_time_seconds"],
        "answer_words": log["generation"]["answer_words"]
    }
    for log in logs
])

print(df.describe())
```

### Visualization
```python
import matplotlib.pyplot as plt

# Plot generation time vs answer length
plt.scatter(df["answer_words"], df["gen_time"])
plt.xlabel("Answer Length (words)")
plt.ylabel("Generation Time (seconds)")
plt.title("Generation Performance")
plt.show()
```

## Tips

1. **Disk Space**: JSON logs can get large. Monitor disk usage and archive/delete old logs regularly.

2. **Privacy**: Query logs contain full question text and answers. Be mindful of sensitive information.

3. **Performance**: Logging adds minimal overhead (~10-50ms per query) to write the JSON file.

4. **Selective Logging**: Only enable logging when needed. Disable for production unless monitoring is required.

5. **Structured Analysis**: Use `jq` or Python/pandas for analyzing large numbers of log files.

## Environment Variables Summary

```bash
# Enable query logging
export LOG_QUERIES=1

# Disable query logging (default)
export LOG_QUERIES=0
# or
unset LOG_QUERIES
```

## Example Workflow

```bash
# 1. Enable logging
export LOG_QUERIES=1

# 2. Run queries on your document
export PGTABLE=llama2_paper
python rag_low_level_m1_16gb_verbose.py --interactive

# 3. Run multiple test queries
# (ask several questions in interactive mode)

# 4. Analyze results
echo "=== Query Performance Summary ==="
echo "Total queries:"
ls query_logs/llama2_paper/*.json | wc -l

echo "Average best score:"
jq '.retrieval.quality_metrics.best_score' query_logs/llama2_paper/*.json | \
  awk '{sum+=$1; count+=1} END {print sum/count}'

echo "Average generation time:"
jq '.generation.generation_time_seconds' query_logs/llama2_paper/*.json | \
  awk '{sum+=$1; count+=1} END {print sum/count " seconds"}'

# 5. Disable logging when done
export LOG_QUERIES=0
```

## Troubleshooting

### Logs Not Being Created
- Check that `LOG_QUERIES=1` is set
- Verify write permissions in current directory
- Check logs for any error messages about file writing

### Large Log Files
- Reduce the number of queries
- Archive and compress old logs
- Consider storing logs on a larger disk

### Missing Data in Logs
- Ensure the query completed successfully
- Check that retrieval returned results
- Verify the LLM generated a response

## Next Steps

- Set up automated log analysis scripts
- Create dashboards for query performance monitoring
- Implement log rotation for long-running systems
- Export logs to a database for easier querying
