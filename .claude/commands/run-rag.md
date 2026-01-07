---
description: Run RAG pipeline with custom parameters
---

# Run RAG Pipeline

Execute the RAG pipeline with specified parameters for indexing and/or querying.

## Usage

```
/run-rag [options]
```

## Options

### Document Selection
- `--doc <path>` - Document or folder to index
- `--table <name>` - PostgreSQL table name

### Chunking
- `--chunk-size <n>` - Chunk size (default: 700)
- `--overlap <n>` - Overlap size (default: 150)
- `--preset <name>` - Use preset: ultra-fine|fine|balanced|contextual|large

### Retrieval
- `--top-k <n>` - Number of chunks to retrieve (default: 4)
- `--query <text>` - Question to ask

### Modes
- `--index-only` - Only index, don't query
- `--query-only` - Only query existing index
- `--interactive` - Interactive query mode
- `--reset` - Reset table before indexing

## Presets

| Preset | Chunk Size | Overlap | Best For |
|--------|------------|---------|----------|
| ultra-fine | 100 | 20 | Chat logs, tweets |
| fine | 300 | 60 | Q&A, facts |
| balanced | 700 | 150 | General (default) |
| contextual | 1200 | 240 | Summaries |
| large | 2000 | 400 | Long explanations |

## Examples

### Index a PDF with balanced settings
```
/run-rag --doc data/document.pdf --table my_doc --reset
```

### Query existing index
```
/run-rag --table my_doc --query-only --query "What is the main topic?"
```

### Fine-grained indexing for Q&A
```
/run-rag --doc data/faq.txt --table faq_index --preset fine --reset
```

### Full pipeline with custom settings
```
/run-rag --doc data/book.pdf --table book_cs500 --chunk-size 500 --overlap 100 --top-k 5 --query "Summarize chapter 1"
```

## Workflow

1. **Parse Parameters**: Extract configuration from command
2. **Validate**: Check paths and settings
3. **Execute**: Run appropriate pipeline mode
4. **Report**: Show results and statistics

## Output

The command will show:
- Document loading progress
- Chunking statistics
- Embedding progress
- Storage confirmation
- Query results (if applicable)
- Performance metrics

## Environment Variables Set

```bash
PDF_PATH=<doc>
PGTABLE=<table>
CHUNK_SIZE=<size>
CHUNK_OVERLAP=<overlap>
TOP_K=<k>
RESET_TABLE=<0|1>
```

## Tips

- Use `--reset` when changing chunk settings
- Use `--query-only` to test queries without re-indexing
- Include chunk config in table name: `doc_cs500_ov100`
- Start with `balanced` preset, adjust based on results
