---
description: Launch and manage the Streamlit web UI
---

# Web UI Management

Launch and manage the Streamlit web interface for the RAG pipeline.

## Usage

```
/web-ui [command]
```

## Commands

### Start Web UI
```bash
streamlit run rag_web.py
```

Opens at: http://localhost:8501

### Start on Custom Port
```bash
streamlit run rag_web.py --server.port 8080
```

### Start in Background
```bash
nohup streamlit run rag_web.py > streamlit.log 2>&1 &
```

### Stop Background Server
```bash
pkill -f "streamlit run rag_web.py"
```

## Features

### Index Documents Page
- Select document or folder from data/
- Configure chunking (presets or custom)
- Choose embedding model
- View real-time progress
- See chunk distribution chart
- Visualize embeddings (t-SNE/UMAP/PCA)

### Query Page
- Select existing index
- Configure TOP_K
- Enter questions
- View retrieved chunks with scores
- See generated answers
- Query history

### View Indexes Page
- List all vector tables
- See row counts and configurations
- Visualize embeddings from any index
- Delete indexes

### Settings Page
- Database connection config
- Test connection
- Clear caches

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill it
kill -9 <PID>
```

### Database Connection Error
1. Check PostgreSQL is running
2. Verify credentials in Settings page
3. Click "Test Connection"

### Slow Embedding Visualization
- Limit to 500 points (default)
- Use PCA instead of t-SNE (faster)
- Use 2D instead of 3D

### Streamlit Cache Issues
1. Go to Settings page
2. Click "Clear All Caches"
3. Refresh browser

## Development Mode

For auto-reload on code changes:
```bash
streamlit run rag_web.py --server.runOnSave true
```

## Configuration

Streamlit config file: `~/.streamlit/config.toml`

```toml
[server]
port = 8501
headless = true

[theme]
base = "dark"
```

## Tips

- Use Chrome/Firefox for best performance
- Embedding viz works best with < 1000 points
- Keep browser tab focused for progress updates
- Use query history to avoid retyping
