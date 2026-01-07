# Scripts Directory

Utility scripts for deployment, benchmarking, visualization, and data processing.

## Deployment Scripts

**RunPod Deployment:**
- `deploy_runpod.sh` - Deploy to RunPod GPU servers
- `runpod_startup.sh` - RunPod container startup script
- `runpod_startup_verbose.sh` - Verbose startup with detailed logging
- `verify_runpod_setup.sh` - Verify RunPod environment setup

**vLLM Server:**
- `start_vllm_server.sh` - Start vLLM server for GPU inference
- `vllm_server_control.sh` - Control vLLM server (start/stop/status)
- `quick_start_vllm.sh` - Quick start with vLLM backend
- `quick_start_optimized.sh` - Optimized quick start configuration

## Indexing Scripts

- `index_bge_small.sh` - Index with BGE-small embedding model
- `index_multilingual_e5.sh` - Index with multilingual-e5 model

## Benchmarking Scripts

- `benchmark_embeddings.py` - Benchmark embedding models
- `compare_embedding_models.py` - Compare different embedding models
- `compare_both_models.sh` - Shell script to compare both BGE and E5
- `compare_models.py` - Python script for model comparison
- `test_query_quality.sh` - Test and evaluate query quality

## Visualization Scripts

- `tensorboard_embeddings.py` - Visualize embeddings in TensorBoard
- `tensorboard_multi.py` - Multi-model TensorBoard visualization
- `atlas_embeddings.py` - Atlas/Nomic visualization of embeddings
- `visualize_rag.py` - RAG pipeline visualization
- `chainlit_app.py` - Chainlit chat UI for RAG

## Data Processing Scripts

- `clean_messenger_html.py` - Clean and process Messenger HTML exports
- `clean_instagram_json.py` - Process Instagram JSON data

## System & Database Scripts

- `database_apply_hnsw.sh` - Apply HNSW indexing to PostgreSQL
- `system_free_memory.sh` - Free system memory
- `monitoring_query.sh` - Monitor query performance

## Configuration Scripts

- `config_optimized.sh` - Optimized configuration settings
- `helper_quick_commands.sh` - Collection of useful quick commands

## Usage Examples

```bash
# Deploy to RunPod
./scripts/deploy_runpod.sh

# Start vLLM server
./scripts/start_vllm_server.sh

# Benchmark embeddings
python scripts/benchmark_embeddings.py

# Visualize with TensorBoard
python scripts/tensorboard_embeddings.py

# Clean Messenger data
python scripts/clean_messenger_html.py
```

## Naming Convention

- **Deployment**: `deploy_*`, `runpod_*`, `quick_start_*`
- **Indexing**: `index_*`
- **Benchmarking**: `benchmark_*`, `compare_*`, `test_*`
- **Visualization**: `*_embeddings.py`, `visualize_*`, `chainlit_*`
- **Data Processing**: `clean_*`
- **System/Database**: `database_*`, `system_*`, `monitoring_*`
- **Configuration**: `config_*`, `helper_*`
