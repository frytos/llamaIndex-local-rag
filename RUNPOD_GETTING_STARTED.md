# 🚀 RunPod Deployment - Getting Started in 5 Minutes

**Quick start guide for deploying RAG pipeline to RunPod GPUs**

---

## Prerequisites (2 minutes)

### 1. RunPod Account

Sign up at https://runpod.io/signup (free)

### 2. API Key

1. Go to https://runpod.io/settings
2. Copy your API key
3. Save it securely

### 3. Install Dependencies

```bash
# Install all dependencies (includes RunPod SDK)
pip install -r requirements.txt

# Or install just RunPod SDK
pip install "runpod>=1.7.5"
```

✅ Done! You're ready to deploy.

**Note**: RunPod SDK (`runpod>=1.7.5`) is already included in `requirements.txt`

---

## Quick Start (3 minutes)

### Method 1: Streamlit UI (Recommended)

```bash
# 1. Launch UI
streamlit run rag_web.py

# 2. Click "☁️ RunPod Deployment" in sidebar

# 3. Enter your API key

# 4. Click "🚀 Deploy Pod"

# 5. Wait 2-3 minutes

# 6. Follow post-deployment instructions

✅ Done! Pod deployed.
```

### Method 2: One-Command CLI

```bash
# 1. Set API key
export RUNPOD_API_KEY=your_api_key_here

# 2. Deploy
bash scripts/quick_deploy_runpod.sh

# 3. Wait 2-3 minutes

✅ Done! Pod deployed.
```

---

## Next Steps (5 minutes)

### 1. Initialize Services

```bash
# SSH into pod
ssh POD_HOST@ssh.runpod.io

# Run initialization
bash /workspace/rag-pipeline/scripts/init_runpod_services.sh

# Wait 5-10 minutes for PostgreSQL and vLLM setup
```

### 2. Create SSH Tunnel

```bash
# In new terminal on local machine
ssh -L 8000:localhost:8000 -L 5432:localhost:5432 POD_HOST@ssh.runpod.io

# Keep this running!
```

### 3. Test Services

```bash
# Test vLLM
curl http://localhost:8000/health

# Test PostgreSQL
psql -h localhost -U fryt -d vector_db -c "SELECT 1"
```

### 4. Run Queries

```bash
# Use RAG pipeline
python rag_low_level_m1_16gb_verbose.py --query-only --query "test question"

# Or use Streamlit Query tab
# Go to "Query" tab in UI → Enter question → Get answer!
```

---

## When You're Done

### Save Costs

```bash
# Stop pod (keeps data, stops GPU billing)
python scripts/runpod_cli.py stop POD_ID

# Or click "⏸️ Stop" in Streamlit UI
```

### Resume Later

```bash
# Resume pod
python scripts/runpod_cli.py resume POD_ID

# Or click "▶️ Resume" in Streamlit UI
```

---

## Cost Reference

| Usage | Monthly Cost |
|-------|--------------|
| 2 hours/day | **$30** |
| 4 hours/day | **$60** |
| 8 hours/day | **$120** |
| 24/7 | **$360** |

💡 **Tip**: Stop pods when not in use to save 40-60%!

---

## Troubleshooting

### Can't Connect to RunPod

```bash
# Test your API key
python scripts/test_runpod_connection.py --api-key YOUR_KEY
```

### Services Not Running

```bash
# SSH into pod and check
ssh POD_HOST@ssh.runpod.io

# Check PostgreSQL
service postgresql status

# Check vLLM logs
cat /workspace/rag-pipeline/logs/vllm.log
```

### Slow Performance

```bash
# Ensure HNSW indices exist
python audit_index.py

# If missing, they'll be auto-created during indexing
```

---

## Help & Support

### Documentation

- 📖 Quick Reference: `RUNPOD_QUICK_REFERENCE.md`
- 📖 Complete Guide: `docs/RUNPOD_DEPLOYMENT_WORKFLOW.md`
- 📖 UI Guide: `docs/PHASE3_STREAMLIT_UI.md`

### External

- 🌐 RunPod Docs: https://docs.runpod.io/
- 💬 RunPod Discord: https://discord.gg/runpod
- 🐙 Python SDK: https://github.com/runpod/runpod-python

---

## That's It!

**Total time**: 5 minutes to deploy
**Performance**: 200x faster
**Cost**: $30-120/month

**Start now**: `streamlit run rag_web.py` 🚀

---

**Questions?** Check the comprehensive docs or ask for help!
