# Quick Start - Web UI

Complete guide to launch the RAG Web UI with Docker PostgreSQL.

## 🚀 One-Command Launch (Recommended)

```bash
./launch.sh
```

This script will:
1. Start PostgreSQL database in Docker
2. Wait for database to be ready
3. Check Python environment
4. Install Streamlit if needed
5. Launch the web UI

## 🛠️ Manual Setup (Step by Step)

### 1. Configure Environment

```bash
# Copy example config
cp config/.env.example .env

# Edit and set your credentials
nano .env
```

**Required variables in `.env`:**
```bash
PGUSER=postgres
PGPASSWORD=your_secure_password
DB_NAME=vector_db
PGHOST=localhost
PGPORT=5432
```

### 2. Start PostgreSQL Database

**Option A: Use the start script (recommended)**
```bash
./start_db.sh
```

**Option B: Manual start**
```bash
# Create symlink so docker-compose can find .env
ln -s ../.env config/.env

# Start database
cd config
docker-compose up -d db
```

**Option C: Specify env-file explicitly**
```bash
cd config
docker-compose --env-file ../.env up -d db
```

**Check it's running:**
```bash
docker ps | grep rag_postgres
cd config && docker-compose logs db
```

### 3. Install Web UI Dependencies

```bash
cd ..
source .venv/bin/activate
pip install streamlit plotly umap-learn watchdog
```

### 4. Launch Web UI

```bash
streamlit run rag_web_enhanced.py
```

Open browser to: **http://localhost:8501**

## 🎛️ Available Commands

### Database Management

```bash
# Start database only
cd config && docker-compose up -d db

# Start full monitoring stack (DB + Grafana + Prometheus)
cd config && docker-compose up -d

# Stop everything
cd config && docker-compose down

# View logs
cd config && docker-compose logs -f db

# Restart database
cd config && docker-compose restart db
```

### Web UI Management

```bash
# Launch enhanced UI (with presets + full controls)
streamlit run rag_web_enhanced.py

# Launch original UI
streamlit run rag_web.py

# Kill Streamlit
# Press Ctrl+C in terminal, or:
pkill -f streamlit
```

## 🔍 Troubleshooting

### "PGPASSWORD is missing a value" Error

**Problem:** Docker Compose can't find `.env` file when running from `config/` directory.

**Solution 1 - Create symlink (recommended):**
```bash
cd /Users/frytos/code/llamaIndex-local-rag
ln -s ../.env config/.env
cd config
docker-compose up -d db
```

**Solution 2 - Use --env-file flag:**
```bash
cd config
docker-compose --env-file ../.env up -d db
```

**Solution 3 - Export variables:**
```bash
cd /Users/frytos/code/llamaIndex-local-rag
source .env
cd config
docker-compose up -d db
```

### Database Won't Start

```bash
# Check Docker is running
docker ps

# Check logs
cd config && docker-compose logs db

# Reset database (WARNING: deletes all data)
cd config
docker-compose down -v
docker-compose up -d db
```

### Connection Refused

```bash
# Check database is accessible
source .env
psql -h localhost -U $PGUSER -d $DB_NAME -c "SELECT version();"

# If fails, check Docker network
docker network ls
docker network inspect config_rag_network
```

### Port Already in Use

```bash
# Check what's using port 5432
lsof -i :5432

# Or use different port in .env
PGPORT=5433

# Then restart:
cd config && docker-compose down && docker-compose up -d db
```

### Streamlit Port in Use (8501)

```bash
# Check what's using port 8501
lsof -i :8501

# Kill existing Streamlit
pkill -f streamlit

# Or use different port
streamlit run rag_web_enhanced.py --server.port 8502
```

## 📊 Full Monitoring Stack

Start complete monitoring with Grafana dashboards:

```bash
cd config
docker-compose up -d
```

**Access:**
- **Database**: localhost:5432
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **cAdvisor**: http://localhost:8080
- **Node Exporter**: http://localhost:9100

## 🧪 Quick Test

Once UI is running:

1. Open http://localhost:8501
2. Go to **Settings** tab
3. Click "Test Connection"
4. Should see: ✓ Connected!
5. Go to **Quick Start** tab
6. Select "Balanced ⚖️" preset
7. Choose a document from `data/` folder
8. Click "🚀 Index Now"

## 📝 Environment Variables Reference

### Database (Required)
- `PGUSER` - Database user (default: postgres)
- `PGPASSWORD` - Database password (REQUIRED)
- `DB_NAME` - Database name (default: vector_db)
- `PGHOST` - Database host (default: localhost)
- `PGPORT` - Database port (default: 5432)

### Indexing
- `CHUNK_SIZE` - Chunk size in characters (default: 700)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 150)
- `EMBED_MODEL` - Embedding model (default: BAAI/bge-small-en)
- `EMBED_DIM` - Embedding dimensions (default: 384)
- `EMBED_BATCH` - Embedding batch size (default: 32)

### Query
- `TOP_K` - Number of chunks to retrieve (default: 4)
- `TEMP` - LLM temperature (default: 0.1)
- `MAX_NEW_TOKENS` - Max generation tokens (default: 256)
- `CTX` - Context window size (default: 3072)

### LLM (llama.cpp)
- `MODEL_URL` - Hugging Face model URL
- `MODEL_PATH` - Local model path (overrides URL)
- `N_GPU_LAYERS` - GPU layers for Metal (default: 24)
- `N_BATCH` - Batch size (default: 256)

## 🆘 Support

**Check logs:**
```bash
# Database logs
cd config && docker-compose logs -f db

# Streamlit logs
# Shown in terminal where you ran streamlit
```

**Reset everything:**
```bash
# Stop containers
cd config && docker-compose down -v

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Restart
./launch.sh
```
