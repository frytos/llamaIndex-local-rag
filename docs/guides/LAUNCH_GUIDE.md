# 🚀 Launch Guide - One Command to Rule Them All

**Last Updated**: January 2026

---

## ⚡ Launch Everything (10 seconds)

```bash
./launch.sh
```

**What starts:**
- ✅ PostgreSQL database (with pgvector)
- ✅ Monitoring stack (Grafana, Prometheus, cAdvisor, Alertmanager)
- ✅ Streamlit Web UI

**Then open:** http://localhost:8501 🎯

---

## 🛑 Stop Everything

```bash
./shutdown.sh
```

---

## 🎯 Access Points

| Service | URL | Login | Purpose |
|---------|-----|-------|---------|
| **Streamlit UI** | http://localhost:8501 | - | **Main interface** |
| **Grafana** | http://localhost:3000 | admin/admin | Monitoring dashboards |
| **Prometheus** | http://localhost:9090 | - | Metrics & alerts |
| **cAdvisor** | http://localhost:8080 | - | Container stats |

---

## 📝 First Time Setup (5 minutes)

### 1. Clone & Setup Environment

```bash
cd /Users/frytos/code/llamaIndex-local-rag

# Create .env file
cp config/.env.example .env
nano .env  # Set PGPASSWORD=your_password

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch!

```bash
./launch.sh
```

**That's it!** Browser opens automatically to http://localhost:8501

---

## 🔄 Daily Usage

```bash
# Every day - just run this:
./launch.sh

# When done:
./shutdown.sh
```

---

## 📥 Index Your First Dataset

Once Streamlit opens (http://localhost:8501):

1. **Quick Start tab** → Select "Balanced ⚖️"
2. **Document path**: `data/inbox_clean`
3. **Table name**: `my_conversations`
4. **Click**: "📥 Index Document"
5. **Wait**: ~2-3 minutes for 831 files

**Progress shows:**
- Loading: 831 files
- Chunking: ~15,000 chunks
- Embedding: Real-time speed (chunks/sec)
- Storing: Database writes

---

## 🔍 Run Your First Query

1. **Query tab** → Select your index
2. **Ensure**: Advanced Features disabled (all sliders at safe defaults)
3. **Ask**: "qui est agathe ?" or any question
4. **Get**: Answer with sources + similarity scores

**Enhanced metadata extracted automatically:**
- Participants, dates, group names
- Attachments, reactions, events
- 23 metadata fields per chunk!

---

## 📊 Monitor Performance

**Grafana**: http://localhost:3000
- Login: `admin` / `admin`
- Explore metrics: `pg_*`, `container_*`, `node_*`
- Create custom dashboards

**Prometheus**: http://localhost:9090
- Status → Targets (all should be UP)
- Graph → Query: `up` to see all services

---

## 🐛 Common Issues

### "Database credentials not set"
```bash
# Check .env exists and has PGPASSWORD set
cat .env | grep PGPASSWORD
```

### "Port 8501 already in use"
```bash
# Kill existing Streamlit
pkill -f 'streamlit run'

# Or use different port
streamlit run rag_web_enhanced.py --server.port 8502
```

### "Docker containers won't start"
```bash
# Remove all and restart
cd config
docker-compose down
docker-compose up -d
```

### "Prometheus keeps restarting"
```bash
# Check logs
docker logs rag_prometheus

# Usually fixed by:
docker restart rag_prometheus
```

---

## 🎛️ Advanced Usage

### Start Only What You Need

```bash
# Just database
./start_db.sh

# Just monitoring (requires database)
cd config && docker-compose up -d prometheus grafana

# Just UI (requires database)
source .venv/bin/activate
streamlit run rag_web_enhanced.py
```

### Custom Configuration

```bash
# Edit settings before launching
nano .env

# Then launch normally
./launch.sh
```

---

## 📂 Project Structure (Simplified)

```
llamaIndex-local-rag/
├── launch.sh              ← Start everything
├── shutdown.sh            ← Stop everything
├── rag_web_enhanced.py    ← Main web UI
├── .env                   ← Your configuration
├── data/                  ← Put documents here
│   ├── inbox_clean/       ← 831 email conversations (11MB)
│   └── messenger_clean_small/  ← 207 chats (80MB)
├── config/
│   ├── docker-compose.yml ← All services
│   └── monitoring/        ← Prometheus/Grafana configs
└── docs/                  ← Full documentation
```

---

## 🎯 Recommended Workflow

### Morning
```bash
./launch.sh
# ☕ Make coffee while services start (10 seconds)
# Open http://localhost:8501
```

### During Day
- Index new documents
- Run queries
- Monitor performance in Grafana

### Evening
```bash
./shutdown.sh
```

---

## 💡 Pro Tips

1. **Keep launch.sh running in a terminal** - you'll see real-time logs
2. **Open Grafana in a tab** - monitor while you query
3. **Use Quick Start presets** - Balanced is best for most use cases
4. **Let Streamlit cache build** - First query is slower, rest are fast
5. **Check "View Indexes" regularly** - Clean empty tables

---

## 🆘 Need Help?

- **This guide**: `LAUNCH_GUIDE.md` (you are here)
- **Web UI guide**: `docs/GUI_USER_GUIDE.md`
- **Developer guide**: `CLAUDE.md`
- **Full docs**: `docs/`

---

## ✨ Summary

**To launch everything:**
```bash
./launch.sh
```

**Then open:**
- **Main UI**: http://localhost:8501
- **Monitoring**: http://localhost:3000

**To stop:**
```bash
./shutdown.sh
```

**That's all you need to know!** 🎉
