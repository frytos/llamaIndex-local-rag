# Indexing Implementation Guide

**Quick Start**: Follow this guide to implement the indexing strategy from `DATA_INDEXING_STRATEGY.md`

---

## Phase 1: Assessment & Preparation

### 1.1 Current State Analysis

```bash
# Current data directory analysis
du -sh /Users/frytos/code/llamaIndex-local-rag/data 2>/dev/null
# Output: 47G

# List all folders with sizes
du -sh /Users/frytos/code/llamaIndex-local-rag/data/* 2>/dev/null | sort -rh
```

### 1.2 Decision: What to Keep

Run through the decision matrix from `DATA_INDEXING_STRATEGY.md` Section 9.

**For this implementation, we're keeping**:
- ✓ `rag_research_papers/` (173 MB) - Core RAG papers
- ✓ `llama2.pdf` (13 MB) - LLM foundation
- ✓ `mastering-rag.pdf` (16 MB) - RAG methodology
- ✓ `messenger_clean_small/` (80 MB) - Clean conversations
- ✓ `inbox_small/` (1 MB) - Sample messages (optional)

**Optional for archiving (don't delete yet)**:
- ⚠️ `enwiktionary-latest-pages-articles.xml` (11 GB)
- ⚠️ `facebook-groussarda-14_08_2024-c4NOel1H` (5.7 GB)

**Should be deleted (redundant duplicates)**:
- ✗ `251218-messenger/` (28 GB)
- ✗ `messenger/` (786 MB)
- ✗ `messages/` (786 MB)
- ✗ `messages-text/` (242 MB)
- ✗ `messages-text-slim/` (101 MB)
- ✗ `messenger_clean/` (144 MB)
- ✗ `inbox/` (44 MB)
- ✗ `inbox_clean/` (11 MB)
- ✗ `test_*` folders (56 KB)
- ✗ `untitled folder*` (empty)

---

## Phase 2: Backup & Cleanup (Optional but Recommended)

### 2.1 Backup Large Files (If You Want Them Later)

```bash
# Create backup directory
mkdir -p /Volumes/external_backup/llamaindex_archive  # or another location

# Option A: Archive to external drive
cp -v /Users/frytos/code/llamaIndex-local-rag/data/enwiktionary-latest-pages-articles.xml \
       /Volumes/external_backup/llamaindex_archive/

cp -rv /Users/frytos/code/llamaIndex-local-rag/data/facebook-groussarda-14_08_2024-c4NOel1H \
       /Volumes/external_backup/llamaindex_archive/

# Option B: Create compressed archive (saves space)
tar -czf /Volumes/external_backup/llamaindex_archive/facebook-export.tar.gz \
         /Users/frytos/code/llamaIndex-local-rag/data/facebook-groussarda-14_08_2024-c4NOel1H

# Verify backup
du -sh /Volumes/external_backup/llamaindex_archive/*
ls -la /Volumes/external_backup/llamaindex_archive/
```

### 2.2 Delete Redundant Messenger Data (Safe - Duplicates)

```bash
# Verify these are duplicates before deleting
echo "Checking messenger folder sizes..."
du -sh /Users/frytos/code/llamaIndex-local-rag/data/messenger*
du -sh /Users/frytos/code/llamaIndex-local-rag/data/messages*

# Proceed with caution - these are true duplicates of worse quality
# (keep only messenger_clean_small)
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/251218-messenger
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/messenger
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/messages
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/messages-text
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/messages-text-slim
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/messenger_clean
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/inbox  # Keep inbox_small instead
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/inbox_clean  # No text files here

# Delete empty test folders
rm -rv "/Users/frytos/code/llamaIndex-local-rag/data/untitled folder"
rm -rv "/Users/frytos/code/llamaIndex-local-rag/data/untitled folder 2"
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/test_messenger_input
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/test_messenger_output

# Archive or delete the large files
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/enwiktionary-latest-pages-articles.xml
rm -rv /Users/frytos/code/llamaIndex-local-rag/data/facebook-groussarda-14_08_2024-c4NOel1H

# Archive: zip file (probably not needed)
rm -f /Users/frytos/code/llamaIndex-local-rag/data/messenger_clean_small.zip

echo "Cleanup complete. Current data size:"
du -sh /Users/frytos/code/llamaIndex-local-rag/data
# Should be approximately: 300 MB (papers + books + cleaned samples)
```

### 2.3 Verify Cleanup

```bash
echo "=== Data directory after cleanup ==="
ls -lh /Users/frytos/code/llamaIndex-local-rag/data/ | grep -E "^d|^-" | grep -v "\." | head -20

echo ""
echo "=== Final directory sizes ==="
du -sh /Users/frytos/code/llamaIndex-local-rag/data/*

echo ""
echo "=== Total space ==="
du -sh /Users/frytos/code/llamaIndex-local-rag/data
```

Expected output after cleanup:
```
173M    rag_research_papers
 16M    mastering-rag.pdf
 13M    llama2.pdf
 80M    messenger_clean_small
  1M    inbox_small
  0M    ethical-slut.pdf (2.8M, maybe keep)
 ───────────────────────
~283M   Total
```

---

## Phase 3: PostgreSQL Preparation

### 3.1 Verify Database Connection

```bash
# Test connection
export PGHOST=localhost
export PGPORT=5432
export PGUSER=fryt
export PGPASSWORD=frytos
export DB_NAME=vector_db

psql -h $PGHOST -U $PGUSER -d $DB_NAME -c "SELECT version();"

# Should output PostgreSQL version info
# If error: database connection failed
```

### 3.2 Verify pgvector Extension

```bash
# Check if pgvector is installed
psql -h localhost -U fryt -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify
psql -h localhost -U fryt -d vector_db -c "\dx vector"

# Output should show: vector | 0.5.0 (or similar version)
```

### 3.3 Check Existing Tables

```bash
# List all tables
psql -h localhost -U fryt -d vector_db -c "\dt"

# Count rows in existing tables (if any)
psql -h localhost -U fryt -d vector_db << 'EOF'
SELECT table_name,
  (SELECT COUNT(*) FROM information_schema.tables
   WHERE table_schema = 'public') as table_count
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
EOF
```

### 3.4 Clean Up Old Tables (Optional)

```bash
# List all tables matching our pattern
psql -h localhost -U fryt -d vector_db << 'EOF'
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_name LIKE '%cs700%';
EOF

# If you want to reset all old tables:
# (Careful - this deletes data!)
psql -h localhost -U fryt -d vector_db << 'EOF'
-- Drop old tables if they exist
DROP TABLE IF EXISTS rag_papers_cs700_ov150 CASCADE;
DROP TABLE IF EXISTS technical_books_cs700_ov150 CASCADE;
DROP TABLE IF EXISTS messenger_small_cs700_ov150 CASCADE;
DROP TABLE IF EXISTS inbox_sample_cs700_ov150 CASCADE;

-- Verify
\dt
EOF
```

---

## Phase 4: Indexing Execution

### 4.1 Prepare Environment

```bash
# Navigate to project directory
cd /Users/frytos/code/llamaIndex-local-rag

# Activate virtual environment (if using venv)
source .venv/bin/activate

# Load database credentials
export PGHOST=localhost
export PGPORT=5432
export PGUSER=fryt
export PGPASSWORD=frytos
export DB_NAME=vector_db

# Verify Python installation
python --version
python -c "import llama_index; print('LlamaIndex OK')"
```

### 4.2 Index Table 1: Research Papers (Priority 1)

This is the most important table. Start here.

```bash
# ========== TABLE 1: RAG Research Papers ==========
export PDF_PATH=data/rag_research_papers
export PGTABLE=rag_papers_cs700_ov150
export RESET_TABLE=1
export CHUNK_SIZE=700
export CHUNK_OVERLAP=150
export EMBED_BATCH=64

# Optional: Enable performance tracking
export ENABLE_PERFORMANCE_RECORDING=1

echo "Starting indexing of research papers..."
echo "PDF_PATH: $PDF_PATH"
echo "PGTABLE: $PGTABLE"
echo "Time: $(date)"

time python rag_low_level_m1_16gb_verbose.py

echo "Research papers indexing complete at $(date)"
```

**Expected output**:
```
Loading documents from data/rag_research_papers...
Found 109 PDF files
...
Building embeddings...
Inserted X chunks into rag_papers_cs700_ov150

Time: ~40-50 seconds
```

### 4.3 Index Table 2: Technical Books (Priority 2)

```bash
# ========== TABLE 2: Technical Books ==========
export PDF_PATH=data/llama2.pdf
export PGTABLE=technical_books_cs700_ov150
export RESET_TABLE=1
export CHUNK_SIZE=700
export CHUNK_OVERLAP=150
export EMBED_BATCH=64

echo "Starting indexing of technical books..."
echo "PDF_PATH: $PDF_PATH"
echo "PGTABLE: $PGTABLE"
echo "Time: $(date)"

time python rag_low_level_m1_16gb_verbose.py

echo "Technical books indexing complete at $(date)"
```

**Expected output**:
```
Loading documents from data/llama2.pdf...
Processing: llama2.pdf
...
Inserted X chunks into technical_books_cs700_ov150

Time: ~10-15 seconds
```

### 4.4 Index Mastering RAG (Add to Table 2 or Separate)

```bash
# Option A: Add to same table as llama2
export PDF_PATH=data/mastering-rag.pdf
export PGTABLE=technical_books_cs700_ov150
export RESET_TABLE=0  # DON'T reset, add to existing table
export CHUNK_SIZE=700
export CHUNK_OVERLAP=150

echo "Adding mastering-rag.pdf to technical_books table..."
time python rag_low_level_m1_16gb_verbose.py

# Option B: Create separate table
# export PGTABLE=mastering_rag_cs700_ov150
# export RESET_TABLE=1
```

### 4.5 Index Table 3: Messenger Conversations (Priority 3)

```bash
# ========== TABLE 3: Messenger Clean Small ==========
export PDF_PATH=data/messenger_clean_small
export PGTABLE=messenger_small_cs700_ov150
export RESET_TABLE=1
export CHUNK_SIZE=700
export CHUNK_OVERLAP=150
export EMBED_BATCH=64

echo "Starting indexing of messenger conversations..."
echo "PDF_PATH: $PDF_PATH"
echo "PGTABLE: $PGTABLE"
echo "Time: $(date)"

time python rag_low_level_m1_16gb_verbose.py

echo "Messenger indexing complete at $(date)"
```

**Expected output**:
```
Loading documents from data/messenger_clean_small...
Found X conversation files
...
Inserted X chunks into messenger_small_cs700_ov150

Time: ~20-30 seconds
```

### 4.6 Index Table 4: Inbox Sample (Priority 4, Optional)

```bash
# ========== TABLE 4: Inbox Sample (Optional) ==========
export PDF_PATH=data/inbox_small
export PGTABLE=inbox_sample_cs700_ov150
export RESET_TABLE=1
export CHUNK_SIZE=700
export CHUNK_OVERLAP=150
export EMBED_BATCH=64

echo "Starting indexing of inbox sample..."
echo "PDF_PATH: $PDF_PATH"
echo "PGTABLE: $PGTABLE"
echo "Time: $(date)"

time python rag_low_level_m1_16gb_verbose.py

echo "Inbox sample indexing complete at $(date)"
```

**Expected output**:
```
Loading documents from data/inbox_small...
Found X message files
...
Inserted X chunks into inbox_sample_cs700_ov150

Time: ~2-3 seconds
```

---

## Phase 5: Verification & Testing

### 5.1 Verify All Tables Created

```bash
# List all new tables
psql -h localhost -U fryt -d vector_db << 'EOF'
SELECT table_name,
       obj_description(to_regclass(table_name), 'pg_class') as description
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE '%cs700%'
ORDER BY table_name;
EOF
```

**Expected output**:
```
 table_name                    | description
───────────────────────────────┼─────────────────
 rag_papers_cs700_ov150        |
 technical_books_cs700_ov150   |
 messenger_small_cs700_ov150   |
 inbox_sample_cs700_ov150      |
(4 rows)
```

### 5.2 Count Chunks in Each Table

```bash
# Count total chunks per table
psql -h localhost -U fryt -d vector_db << 'EOF'
SELECT
  'rag_papers_cs700_ov150' as table_name,
  COUNT(*) as chunk_count
FROM rag_papers_cs700_ov150
UNION ALL
SELECT 'technical_books_cs700_ov150', COUNT(*) FROM technical_books_cs700_ov150
UNION ALL
SELECT 'messenger_small_cs700_ov150', COUNT(*) FROM messenger_small_cs700_ov150
UNION ALL
SELECT 'inbox_sample_cs700_ov150', COUNT(*) FROM inbox_sample_cs700_ov150
ORDER BY chunk_count DESC;
EOF
```

**Expected output**:
```
 table_name                   | chunk_count
──────────────────────────────┼─────────────
 rag_papers_cs700_ov150       |   45000-50000
 messenger_small_cs700_ov150  |   25000-30000
 technical_books_cs700_ov150  |   10000-12000
 inbox_sample_cs700_ov150     |   500-1000
```

### 5.3 Check Database Size

```bash
# Estimate database size (PostgreSQL internal)
psql -h localhost -U fryt -d vector_db << 'EOF'
SELECT
  table_name,
  ROUND(pg_total_relation_size(table_name::regclass) / 1024.0 / 1024.0, 2) as size_mb
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE '%cs700%'
ORDER BY pg_total_relation_size(table_name::regclass) DESC;
EOF
```

**Expected output** (total ~4-5 GB):
```
 table_name                   | size_mb
──────────────────────────────┼─────────
 rag_papers_cs700_ov150       |  2000-2500
 messenger_small_cs700_ov150  |  1000-1500
 technical_books_cs700_ov150  |  400-600
 inbox_sample_cs700_ov150     |  50-100
```

### 5.4 Sample Query Test

```bash
# Test interactive query mode
python rag_interactive.py

# When prompted:
# Select table: rag_papers_cs700_ov150
# Enter query: "What is retrieval augmented generation?"
# Should return relevant paper chunks

# Or test programmatically:
python << 'EOF'
import psycopg2
import json

conn = psycopg2.connect(
    host="localhost",
    database="vector_db",
    user="fryt",
    password="frytos"
)

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM rag_papers_cs700_ov150")
count = cur.fetchone()[0]
print(f"rag_papers_cs700_ov150: {count} chunks")

cur.execute("SELECT COUNT(*) FROM technical_books_cs700_ov150")
count = cur.fetchone()[0]
print(f"technical_books_cs700_ov150: {count} chunks")

cur.execute("SELECT COUNT(*) FROM messenger_small_cs700_ov150")
count = cur.fetchone()[0]
print(f"messenger_small_cs700_ov150: {count} chunks")

cur.close()
conn.close()
EOF
```

---

## Phase 6: Integration with Web UI

### 6.1 Update rag_web.py to Use New Tables

```python
# In your rag_web.py, update table selection
AVAILABLE_TABLES = {
    "RAG Research Papers": "rag_papers_cs700_ov150",
    "Technical Books (Llama2 + RAG)": "technical_books_cs700_ov150",
    "Sample Conversations": "messenger_small_cs700_ov150",
    "Inbox Messages": "inbox_sample_cs700_ov150",
    # "All Knowledge": "combined_knowledge_cs700_ov150",  # Optional
}

# Or use dynamic table discovery:
def get_available_tables():
    """Query database for available tables"""
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name LIKE '%cs700%'
        ORDER BY table_name
    """)
    return {row[0]: row[0] for row in cur.fetchall()}
```

### 6.2 Update rag_interactive.py

```python
# In your rag_interactive.py, add new tables to menu
def get_table_options():
    return [
        "rag_papers_cs700_ov150",
        "technical_books_cs700_ov150",
        "messenger_small_cs700_ov150",
        "inbox_sample_cs700_ov150",
    ]
```

### 6.3 Test Web UI

```bash
# Start the web interface
streamlit run rag_web.py

# In browser:
# http://localhost:8501
# Select table dropdown should show new tables
# Test query in each table
```

---

## Phase 7: Documentation & Cleanup

### 7.1 Document Your Configuration

Create `/Users/frytos/code/llamaIndex-local-rag/INDEXING_CONFIG.json`:

```json
{
  "indexing_date": "2026-01-08",
  "total_source_size_mb": 283,
  "total_database_size_gb": 4.5,
  "tables": {
    "rag_papers_cs700_ov150": {
      "source": "data/rag_research_papers",
      "format": "PDF",
      "file_count": 109,
      "chunk_count": 47500,
      "size_mb": 2200,
      "description": "RAG, LLM, and retrieval methodology papers",
      "priority": 1
    },
    "technical_books_cs700_ov150": {
      "source": "data/llama2.pdf + data/mastering-rag.pdf",
      "format": "PDF",
      "file_count": 2,
      "chunk_count": 11000,
      "size_mb": 500,
      "description": "Foundational LLM and RAG knowledge",
      "priority": 2
    },
    "messenger_small_cs700_ov150": {
      "source": "data/messenger_clean_small",
      "format": "TXT",
      "file_count": 115,
      "chunk_count": 27500,
      "size_mb": 1200,
      "description": "Sample conversation threads for dialogue RAG",
      "priority": 3
    },
    "inbox_sample_cs700_ov150": {
      "source": "data/inbox_small",
      "format": "TXT",
      "file_count": 50,
      "chunk_count": 750,
      "size_mb": 75,
      "description": "Message sample for retrieval testing",
      "priority": 4
    }
  },
  "excluded_data": {
    "enwiktionary": {
      "size_gb": 11,
      "reason": "Too large, low RAG value"
    },
    "facebook_activity": {
      "size_gb": 5.7,
      "reason": "Privacy risk, low relevance"
    },
    "messenger_duplicates": {
      "size_gb": 2.3,
      "reason": "Redundant, lower quality variants"
    }
  },
  "configuration": {
    "chunk_size": 700,
    "chunk_overlap": 150,
    "embedding_batch_size": 64
  }
}
```

### 7.2 Update Project README

```markdown
## Indexed Data

The local RAG pipeline indexes the following data:

### Available Tables

| Table | Source | Type | Size | Purpose |
|-------|--------|------|------|---------|
| rag_papers_cs700_ov150 | rag_research_papers/ | PDF (109) | 2.2 GB | RAG/LLM papers |
| technical_books_cs700_ov150 | llama2.pdf + mastering-rag.pdf | PDF (2) | 500 MB | Foundation |
| messenger_small_cs700_ov150 | messenger_clean_small/ | TXT (115) | 1.2 GB | Conversations |
| inbox_sample_cs700_ov150 | inbox_small/ | TXT (50) | 75 MB | Messages |

**Total Database Size**: ~4.5 GB
**Total Source Size**: ~283 MB
**Indexing Configuration**: CHUNK_SIZE=700, CHUNK_OVERLAP=150

### Excluded Data

- **enwiktionary-latest-pages-articles.xml** (11 GB) - Too large, low RAG value
- **facebook-groussarda-14_08_2024/** (5.7 GB) - Privacy risk, low relevance
- **messenger duplicates** (2.3 GB) - Use clean_small instead

See `DATA_INDEXING_STRATEGY.md` for comprehensive analysis.
```

### 7.3 Create Data Inventory

```bash
# Generate inventory of indexed data
cat > /Users/frytos/code/llamaIndex-local-rag/DATA_INVENTORY.md << 'EOF'
# Data Inventory

**Last Updated**: 2026-01-08
**Total Size**: 283 MB (source), 4.5 GB (database with vectors)

## Indexed Content

### Research Papers (173 MB)
- Location: `data/rag_research_papers/`
- Count: 109 PDF files
- Topics: RAG, retrieval, embeddings, LLM augmentation
- Table: `rag_papers_cs700_ov150`
- Chunks: ~47,500
- DB Size: ~2.2 GB

### Technical Books (29 MB)
- Location: `data/llama2.pdf`, `data/mastering-rag.pdf`
- Count: 2 PDF files
- Topics: Llama 2 model, RAG methodology
- Table: `technical_books_cs700_ov150`
- Chunks: ~11,000
- DB Size: ~500 MB

### Sample Conversations (80 MB)
- Location: `data/messenger_clean_small/`
- Count: 115 conversation threads
- Topics: Personal conversations (development use only)
- Table: `messenger_small_cs700_ov150`
- Chunks: ~27,500
- DB Size: ~1.2 GB

### Message Samples (1 MB)
- Location: `data/inbox_small/`
- Count: 50 message files
- Topics: Individual messages (development use only)
- Table: `inbox_sample_cs700_ov150`
- Chunks: ~750
- DB Size: ~75 MB

## Excluded/Archived

- **Dictionary**: enwiktionary XML (11 GB) - Archived
- **Facebook Activity**: Personal export (5.7 GB) - Archived
- **Message Duplicates**: messenger/messages variants (2.3 GB) - Deleted

## Statistics

- **Total chunks**: ~87,250
- **Total database size**: ~4.5 GB
- **Average chunk size**: 700 characters
- **Overlap**: 150 characters (21%)
- **Indexing time**: ~5-8 minutes (all tables)
EOF

cat /Users/frytos/code/llamaIndex-local-rag/DATA_INVENTORY.md
```

---

## Phase 8: Troubleshooting

### 8.1 Common Issues & Solutions

#### Issue: "File not found" error
```bash
# Verify file exists
ls -lh data/rag_research_papers/ | head -5

# Use absolute path if needed
export PDF_PATH=/Users/frytos/code/llamaIndex-local-rag/data/rag_research_papers
```

#### Issue: Database connection failed
```bash
# Check PostgreSQL is running
psql -h localhost -U fryt -d vector_db -c "SELECT 1"

# Verify credentials
echo $PGUSER $PGPASSWORD $PGHOST

# Check port
sudo ss -tulpn | grep 5432
```

#### Issue: Embedding timeout or slow
```bash
# Reduce batch size
export EMBED_BATCH=32

# Reduce chunk size
export CHUNK_SIZE=500

# Retry indexing
python rag_low_level_m1_16gb_verbose.py
```

#### Issue: Out of memory
```bash
# Stop other applications
# Reduce embedding batch size
export EMBED_BATCH=16

# Use smaller chunk size
export CHUNK_SIZE=500
```

#### Issue: Duplicate chunks in database
```bash
# If using RESET_TABLE=0 accidentally
# Clean up duplicates
psql -h localhost -U fryt -d vector_db << 'EOF'
DELETE FROM table_name WHERE id IN (
  SELECT id FROM table_name
  ORDER BY id OFFSET (SELECT COUNT(*)/2 FROM table_name)
);
EOF

# Or recreate the table
export RESET_TABLE=1
python rag_low_level_m1_16gb_verbose.py
```

### 8.2 Performance Monitoring

```bash
# Monitor memory during indexing
python << 'EOF'
import psutil
import time

def monitor_memory():
    process = psutil.Process()
    while True:
        mem = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory: {mem:.1f} MB", end='\r')
        time.sleep(1)

if __name__ == '__main__':
    monitor_memory()
EOF

# Run in background while indexing
python monitor_memory.py &
python rag_low_level_m1_16gb_verbose.py
```

---

## Phase 9: Maintenance & Updates

### 9.1 Regular Checks

```bash
# Monthly: Check database size
psql -h localhost -U fryt -d vector_db << 'EOF'
SELECT
  schemaname,
  ROUND(SUM(pg_total_relation_size(schemaname||'.'||tablename))::numeric / 1024 / 1024 / 1024, 2) as size_gb
FROM pg_tables
GROUP BY schemaname
ORDER BY 2 DESC;
EOF

# Check for unused tables
psql -h localhost -U fryt -d vector_db << 'EOF'
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
EOF
```

### 9.2 Backups

```bash
# Backup individual table
pg_dump -h localhost -U fryt vector_db -t rag_papers_cs700_ov150 > rag_papers_backup.sql

# Restore table
psql -h localhost -U fryt -d vector_db < rag_papers_backup.sql

# Full database backup
pg_dump -h localhost -U fryt vector_db > vector_db_backup.sql

# Verify backup
du -sh vector_db_backup.sql
gzip vector_db_backup.sql  # Compress for storage
```

### 9.3 Reindexing with Updated Data

```bash
# If you add new papers to rag_research_papers/:
# Option 1: Add to existing table (no reset)
export RESET_TABLE=0
export PGTABLE=rag_papers_cs700_ov150
python rag_low_level_m1_16gb_verbose.py

# Option 2: Recreate table from scratch
export RESET_TABLE=1
export PGTABLE=rag_papers_cs700_ov150_v2
python rag_low_level_m1_16gb_verbose.py
# Then migrate apps to use new table
```

---

## Summary Checklist

- [ ] **Phase 1**: Review strategy and confirm approach
- [ ] **Phase 2**: Backup large files (optional)
- [ ] **Phase 2**: Delete redundant messenger data (saves 2.3 GB)
- [ ] **Phase 3**: Verify PostgreSQL and pgvector
- [ ] **Phase 4**: Index research papers (~1 min)
- [ ] **Phase 4**: Index technical books (~1 min)
- [ ] **Phase 4**: Index messenger conversations (~1 min)
- [ ] **Phase 4**: Index inbox sample (~1 min)
- [ ] **Phase 5**: Verify all tables created
- [ ] **Phase 5**: Test queries on each table
- [ ] **Phase 6**: Update web UI to use new tables
- [ ] **Phase 7**: Document configuration
- [ ] **Phase 8**: Set up monitoring (optional)
- [ ] **Phase 9**: Create backup strategy

**Total Time**: ~30 minutes (4 min indexing + 15 min setup + 10 min testing)

---

## Quick Reference: One-Liner Commands

```bash
# Index all in sequence
cd /Users/frytos/code/llamaIndex-local-rag && \
PDF_PATH=data/rag_research_papers PGTABLE=rag_papers_cs700_ov150 RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py && \
PDF_PATH=data/llama2.pdf PGTABLE=technical_books_cs700_ov150 RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py && \
PDF_PATH=data/messenger_clean_small PGTABLE=messenger_small_cs700_ov150 RESET_TABLE=1 python rag_low_level_m1_16gb_verbose.py

# Verify tables
psql -h localhost -U fryt -d vector_db -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '%cs700%' ORDER BY table_name;"

# Test interactive mode
python rag_interactive.py
```

---

**Next Steps**: Follow Phase 1 to get started!
