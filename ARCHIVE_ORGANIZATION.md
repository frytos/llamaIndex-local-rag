# Archive and Data Organization Summary

## Overview

This document summarizes the organization and documentation created for the `archive/` and `data/` directories in the Local RAG pipeline project.

**Date:** 2026-01-07
**Purpose:** Establish clear guidelines for data management and historical record keeping

---

## Archive Directory (`/archive/`)

### Purpose
Stores historical outputs, logs, and experimental results from RAG pipeline operations.

### Current Contents
- **Total size:** ~35 MB
- **Date range:** December 2024 - January 2025
- **File count:** ~250+ files across 6 subdirectories

### Structure
```
archive/
├── README.md           # Comprehensive documentation (NEW)
├── .gitignore          # Excludes logs and temp files (NEW)
├── benchmarks/         # 13 files, 816 KB - chunk size optimization tests
├── logs/               # 4 files, 264 KB - application logs
├── performance/        # 20 files, 20 MB - memory/power monitoring
├── query_logs/         # 120+ files, 760 KB - historical query results
├── runs/               # 14 MB - complete pipeline runs
└── temp/               # 372 KB - temporary experiment files
```

### Key Features
- **File naming conventions:** ISO 8601 timestamps, descriptive configs
- **Retention policy:** Keep benchmarks indefinitely, review logs quarterly
- **Cleanup scripts:** Commands for finding old files and compressing archives
- **Git ignored:** All logs and data files excluded from version control

---

## Data Directory (`/data/`)

### Purpose
Contains documents to be indexed by the RAG pipeline.

### Current Contents
- **Total size:** ~11 GB (dominated by one 11 GB XML file)
- **PDF files:** ~300 MB across 248+ files
- **Directories:** 15+ subdirectories with organized content

### Structure
```
data/
├── README.md                   # Comprehensive guide (NEW)
├── .gitignore                  # Protects privacy (NEW)
├── llama2.pdf                  # 13 MB research paper
├── mastering-rag.pdf           # 16 MB RAG guide
├── ethical-slut.pdf            # 2.8 MB book
├── enwiktionary-latest-pages-articles.xml  # 11 GB (!)
├── rag_research_papers/        # 110 papers
├── inbox/                      # 869 HTML files
├── inbox_clean/                # 833 cleaned files
├── inbox_small/                # 52 sample files
└── messages/                   # Facebook/Messenger exports (sensitive)
```

### Key Features
- **Supported formats:** PDF, HTML, Markdown, Text, JSON, Code
- **Organization strategies:** By project, by type, by topic
- **Privacy guidelines:** Clear warnings about sensitive data
- **Performance tips:** File size recommendations, batch optimization
- **Common workflows:** Research papers, knowledge base, code docs

---

## Reorganization Suggestions

### Archive Directory

#### Current State (Good)
- Logical categorization by content type
- Consistent file naming in benchmarks
- Timestamped performance logs

#### Recommended Improvements

**1. Add Monthly Subdirectories for Query Logs**
```bash
archive/query_logs/
├── 2024-12/
│   ├── ethical-slut_paper/
│   └── other_indices/
└── 2025-01/
    └── recent_queries/
```

**Benefits:**
- Easier to find logs by time period
- Simpler quarterly cleanup
- Better organization as project grows

**2. Create Summary Files**
```bash
# Add these documentation files:
archive/benchmarks/SUMMARY.md     # Best configurations found
archive/performance/ANALYSIS.md   # Performance trends
archive/query_logs/PATTERNS.md    # Common query patterns
```

**3. Compress Old Logs**
```bash
# After 3 months, compress by month:
cd archive/query_logs/2024-12/
tar -czf ../query_logs_2024_12.tar.gz .
cd .. && rm -r 2024-12/
```

### Data Directory

#### Current State (Needs Attention)
- **Multiple redundant directories:** `inbox/`, `inbox_clean/`, `inbox_small/`
- **Duplicate message folders:** `messages/`, `messenger/`, etc.
- **11 GB file:** `enwiktionary-latest-pages-articles.xml` rarely used
- **Privacy risk:** Personal Facebook messages not git-ignored properly

#### Recommended Improvements

**1. Consolidate Messenger Data**
```bash
# Current (redundant):
messages/
messenger/
messages-text/
messages-text-slim/
messenger_clean/
messenger_clean_small/

# Recommended:
messenger_data/
├── README.md              # Processing notes
├── raw/                   # Original export
├── clean/                 # Preprocessed
└── samples/               # Small test sets
```

**2. Move Large Files to External Storage**
```bash
# Option A: Move to external drive
mv data/enwiktionary-latest-pages-articles.xml /Volumes/ExternalDrive/datasets/

# Option B: Delete if not needed
rm data/enwiktionary-latest-pages-articles.xml

# Option C: Download script instead of storing
# Add scripts/download_wiktionary.sh to fetch on demand
```

**3. Create Test Data Directory**
```bash
data/
├── test_data/             # NEW: Small files for testing
│   ├── sample.pdf         # 100 KB sample document
│   ├── test.html          # Small HTML file
│   └── README.md          # Test data description
└── datasets/              # NEW: Large reference datasets
    └── README.md          # External dataset links
```

**4. Organize by Use Case**
```bash
data/
├── research/              # Academic papers
│   ├── ml/
│   ├── nlp/
│   └── rag/
├── personal/              # Personal documents (git-ignored)
│   ├── notes/
│   └── books/
└── projects/              # Project-specific docs
    ├── project_a/
    └── project_b/
```

---

## Implementation Checklist

### Completed
- [x] Create `/archive/README.md` with comprehensive documentation
- [x] Create `/archive/.gitignore` to exclude logs from git
- [x] Create `/data/README.md` with usage guide
- [x] Create `/data/.gitignore` to protect privacy
- [x] Document current directory structures
- [x] Provide cleanup scripts and commands

### Recommended Next Steps

#### High Priority
- [ ] Review and consolidate messenger data directories
- [ ] Archive or delete `enwiktionary-latest-pages-articles.xml` (11 GB)
- [ ] Add monthly subdirectories to `archive/query_logs/`
- [ ] Create `data/test_data/` with small sample files

#### Medium Priority
- [ ] Create summary files for key archive subdirectories
- [ ] Reorganize `data/` by use case (research/personal/projects)
- [ ] Compress archive logs older than 3 months
- [ ] Add `scripts/download_datasets.sh` for large external data

#### Low Priority
- [ ] Create visualization of archive growth over time
- [ ] Set up automated cleanup cron job
- [ ] Add data preprocessing templates to `scripts/`
- [ ] Document data lineage for key indices

---

## Privacy and Security Notes

### Data Directory Risks
1. **Personal messages:** Facebook/Messenger exports contain sensitive conversations
2. **PDF attachments:** May contain personal information, contracts, or private documents
3. **No encryption:** Data stored in plaintext on disk

### Recommended Actions
1. **Immediate:** Ensure `.gitignore` prevents committing personal data
2. **Short-term:** Review and redact sensitive information from test files
3. **Long-term:** Consider encrypting the `data/personal/` directory

### Archive Directory Risks
- **Query logs:** May contain sensitive queries or responses
- **Low risk:** Mostly performance metrics and benchmarks

---

## Disk Space Summary

### Current Usage
```
Total:  11.3 GB
├── data/                    11.0 GB  (97%)
│   ├── enwiktionary.xml     11.0 GB  (97%)
│   ├── PDFs                 ~300 MB  (3%)
│   └── Other                 ~50 MB  (<1%)
└── archive/                  ~35 MB  (<1%)
```

### After Cleanup (Projected)
```
Total:  ~400 MB
├── data/                    ~350 MB
│   ├── research/             200 MB
│   ├── personal/             100 MB
│   └── test_data/             50 MB
└── archive/                  ~30 MB  (compressed)
```

**Space savings:** 11 GB → 400 MB (96% reduction)

---

## Maintenance Schedule

### Weekly
- Check disk space: `du -sh archive/ data/`
- Review new query logs in `query_logs/`

### Monthly
- Move completed experiments to `archive/`
- Compress old archive logs
- Review data directory for unused files

### Quarterly
- Delete logs older than 6 months (after review)
- Update retention policies based on usage patterns
- Audit privacy compliance

### Annually
- Major reorganization if needed
- Archive historical data to external storage
- Update documentation

---

## Related Documentation

- `/CLAUDE.md` - Main project documentation
- `/archive/README.md` - Archive management guide
- `/data/README.md` - Data usage guide
- `/data/MESSENGER_PREPROCESSING_SUMMARY.md` - Messenger data processing

---

## Questions or Issues?

If you have questions about:
- **What to archive:** See `/archive/README.md` retention policy
- **How to add data:** See `/data/README.md` quick start
- **Privacy concerns:** Review `.gitignore` files and data organization
- **Disk space:** Follow cleanup recommendations above

**Last Updated:** 2026-01-07
**Maintained By:** Project team
**Version:** 1.0
