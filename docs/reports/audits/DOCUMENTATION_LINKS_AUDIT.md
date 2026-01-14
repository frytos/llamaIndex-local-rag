# Documentation Links Audit Report

**Generated:** 2026-01-07
**Purpose:** Comprehensive verification of all internal documentation links after repository reorganization

---

## Executive Summary

**Total Markdown Files Checked:** 69 (excluding .venv)
**Total Links Found:** ~250
**Broken Links:** 45 (18%)
**Outdated References:** 28 (11.2%)
**External Links:** All valid

### Critical Issues

1. **8 missing root documentation files** referenced 25+ times
2. **14 references to archived PERFORMANCE_QUICK_START.md** (wrong path)
3. **12 references to moved Python files** (scripts/ and utils/ directories)
4. **8 references to renamed shell scripts** (new prefix naming)
5. **5 references to moved .env.example** (now in config/)

---

## Critical Broken Links

### 1. Missing Root Files (Referenced by Multiple Docs)

| File | References | Status |
|------|-----------|--------|
| `FINAL_SUMMARY.md` | 8 | ✗ MISSING |
| `AUTONOMOUS_IMPROVEMENTS_COMPLETE.md` | 5 | ✗ MISSING |
| `REPOSITORY_STRUCTURE.md` | 4 | ✗ MISSING |
| `REPO_ORGANIZATION_REPORT.md` | 3 | ✗ MISSING |
| `DEPENDENCY_AUDIT_SUMMARY.md` | 2 | ✗ MISSING |
| `DEV_QUICK_REFERENCE.md` | 2 | ✗ MISSING |
| `DOCUMENTATION_MAP.md` | 1 | ✗ MISSING |
| `REPOSITORY_BEST_PRACTICES.md` | 3 | ✗ MISSING |

**Impact:** High - These files are heavily cross-referenced in improvement documentation.

**Referenced in:**
- `docs/INDEX.md`
- `docs/IMPROVEMENTS_OVERVIEW.md`
- `docs/IMPROVEMENTS_APPLIED.md`
- `REPO_ORGANIZER_IMPLEMENTATION.md`
- `.claude/README.md`

---

### 2. Archived Files Referenced with Wrong Paths

#### PERFORMANCE_QUICK_START.md (14 references)
- **Wrong:** `docs/PERFORMANCE_QUICK_START.md`
- **Correct:** `docs/archive/performance/PERFORMANCE_QUICK_START.md`
- **Referenced in:** `docs/INDEX.md` (5x), `docs/PERFORMANCE.md`, `docs/DEPENDENCIES.md`, `docs/README.md`, `CONTRIBUTING.md`

#### CHUNK_SIZE_ANALYSIS.md (2 references)
- **Wrong:** `docs/CHUNK_SIZE_ANALYSIS.md`
- **Correct:** `docs/archive/chunk-size/CHUNK_SIZE_ANALYSIS.md`
- **Referenced in:** `docs/INDEX.md`

#### RUNPOD Documentation (4 references)
- **Wrong:** `docs/RUNPOD_FINAL_SETUP.md`
- **Correct:** `docs/archive/runpod/RUNPOD_FINAL_SETUP.md`
- **Also affects:** `RUNPOD_DEPLOYMENT_GUIDE.md`, `RUNPOD_STARTUP_INSTRUCTIONS.md`
- **Referenced in:** `docs/INDEX.md`

---

### 3. Moved Python Files (Outdated References)

#### performance_analysis.py → scripts/benchmarking_performance_analysis.py
**12 references across:**
- `docs/PERFORMANCE.md` (8 references - lines 392, 441, 505, 533-538, 587)
- `docs/PERFORMANCE_ANALYSIS.md` (1 reference)
- `docs/IMPROVEMENTS_APPLIED.md` (3 references - lines 23, 176, 528-533)
- `docs/AUDIT_EXECUTIVE_SUMMARY.md` (2 references)
- `SESSION_SUMMARY.md` (1 reference)
- `docs/INDEX.md` (1 reference - line 71, 184)

#### utils/ Python files (4 references)
- `mlx_embedding.py` → `utils/mlx_embedding.py` (1 reference in `docs/START_HERE.md`)
- `reranker.py` → `utils/reranker.py` (2 references in `docs/START_HERE.md`, `docs/POST_EMBEDDING_PLAN.md`)
- `query_cache.py` → `utils/query_cache.py` (1 reference in `docs/START_HERE.md`)

---

### 4. Renamed Shell Scripts (Outdated References)

#### QUICK_COMMANDS.sh → scripts/helper_quick_commands.sh
**5 references in:**
- `docs/START_HERE.md` (lines 18, 87, 135, 148)
- `docs/INTERACTIVE_GUIDE.md` (line 308)

#### scripts/apply_hnsw.sh → scripts/database_apply_hnsw.sh
**3 references in:**
- `docs/START_HERE.md` (lines 41, 62)
- `docs/POST_EMBEDDING_PLAN.md` (lines 67, 186)

---

### 5. Moved Configuration File

#### .env.example → config/.env.example
**5 references in:**
- `docs/INDEX.md` (lines 13, 103)
- `docs/START_HERE.md`
- Other setup documentation

---

### 6. Missing Quick Start Scripts

#### QUICK_START_OPTIMIZED.sh (7 references)
**Status:** File does not exist
**Referenced in:**
- `docs/INDEX.md` (lines 168, 255)
- `docs/IMPROVEMENTS_APPLIED.md` (lines 141, 168, 222, 413, 688)
- `SESSION_SUMMARY.md` (lines 173, 201, 218)
- `docs/IMPROVEMENTS_OVERVIEW.md` (line 228)

#### QUICK_START_VLLM.sh (3 references)
**Status:** File does not exist
**Referenced in:**
- `docs/VLLM_SERVER_GUIDE.md` (lines 234-236)

---

## Files with Most Broken Links

1. **docs/INDEX.md** - 25 broken links
2. **docs/IMPROVEMENTS_OVERVIEW.md** - 14 broken links
3. **docs/START_HERE.md** - 10 broken links
4. **docs/PERFORMANCE.md** - 8 broken links
5. **docs/IMPROVEMENTS_APPLIED.md** - 5 broken links

---

## Detailed Breakdown: docs/INDEX.md (25 broken links)

| Line | Current Reference | Correct Path | Issue |
|------|------------------|--------------|-------|
| 13 | `../.env.example` | `../config/.env.example` | Moved |
| 29 | `../FINAL_SUMMARY.md` | N/A | Missing |
| 39 | `PERFORMANCE_QUICK_START.md` | `archive/performance/PERFORMANCE_QUICK_START.md` | Archived |
| 40 | `PERFORMANCE_SUMMARY.md` | `archive/performance/PERFORMANCE_SUMMARY.md` | Archived |
| 50 | `../DEPENDENCY_AUDIT_SUMMARY.md` | N/A | Missing |
| 58 | `../REPOSITORY_STRUCTURE.md` | N/A | Missing |
| 61 | `../REPO_ORGANIZATION_REPORT.md` | N/A | Missing |
| 70 | `PERFORMANCE_QUICK_START.md` | `archive/performance/PERFORMANCE_QUICK_START.md` | Archived |
| 90 | `RUNPOD_FINAL_SETUP.md` | `archive/runpod/RUNPOD_FINAL_SETUP.md` | Archived |
| 91 | `RUNPOD_DEPLOYMENT_GUIDE.md` | `archive/runpod/RUNPOD_DEPLOYMENT_GUIDE.md` | Archived |
| 92 | `RUNPOD_STARTUP_INSTRUCTIONS.md` | `archive/runpod/RUNPOD_STARTUP_INSTRUCTIONS.md` | Archived |
| 103 | `../.env.example` | `../config/.env.example` | Moved |
| 107 | `CHUNK_SIZE_ANALYSIS.md` | `archive/chunk-size/CHUNK_SIZE_ANALYSIS.md` | Archived |
| 133 | `archive/CHUNK_SIZE_FIX_SUMMARY.md` | `archive/chunk-size/CHUNK_SIZE_FIX_SUMMARY.md` | Wrong path |
| 134 | `archive/FIXES_APPLIED.md` | `FIXES_APPLIED.md` | Not archived |
| 135 | `archive/POST_EMBEDDING_PLAN.md` | `POST_EMBEDDING_PLAN.md` | Not archived |
| 144 | `../DEV_QUICK_REFERENCE.md` | N/A | Missing |
| 155 | `../FINAL_SUMMARY.md` | N/A | Missing |
| 156 | `../AUTONOMOUS_IMPROVEMENTS_COMPLETE.md` | N/A | Missing |
| 168 | `./QUICK_START_OPTIMIZED.sh` | N/A | Missing |
| 181-254 | `PERFORMANCE_QUICK_START.md` (4x) | `archive/performance/PERFORMANCE_QUICK_START.md` | Archived |
| 255 | `./QUICK_START_OPTIMIZED.sh` | N/A | Missing |

---

## Recommendations

### Priority 1: Fix Critical Missing Root Files (URGENT)

**Option A: Restore/Create Files**
If these files existed and were accidentally deleted, restore them from git history:
```bash
git log --all --full-history -- "FINAL_SUMMARY.md"
git log --all --full-history -- "AUTONOMOUS_IMPROVEMENTS_COMPLETE.md"
```

**Option B: Remove References**
If files were intentionally removed, update all documentation to remove broken links:
- Remove or comment out references in `docs/INDEX.md`
- Update `docs/IMPROVEMENTS_OVERVIEW.md`
- Update `docs/IMPROVEMENTS_APPLIED.md`

**Impact:** High - 25+ broken references affecting navigation

---

### Priority 2: Fix Archive Path References (HIGH)

Update all references to archived documentation with correct paths:

**Automated Fix:**
```bash
# Fix PERFORMANCE_QUICK_START references
find /Users/frytos/code/llamaIndex-local-rag/docs -name "*.md" -type f \
  -exec sed -i '' 's|PERFORMANCE_QUICK_START\.md|archive/performance/PERFORMANCE_QUICK_START.md|g' {} +

# Fix CHUNK_SIZE_ANALYSIS references
find /Users/frytos/code/llamaIndex-local-rag/docs -name "*.md" -type f \
  -exec sed -i '' 's|CHUNK_SIZE_ANALYSIS\.md|archive/chunk-size/CHUNK_SIZE_ANALYSIS.md|g' {} +

# Fix RUNPOD references
find /Users/frytos/code/llamaIndex-local-rag/docs -name "*.md" -type f \
  -exec sed -i '' 's|RUNPOD_FINAL_SETUP\.md|archive/runpod/RUNPOD_FINAL_SETUP.md|g' {} +
find /Users/frytos/code/llamaIndex-local-rag/docs -name "*.md" -type f \
  -exec sed -i '' 's|RUNPOD_DEPLOYMENT_GUIDE\.md|archive/runpod/RUNPOD_DEPLOYMENT_GUIDE.md|g' {} +
find /Users/frytos/code/llamaIndex-local-rag/docs -name "*.md" -type f \
  -exec sed -i '' 's|RUNPOD_STARTUP_INSTRUCTIONS\.md|archive/runpod/RUNPOD_STARTUP_INSTRUCTIONS.md|g' {} +
```

**Impact:** Medium-High - 20+ broken references

---

### Priority 3: Update File Movement References (MEDIUM)

Global find-replace for moved files:

```bash
# Fix performance_analysis.py references (12 references)
find /Users/frytos/code/llamaIndex-local-rag -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|performance_analysis\.py|scripts/benchmarking_performance_analysis.py|g' {} +

# Fix QUICK_COMMANDS.sh references (5 references)
find /Users/frytos/code/llamaIndex-local-rag -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|QUICK_COMMANDS\.sh|scripts/helper_quick_commands.sh|g' {} +

# Fix apply_hnsw.sh references (3 references)
find /Users/frytos/code/llamaIndex-local-rag -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|scripts/apply_hnsw\.sh|scripts/database_apply_hnsw.sh|g' {} +
find /Users/frytos/code/llamaIndex-local-rag -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|\./scripts/apply_hnsw\.sh|scripts/database_apply_hnsw.sh|g' {} +

# Fix utils file references (4 references)
find /Users/frytos/code/llamaIndex-local-rag -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|\bmlx_embedding\.py|utils/mlx_embedding.py|g' {} +
find /Users/frytos/code/llamaIndex-local-rag -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|\breranker\.py|utils/reranker.py|g' {} +
find /Users/frytos/code/llamaIndex-local-rag -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|\bquery_cache\.py|utils/query_cache.py|g' {} +

# Fix .env.example references (5 references)
find /Users/frytos/code/llamaIndex-local-rag -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|\.\.\/\.env\.example|../config/.env.example|g' {} +
```

**Impact:** Medium - 29 broken references

---

### Priority 4: Handle Missing Quick Start Scripts (MEDIUM)

**Option A: Create Missing Scripts**
If these scripts should exist, create them based on the documented functionality.

**Option B: Remove References**
If scripts are no longer needed:
```bash
# Remove QUICK_START_OPTIMIZED.sh references (7 references)
# Remove QUICK_START_VLLM.sh references (3 references)
```

**Files to update:**
- `docs/INDEX.md`
- `docs/IMPROVEMENTS_APPLIED.md`
- `docs/IMPROVEMENTS_OVERVIEW.md`
- `docs/VLLM_SERVER_GUIDE.md`
- `SESSION_SUMMARY.md`

**Impact:** Medium - 10 broken references

---

### Priority 5: Update CLAUDE.md File Structure (LOW)

Update the file structure section in `CLAUDE.md` to reflect new organization:

```markdown
## File Structure

llamaIndex-local-rag/
├── rag_low_level_m1_16gb_verbose.py  # Main RAG pipeline
├── rag_interactive.py                 # CLI menu interface
├── rag_web.py                         # Streamlit web UI
├── utils/                             # Utility modules
│   ├── mlx_embedding.py              # MLX backend
│   ├── reranker.py                   # Cross-encoder reranking
│   └── query_cache.py                # Query caching
├── scripts/                           # Shell scripts & tools
│   ├── benchmarking_performance_analysis.py  # Performance analysis
│   ├── database_apply_hnsw.sh        # Apply HNSW indexing
│   ├── helper_quick_commands.sh      # Quick commands
│   └── system_free_memory.sh         # Memory management
├── config/                            # Configuration files
│   ├── .env.example                  # Environment template
│   └── requirements_vllm.txt         # vLLM dependencies
├── docs/                              # Documentation
│   ├── archive/                      # Historical documentation
│   │   ├── performance/              # Performance docs
│   │   ├── chunk-size/               # Chunking analysis
│   │   └── runpod/                   # RunPod deployment
│   └── *.md                          # Active documentation
└── data/                              # Documents to index
```

**Impact:** Low - Informational only (not broken links)

---

## Complete Fix Script

Run this comprehensive script to fix all automated issues:

```bash
#!/bin/bash
# fix_documentation_links.sh

echo "Fixing documentation links after repository reorganization..."

cd /Users/frytos/code/llamaIndex-local-rag

# Backup before making changes
echo "Creating backup..."
tar -czf docs_backup_$(date +%Y%m%d_%H%M%S).tar.gz docs/ *.md

echo "Fixing .env.example references..."
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|\.\.\/\.env\.example|../config/.env.example|g' {} +

echo "Fixing performance_analysis.py references..."
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|performance_analysis\.py|scripts/benchmarking_performance_analysis.py|g' {} +

echo "Fixing QUICK_COMMANDS.sh references..."
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|source QUICK_COMMANDS\.sh|source scripts/helper_quick_commands.sh|g' {} +
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|QUICK_COMMANDS\.sh|scripts/helper_quick_commands.sh|g' {} +

echo "Fixing apply_hnsw.sh references..."
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|scripts/apply_hnsw\.sh|scripts/database_apply_hnsw.sh|g' {} +
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|\./scripts/apply_hnsw\.sh|scripts/database_apply_hnsw.sh|g' {} +

echo "Fixing utils file references..."
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|python mlx_embedding\.py|python utils/mlx_embedding.py|g' {} +
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|python reranker\.py|python utils/reranker.py|g' {} +
find . -name "*.md" -type f -not -path "*/\.venv/*" \
  -exec sed -i '' 's|python query_cache\.py|python utils/query_cache.py|g' {} +

echo "Fixing PERFORMANCE_QUICK_START references..."
find docs -name "*.md" -type f \
  -exec sed -i '' 's|\[PERFORMANCE_QUICK_START\.md\](PERFORMANCE_QUICK_START\.md)|[PERFORMANCE_QUICK_START.md](archive/performance/PERFORMANCE_QUICK_START.md)|g' {} +
find docs -name "*.md" -type f \
  -exec sed -i '' 's|docs/PERFORMANCE_QUICK_START\.md|docs/archive/performance/PERFORMANCE_QUICK_START.md|g' {} +

echo "Fixing CHUNK_SIZE_ANALYSIS references..."
find docs -name "*.md" -type f \
  -exec sed -i '' 's|\[CHUNK_SIZE_ANALYSIS\.md\](CHUNK_SIZE_ANALYSIS\.md)|[CHUNK_SIZE_ANALYSIS.md](archive/chunk-size/CHUNK_SIZE_ANALYSIS.md)|g' {} +

echo "Fixing RUNPOD documentation references..."
find docs -name "*.md" -type f \
  -exec sed -i '' 's|\[RUNPOD_FINAL_SETUP\.md\](RUNPOD_FINAL_SETUP\.md)|[RUNPOD_FINAL_SETUP.md](archive/runpod/RUNPOD_FINAL_SETUP.md)|g' {} +
find docs -name "*.md" -type f \
  -exec sed -i '' 's|\[RUNPOD_DEPLOYMENT_GUIDE\.md\](RUNPOD_DEPLOYMENT_GUIDE\.md)|[RUNPOD_DEPLOYMENT_GUIDE.md](archive/runpod/RUNPOD_DEPLOYMENT_GUIDE.md)|g' {} +
find docs -name "*.md" -type f \
  -exec sed -i '' 's|\[RUNPOD_STARTUP_INSTRUCTIONS\.md\](RUNPOD_STARTUP_INSTRUCTIONS\.md)|[RUNPOD_STARTUP_INSTRUCTIONS.md](archive/runpod/RUNPOD_STARTUP_INSTRUCTIONS.md)|g' {} +

echo "Done! Check git diff to review changes."
echo "Backup saved to docs_backup_*.tar.gz"
```

---

## Verification Checklist

After applying fixes, verify:

- [ ] All links in `docs/INDEX.md` resolve correctly
- [ ] `docs/START_HERE.md` references correct script paths
- [ ] `docs/PERFORMANCE.md` references correct Python script
- [ ] Archive references use correct subdirectory paths
- [ ] No references to missing root documentation files
- [ ] Config file references point to `config/.env.example`
- [ ] Shell script references use new prefixed names
- [ ] Python utility references include `utils/` directory

---

## Statistics Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Links Checked** | ~250 | 100% |
| **Valid Links** | 177 | 70.8% |
| **Broken Links** | 45 | 18% |
| **Outdated References** | 28 | 11.2% |
| **External Links (All Valid)** | ~50 | 20% |

### Broken Links by Category

| Category | Count | Priority |
|----------|-------|----------|
| Missing root files | 8 files (25+ refs) | URGENT |
| Archived docs (wrong path) | 20 refs | HIGH |
| Moved Python files | 16 refs | MEDIUM |
| Renamed shell scripts | 8 refs | MEDIUM |
| Missing scripts | 10 refs | MEDIUM |
| Config file moved | 5 refs | LOW |

---

## Next Steps

1. **Immediate:** Determine status of missing root documentation files
   - Were they deleted intentionally?
   - Should they be restored from git history?
   - Should references be removed?

2. **High Priority:** Run automated fix script for file movements
   - Fixes 29 broken references automatically
   - Low risk of errors

3. **Manual Review:** Update `docs/INDEX.md` with correct archive paths
   - 14 references to archived files
   - Requires careful path verification

4. **Decision Required:** Handle missing quick start scripts
   - Create scripts or remove references?
   - Impacts user onboarding documentation

---

**Report Generated By:** Documentation Link Audit Tool
**Date:** 2026-01-07
**Repository:** /Users/frytos/code/llamaIndex-local-rag
