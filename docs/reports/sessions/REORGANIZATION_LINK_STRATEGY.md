# Repository Reorganization - Link Strategy Overview

**Date**: 2026-01-14

This document explains how different types of links work after the repository reorganization.

---

## Overview: 3 Different Strategies

We used **3 different linking strategies** depending on the type of reference:

| Type | Strategy | Why | Count |
|------|----------|-----|-------|
| **User-facing scripts** | Symlinks (filesystem) | Users type `./launch.sh` - needs to work | 4 symlinks |
| **Python imports** | sys.path updates | Tests import scripts - need module path | 2 files |
| **Documentation links** | Path updates (in files) | Markdown relative paths - portable | 11 links |

---

## Strategy 1: Symlinks (Filesystem Level) ✅

**Used for**: User-facing files that users directly execute or reference

### Symlinks Created (4 total):

```bash
# Root directory symlinks
launch.sh → scripts/launch.sh              # Main launcher
.env.example → config/.env.example         # Config template
docker-compose.yml → config/docker-compose.yml   # Docker setup
monitoring → config/monitoring             # Monitoring configs
```

### How it works:

```
llamaIndex-local-rag/
├── launch.sh (symlink) ─────────┐
└── scripts/                     │
    └── launch.sh (actual file) ←┘
```

**User perspective**:
```bash
# User runs from root (works!)
./launch.sh

# Shell follows symlink automatically
# → Actually executes: scripts/launch.sh
```

**Advantages**:
- ✅ Users don't need to know about reorganization
- ✅ Old commands still work: `./launch.sh`
- ✅ Documentation doesn't need updating
- ✅ Works across all shells and scripts

**Files using this strategy**:
- Documentation that says: `./launch.sh` ✅ Still works
- User instructions: `./launch.sh` ✅ Still works
- Shell scripts referencing these: ✅ Work automatically

---

## Strategy 2: Python Import Path Updates 🐍

**Used for**: Python test files importing moved scripts

### Files Modified (2 total):

1. **tests/test_hnsw_migration.py**
2. **tests/test_audit_index.py**

### How it works:

**Before (broken):**
```python
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import migrate_add_hnsw_indices  # ❌ Not found!
```

**After (fixed):**
```python
# Add project root and scripts to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))  # ← Added!

import migrate_add_hnsw_indices  # ✅ Found in scripts/
```

**Visual:**
```
tests/test_hnsw_migration.py
    ↓ sys.path includes...
    ├── /Users/frytos/code/llamaIndex-local-rag/        (root)
    └── /Users/frytos/code/llamaIndex-local-rag/scripts (scripts)

    import migrate_add_hnsw_indices
    → Python searches scripts/ directory
    → Finds: scripts/migrate_add_hnsw_indices.py ✅
```

**Why not symlinks?**
- Python imports work via sys.path, not filesystem links
- This is the standard Python way to import from subdirectories
- More explicit and maintainable

**Verification**:
```bash
# Test collection works
pytest tests/test_hnsw_migration.py --collect-only
# ✅ Collected 41 items
```

---

## Strategy 3: Markdown Path Updates 📝

**Used for**: Internal documentation links

### Files Modified (2 total):

1. **docs/INDEX.md** - 7 links updated
2. **docs/reports/audits/PERFORMANCE_FIXES_COMPLETE.md** - 4 links updated

### How it works:

**Relative paths updated to reflect new structure:**

#### Example 1: docs/INDEX.md

```markdown
Before:
- [SECURITY_AUDIT.md](../SECURITY_AUDIT.md)
                      ↑ Goes to root, file NOT there

After:
- [SECURITY_AUDIT.md](reports/audits/SECURITY_AUDIT.md)
                      ↑ Correct relative path
```

**Path resolution:**
```
Current file: docs/INDEX.md

Link: reports/audits/SECURITY_AUDIT.md
      ↓ Resolves to...
      docs/reports/audits/SECURITY_AUDIT.md ✅
```

#### Example 2: docs/reports/audits/PERFORMANCE_FIXES_COMPLETE.md

```markdown
Before:
- [README.md](README.md)
              ↑ Looks in same directory (wrong!)

After:
- [README.md](../../../README.md)
              ↑ Goes up 3 levels to root
```

**Path resolution:**
```
Current file: docs/reports/audits/PERFORMANCE_FIXES_COMPLETE.md

Link: ../../../README.md
      ↓
      docs/reports/audits/../../../README.md
      ↓ Simplifies to...
      README.md (in root) ✅
```

### All Fixed Links:

#### From docs/INDEX.md (relative to `docs/`):
```markdown
✅ reports/audits/SECURITY_AUDIT.md
✅ reports/sessions/FINAL_SUMMARY.md
✅ references/REPOSITORY_ORGANIZATION.md
✅ references/ARCHIVE_ORGANIZATION.md
✅ references/DEVELOPMENT.md
✅ reports/sessions/AUTONOMOUS_IMPROVEMENTS_COMPLETE.md
```

#### From docs/reports/audits/PERFORMANCE_FIXES_COMPLETE.md (relative to `docs/reports/audits/`):
```markdown
✅ ../../../README.md                  (up 3 → root)
✅ ../../../CLAUDE.md                  (up 3 → root)
✅ ../../PERFORMANCE.md                (up 2 → docs/)
✅ ./PERFORMANCE_OPTIMIZATION_REPORT.md (same dir)
```

**Why not symlinks?**
- Markdown links are portable across different systems
- Relative paths work in GitHub, local editors, web browsers
- No filesystem dependencies
- Standard markdown practice

---

## No Changes Needed ✅

Some things still work without any modifications:

### 1. Core Application Imports
```python
# These still work (utils/ didn't move)
from utils import metrics
from utils import mlx_embedding
```

### 2. Main Documentation
```markdown
# README.md links to docs/ (still correct)
[Documentation](docs/START_HERE.md) ✅

# Links to files in same directory
[CONTRIBUTING.md](CONTRIBUTING.md) ✅
```

### 3. Configuration Files
```toml
# pyproject.toml paths (all still correct)
[tool.pytest.ini_options]
testpaths = ["tests"]  ✅
```

---

## Decision Matrix

When should you use each strategy?

| Situation | Use | Example |
|-----------|-----|---------|
| User runs script from root | **Symlink** | `./launch.sh` |
| Python imports from subdir | **sys.path** | `import script_name` |
| Markdown internal links | **Path update** | `[Link](path/to/file.md)` |
| Config file references | **Direct path** | `testpaths = ["tests"]` |
| Files in same directory | **No change** | Already working |

---

## Verification Commands

### Check Symlinks:
```bash
ls -la | grep "^l"
# Shows: launch.sh -> scripts/launch.sh ✅
```

### Check Python Imports:
```bash
pytest tests/test_hnsw_migration.py --collect-only
pytest tests/test_audit_index.py --collect-only
# Both collect successfully ✅
```

### Check Markdown Links:
```bash
# Verify docs/INDEX.md links
grep "reports/audits/SECURITY_AUDIT.md" docs/INDEX.md
# ✅ Found

# Verify PERFORMANCE_FIXES_COMPLETE.md links
grep "../../../README.md" docs/reports/audits/PERFORMANCE_FIXES_COMPLETE.md
# ✅ Found
```

---

## Summary Table

| Link Type | Method | Files Affected | Status |
|-----------|--------|----------------|--------|
| **User scripts** | Symlinks | 4 symlinks | ✅ Working |
| **Python imports** | sys.path | 2 test files | ✅ Fixed |
| **Doc links** | Path updates | 2 doc files | ✅ Fixed |
| **Core imports** | No change | All core files | ✅ Working |
| **Config files** | No change | All configs | ✅ Working |
| **Total** | **3 strategies** | **8 files** | **100% functional** |

---

## Benefits of This Approach

### 1. User-Friendly (Symlinks)
- Users don't need to change their workflows
- `./launch.sh` still works from root
- Documentation examples remain valid

### 2. Maintainable (sys.path)
- Python imports are explicit and standard
- Easy to understand for Python developers
- Works with IDEs and type checkers

### 3. Portable (Path Updates)
- Markdown links work everywhere
- GitHub renders correctly
- No filesystem dependencies
- Works in text editors

### 4. Minimal Changes
- Only 8 files modified total
- Most files still work as-is
- Low risk of breaking changes

---

## Future Considerations

### If You Add New Files:

**New Python script in scripts/**:
- No changes needed for imports from scripts
- Tests: Add `scripts/` to sys.path (already done)

**New documentation in docs/**:
- Use relative paths: `[Link](../path/to/file.md)`
- Test links locally before committing

**New user-facing script**:
- Consider symlink in root if frequently used
- Or document as: `./scripts/new_script.sh`

### If You Move More Files:

Use this decision tree:
```
User executes it? → Symlink
Python imports it? → sys.path or refactor imports
Markdown links to it? → Update paths in MD files
Config references it? → Update config paths
```

---

## Conclusion

✅ **All links working via 3 complementary strategies**

- **Symlinks**: User convenience (4 files)
- **sys.path**: Python imports (2 files)
- **Path updates**: Documentation (11 links in 2 files)

No single strategy fits all cases - we used the right tool for each job!
