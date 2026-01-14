# Shell Script Reference Audit Report

**Generated**: 2026-01-07
**Task**: Verify all shell script references use correct paths after renaming

---

## Executive Summary

### Status Overview
- ✅ **All renamed scripts exist** at correct locations in `/scripts/` directory
- ✅ **All old script names removed** from filesystem (no orphaned files)
- ✅ **All scripts are executable** with proper permissions (`-rwx--x--x`)
- ✅ **All scripts have proper shebang** lines (`#!/bin/bash` or `#!/usr/bin/env bash`)
- ❌ **Documentation not updated** - 17 files with 30+ broken references

### Critical Finding
While the physical renaming is complete and correct, **documentation lags behind**, creating a disconnect between what users read and what actually exists on disk.

---

## Renamed Scripts (Completed)

| Old Name | New Name | Status |
|----------|----------|--------|
| `QUICK_COMMANDS.sh` | `scripts/helper_quick_commands.sh` | ✅ Renamed |
| `QUICK_START_OPTIMIZED.sh` | `scripts/quick_start_optimized.sh` | ✅ Renamed |
| `QUICK_START_VLLM.sh` | `scripts/quick_start_vllm.sh` | ✅ Renamed |
| `scripts/apply_hnsw.sh` | `scripts/database_apply_hnsw.sh` | ✅ Renamed |
| `scripts/free_memory.sh` | `scripts/system_free_memory.sh` | ✅ Renamed |
| `scripts/monitor_query.sh` | `scripts/monitoring_query.sh` | ✅ Renamed |
| `scripts/optimized_config.sh` | `scripts/config_optimized.sh` | ✅ Renamed |
| `scripts/deploy_to_runpod.sh` | `scripts/deploy_runpod.sh` | ✅ Renamed |

---

## Broken References by File

### Critical User-Facing Files

#### 1. `/Users/frytos/code/llamaIndex-local-rag/docs/START_HERE.md` (7 references)
**Impact**: HIGH - Primary onboarding document

```
Old Reference                          | Should Be
---------------------------------------|----------------------------------------
source QUICK_COMMANDS.sh               | source scripts/helper_quick_commands.sh
QUICK_COMMANDS.sh - Interactive...     | helper_quick_commands.sh - Interactive...
source QUICK_COMMANDS.sh && run_all    | source scripts/helper_quick_commands.sh && run_all
source QUICK_COMMANDS.sh && show_usage | source scripts/helper_quick_commands.sh && show_usage
./scripts/apply_hnsw.sh inbox_clean    | ./scripts/database_apply_hnsw.sh inbox_clean
./scripts/apply_hnsw.sh inbox_mlx...   | ./scripts/database_apply_hnsw.sh inbox_mlx...
scripts/apply_hnsw.sh - HNSW...        | scripts/database_apply_hnsw.sh - HNSW...
```

#### 2. `/Users/frytos/code/llamaIndex-local-rag/docs/VLLM_SERVER_GUIDE.md` (3 references)
**Impact**: HIGH - Key feature documentation

```
Old Reference                                        | Should Be
-----------------------------------------------------|---------------------------------------------
./QUICK_START_VLLM.sh query when did I go to New York | ./scripts/quick_start_vllm.sh query when did I go to New York
./QUICK_START_VLLM.sh query restaurants parisiens     | ./scripts/quick_start_vllm.sh query restaurants parisiens
./QUICK_START_VLLM.sh query quels sont les sports     | ./scripts/quick_start_vllm.sh query quels sont les sports
```

#### 3. `/Users/frytos/code/llamaIndex-local-rag/scripts/helper_quick_commands.sh` (5 references)
**Impact**: MEDIUM - Self-referencing in usage text (lines 204, 210, 216)

```bash
# Current (incorrect):
echo "  source QUICK_COMMANDS.sh"
echo "  source QUICK_COMMANDS.sh && run_all"
echo "This script should be sourced:"
echo "  source QUICK_COMMANDS.sh"

# Should be:
echo "  source scripts/helper_quick_commands.sh"
echo "  source scripts/helper_quick_commands.sh && run_all"
echo "This script should be sourced:"
echo "  source scripts/helper_quick_commands.sh"
```

### Secondary Documentation Files

| File | Old Refs | Primary Issues |
|------|----------|----------------|
| `/Users/frytos/code/llamaIndex-local-rag/docs/IMPROVEMENTS_APPLIED.md` | 5 | `QUICK_START_OPTIMIZED.sh` references |
| `/Users/frytos/code/llamaIndex-local-rag/docs/POST_EMBEDDING_PLAN.md` | 2 | `apply_hnsw.sh` references |
| `/Users/frytos/code/llamaIndex-local-rag/SESSION_SUMMARY.md` | 3 | `QUICK_START_OPTIMIZED.sh` references |
| `/Users/frytos/code/llamaIndex-local-rag/docs/INDEX.md` | 2 | `QUICK_START_OPTIMIZED.sh` references |
| `/Users/frytos/code/llamaIndex-local-rag/docs/IMPROVEMENTS_OVERVIEW.md` | 1 | `QUICK_START_OPTIMIZED.sh` reference |
| `/Users/frytos/code/llamaIndex-local-rag/docs/INTERACTIVE_GUIDE.md` | 1 | `QUICK_COMMANDS.sh` reference |
| `/Users/frytos/code/llamaIndex-local-rag/data/MESSENGER_PREPROCESSING_SUMMARY.md` | 1 | `optimized_config.sh` reference |

---

## Correctly Updated Files ✅

These files already use the correct new names:

- `/Users/frytos/code/llamaIndex-local-rag/scripts/README.md` - Uses all new names
- `/Users/frytos/code/llamaIndex-local-rag/REPOSITORY_ORGANIZATION.md` - Documents the renaming
- `/Users/frytos/code/llamaIndex-local-rag/CLAUDE.md` - References `quick_start_optimized.sh` and `quick_start_vllm.sh` correctly

---

## Script Verification Results

### All Scripts Exist ✅
```bash
✅ scripts/helper_quick_commands.sh
✅ scripts/quick_start_optimized.sh
✅ scripts/quick_start_vllm.sh
✅ scripts/database_apply_hnsw.sh
✅ scripts/system_free_memory.sh
✅ scripts/monitoring_query.sh
✅ scripts/config_optimized.sh
✅ scripts/deploy_runpod.sh
✅ scripts/start_vllm_server.sh
```

### Old Names Correctly Removed ✅
```bash
✅ QUICK_COMMANDS.sh - correctly removed
✅ QUICK_START_OPTIMIZED.sh - correctly removed
✅ QUICK_START_VLLM.sh - correctly removed
✅ scripts/apply_hnsw.sh - correctly removed
✅ scripts/free_memory.sh - correctly removed
✅ scripts/optimized_config.sh - correctly removed
✅ scripts/deploy_to_runpod.sh - correctly removed
```

### All Scripts Executable ✅
```bash
-rwx--x--x  scripts/compare_both_models.sh
-rwx--x--x  scripts/config_optimized.sh
-rwx--x--x  scripts/database_apply_hnsw.sh
-rwx--x--x  scripts/deploy_runpod.sh
-rwx--x--x  scripts/helper_quick_commands.sh
-rwx--x--x  scripts/index_bge_small.sh
-rwx--x--x  scripts/index_multilingual_e5.sh
-rwx--x--x  scripts/monitoring_query.sh
-rwx--x--x  scripts/quick_start_optimized.sh
-rwx--x--x  scripts/quick_start_vllm.sh
-rwx--x--x  scripts/runpod_startup_verbose.sh
-rwx--x--x  scripts/runpod_startup.sh
-rwx--x--x  scripts/start_vllm_server.sh
-rwx--x--x  scripts/system_free_memory.sh
-rwx--x--x  scripts/test_query_quality.sh
-rwx--x--x  scripts/verify_runpod_setup.sh
-rwx--x--x  scripts/vllm_server_control.sh
```

---

## Automated Fix Commands

### Option 1: Update All Documentation (Recommended)

```bash
# Navigate to project root
cd /Users/frytos/code/llamaIndex-local-rag

# Update QUICK_COMMANDS.sh → helper_quick_commands.sh
find docs/ -name "*.md" -type f -exec sed -i '' 's/QUICK_COMMANDS\.sh/helper_quick_commands.sh/g' {} \;
find docs/ -name "*.md" -type f -exec sed -i '' 's/source QUICK_COMMANDS/source scripts\/helper_quick_commands/g' {} \;

# Update QUICK_START_OPTIMIZED.sh → quick_start_optimized.sh
find docs/ -name "*.md" -type f -exec sed -i '' 's/QUICK_START_OPTIMIZED\.sh/quick_start_optimized.sh/g' {} \;
find docs/ -name "*.md" -type f -exec sed -i '' 's/\.\/QUICK_START_OPTIMIZED/\.\/scripts\/quick_start_optimized/g' {} \;

# Update QUICK_START_VLLM.sh → quick_start_vllm.sh
find docs/ -name "*.md" -type f -exec sed -i '' 's/QUICK_START_VLLM\.sh/quick_start_vllm.sh/g' {} \;
find docs/ -name "*.md" -type f -exec sed -i '' 's/\.\/QUICK_START_VLLM/\.\/scripts\/quick_start_vllm/g' {} \;

# Update apply_hnsw.sh → database_apply_hnsw.sh
find docs/ -name "*.md" -type f -exec sed -i '' 's/apply_hnsw\.sh/database_apply_hnsw.sh/g' {} \;

# Update optimized_config.sh → config_optimized.sh
find . -name "*.md" -type f -exec sed -i '' 's/optimized_config\.sh/config_optimized.sh/g' {} \;
find . -name "*.md" -type f -exec sed -i '' 's/source optimized_config/source scripts\/config_optimized/g' {} \;

# Update self-references in helper_quick_commands.sh
sed -i '' 's/QUICK_COMMANDS\.sh/helper_quick_commands.sh/g' scripts/helper_quick_commands.sh
sed -i '' 's/source QUICK_COMMANDS/source scripts\/helper_quick_commands/g' scripts/helper_quick_commands.sh

# Verify no old references remain
echo "=== VERIFICATION ==="
grep -r "QUICK_COMMANDS\.sh\|QUICK_START_OPTIMIZED\.sh\|QUICK_START_VLLM\.sh\|apply_hnsw\.sh\|optimized_config\.sh" docs/ scripts/ SESSION_SUMMARY.md data/*.md 2>/dev/null | wc -l
echo "Should be 0 ^"
```

### Option 2: Manual Updates (Priority Files Only)

1. **docs/START_HERE.md** - Replace all 7 references
2. **docs/VLLM_SERVER_GUIDE.md** - Replace 3 command examples
3. **scripts/helper_quick_commands.sh** - Update usage text (lines 204, 210, 216)

---

## Recommendations

### Immediate Actions (Priority 1)
1. **Run automated fix commands** to update all documentation
2. **Verify fixes** with grep scan
3. **Test critical user flows**:
   - New user following `docs/START_HERE.md`
   - Developer using vLLM via `docs/VLLM_SERVER_GUIDE.md`
   - Helper commands via `scripts/helper_quick_commands.sh`

### Follow-up Actions (Priority 2)
4. **Update archive files** if still actively referenced
5. **Add redirect notices** in old locations (if symlinks desired)
6. **Update external documentation** (README, tutorials, etc.)

### Prevention Measures (Priority 3)
7. **Add CI check** to detect uppercase script names in docs
8. **Document naming convention** in `DEVELOPMENT.md` or `CLAUDE.md`
9. **Create test** to validate all documented scripts exist

---

## Impact Analysis

### User Experience Impact
- **New users**: Will encounter broken commands when following `START_HERE.md`
- **Existing users**: May have muscle memory for old script names (now gone)
- **Developers**: Self-referencing usage text in `helper_quick_commands.sh` is misleading

### Risk Level
- **Severity**: MEDIUM - Scripts work when called correctly, but documentation is outdated
- **Frequency**: HIGH - Affects all onboarding and key feature documentation
- **Detectability**: HIGH - Errors occur immediately when user tries documented commands

---

## Verification Checklist

- [ ] Run automated fix commands
- [ ] Verify with: `grep -r "QUICK_COMMANDS\|QUICK_START_OPTIMIZED\|QUICK_START_VLLM\|apply_hnsw\.sh\|optimized_config\.sh" docs/`
- [ ] Test actual script execution: `./scripts/quick_start_optimized.sh`
- [ ] Verify helper commands: `source scripts/helper_quick_commands.sh && show_usage`
- [ ] Update `SESSION_SUMMARY.md` if actively maintained
- [ ] Archive old references in `docs/archive/` if needed
- [ ] Add redirect notices or symlinks if backward compatibility required
- [ ] Update external tutorials or blog posts

---

## Appendix: Complete File List

### Files Needing Updates (17 total)
1. `/Users/frytos/code/llamaIndex-local-rag/docs/INTERACTIVE_GUIDE.md`
2. `/Users/frytos/code/llamaIndex-local-rag/docs/START_HERE.md`
3. `/Users/frytos/code/llamaIndex-local-rag/docs/IMPROVEMENTS_OVERVIEW.md`
4. `/Users/frytos/code/llamaIndex-local-rag/docs/INDEX.md`
5. `/Users/frytos/code/llamaIndex-local-rag/docs/IMPROVEMENTS_APPLIED.md`
6. `/Users/frytos/code/llamaIndex-local-rag/docs/VLLM_SERVER_GUIDE.md`
7. `/Users/frytos/code/llamaIndex-local-rag/docs/POST_EMBEDDING_PLAN.md`
8. `/Users/frytos/code/llamaIndex-local-rag/SESSION_SUMMARY.md`
9. `/Users/frytos/code/llamaIndex-local-rag/data/MESSENGER_PREPROCESSING_SUMMARY.md`
10. `/Users/frytos/code/llamaIndex-local-rag/scripts/helper_quick_commands.sh`
11. Additional references in `docs/archive/performance/` (multiple files)

### Total Reference Count
- **QUICK_COMMANDS.sh**: 6 references
- **QUICK_START_OPTIMIZED.sh**: 11 references
- **QUICK_START_VLLM.sh**: 3 references
- **apply_hnsw.sh**: 7 references
- **optimized_config.sh**: 1 reference
- **Total**: 30+ broken references across 17 files

---

**Report Generated**: 2026-01-07
**Audit Tool**: Claude Code Agent with systematic grep/find analysis
**Scope**: All `.md`, `.sh`, and `.py` files in repository
