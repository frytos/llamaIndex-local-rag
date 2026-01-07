# Repository Organization Guide

**Quick reference for maintaining professional repository structure**

## Quick Commands

```bash
# Audit repository organization
python scripts/validate_repo_organization.py

# Get markdown report
python scripts/validate_repo_organization.py --format=markdown --output=docs/AUDIT.md

# Claude Code commands
/organize-repo                    # Audit mode (analyze only)
/organize-repo --mode=quick-fix   # Apply safe improvements
/organize-repo --mode=full        # Complete reorganization

# Use the repo-organizer agent
# In Claude Code, invoke the Task tool with subagent_type='repo-organizer'
```

## When to Use What

### Daily Maintenance
```bash
# Check if new files need organizing
python scripts/validate_repo_organization.py

# If score < 8.0, fix issues:
/organize-repo --mode=quick-fix
```

### Starting New Project
```bash
# Full setup with best practices
/organize-repo --mode=full --lang=python
```

### Before Major Release
```bash
# Complete audit and fix all issues
python scripts/validate_repo_organization.py --format=markdown > AUDIT.md
/organize-repo --mode=full
```

### Inheriting Legacy Code
```bash
# Understand current state
/organize-repo  # Audit mode

# Plan reorganization
# Review report, then apply in phases
/organize-repo --mode=quick-fix    # Phase 1
# Manual code review and testing
/organize-repo --mode=full         # Phase 2
```

## Organization Checklist

### Essential Files (Required)
- [ ] `README.md` - Project overview with quick start
- [ ] `LICENSE` - Legal requirement for open source
- [ ] `.gitignore` - Prevent accidental commits
- [ ] `CONTRIBUTING.md` - Contribution guidelines
- [ ] `pyproject.toml` (Python) or equivalent for other languages

### Professional Files (Recommended)
- [ ] `CHANGELOG.md` - Version history
- [ ] `SECURITY.md` - Security policy and reporting
- [ ] `CODE_OF_CONDUCT.md` - Community guidelines
- [ ] `Makefile` - Build automation
- [ ] `.github/workflows/ci.yml` - CI/CD

### Directory Structure (Standard)
- [ ] `src/` or `lib/` - Source code
- [ ] `tests/` - Test suite
- [ ] `docs/` - Documentation (3+ guides)
- [ ] `config/` - Configuration files
- [ ] `scripts/` - Automation scripts
- [ ] `examples/` - Usage examples (if applicable)

### Git Hygiene (Critical)
- [ ] No secrets in repository (.env gitignored)
- [ ] No large binaries (use Git LFS if needed)
- [ ] Comprehensive .gitignore
- [ ] Clean commit history
- [ ] Proper branch strategy documented

## Directory Layout Reference

### Python Project (Recommended)
```
project/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml
├── Makefile
├── .gitignore
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── core/
│       ├── utils/
│       └── config.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   └── integration/
├── docs/
│   ├── README.md
│   ├── installation.md
│   ├── configuration.md
│   └── api-reference.md
├── config/
│   ├── .env.example
│   └── docker-compose.yml
├── scripts/
│   ├── setup_environment.sh
│   └── validate_repo_organization.py
└── .github/
    └── workflows/
        └── ci.yml
```

### Data Science Project
```
project/
├── README.md
├── notebooks/              # Jupyter notebooks
│   ├── 01-exploration.ipynb
│   └── 02-modeling.ipynb
├── src/
│   └── project/
│       ├── data/          # Data loading
│       ├── features/      # Feature engineering
│       └── models/        # Model code
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/                # Trained models
├── reports/              # Analysis outputs
└── references/           # Papers, manuals
```

### Web API Project
```
project/
├── README.md
├── src/
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── routes/
│       ├── models/
│       ├── services/
│       └── middleware/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── config/
│   ├── docker-compose.yml
│   └── .env.example
└── docs/
    └── api-reference.md
```

## File Naming Standards

### Source Code
```python
# Python
user_authentication.py     # snake_case
UserModel.py              # PascalCase for classes file
string_utils.py           # descriptive, not utils.py

# JavaScript
userAuthentication.js     # camelCase
UserModel.ts              # PascalCase for classes
stringUtils.js            # descriptive
```

### Documentation
```markdown
README.md                 # Always uppercase
CONTRIBUTING.md           # Uppercase for important docs
installation.md           # Lowercase for guides
api-reference.md          # Kebab-case for multi-word
```

### Scripts
```bash
# Use prefixes for grouping
setup_environment.sh      # setup_*
setup_database.sh
deploy_production.sh      # deploy_*
deploy_staging.sh
test_integration.sh       # test_*
db_migrate.sh            # db_*
system_monitor.sh        # system_*
```

### Configuration
```yaml
.env.example             # Template (committed)
.env                     # Actual (gitignored)
docker-compose.yml       # Lowercase with hyphens
pytest.ini              # Tool-specific naming
```

## Common Migration Patterns

### Pattern 1: Docs Scattered in Root
```bash
# Before
README.md, SETUP.md, API.md, GUIDE.md, HOWTO.md, ...

# After
README.md                    # Keep in root
docs/
  ├── installation.md        # SETUP.md → installation.md
  ├── api-reference.md       # API.md → api-reference.md
  ├── user-guide.md          # GUIDE.md → user-guide.md
  └── how-to/                # HOWTO.md → how-to/
```

### Pattern 2: Config Files Mixed
```bash
# Before
docker-compose.yml, .env.example, runpod_config.env, pytest.ini

# After
config/
  ├── docker-compose.yml
  ├── .env.example
  ├── runpod_config.env
  └── pytest.ini
docker-compose.yml → config/docker-compose.yml  # Symlink in root
```

### Pattern 3: Scripts Unorganized
```bash
# Before
helper1.py, helper2.py, deploy.sh, test.sh, monitor.py

# After
scripts/
  ├── setup_environment.sh    # helper1 renamed
  ├── deploy_production.sh    # deploy.sh renamed
  ├── test_integration.sh     # test.sh renamed
  └── system_monitor.py       # monitor.py renamed
```

### Pattern 4: Utils at Root
```bash
# Before (root directory)
utils.py, helpers.py, formatting.py

# After
src/project_name/utils/
  ├── __init__.py
  ├── string_utils.py     # utils.py split
  ├── date_utils.py
  └── formatting.py       # kept descriptive name
```

## Score Interpretation

| Score | Status | Action |
|-------|--------|--------|
| 9.0-10.0 | Excellent | Minor tweaks only |
| 8.0-8.9 | Good | Address medium-priority issues |
| 6.0-7.9 | Needs Work | Apply high-priority fixes |
| 4.0-5.9 | Poor | Major reorganization needed |
| 0.0-3.9 | Critical | Immediate action required |

## Quick Fixes (< 30 minutes)

### Add Missing LICENSE
```bash
# Choose appropriate license
# MIT (permissive), Apache-2.0 (permissive), GPL-3.0 (copyleft)

# MIT License template
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 Your Name

Permission is hereby granted, free of charge...
EOF
```

### Enhance .gitignore
```bash
# Append common patterns
cat >> .gitignore << 'EOF'

# Data
data/raw/*
data/processed/*
!data/README.md

# Logs
logs/
*.log

# Results
results/
output/
*.csv
!examples/*.csv

# Environment
.env
.env.local
*.env
!.env.example
EOF
```

### Create Standard Directories
```bash
mkdir -p {src,tests,docs,config,scripts,benchmarks,logs,archive}
touch src/__init__.py tests/__init__.py
echo "# Documentation" > docs/README.md
```

### Add Basic Makefile
```bash
cat > Makefile << 'EOF'
.PHONY: help install test

help:
	@echo "make install - Install dependencies"
	@echo "make test    - Run tests"

install:
	pip install -e .

test:
	pytest tests/ -v
EOF
```

## Maintenance Schedule

### Weekly
- Check for new files in root that should be organized
- Review and clean up temporary files
- Update .gitignore if new patterns emerge

### Monthly
- Run `python scripts/validate_repo_organization.py`
- Review and clean archive/ directory
- Update documentation for accuracy
- Check for outdated dependencies

### Quarterly
- Full organization audit with `/organize-repo`
- Review and update CONTRIBUTING.md
- Clean up stale branches
- Update copyright years in LICENSE

### Annually
- Review and update project goals in README
- Archive old documentation
- Major dependency updates
- Comprehensive security audit

## Troubleshooting

### Issue: "Tests fail after reorganization"
**Cause**: Imports not updated after moving files
**Fix**:
```python
# Update imports from:
from module import function

# To:
from src.project_name.module import function

# Or use relative imports within package
from .module import function
```

### Issue: "Makefile targets don't work"
**Cause**: Paths changed after reorganization
**Fix**: Update Makefile paths to new locations

### Issue: "CI failing after organization"
**Cause**: CI configuration has old paths
**Fix**: Update `.github/workflows/` with new paths

### Issue: "Import errors after creating src/"
**Cause**: Python path doesn't include src/
**Fix**: Either:
1. Use `pip install -e .` (recommended)
2. Add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:src"`

## Resources

### Online References
- [GitHub Repository Best Practices](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories)
- [Python Project Structure Guide](https://docs.python-guide.org/writing/structure/)
- [Modern Python Practices](https://www.stuartellis.name/articles/python-modern-practices/)
- [Repository Cleanup Guide](https://www.codeac.io/blog/make-your-repository-lean-and-clean.html)

### Project Documentation
- [REPOSITORY_BEST_PRACTICES.md](../REPOSITORY_BEST_PRACTICES.md) - Comprehensive guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [DEVELOPMENT.md](../DEVELOPMENT.md) - Development workflow

### Tools
- `git-filter-repo`: Remove large files from history
- `BFG Repo-Cleaner`: Fast cleanup of secrets
- `git-sizer`: Analyze repository size
- `ruff`: Python linting
- `black`: Python formatting

## FAQ

### Q: Will reorganization break my code?
**A**: Not if done properly. The agent:
- Uses `git mv` to preserve history
- Makes incremental commits
- Updates imports automatically
- Validates tests pass after each phase

### Q: How long does full reorganization take?
**A**: Depends on project size:
- Small project (<50 files): 15-30 minutes
- Medium project (50-200 files): 1-2 hours
- Large project (200+ files): 2-4 hours

### Q: Should I reorganize an active project?
**A**: Yes, but:
1. Create feature branch
2. Notify team
3. Apply during low-activity period
4. Test thoroughly before merging
5. Update team documentation

### Q: What if I don't like the suggested structure?
**A**: The structure is customizable:
1. Review audit report
2. Apply changes manually with guidance
3. Keep what works for your team
4. Document deviations in DEVELOPMENT.md

### Q: How do I prevent disorganization?
**A**:
1. Add to PR checklist: "Files in correct directories"
2. Run validation script in CI
3. Monthly organization reviews
4. Document organization rules in CONTRIBUTING.md

## Example: Full Reorganization Workflow

```bash
# 1. Audit current state
python scripts/validate_repo_organization.py > BEFORE_AUDIT.txt
# Review issues, score: 5.2/10

# 2. Create backup
git checkout -b backup/pre-organization
git push origin backup/pre-organization

# 3. Create work branch
git checkout -b refactor/organize-repository

# 4. Apply quick fixes first
/organize-repo --mode=quick-fix
# Adds: LICENSE, .gitignore improvements, empty directories

# 5. Test still works
make test  # or python -m pytest

# 6. Apply full reorganization
/organize-repo --mode=full

# 7. Validate
python scripts/validate_repo_organization.py
# New score: 9.1/10 ✅

make test  # Ensure tests pass

# 8. Review changes
git log --oneline -20
git diff backup/pre-organization

# 9. Create PR
git push origin refactor/organize-repository
gh pr create --title "Reorganize repository structure" \
  --body "$(cat docs/REPO_ORGANIZATION_REPORT.md)"

# 10. After merge, update team
# - Update documentation links
# - Notify team of new structure
# - Update onboarding guide
```

## Tips for Success

### Start Small
- Begin with audit mode to understand scope
- Apply quick-fix mode for immediate improvements
- Plan full reorganization for larger changes

### Communicate
- Share audit report with team before reorganizing
- Use PR for review and discussion
- Update internal documentation
- Provide migration guide for team

### Test Thoroughly
- Run full test suite after each phase
- Validate build process works
- Check CI/CD pipeline
- Ensure imports updated correctly

### Document Decisions
- Keep organization rationale in DEVELOPMENT.md
- Document any deviations from standards
- Update CONTRIBUTING.md with new structure
- Maintain CHANGELOG.md

### Maintain Discipline
- Review new files weekly
- Run validation monthly
- Enforce in code reviews
- Update onboarding materials

## Integration with Development Workflow

### Code Review Checklist
```markdown
- [ ] Files in correct directories (src/, tests/, docs/, config/, scripts/)
- [ ] Naming conventions followed
- [ ] Documentation updated if structure changed
- [ ] Tests in tests/ directory
- [ ] No new files in root unless justified
```

### CI/CD Integration
```yaml
# .github/workflows/organization-check.yml
name: Organization Check
on: [pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate organization
        run: |
          python scripts/validate_repo_organization.py
      - name: Fail if score < 8.0
        run: |
          # Exit code 1 if score < 8.0
```

### Pre-commit Hook (Optional)
```bash
# .git/hooks/pre-commit
#!/bin/bash
score=$(python scripts/validate_repo_organization.py --format=json | jq '.score')
if (( $(echo "$score < 7.0" | bc -l) )); then
  echo "⚠️  Repository organization score: $score/10"
  echo "Run 'python scripts/validate_repo_organization.py' for details"
  exit 1
fi
```

## Success Stories

### Before: Prototype Chaos
```
Root: 47 files
Structure: Flat
Docs: None
Build: Manual commands
Onboarding: 2+ hours
Score: 3.8/10
```

### After: Professional Project
```
Root: 11 files
Structure: Standard directories (src/, tests/, docs/, config/)
Docs: README + 8 guides
Build: make install && make test
Onboarding: <30 minutes
Score: 9.2/10
```

**Impact**:
- 73% reduction in root clutter
- Onboarding time: 2 hours → 30 minutes
- Developer satisfaction: Significantly improved
- OSS contributions: Enabled (was blocked)

## Next Steps

After organizing your repository:

1. **Set up CI/CD**: Use GitHub Actions with new structure
2. **Enable Dependabot**: Automate dependency updates
3. **Add Pre-commit Hooks**: Enforce formatting and linting
4. **Create Project Board**: Track improvements
5. **Document Patterns**: Update CLAUDE.md with learnings
6. **Celebrate**: Share improvements with team!

---

**Remember**: Organization is not a one-time task. Maintain discipline through code reviews, regular audits, and team culture.
