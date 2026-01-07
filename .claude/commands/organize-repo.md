---
description: Quick repository organization - analyze structure and apply best practices
---

# Organize Repository Command

Analyze repository structure and apply industry best practices for organization, documentation, and automation.

## Usage

```bash
/organize-repo [--mode=audit|quick-fix|full] [--lang=python|javascript|go|rust]
```

## Modes

### Audit Mode (Default)
Analyze repository and generate report without making changes.

**Output**: Organization assessment with recommendations

### Quick Fix Mode
Apply low-risk improvements immediately:
- Add missing .gitignore patterns
- Create empty standard directories
- Generate template documentation files
- Create basic Makefile

**Output**: Summary of changes made

### Full Mode
Complete reorganization with file migrations:
- Create full directory structure
- Migrate files to proper locations
- Generate all essential documentation
- Set up build automation
- Update imports and references

**Output**: Detailed migration report

## What This Command Does

### 1. Structure Analysis
- Scans repository with `tree` and `git ls-files`
- Identifies root clutter (target: <15 files in root)
- Checks for essential files (README, LICENSE, etc.)
- Detects language/framework from config files
- Maps current organization against best practices

### 2. Issue Detection

**Critical Issues** (Block OSS release):
- Missing LICENSE file
- Secrets committed to repository (.env, api_keys)
- Large binaries in git history
- No README.md

**High Priority** (Impact productivity):
- Missing pyproject.toml (Python) or equivalent
- No build automation (Makefile, package.json scripts)
- Poor .gitignore (common patterns missing)
- Documentation scattered in root

**Medium Priority** (Technical debt):
- Inconsistent naming conventions
- Missing CONTRIBUTING.md
- No dependency documentation
- Outdated configuration patterns

### 3. Best Practice Application

Applies standards from:
- [GitHub Repository Best Practices](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories)
- Language-specific conventions (PEP 8, JavaScript Standard, Go standards)
- Modern tooling (pyproject.toml, GitHub Actions)
- Security best practices (secret scanning, .gitignore)

### 4. Documentation Generation

Creates missing essential files:

**README.md**:
```markdown
# Project Name

Brief description

## Quick Start
```bash
# 3-5 commands
```

## Documentation
- [Installation](docs/installation.md)
- [Configuration](docs/configuration.md)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md)

## License
[LICENSE](LICENSE)
```

**CONTRIBUTING.md**:
- Development setup instructions
- Code standards
- PR process
- Testing requirements

**pyproject.toml** (Python):
- Modern project metadata (PEP 518)
- Dependency declarations
- Tool configurations (ruff, pytest, mypy)
- Build system specification

**Makefile**:
- help, install, test, lint, format, clean targets
- Self-documenting with `make help`

### 5. Directory Organization

Creates standard structure:

```
project/
├── README.md              # Essential root files
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml         # or package.json, go.mod, Cargo.toml
├── Makefile
├── .gitignore
├── src/ or lib/           # Source code
├── tests/                 # Test suite
├── docs/                  # Documentation
├── config/                # Configuration files
├── scripts/               # Automation scripts
├── examples/              # Usage examples (optional)
├── benchmarks/            # Performance tests (optional)
├── archive/               # Deprecated code (optional)
└── .github/               # GitHub-specific files
    ├── workflows/         # CI/CD
    └── ISSUE_TEMPLATE/
```

## Example Usage

### Quick Audit
```bash
/organize-repo
```
**Output**: Analysis report with recommendations, no files changed

### Apply Quick Fixes
```bash
/organize-repo --mode=quick-fix
```
**Output**:
- Enhanced .gitignore
- Created empty dirs (docs/, config/, scripts/, tests/)
- Added template README if missing
- Created basic Makefile
- Files remain in place

### Full Reorganization
```bash
/organize-repo --mode=full --lang=python
```
**Output**:
- Complete directory structure
- Files migrated to proper locations
- All essential docs generated
- Build automation configured
- Imports updated
- Git commits created incrementally

## Safety Guarantees

### What This Command Will Do
✅ Create new directories
✅ Generate missing documentation
✅ Enhance .gitignore
✅ Create build automation (Makefile)
✅ Move files with `git mv` (preserves history)
✅ Make incremental commits

### What This Command Won't Do
❌ Delete files without confirmation
❌ Modify existing code logic
❌ Change git history (rebase/squash)
❌ Push to remote automatically
❌ Overwrite existing documentation

### Backup Strategy
Before full reorganization:
1. Creates backup branch: `backup/pre-organization`
2. Works on feature branch: `refactor/repository-organization`
3. Makes incremental commits (easy to revert)
4. Validates tests pass after each phase

## Language-Specific Behavior

### Python Projects
- Creates pyproject.toml (PEP 518 standard)
- Suggests src/ layout over flat layout
- Splits requirements.txt → requirements.txt + requirements-dev.txt
- Adds ruff, black, mypy configurations
- Creates pytest configuration

### JavaScript/TypeScript
- Validates package.json structure
- Suggests npm scripts for common tasks
- Creates tsconfig.json if missing
- Adds .eslintrc, .prettierrc
- Configures Jest for testing

### Go Projects
- Validates go.mod structure
- Creates cmd/, pkg/, internal/ structure
- Suggests Makefile for building
- Adds .golangci.yml for linting

### Rust Projects
- Validates Cargo.toml structure
- Creates src/bin/ for binaries
- Adds clippy, rustfmt configuration

## Integration with Existing Workflows

### Git Workflow
```bash
# Before running
git checkout -b refactor/organize-repo

# After command completes
git log --oneline  # Review incremental commits
git diff main      # Review all changes
make test          # Validate

# If satisfied
git push origin refactor/organize-repo
# Create PR for review
```

### CI/CD Integration
Generates GitHub Actions workflow:
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: make install
      - name: Run tests
        run: make test
      - name: Lint
        run: make lint
```

## Success Criteria

After running this command, your repository should:

✅ **Pass Professional Review**
- All essential files present
- Industry-standard structure
- Comprehensive documentation

✅ **Easy Onboarding**
- New developer productive in <1 hour
- Clear README with quick start
- CONTRIBUTING.md with guidelines

✅ **Automated Workflows**
- `make install` sets up environment
- `make test` runs test suite
- `make lint` checks code quality

✅ **Maintainable**
- Logical file organization
- Clear separation of concerns
- Consistent naming conventions

✅ **Production-Ready**
- Security best practices applied
- No secrets in repository
- Proper dependency management

## Example Report (Audit Mode)

```markdown
# Repository Organization Audit
Generated: 2026-01-07

## Overall Score: 6.5/10

### Strengths ✅
- README.md present and informative
- Active development (recent commits)
- Tests directory exists
- Good Python code quality

### Issues Found

#### Critical (P0) - 2 issues
1. **Missing LICENSE file**
   - Impact: Legal uncertainty for users
   - Fix: Add MIT or Apache-2.0 license
   - Effort: 2 minutes

2. **Secrets in repository**
   - Files: .env (committed)
   - Impact: Security risk if public
   - Fix: Remove from git, add to .gitignore
   - Effort: 5 minutes

#### High Priority (P1) - 5 issues
1. **Root directory clutter (38 files)**
   - Should be: <15 files
   - Fix: Organize into docs/, config/, scripts/
   - Effort: 30 minutes

2. **Missing pyproject.toml**
   - Current: Only setup.py
   - Fix: Migrate to modern standard
   - Effort: 15 minutes

3. **Incomplete .gitignore**
   - Missing: data/, logs/, results/ patterns
   - Fix: Enhance .gitignore
   - Effort: 5 minutes

4. **No CONTRIBUTING.md**
   - Impact: Unclear how to contribute
   - Fix: Add contribution guidelines
   - Effort: 20 minutes

5. **No build automation**
   - Current: Manual commands in README
   - Fix: Create Makefile
   - Effort: 30 minutes

#### Medium Priority (P2) - 8 issues
1. Mixed naming conventions (3 patterns detected)
2. Documentation scattered (15 files in root)
3. Configuration files mixed with code
4. Missing dependency documentation
5. No GitHub Actions CI
6. Missing SECURITY.md
7. No CODE_OF_CONDUCT.md
8. Outdated requirements.txt format

## Recommended Actions

### Immediate (< 30 min)
```bash
/organize-repo --mode=quick-fix
```
This will:
- Add comprehensive .gitignore
- Create LICENSE file
- Generate basic documentation templates
- Create standard directories

### Short-term (< 2 hours)
```bash
/organize-repo --mode=full
```
This will:
- Full directory reorganization
- Migrate all files to proper locations
- Generate complete documentation
- Set up build automation
- Update imports and tests

### Follow-up
1. Review and merge reorganization PR
2. Set up CI/CD with GitHub Actions
3. Configure branch protection on main
4. Enable Dependabot for dependency updates
```

## Expert Tips

### Start Small
- Run audit mode first to understand scope
- Apply quick-fix mode for immediate improvements
- Plan full reorganization for larger changes

### Communicate Changes
- Share audit report with team before reorganizing
- Use PR for review and discussion
- Update internal documentation links
- Notify team of new structure

### Maintain Organization
- Monthly: Review new files in root, move to proper directories
- Quarterly: Re-run audit to catch regressions
- Update documentation when structure changes
- Enforce organization in code reviews

### Customize for Your Project
The command provides templates, but you should:
- Adjust to match team conventions
- Add project-specific directories
- Customize documentation for your domain
- Adapt Makefile targets to your workflow

## Related Commands

- `/comprehensive-audit` - Full codebase audit (security, performance, quality)
- `/document-feature` - Generate feature-specific documentation
- `/review-pr` - Review PRs for organization compliance

## References

Based on industry best practices from:
- [GitHub Repository Best Practices](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories)
- [Python Project Structure Best Practices](https://docs.python-guide.org/writing/structure/)
- [Modern Python Development Practices](https://www.stuartellis.name/articles/python-modern-practices/)
- [Repository Cleanup Best Practices](https://www.codeac.io/blog/make-your-repository-lean-and-clean.html)
