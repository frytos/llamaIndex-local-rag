# Claude Code Configuration

This directory contains custom agents and commands (skills) for Claude Code.

## Directory Structure

```
.claude/
├── agents/                 # Specialized agents for complex tasks
│   ├── rag-optimizer.md
│   ├── code-refactor.md
│   ├── code-refactor-maintainer.md
│   └── repo-organizer.md
├── commands/               # Quick command skills
│   ├── run-rag.md
│   ├── optimize-rag.md
│   ├── audit-index.md
│   ├── compare-chunks.md
│   ├── comprehensive-audit.md
│   ├── document-feature.md
│   ├── parallel-test.md
│   ├── review-pr.md
│   ├── search-and-refactor.md
│   ├── tdd-feature.md
│   ├── web-ui.md
│   └── organize-repo.md
└── settings.local.json     # Local Claude Code settings
```

## Agents

Agents are specialized assistants for complex, multi-step tasks.

### `repo-organizer`
**Purpose**: Transform chaotic repositories into professionally organized projects

**Use when**:
- Starting new projects
- Inheriting legacy code
- Preparing for open source release
- Onboarding is difficult due to poor organization

**Capabilities**:
- Analyzes repository structure
- Applies industry best practices
- Generates essential documentation
- Reorganizes files into standard directories
- Creates build automation
- Validates git hygiene

**Example**: Invoke via Task tool with `subagent_type='repo-organizer'`

### `rag-optimizer`
**Purpose**: Optimize RAG pipeline performance and quality

**Use when**:
- After indexing documents
- Retrieval quality seems poor
- Before production deployment
- Experimenting with parameters

**Capabilities**:
- Analyzes chunk configurations
- Evaluates embedding quality
- Optimizes retrieval performance
- Detects configuration issues

### `code-refactor`
**Purpose**: Improve code quality and maintainability

**Use when**:
- After multiple AI-assisted iterations
- Technical debt accumulated
- Before major releases
- Code is hard to understand

**Capabilities**:
- Consolidates duplicated code
- Applies DRY principles
- Adds type hints
- Improves organization

## Commands (Skills)

Commands are quick, focused operations invoked with `/command-name`.

### Repository & Organization

**`/organize-repo`** - Quick repository organization
- Modes: audit, quick-fix, full
- Analyzes structure and applies best practices
- Generates professional documentation

**`/comprehensive-audit`** - Full codebase audit
- Security, performance, quality, testing
- Parallel execution (4 agents)
- Actionable recommendations

### RAG-Specific

**`/run-rag`** - Run RAG pipeline with parameters
**`/optimize-rag`** - Analyze and optimize RAG configuration
**`/audit-index`** - Check index health and consistency
**`/compare-chunks`** - Compare different chunk configurations

### Development

**`/review-pr`** - Review pull request with analysis
**`/tdd-feature`** - Build feature using Test-Driven Development
**`/parallel-test`** - Run test suites in parallel
**`/document-feature`** - Generate feature documentation
**`/search-and-refactor`** - Find patterns and refactor systematically

### User Interface

**`/web-ui`** - Launch Streamlit web UI

## Usage Patterns

### Invoke Commands
```bash
# In Claude Code CLI
/organize-repo
/organize-repo --mode=quick-fix
/run-rag PGTABLE=myindex
/optimize-rag
```

### Invoke Agents
Agents are invoked through the Task tool in conversations:
```python
# Example: Use repo-organizer agent
Task(
    subagent_type='repo-organizer',
    description='Organize repository structure',
    prompt='Analyze and reorganize this repository following best practices'
)
```

## Creating New Skills

### Agent Template
```markdown
---
name: my-agent
description: |
  Brief description of what this agent does
model: sonnet  # or haiku for simpler tasks
color: blue    # purple, yellow, green, etc.
---

# Agent Name

Agent description and responsibilities...

## Core Responsibilities
1. Responsibility 1
2. Responsibility 2

## When to Use This Agent
- Scenario 1
- Scenario 2

## Workflow
### Phase 1: ANALYSIS
### Phase 2: EXECUTION
### Phase 3: VALIDATION
```

### Command Template
```markdown
---
description: Brief description of command
---

# Command Name

Command description...

## Usage
```bash
/command-name [options]
```

## What This Command Does
1. Step 1
2. Step 2

## Options
- `--option1`: Description
- `--option2`: Description

## Examples
```bash
/command-name --option1=value
```
```

## Best Practices

### When to Create an Agent vs Command

**Create an Agent when**:
- Task requires multiple phases
- Needs complex analysis
- Involves multiple domains
- Benefits from specialized expertise
- Requires iterative refinement

**Create a Command when**:
- Task is focused and specific
- Quick execution (< 5 minutes)
- Well-defined inputs/outputs
- Frequently repeated operation

### Naming Conventions

**Agents**: Noun describing role
- `rag-optimizer`, `code-refactor`, `repo-organizer`

**Commands**: Verb-noun pattern
- `run-rag`, `optimize-rag`, `organize-repo`, `review-pr`

### Documentation

Every skill should include:
- Clear description
- When to use it
- What it does
- Examples
- Expected outputs

## Contributing New Skills

1. **Identify Need**: What gap does this fill?
2. **Choose Type**: Agent (complex) or Command (simple)?
3. **Write Spec**: Document responsibilities and workflow
4. **Test**: Validate with real scenarios
5. **Document**: Add to this README
6. **Share**: Commit and push

## Related Documentation

- [REPOSITORY_BEST_PRACTICES.md](../REPOSITORY_BEST_PRACTICES.md) - General organization guide
- [REPO_ORGANIZATION_GUIDE.md](../docs/REPO_ORGANIZATION_GUIDE.md) - Quick reference
- [DEVELOPMENT.md](../DEVELOPMENT.md) - Development workflow
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
