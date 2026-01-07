---
description: Full codebase audit (security, performance, quality, tests)
---

# Comprehensive Codebase Audit

Launch 4 parallel sub-agents to perform a thorough audit of different aspects of your codebase.

## Benefits
- **Comprehensive Coverage**: All major areas audited simultaneously
- **3-4x Faster**: Parallel execution vs sequential audits
- **Actionable Insights**: Specific recommendations for improvements
- **Multiple Perspectives**: Security, performance, quality, and testing angles

## When to Use
- Before major releases
- During quarterly code reviews
- After significant feature additions
- When onboarding new team members
- Before refactoring initiatives
- After dependency updates

## Usage
Invoke with `/comprehensive-audit`

## Audit Areas

### 1. Security Audit

**Focus Areas**:
- Vulnerable dependencies (pip audit / safety check)
- Input validation and sanitization
- Authentication and authorization logic
- Secrets in code or config files
- SQL injection risks (especially for pgvector queries)
- Sensitive data exposure
- Database connection security

**Output**:
- List of security vulnerabilities with severity
- Vulnerable dependencies with CVE numbers
- Recommendations for fixes
- Priority ranking (critical, high, medium, low)

### 2. Performance Audit

**Focus Areas**:
- Embedding throughput bottlenecks
- Database query optimization
- Memory usage patterns
- GPU/MPS utilization
- Batch size optimization
- Vector index efficiency
- LLM inference speed

**Output**:
- Performance bottlenecks with measurements
- Memory usage analysis
- Optimization opportunities
- Quick wins vs long-term improvements

### 3. Code Quality Audit

**Focus Areas**:
- Code duplication (DRY violations)
- Complex functions (high cyclomatic complexity)
- Dead code / unused imports
- Inconsistent patterns
- Type hint coverage
- Long files (>500 lines)
- Deep nesting (>4 levels)
- Magic numbers and strings
- Missing error handling

**Output**:
- Code smells with file locations
- Refactoring candidates
- Consistency issues
- Technical debt estimates

### 4. Test Coverage Audit

**Focus Areas**:
- Missing tests for critical paths
- Low coverage areas
- Missing edge cases
- Integration test gaps
- RAG pipeline test coverage

**Output**:
- Coverage report by area
- Critical paths without tests
- Test quality issues
- Recommendations for test improvements

## Workflow

1. **Launch Sub-Agents**: Spawn 4 specialized agents simultaneously
2. **Parallel Analysis**: Each agent audits their assigned area
3. **Generate Reports**: Each agent produces detailed findings
4. **Aggregate Results**: Main agent combines all reports
5. **Prioritize**: Rank issues by severity and impact
6. **Recommend**: Provide actionable next steps

## Example Output

```markdown
# Comprehensive Audit Report
Generated: 2024-01-15 14:30 UTC

## Executive Summary
- 3 Critical Issues
- 12 Medium Priority Issues
- 8 Low Priority Issues
- 47 Areas Performing Well

---

## Security Audit Results

### Critical Issues (3)
1. **Hardcoded Database Password**
   - Location: rag_low_level_m1_16gb_verbose.py:255
   - Fix: Move to environment variable
   - Impact: Credential exposure if code leaked

2. **SQL Injection Risk in Custom Query**
   - Location: scripts/custom_query.py:45-52
   - Fix: Use parameterized queries
   - Impact: Database compromise

### Recommendations
- Run `pip audit` regularly
- Implement secret scanning in CI/CD
- Add input validation for user queries

---

## Performance Audit Results

### High Impact Issues (5)
1. **Low GPU Utilization**
   - Current N_GPU_LAYERS: 16
   - Recommended: 24-28 for M1 16GB
   - Expected improvement: +30% generation speed

2. **Small Embedding Batch Size**
   - Current: 16
   - Recommended: 64
   - Expected improvement: +40% embedding throughput

3. **Missing HNSW Indexes**
   - Tables affected: 20 of 21
   - Impact: O(n) vs O(log n) retrieval

### Recommendations
- Increase N_GPU_LAYERS to 24
- Increase EMBED_BATCH to 64
- Create HNSW indexes on frequently queried tables

---

## Code Quality Audit Results

### Refactoring Candidates (8)
1. **High Complexity: load_documents()**
   - Location: rag_low_level_m1_16gb_verbose.py:943-1128
   - Lines: 185
   - Fix: Extract format-specific loaders

2. **Code Duplication: DB Connection Logic**
   - Locations: 3 files
   - Duplicate logic: 60 lines
   - Fix: Extract to shared utility

### Recommendations
- Extract document loaders to separate module
- Add type hints to remaining functions
- Create shared database utilities

---

## Test Coverage Audit Results

### Coverage Summary
- Statements: 45% (target: 80%)
- Functions: 30% (target: 80%)

### Critical Gaps (4)
1. **No Tests: RAG Pipeline**
   - Location: rag_low_level_m1_16gb_verbose.py
   - Risk: High (core functionality)
   - Recommendation: Add integration tests

2. **No Tests: Web UI**
   - Location: rag_web.py
   - Risk: Medium

### Recommendations
- Add pytest tests for core functions
- Add integration tests for RAG pipeline
- Set up CI coverage threshold

---

## Priority Matrix

| Priority | Category | Issue | Effort |
|----------|----------|-------|--------|
| P0 | Security | Remove hardcoded password | Low |
| P0 | Performance | Increase GPU layers | Low |
| P1 | Performance | Create HNSW indexes | Medium |
| P1 | Quality | Extract document loaders | Medium |
| P2 | Testing | Add core tests | High |

---

## Recommended Action Plan

### Week 1: Critical Security & Quick Wins
1. Move secrets to environment variables
2. Increase N_GPU_LAYERS to 24
3. Increase EMBED_BATCH to 64

### Week 2: Performance
1. Create HNSW indexes for main tables
2. Optimize database queries

### Week 3: Code Quality
1. Extract document loaders
2. Add type hints

### Week 4: Testing
1. Add core pipeline tests
2. Set up CI with coverage

---

## Metrics to Track

- Security: 0 critical vulnerabilities
- Performance: >60 chunks/s embedding, >12 tok/s generation
- Quality: Cyclomatic complexity <10
- Testing: 80%+ coverage

Next audit recommended: 3 months
```

## Best Practices

1. **Run Regularly**: Quarterly or before major releases
2. **Track Progress**: Keep audit reports to measure improvements
3. **Prioritize**: Focus on critical issues first
4. **Automate**: Add checks to CI/CD where possible
5. **Team Review**: Discuss findings with the team
6. **Action Plan**: Don't just audit - plan improvements

## After the Audit

1. **Create Issues**: File GitHub issues for each finding
2. **Prioritize**: Label with P0/P1/P2 priorities
3. **Assign**: Distribute work across team
4. **Track**: Use project board to monitor progress
5. **Follow Up**: Re-audit in 3 months
