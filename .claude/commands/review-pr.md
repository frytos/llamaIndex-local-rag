---
description: Review pull request with detailed analysis
---

# Pull Request Review

Perform comprehensive code review of pull request changes with detailed feedback on code quality, testing, performance, security, and best practices.

## Benefits
- **Comprehensive Analysis**: Reviews multiple aspects automatically
- **Consistent Standards**: Applies project patterns consistently
- **Fast Feedback**: Immediate review without waiting for human reviewer
- **Learning Tool**: Explains issues and suggests improvements
- **Catches Common Issues**: Security, performance, style problems

## When to Use
- Before requesting human review (pre-review check)
- After making changes to verify quality
- When learning codebase patterns
- For self-review before committing
- As first-pass review on team PRs

## Usage
```bash
/review-pr [PR-number]
/review-pr [branch-name]
/review-pr  # Reviews current branch vs main
```

## Review Areas

### 1. Code Quality
- **Readability**: Clear naming, appropriate complexity
- **Maintainability**: Code organization, modularity
- **Consistency**: Follows project patterns and style
- **Best Practices**: Adheres to Python/RAG conventions
- **Type Hints**: Proper typing for better IDE support
- **Duplication**: DRY principle violations

### 2. Testing
- **Coverage**: New code has tests
- **Quality**: Tests are meaningful and well-structured
- **Edge Cases**: Important scenarios covered
- **Test Names**: Descriptive and clear

### 3. Performance
- **Algorithms**: Efficient implementation
- **Memory**: No memory leaks or excessive allocation
- **Database**: Efficient queries, proper indexing
- **Embedding/LLM**: Optimal batch sizes and parameters

### 4. Security
- **Input Validation**: User input properly validated
- **SQL Injection**: Parameterized queries used
- **Secrets**: No hardcoded credentials
- **Dependencies**: No known vulnerabilities

### 5. Documentation
- **Docstrings**: Functions documented
- **Comments**: Complex logic explained
- **CLAUDE.md**: Updated if needed

### 6. Breaking Changes
- **API Changes**: Breaking API modifications identified
- **Configuration**: Config changes documented
- **Migration Path**: Upgrade path provided

## Review Output Format

```markdown
# PR Review: [PR Title]

## Summary
Brief overview of changes and overall assessment

**Overall Rating**: Approved | Needs Minor Changes | Needs Major Changes

**Key Strengths**: [Positive aspects]
**Main Concerns**: [Critical issues]

---

## Detailed Review

### What's Good
1. Well-documented new function with type hints
2. Clear separation of concerns
3. Proper error handling

### Issues Found

#### Critical (Must Fix)
1. **Security: Hardcoded API Key**
   - Location: `src/services/api.py:12`
   - Fix: Move to environment variable
   - Impact: Credential exposure if code leaked

2. **Bug: Race Condition**
   - Location: `src/utils/async_helpers.py:23`
   - Issue: Async state update without cleanup
   - Fix: Add proper cleanup

#### Medium Priority (Should Fix)
1. **Performance: Inefficient Query**
   - Location: `src/db/queries.py:67`
   - Impact: Slow response with many rows
   - Suggestion: Add index or batch loading

2. **Code Quality: High Complexity**
   - Location: `src/utils/processing.py:45-89`
   - Cyclomatic complexity: 12 (threshold: 10)
   - Suggestion: Extract into smaller functions

#### Low Priority (Nice to Have)
1. **Style: Missing Type Hints**
   - Location: `src/utils/validators.py`
   - Suggestion: Add type annotations

### Metrics
- **Files Changed**: 12
- **Lines Added**: +456
- **Lines Removed**: -123

### Testing Assessment
- Unit tests present
- Missing edge case tests for error handling
- Suggestion: Add tests for invalid input

### Performance Analysis
- No obvious performance issues
- Embedding batch size could be optimized

### Security Check
- No hardcoded secrets found
- Input validation present
- Dependencies up to date

---

## Required Changes

Before merging, please address:

1. **Fix race condition** (src/utils/async_helpers.py:23)
2. **Add missing edge case tests**

## Recommended Improvements

Consider for follow-up:

1. Optimize query performance
2. Refactor high-complexity function
3. Add type hints to utility functions

---

## Checklist

**Author**:
- [ ] All critical issues fixed
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Ready for human review

**Reviewer**:
- [ ] Code follows project standards
- [ ] Tests are adequate
- [ ] No security concerns
- [ ] Performance acceptable
- [ ] Ready to merge

---

## Next Steps

1. Address critical issues
2. Add missing tests
3. Request human review

**Estimated time to address**: 1-2 hours
```

## Review Checklist

The review covers this comprehensive checklist:

### Code Quality
- [ ] Functions are appropriately sized (<50 lines)
- [ ] Files are manageable (<500 lines)
- [ ] Naming is clear and consistent
- [ ] No magic numbers or strings
- [ ] Appropriate use of constants
- [ ] Error handling present
- [ ] No commented-out code
- [ ] Imports organized properly

### Testing
- [ ] New code has tests
- [ ] Tests are meaningful
- [ ] Edge cases covered
- [ ] Error scenarios tested

### Performance
- [ ] No obvious O(n^2) algorithms
- [ ] Async operations handled properly
- [ ] Database queries optimized
- [ ] Proper batching for embeddings/LLM

### Security
- [ ] Input validation present
- [ ] No SQL injection risks
- [ ] Secrets not hardcoded
- [ ] Dependencies not vulnerable

### Documentation
- [ ] Docstrings present
- [ ] Complex logic commented
- [ ] CLAUDE.md updated if needed

## Best Practices

### Before Review
1. Ensure your branch is up to date with main
2. Run all tests locally
3. Run linter
4. Self-review your changes first

### During Review
1. Read through Claude's feedback carefully
2. Understand the reasoning behind suggestions
3. Ask questions if feedback is unclear
4. Prioritize critical issues first

### After Review
1. Address critical issues immediately
2. Plan follow-up for medium priority items
3. Consider low priority suggestions for quality
4. Run tests after making changes

## Integration with Development Workflow

```bash
# 1. Make changes on feature branch
git checkout -b feat/new-feature

# 2. Commit changes
git commit -m "feat: add new feature"

# 3. Self-review before pushing
/review-pr

# 4. Address issues found
[fix issues]

# 5. Review again
/review-pr

# 6. Push and create PR when clean
git push origin feat/new-feature

# 7. Request human review
```

## Tips for Authors

1. **Use early**: Run review before requesting human review
2. **Iterate**: Fix issues and review again
3. **Learn**: Pay attention to repeated feedback
4. **Question**: If you disagree, understand why first
5. **Document**: Add comments for non-obvious decisions

## Limitations

Claude's review cannot:
- Understand full business context
- Test the feature manually
- Judge UX decisions
- Evaluate architectural fit
- Replace human code review

Always follow up with human review for:
- Business logic correctness
- Architecture decisions
- UX considerations
