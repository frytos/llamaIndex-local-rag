---
description: Run test suites in parallel using sub-agents
---

# Parallel Test Execution

Launch multiple test sub-agents to run different test suites simultaneously for faster feedback.

## Benefits
- **3-4x faster** than sequential testing
- Independent test execution reduces bottlenecks
- Parallel feedback on multiple areas
- Efficient use of Claude's parallel processing

## When to Use
- Before committing large changes
- After major refactoring
- During CI/CD verification
- When time-sensitive feedback is needed
- Testing multiple independent modules

## Usage
Invoke with `/parallel-test`

## Workflow

1. **Identify Test Suites**: Analyze the project structure to identify independent test areas
2. **Launch Sub-Agents**: Spawn 3-4 sub-agents, each assigned to a different test suite
3. **Parallel Execution**: Each agent runs its assigned tests independently
4. **Aggregate Results**: Main agent collects and summarizes all results
5. **Report**: Present combined test results with any failures highlighted

## Example Distribution

For this RAG project:
- **Agent 1**: Test core RAG functions (load, chunk, embed)
- **Agent 2**: Test database operations (insert, query, retrieval)
- **Agent 3**: Test utilities and helpers
- **Agent 4**: Test web UI components

For a general Python project:
- **Agent 1**: Test `src/services/`
- **Agent 2**: Test `src/utils/`
- **Agent 3**: Test `src/models/`
- **Agent 4**: Test `src/api/`

## Performance Comparison

```
Sequential Testing:
  core_tests/       (2 min)
+ db_tests/         (1.5 min)
+ utils_tests/      (1 min)
+ web_tests/        (1.5 min)
= 6 minutes total

Parallel Testing:
  All four simultaneously = ~2 minutes (3x faster)
```

## Output Format

Each sub-agent reports:
- Number of tests passed
- Number of tests failed
- Execution time
- Details of any failures

Main agent provides:
- Summary of all test results
- Total test count and pass rate
- List of all failures with file locations
- Recommendations for fixing failures

## Best Practices

- Ensure test suites are truly independent (no shared state)
- Distribute tests evenly across agents (similar execution time)
- Use for projects with many tests for meaningful speedup
- Don't spawn too many agents (3-4 is optimal)

## Example Invocation

```
User: /parallel-test

Claude: I'll run your test suite in parallel. Let me analyze the project structure...

[Spawns 4 sub-agents]

Agent 1: Testing core functions... 45 tests passed in 1.8 min
Agent 2: Testing database ops... 23 tests passed in 1.2 min
Agent 3: Testing utilities... 2 failures, 28 passed in 2.1 min
Agent 4: Testing web UI... 15 tests passed in 1.5 min

Summary: 109 tests total, 107 passed, 2 failed in 2.1 minutes

Failures:
1. test_chunk_overlap_validation - AssertionError at line 45
2. test_empty_document_handling - TypeError at line 78

Recommendations:
- Fix overlap validation in chunk_documents()
- Add null check in load_documents()
```

## Test Categories

### Unit Tests
Fast, isolated tests for individual functions:
```python
def test_format_score():
    assert format_score(0.8567) == "0.86"
```

### Integration Tests
Tests that verify component interactions:
```python
def test_full_rag_pipeline():
    docs = load_documents("test.pdf")
    chunks = chunk_documents(docs)
    nodes = embed_nodes(model, chunks)
    assert len(nodes) > 0
```

### Database Tests
Tests requiring database connection:
```python
def test_insert_and_retrieve():
    insert_nodes(store, test_nodes)
    results = retriever.retrieve("test query")
    assert len(results) == TOP_K
```

## Handling Test Failures

When failures occur:
1. Review the failure details
2. Check if it's a flaky test or real bug
3. Run the specific test in isolation
4. Fix the issue
5. Re-run parallel tests to verify

## Integration with CI/CD

```yaml
# GitHub Actions example
- name: Run Parallel Tests
  run: |
    pytest tests/core/ &
    pytest tests/db/ &
    pytest tests/utils/ &
    pytest tests/web/ &
    wait
```

## Tips for Effective Parallel Testing

1. **Group by Independence**: Tests that share no state can run in parallel
2. **Balance Load**: Try to make each group take similar time
3. **Handle Shared Resources**: Use fixtures for database setup/teardown
4. **Monitor Memory**: Parallel tests use more memory
5. **Check for Race Conditions**: Ensure no test depends on another's side effects
