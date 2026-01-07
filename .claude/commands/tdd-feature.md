---
description: Build feature using Test-Driven Development
---

# TDD Feature Development

Implement new features following the Test-Driven Development (TDD) methodology using the Red-Green-Refactor cycle.

## What is TDD?

Test-Driven Development is a software development approach where you:
1. Write a failing test (RED)
2. Write minimal code to make it pass (GREEN)
3. Improve the code while keeping tests green (REFACTOR)

## Benefits
- **Better Design**: Tests force you to think about API design first
- **Higher Confidence**: Code is tested from the start
- **Living Documentation**: Tests serve as usage examples
- **Prevents Over-Engineering**: Write only code needed to pass tests
- **Easier Refactoring**: Tests catch regressions immediately

## When to Use
- Building new features from scratch
- Adding new functionality to existing code
- When requirements are clear and well-defined
- For complex logic that benefits from incremental development

## Usage
Invoke with `/tdd-feature`

## Workflow

### Phase 1: RED - Write Failing Test

1. **Understand Requirements**: Clarify what the feature should do
2. **Write Test First**: Create test that describes desired behavior
3. **Run Test**: Verify it fails (proves test is actually testing something)

**Example**:
```python
import pytest
from rag_utils import calculate_chunk_overlap

def test_calculate_overlap_percentage():
    """Should calculate overlap as percentage of chunk size."""
    result = calculate_chunk_overlap(chunk_size=700, overlap_ratio=0.2)
    assert result == 140

# Run test: FAIL - calculate_chunk_overlap is not defined
```

### Phase 2: GREEN - Make Test Pass

1. **Write Minimal Code**: Implement simplest solution to pass test
2. **Run Test**: Verify it passes
3. **Resist Temptation**: Don't add features not required by tests

**Example**:
```python
def calculate_chunk_overlap(chunk_size: int, overlap_ratio: float) -> int:
    """Calculate chunk overlap from size and ratio."""
    return int(chunk_size * overlap_ratio)

# Run test: PASS
```

### Phase 3: REFACTOR - Improve Code

1. **Improve Code Quality**: Extract functions, rename variables, simplify logic
2. **Keep Tests Green**: Run tests after each change
3. **Add More Tests**: Cover edge cases and additional scenarios

**Example**:
```python
# Add more test cases
def test_overlap_with_zero_ratio():
    result = calculate_chunk_overlap(chunk_size=700, overlap_ratio=0)
    assert result == 0

def test_overlap_validates_ratio_range():
    with pytest.raises(ValueError):
        calculate_chunk_overlap(chunk_size=700, overlap_ratio=1.5)

# Refactor for validation
def calculate_chunk_overlap(
    chunk_size: int,
    overlap_ratio: float,
) -> int:
    """Calculate chunk overlap from size and ratio.

    Args:
        chunk_size: Size of each chunk in characters
        overlap_ratio: Overlap as ratio (0.0 to 1.0)

    Returns:
        Overlap size in characters

    Raises:
        ValueError: If overlap_ratio not in valid range
    """
    if not 0 <= overlap_ratio <= 1:
        raise ValueError(f"overlap_ratio must be 0-1, got {overlap_ratio}")
    return int(chunk_size * overlap_ratio)
```

### Phase 4: DOCUMENT - Add Documentation

1. **Add Docstrings**: Document function purpose and parameters
2. **Update CLAUDE.md**: Add feature to documentation if appropriate
3. **Code Comments**: Explain non-obvious business logic

## TDD Cycle Diagram

```
     +----------+
     |   RED    |  Write failing test
     |          |
     +----+-----+
          |
          v
     +----------+
     |  GREEN   |  Make test pass
     |          |
     +----+-----+
          |
          v
     +----------+
     | REFACTOR |  Improve code
     |          |
     +----+-----+
          |
          | (Add more tests)
          +--------------+
```

## Example Session

```
User: /tdd-feature "Add similarity threshold filtering to retrieval"

Claude: Let's build similarity threshold filtering using TDD.

=== PHASE 1: RED ===

First, let's write a failing test:

```python
def test_filter_by_similarity_threshold():
    """Should filter out chunks below similarity threshold."""
    chunks = [
        NodeWithScore(node=TextNode(text="high"), score=0.85),
        NodeWithScore(node=TextNode(text="medium"), score=0.55),
        NodeWithScore(node=TextNode(text="low"), score=0.25),
    ]

    filtered = filter_by_similarity(chunks, threshold=0.5)

    assert len(filtered) == 2
    assert filtered[0].score == 0.85
    assert filtered[1].score == 0.55
```

Running test... FAIL: filter_by_similarity is not defined

=== PHASE 2: GREEN ===

Now let's implement the minimal code to pass:

```python
def filter_by_similarity(
    chunks: List[NodeWithScore],
    threshold: float,
) -> List[NodeWithScore]:
    return [c for c in chunks if c.score >= threshold]
```

Running test... PASS

=== PHASE 3: REFACTOR ===

Let's add edge cases and improve:

[Adds tests for empty list, all filtered, threshold=0, threshold=1]
[Adds type hints and docstring]
[Adds validation for threshold range]

=== PHASE 4: DOCUMENT ===

[Adds JSDoc and usage examples]

Feature complete! All 6 tests passing.
```

## Best Practices

1. **Start Simple**: Begin with the simplest test case (happy path)
2. **One Test at a Time**: Don't write multiple tests before implementing
3. **Small Steps**: Each cycle should take 5-10 minutes max
4. **Run Tests Frequently**: After every small change
5. **Refactor Continuously**: Don't accumulate technical debt
6. **Test Behavior**: Focus on what code does, not how it does it

## Common Mistakes to Avoid

- **Writing implementation first**: Defeats the purpose of TDD
- **Writing too many tests at once**: Can't follow red-green-refactor
- **Skipping the refactor phase**: Leads to messy code
- **Testing implementation details**: Makes tests brittle
- **Not running tests frequently**: Delayed feedback

## TDD for Different Scenarios

### Pure Functions
Perfect for TDD - deterministic, easy to test
```python
# Test first
def test_format_chunk_info():
    assert format_chunk_info("hello", 1) == "Chunk 1: 5 chars"

# Then implement
def format_chunk_info(text: str, idx: int) -> str:
    return f"Chunk {idx}: {len(text)} chars"
```

### Database Operations
Test with mocks or test database
```python
# Test first
def test_insert_nodes_to_db(mock_db):
    nodes = [TextNode(text="test")]
    insert_nodes(mock_db, nodes)
    mock_db.add.assert_called_once()
```

### RAG Pipeline Components
Test each stage independently
```python
# Test first
def test_chunking_respects_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # Check chunks overlap by 20 chars
    assert chunks[0][-20:] == chunks[1][:20]
```

## Resources

- **Test Framework Docs**: Refer to pytest documentation
- **Project Tests**: Look at existing tests for patterns
- **TDD by Example**: Kent Beck's classic book

## Next Steps After TDD

1. All tests pass
2. Documentation complete
3. Code review (if working with team)
4. Ready to commit and push!
