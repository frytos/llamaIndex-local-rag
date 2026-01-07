---
description: Find patterns and refactor systematically
---

# Search and Refactor

Two-stage workflow that uses a search sub-agent to find all occurrences of a pattern, then the main agent reviews and refactors systematically.

## Benefits
- **Complete Coverage**: Won't miss any occurrences
- **Context Awareness**: Main agent reviews results before refactoring
- **Safe**: Systematic approach reduces risk of breaking changes
- **Efficient**: Sub-agent handles time-consuming search

## When to Use
- Renaming functions, variables, or classes across codebase
- Updating deprecated API usage
- Standardizing code patterns
- Removing technical debt
- Applying consistent naming conventions
- Migrating to new libraries or frameworks

## Usage
Invoke with `/search-and-refactor`

## Workflow

### Stage 1: Search (Sub-Agent)

The search sub-agent performs comprehensive search:

1. **Identify Pattern**: Understand what to search for
2. **Comprehensive Search**: Find all occurrences across codebase
3. **Gather Context**: Collect surrounding code for each occurrence
4. **Report Results**: Return structured list with locations and context

**Example Search Tasks**:
- Find all uses of `get_db_connection()` function
- Locate all instances of deprecated `chunk_documents()` call
- Find functions without type hints
- Search for hardcoded values that should be constants

### Stage 2: Refactor (Main Agent)

The main agent reviews and refactors:

1. **Review Results**: Examine all found occurrences
2. **Plan Refactoring**: Determine how to update each occurrence
3. **Apply Changes**: Systematically update code
4. **Run Tests**: Verify nothing broke
5. **Report**: Summarize changes made

## Example Usage

### Example 1: Rename Function

```
User: /search-and-refactor "Rename embed_nodes to compute_embeddings"

=== STAGE 1: SEARCH ===
Sub-agent searching for 'embed_nodes'...

Found 8 occurrences across 4 files:

1. rag_low_level_m1_16gb_verbose.py:1234
   def embed_nodes(embed_model, nodes):

2. rag_low_level_m1_16gb_verbose.py:1680
   embed_nodes(embed_model, nodes)

3. rag_web.py:571
   from rag_low_level_m1_16gb_verbose import embed_nodes

4. rag_interactive.py:89
   embed_nodes(model, chunks)

[... 4 more occurrences ...]

=== STAGE 2: REFACTOR ===
Main agent reviewing results...

Planning refactoring strategy:
1. Update function definition
2. Update all 7 call sites
3. Update import statements
4. Run tests to verify

Applying changes...
- Updated function definition
- Updated 7 call sites
- Updated 2 import statements
- Tests pass

Summary: Renamed embed_nodes -> compute_embeddings (8 locations)
```

### Example 2: Standardize Error Handling

```
User: /search-and-refactor "Standardize error handling to use custom RAGError"

=== STAGE 1: SEARCH ===
Found 15 different error handling patterns:

Pattern 1: bare try-except with log.error (8 locations)
Pattern 2: raise Exception with string (5 locations)
Pattern 3: return None on error (2 locations)

=== STAGE 2: REFACTOR ===
Standardizing all to RAGError pattern...

Created: src/errors.py with RAGError class
Updated 15 locations:
  - Replaced bare exceptions with specific RAGError
  - Added proper error context
  - Standardized error messages

- All functionality preserved
- Error handling now consistent

Summary: Unified error handling across codebase
```

### Example 3: Add Type Hints

```
User: /search-and-refactor "Add type hints to all public functions"

=== STAGE 1: SEARCH ===
Found 23 functions without type hints:

1. load_documents(doc_path)
2. chunk_documents(docs)
3. build_nodes(docs, chunks, doc_idxs)
[... 20 more ...]

=== STAGE 2: REFACTOR ===
Adding type hints...

- Added return types
- Added parameter types
- Created type aliases for common types

Updated 23 functions:
  - All now have complete type annotations
  - Created types.py for shared types

Summary: Added type hints to 23 functions
```

## Search Patterns

### Function Calls
```python
# Find all calls to a specific function
pattern: "function_name("
```

### Import Statements
```python
# Find all imports from a module
pattern: "from module import"
```

### Class Usage
```python
# Find all uses of a class
pattern: "ClassName("
```

### Configuration Values
```python
# Find hardcoded values
pattern: "chunk_size = 700"
```

## Refactoring Strategies

### 1. Direct Replacement
Simple find-and-replace when context doesn't matter
```python
# Before: get_data
# After: fetch_data
```

### 2. Context-Aware Replacement
Different replacement based on context
```python
# In main code: get_data -> fetch_data
# In tests: get_data -> mock_fetch_data
```

### 3. Structural Changes
Requires rewriting code structure
```python
# Before: multiple separate functions
# After: consolidated class with methods
```

### 4. Incremental Migration
Gradual replacement with backward compatibility
```python
# Step 1: Add new function alongside old
# Step 2: Migrate callers one by one
# Step 3: Remove old function when complete
```

## Best Practices

### Before Refactoring
1. **Commit Current Work**: Start with clean working directory
2. **Run Tests**: Ensure everything passes before changes
3. **Create Branch**: `git checkout -b refactor/description`
4. **Backup**: Consider creating a backup branch

### During Refactoring
1. **Review Search Results**: Ensure search found everything
2. **Plan Approach**: Decide on refactoring strategy
3. **Make Atomic Changes**: One logical change at a time
4. **Test Frequently**: Run tests after each change
5. **Handle Edge Cases**: Address variations in usage

### After Refactoring
1. **Run Full Test Suite**: Ensure nothing broke
2. **Manual Testing**: Test affected features
3. **Code Review**: Have teammate review changes
4. **Update Documentation**: Reflect new patterns

## Safety Checks

The command includes built-in safety:
- Requires test suite to pass before finalizing
- Main agent reviews all changes before applying
- Highlights potential breaking changes
- Suggests rollback if issues detected

## Common Use Cases

### Renaming
- Functions, classes, methods
- Variables and constants
- Files and directories

### Updating Patterns
- Deprecated API usage
- Old library patterns -> new patterns
- Inconsistent code styles
- Anti-patterns -> best practices

### Migration
- Library upgrades
- Python version features
- Testing frameworks
- Configuration formats

## Integration with Other Commands

Combine with:
- `/parallel-test` - Test refactoring quickly
- `/review-pr` - Review refactoring changes
- `/comprehensive-audit` - Identify refactoring needs

## Example Workflow

```bash
# 1. Identify refactoring need
/comprehensive-audit  # Finds code duplication

# 2. Perform systematic refactoring
/search-and-refactor "Extract duplicate validation logic"

# 3. Verify changes
/parallel-test

# 4. Review before committing
/review-pr

# 5. Commit
git add .
git commit -m "refactor: extract common validation logic"
```
