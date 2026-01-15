# Coding Conventions

**Analysis Date:** 2026-01-15

## Naming Patterns

**Files:**
- snake_case for all files: `rag_low_level_m1_16gb_verbose.py`, `metadata_extractor.py`
- Test files: `test_*.py` prefix (e.g., `test_core_config.py`, `test_chunking.py`)
- Descriptive names including context

**Functions:**
- snake_case for all functions: `build_embed_model()`, `chunk_documents()`, `run_query()`
- Verb-first for actions: `load_documents()`, `insert_nodes()`, `create_hnsw_index()`
- No special prefix for async functions

**Variables:**
- snake_case for variables: `chunk_size`, `embed_model`, `participant_colors`
- UPPER_SNAKE_CASE for constants: `COLORS`, `RESET`, `BOLD`, `DB_CONFIG`
- Private members prefixed with underscore: `_generate_embedding()`, `_extract_model_short_name`

**Types:**
- PascalCase for classes: `Settings`, `VectorDBRetriever`, `HybridRetriever`, `Reranker`
- PascalCase for interfaces: `*Retriever` suffix (follows LlamaIndex convention)
- PascalCase for dataclasses: `ChunkConfig`, `SimilarityThresholds`, `LLMConfig`

## Code Style

**Formatting:**
- Black formatter with `.editorconfig` and `pyproject.toml` configuration
- 100 character line length
- 4 space indentation
- Double quotes for strings
- Triple double quotes for docstrings (`"""`)
- UTF-8 encoding, LF line endings, final newline required

**Linting:**
- Ruff 0.6+ - Multi-purpose linter (`pyproject.toml` lines 95-147)
- Enabled rules: E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, RET, TRY, PLR, PLW
- Ignored rules: E501 (line length), B008, B905, TRY003, PLR0913, PLR2004
- Per-file ignores: F401 in `__init__.py`, T201 (print) in scripts

## Import Organization

**Order:**
1. Standard library imports (alphabetical)
2. Third-party imports (alphabetical)
3. LlamaIndex imports
4. Local imports - configuration
5. Local imports - utilities

**Grouping:**
- Blank line between groups
- Within groups: Related imports on same line
- Type imports last within each group

**Path Aliases:**
- No path aliases defined
- Use relative imports for local modules

## Error Handling

**Patterns:**
- Throw errors, catch at boundaries (main functions, CLI handlers)
- Custom errors: Extend Error class (e.g., `ValidationError`, `ConfigError`)
- Async functions use try/catch, no .catch() chains
- Optional dependencies with graceful fallback (try/except ImportError)

**Error Types:**
- Throw on invalid input, missing dependencies, invariant violations
- Log error with context before throwing
- Include cause in error message when re-throwing

## Logging

**Framework:**
- Python logging module with configurable level
- Levels: DEBUG, INFO, WARNING, ERROR (no TRACE)

**Patterns:**
- Structured logging with f-strings: `log.info(f"Indexed {len(nodes)} chunks")`
- Log at service boundaries, not in utility functions
- Log state transitions, external calls, errors
- No print() in committed production code (only in scripts/CLI for user output)

## Comments

**When to Comment:**
- Explain why, not what
- Document business logic, algorithms, edge cases
- Avoid obvious comments (e.g., `# increment counter`)

**JSDoc/TSDoc:**
- Google-style docstrings for all public functions
- Optional for internal functions if signature is self-explanatory
- Use `Args:`, `Returns:`, `Raises:`, `Examples:` sections

**TODO Comments:**
- Format: `# TODO: description` (no username)
- Link to issue if exists: `# TODO: Fix race condition (issue #123)`

## Function Design

**Size:**
- Keep under 200 lines (main functions may be longer)
- Extract helpers for complex logic
- One level of abstraction per function

**Parameters:**
- Max 5 parameters recommended
- Use dataclass for 6+ parameters: `Settings` object
- Destructure in parameter list where appropriate

**Return Values:**
- Explicit return statements
- Return early for guard clauses
- Type annotations always present: `-> str`, `-> List[float]`, `-> None`

## Module Design

**Exports:**
- Named exports preferred
- No default exports (Python doesn't support)
- Export public API from `__init__.py` for packages

**Barrel Files:**
- `__init__.py` re-exports public API
- Keep internal helpers private (don't export from `__init__.py`)
- Avoid circular dependencies

## Type Hints

**Usage:**
- Used extensively in function signatures
- Optional types: `Optional[str]`
- Collection types: `List[]`, `Dict[]`, `Tuple[]`, `Set[]`
- Generic types: `Any`, `Callable`
- Return type annotations always present

**MyPy Configuration:**
- Python version: 3.11 (`pyproject.toml` line 151)
- Mode: basic (line 255)
- Third-party stubs ignored for: llama_index, llama_cpp, sentence_transformers

---

*Convention analysis: 2026-01-15*
*Update when patterns change*
