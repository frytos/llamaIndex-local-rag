# Code Review Report: Critical Table Naming Bug

**Date**: 2026-01-08
**Reviewer**: Code Reviewer Agent
**Severity**: CRITICAL (P0)
**Status**: PARTIALLY FIXED - Additional fixes required

---

## Executive Summary

A critical bug in table naming causes query failures due to PGVectorStore's automatic "data_" prefix prepending. When a table name already starts with "data_", the system creates a double-prefixed table (e.g., `data_data_messages-text-slim_fast_1883_260108`), resulting in 0 query results despite having indexed data.

**Current State:**
- Partial fix applied in `rag_web_enhanced.py:run_query()` (lines 1308-1347)
- **Bug still exists** in other critical locations
- No defensive checks to prevent empty table creation
- Missing unit tests for edge cases
- Root cause not addressed in table name generation

---

## 1. Root Cause Analysis

### PGVectorStore Behavior
```python
# LlamaIndex's PGVectorStore.from_params() internally prepends "data_" to all table names
# Source: llama_index.vector_stores.postgres.PGVectorStore

# User provides: "messages-text-slim_fast_1883_260108"
# PGVectorStore creates: "data_messages-text-slim_fast_1883_260108"

# If user provides: "data_messages-text-slim_fast_1883_260108"
# PGVectorStore creates: "data_data_messages-text-slim_fast_1883_260108" ❌
```

### Bug Manifestation
1. **Indexing Phase**: Table created with user-provided name (e.g., `data_messages...`)
2. **PGVectorStore**: Automatically prepends "data_" → creates `data_data_messages...`
3. **Query Phase**: Searches in `data_data_messages...` (empty) instead of `data_messages...` (has data)
4. **Result**: 0 query results, users confused

---

## 2. Current Fix Review

### Location: `rag_web_enhanced.py` lines 1308-1347

```python
# CRITICAL FIX: PGVectorStore auto-prepends "data_" prefix!
# If table name already starts with "data_", strip it to avoid double prefix
query_table_name = table_name
if table_name.startswith("data_"):
    query_table_name = table_name[5:]  # Remove "data_" prefix
    st.caption(f"🔧 Table name adjusted: `{table_name}` → `{query_table_name}` (PGVectorStore adds 'data_' prefix)")

# Update settings with corrected table name
rag.S.table = query_table_name
```

### Review Assessment: ⚠️ INCOMPLETE

**Strengths:**
- ✅ Correctly identifies the problem
- ✅ Strips "data_" prefix to prevent double-prefixing
- ✅ Provides user feedback
- ✅ Fixes query functionality in `rag_web_enhanced.py`

**Critical Issues:**
1. ❌ **Only fixes ONE location** (query page in enhanced web UI)
2. ❌ **Does not fix indexing phase** - still creates double-prefixed tables
3. ❌ **Other query locations still broken** (`rag_web.py`, `rag_interactive.py`)
4. ❌ **Band-aid fix** - doesn't prevent problem at source
5. ❌ **Edge cases not handled** (see section 3)

---

## 3. Edge Cases Not Handled

### 3.1 Table Names with "data_" in Middle
```python
# Current fix only checks startswith()
"my_data_table"  # ✅ Not affected (correct)
"table_data_v2"  # ✅ Not affected (correct)
```
**Status**: ✅ Handled correctly (no change needed)

### 3.2 Multiple "data_" Prefixes
```python
"data_data_table"  # Current: strips to "data_table" ✅
                   # PGVectorStore: creates "data_data_table" ✅ (matches original intent)
```
**Status**: ✅ Handled correctly

### 3.3 Tables NOT Starting with "data_"
```python
"messages_cs500_ov100"  # Current: no strip
                        # PGVectorStore: creates "data_messages_cs500_ov100" ✅
```
**Status**: ✅ Handled correctly

### 3.4 Empty Table Name
```python
""  # Current: no check
    # Result: undefined behavior
```
**Status**: ❌ NOT HANDLED - needs validation

### 3.5 Table Names with Hyphens
```python
"data_messages-text-slim_fast"  # PostgreSQL allows hyphens in identifiers
                                 # Must use double quotes: "data_messages-text-slim_fast"
```
**Status**: ⚠️ PARTIALLY HANDLED - SQL uses sql.Identifier() but manual DROP uses raw strings

---

## 4. Affected Code Locations

### 4.1 Query Functions (WHERE BUG MANIFESTS)

| File | Function | Line | Status | Severity |
|------|----------|------|--------|----------|
| `rag_web_enhanced.py` | `run_query()` | 1308-1347 | ✅ FIXED | P0 |
| `rag_web.py` | `run_query()` | 737-755 | ❌ BROKEN | P0 |
| `rag_interactive.py` | Query menu option | N/A | ❌ BROKEN | P1 |
| `test_retrieval_direct.py` | Test script | 45 | ❌ BROKEN | P2 |

### 4.2 Indexing Functions (WHERE BUG ORIGINATES)

| File | Function | Line | Status | Severity |
|------|----------|------|--------|----------|
| `rag_web_enhanced.py` | `run_indexing()` | 766 | ❌ NO FIX | P0 |
| `rag_web.py` | `run_indexing()` | 534 | ❌ NO FIX | P0 |
| `rag_low_level_m1_16gb_verbose.py` | `main()` | 718-726 | ❌ NO FIX | P0 |

### 4.3 Table Management Functions

| File | Function | Line | Status | Risk |
|------|----------|------|--------|------|
| `rag_web_enhanced.py` | `list_vector_tables()` | 225-275 | ⚠️ Shows both | Medium |
| `rag_web_enhanced.py` | `delete_table()` | 280-291 | ⚠️ Direct SQL | Medium |
| `rag_web_enhanced.py` | `delete_empty_tables()` | 293-321 | ⚠️ Direct SQL | Low |
| `rag_low_level_m1_16gb_verbose.py` | `reset_table()` | 896 | ⚠️ Direct SQL | Medium |

### 4.4 Table Name Generation

| File | Function | Line | Status | Action Needed |
|------|----------|------|--------|---------------|
| `utils/naming.py` | `generate_table_name()` | 69-103 | ⚠️ OK | Add prefix check |
| `utils/naming.py` | `sanitize_table_name()` | 11-33 | ✅ OK | No change |

---

## 5. Recommended Fixes

### 5.1 Fix Priority P0: Add Helper Function

**File**: `utils/naming.py`
**Action**: Create centralized table name normalization

```python
def normalize_table_name_for_pgvector(table_name: str) -> str:
    """Normalize table name for PGVectorStore compatibility.

    PGVectorStore automatically prepends "data_" to all table names.
    If the provided name already starts with "data_", we strip it
    to prevent double-prefixing (e.g., "data_data_table").

    Args:
        table_name: Raw table name (may or may not start with "data_")

    Returns:
        Normalized table name safe for PGVectorStore

    Examples:
        >>> normalize_table_name_for_pgvector("data_messages_cs500")
        'messages_cs500'
        >>> normalize_table_name_for_pgvector("messages_cs500")
        'messages_cs500'
        >>> normalize_table_name_for_pgvector("")
        Raises ValueError

    Raises:
        ValueError: If table_name is empty or invalid
    """
    if not table_name or not table_name.strip():
        raise ValueError("Table name cannot be empty")

    table_name = table_name.strip()

    # Strip "data_" prefix if present (PGVectorStore will add it back)
    if table_name.startswith("data_"):
        return table_name[5:]  # Remove "data_" prefix

    return table_name


def get_actual_table_name(table_name: str) -> str:
    """Get the actual PostgreSQL table name after PGVectorStore processing.

    Use this when you need to query the database directly (not through PGVectorStore).

    Args:
        table_name: Table name as provided by user

    Returns:
        Actual table name in PostgreSQL (with "data_" prefix)

    Examples:
        >>> get_actual_table_name("messages_cs500")
        'data_messages_cs500'
        >>> get_actual_table_name("data_messages_cs500")
        'data_messages_cs500'
    """
    if not table_name.startswith("data_"):
        return f"data_{table_name}"
    return table_name
```

### 5.2 Fix Priority P0: Update All Query Functions

**File**: `rag_web.py` line 737-755

```python
def run_query(table_name: str, query: str, top_k: int, show_sources: bool):
    """Run a query against the index."""

    import rag_low_level_m1_16gb_verbose as rag
    from utils.naming import normalize_table_name_for_pgvector

    # CRITICAL FIX: Normalize table name for PGVectorStore
    normalized_table = normalize_table_name_for_pgvector(table_name)
    rag.S.table = normalized_table
    rag.S.top_k = top_k

    with st.spinner("Searching..."):
        try:
            # Build retriever
            embed_model = get_embed_model(rag.S.embed_model_name)
            vector_store = make_vector_store()  # Will prepend "data_" internally
            retriever = VectorDBRetriever(vector_store, embed_model, similarity_top_k=top_k)

            # Retrieve
            results = retriever._retrieve(QueryBundle(query_str=query))
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            return
    # ... rest of function
```

### 5.3 Fix Priority P0: Update All Indexing Functions

**File**: `rag_web_enhanced.py` line 766

```python
def run_indexing(doc_path: Path, table_name: str, chunk_size: int, chunk_overlap: int,
                 embed_model_name: str, embed_dim: int, embed_batch: int, reset_table: bool):
    """Run indexing pipeline."""
    import time
    from utils.naming import normalize_table_name_for_pgvector, get_actual_table_name

    # Update settings
    import rag_low_level_m1_16gb_verbose as rag
    rag.S.pdf_path = str(doc_path)

    # CRITICAL FIX: Normalize table name for PGVectorStore
    normalized_table = normalize_table_name_for_pgvector(table_name)
    rag.S.table = normalized_table

    rag.S.chunk_size = chunk_size
    rag.S.chunk_overlap = chunk_overlap
    rag.S.embed_model_name = embed_model_name
    rag.S.embed_dim = embed_dim
    rag.S.embed_batch = embed_batch
    rag.S.reset_table = reset_table

    # ... continue with rest of function

    # When dropping table, use actual name
    if reset_table:
        st.caption("Dropping existing table...")
        try:
            actual_table = get_actual_table_name(normalized_table)
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE').format(sql.Identifier(actual_table)))
            conn.commit()
            cur.close()
            conn.close()
            st.caption(f"✓ Dropped table `{actual_table}`")
        except:
            pass
```

### 5.4 Fix Priority P0: Update Core Script

**File**: `rag_low_level_m1_16gb_verbose.py` lines 715-727

```python
from utils.naming import generate_table_name, normalize_table_name_for_pgvector

# Auto-generate table name if PGTABLE was not explicitly set
if not os.getenv("PGTABLE"):
    generated_table = generate_table_name(
        S.pdf_path,
        S.chunk_size,
        S.chunk_overlap,
        S.embed_model_name
    )
    # Normalize for PGVectorStore compatibility
    S.table = normalize_table_name_for_pgvector(generated_table)
    log.debug(f"Auto-generated table name: {generated_table} → normalized: {S.table}")
else:
    raw_table = os.getenv("PGTABLE")
    S.table = normalize_table_name_for_pgvector(raw_table)
    if raw_table.startswith("data_"):
        log.warning(f"Table name normalized: {raw_table} → {S.table} (PGVectorStore adds 'data_' prefix)")
```

### 5.5 Fix Priority P1: Add Defensive Checks

**File**: `rag_low_level_m1_16gb_verbose.py` after line 2293

```python
def make_vector_store() -> PGVectorStore:
    """
    Create the vector store client that uses Postgres + pgvector.

    IMPORTANT: PGVectorStore automatically prepends "data_" to table names.
    Ensure S.table does NOT start with "data_" to avoid double-prefixing.
    """
    from utils.naming import normalize_table_name_for_pgvector

    # Defensive check: warn if table name starts with "data_"
    if S.table.startswith("data_"):
        log.warning(
            f"⚠️  Table name '{S.table}' starts with 'data_' prefix. "
            f"PGVectorStore will create table 'data_{S.table}'. "
            f"Consider using normalize_table_name_for_pgvector() to avoid double-prefixing."
        )

    log.info(f"Connecting to Postgres vector store: db={S.db_name} host={S.host}:{S.port} user={S.user} table={S.table}")
    log.info(f"⚠️  Note: PGVectorStore will create table 'data_{S.table}' in PostgreSQL")

    store = PGVectorStore.from_params(
        database=S.db_name,
        host=S.host,
        port=S.port,
        user=S.user,
        password=S.password,
        table_name=S.table,
        embed_dim=S.embed_dim,
    )
    return store
```

### 5.6 Fix Priority P1: Add Table Existence Check

**File**: New function in `rag_low_level_m1_16gb_verbose.py`

```python
def verify_table_not_empty(table_name: str) -> bool:
    """Check if table exists and has data before querying.

    Args:
        table_name: Table name to check (will check both with and without "data_" prefix)

    Returns:
        True if table exists and has rows, False otherwise

    Raises:
        ValueError: If table doesn't exist or is empty
    """
    from utils.naming import get_actual_table_name

    actual_table = get_actual_table_name(table_name)

    try:
        conn = db_conn()
        cur = conn.cursor()

        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
        """, (actual_table,))

        exists = cur.fetchone()[0]

        if not exists:
            cur.close()
            conn.close()
            raise ValueError(f"Table '{actual_table}' does not exist in database")

        # Check if table has data
        cur.execute(sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(actual_table)))
        count = cur.fetchone()[0]

        cur.close()
        conn.close()

        if count == 0:
            raise ValueError(f"Table '{actual_table}' exists but is empty (0 rows)")

        log.info(f"✅ Table '{actual_table}' verified: {count} rows")
        return True

    except Exception as e:
        log.error(f"❌ Table verification failed: {e}")
        raise
```

---

## 6. Comprehensive Test Suite

### 6.1 Unit Tests for Table Name Normalization

**File**: `tests/test_table_name_normalization.py`

```python
"""Tests for table name normalization to fix PGVectorStore double-prefix bug."""

import pytest
from utils.naming import normalize_table_name_for_pgvector, get_actual_table_name


class TestNormalizeTableName:
    """Test table name normalization for PGVectorStore compatibility."""

    def test_strips_data_prefix(self):
        """Test that 'data_' prefix is stripped."""
        assert normalize_table_name_for_pgvector("data_messages_cs500") == "messages_cs500"
        assert normalize_table_name_for_pgvector("data_test_table") == "test_table"

    def test_keeps_non_prefixed_names(self):
        """Test that names without 'data_' prefix are unchanged."""
        assert normalize_table_name_for_pgvector("messages_cs500") == "messages_cs500"
        assert normalize_table_name_for_pgvector("test_table") == "test_table"

    def test_handles_multiple_data_prefixes(self):
        """Test handling of multiple 'data_' prefixes."""
        # Only strip the first one
        assert normalize_table_name_for_pgvector("data_data_table") == "data_table"
        assert normalize_table_name_for_pgvector("data_data_data_table") == "data_data_table"

    def test_data_in_middle_not_affected(self):
        """Test that 'data_' in middle of name is preserved."""
        assert normalize_table_name_for_pgvector("my_data_table") == "my_data_table"
        assert normalize_table_name_for_pgvector("table_data_v2") == "table_data_v2"

    def test_empty_name_raises_error(self):
        """Test that empty table name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_table_name_for_pgvector("")

        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_table_name_for_pgvector("   ")

    def test_whitespace_trimmed(self):
        """Test that whitespace is trimmed."""
        assert normalize_table_name_for_pgvector("  data_table  ") == "table"
        assert normalize_table_name_for_pgvector("  table  ") == "table"

    def test_case_sensitivity(self):
        """Test that function is case-sensitive (PostgreSQL convention)."""
        # Should NOT strip "DATA_" (uppercase)
        assert normalize_table_name_for_pgvector("DATA_table") == "DATA_table"
        assert normalize_table_name_for_pgvector("Data_table") == "Data_table"

    def test_hyphens_preserved(self):
        """Test that hyphens in table names are preserved."""
        assert normalize_table_name_for_pgvector("data_messages-text-slim") == "messages-text-slim"
        assert normalize_table_name_for_pgvector("my-table-name") == "my-table-name"


class TestGetActualTableName:
    """Test getting actual PostgreSQL table name after PGVectorStore processing."""

    def test_adds_prefix_when_missing(self):
        """Test that 'data_' prefix is added when missing."""
        assert get_actual_table_name("messages_cs500") == "data_messages_cs500"
        assert get_actual_table_name("test_table") == "data_test_table"

    def test_preserves_existing_prefix(self):
        """Test that existing 'data_' prefix is preserved."""
        assert get_actual_table_name("data_messages_cs500") == "data_messages_cs500"
        assert get_actual_table_name("data_test_table") == "data_test_table"

    def test_idempotent_with_normalize(self):
        """Test that normalize → get_actual is idempotent."""
        original = "data_messages_cs500"
        normalized = normalize_table_name_for_pgvector(original)
        actual = get_actual_table_name(normalized)
        assert actual == original


class TestRoundTripConversion:
    """Test round-trip conversion scenarios."""

    def test_user_provides_no_prefix(self):
        """User provides: 'messages_cs500' → PostgreSQL: 'data_messages_cs500'"""
        user_input = "messages_cs500"
        normalized = normalize_table_name_for_pgvector(user_input)
        assert normalized == "messages_cs500"

        actual = get_actual_table_name(normalized)
        assert actual == "data_messages_cs500"

    def test_user_provides_with_prefix(self):
        """User provides: 'data_messages_cs500' → PostgreSQL: 'data_messages_cs500'"""
        user_input = "data_messages_cs500"
        normalized = normalize_table_name_for_pgvector(user_input)
        assert normalized == "messages_cs500"

        actual = get_actual_table_name(normalized)
        assert actual == "data_messages_cs500"

    def test_double_prefix_prevented(self):
        """Verify double prefix is prevented."""
        # Without fix, this would create "data_data_messages_cs500"
        user_input = "data_messages_cs500"
        normalized = normalize_table_name_for_pgvector(user_input)
        actual = get_actual_table_name(normalized)

        # Should NOT be double-prefixed
        assert actual == "data_messages_cs500"
        assert actual.count("data_") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 6.2 Integration Tests

**File**: `tests/test_table_name_bug_integration.py`

```python
"""Integration tests for table name bug fix."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestTableNameBugIntegration:
    """Integration tests for the table naming bug fix."""

    @patch('rag_low_level_m1_16gb_verbose.db_conn')
    @patch('rag_low_level_m1_16gb_verbose.PGVectorStore')
    def test_indexing_with_data_prefix(self, mock_pgvector, mock_db_conn):
        """Test indexing with table name starting with 'data_'."""
        from rag_low_level_m1_16gb_verbose import S, make_vector_store
        from utils.naming import normalize_table_name_for_pgvector

        # User provides table with "data_" prefix
        user_table = "data_messages_cs500"
        S.table = normalize_table_name_for_pgvector(user_table)

        # Should strip prefix
        assert S.table == "messages_cs500"

        # Make vector store
        mock_pgvector.from_params.return_value = Mock()
        store = make_vector_store()

        # Verify PGVectorStore called with normalized name
        mock_pgvector.from_params.assert_called_once()
        call_kwargs = mock_pgvector.from_params.call_args[1]
        assert call_kwargs['table_name'] == "messages_cs500"

    @patch('rag_low_level_m1_16gb_verbose.db_conn')
    def test_query_table_verification(self, mock_db_conn):
        """Test table existence verification before querying."""
        from rag_low_level_m1_16gb_verbose import verify_table_not_empty

        # Mock database cursor
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        # Table exists with data
        mock_cursor.fetchone.side_effect = [(True,), (1000,)]

        result = verify_table_not_empty("messages_cs500")
        assert result is True

        # Verify checked for "data_messages_cs500"
        calls = mock_cursor.execute.call_args_list
        assert any("data_messages_cs500" in str(call) for call in calls)

    @patch('rag_low_level_m1_16gb_verbose.db_conn')
    def test_empty_table_detection(self, mock_db_conn):
        """Test that empty tables are detected and error is raised."""
        from rag_low_level_m1_16gb_verbose import verify_table_not_empty

        # Mock database cursor
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        # Table exists but is empty
        mock_cursor.fetchone.side_effect = [(True,), (0,)]

        with pytest.raises(ValueError, match="empty"):
            verify_table_not_empty("messages_cs500")

    def test_generate_table_name_compatibility(self):
        """Test that generated table names work with normalization."""
        from utils.naming import generate_table_name, normalize_table_name_for_pgvector

        doc_path = Path("test.pdf")
        generated = generate_table_name(doc_path, 700, 150, "BAAI/bge-small-en")

        # Should not start with "data_"
        assert not generated.startswith("data_")

        # Normalization should return unchanged
        normalized = normalize_table_name_for_pgvector(generated)
        assert normalized == generated


class TestWebUIIntegration:
    """Test web UI integration with fix."""

    @patch('rag_web_enhanced.make_vector_store')
    @patch('rag_web_enhanced.get_embed_model')
    def test_query_page_normalizes_table_name(self, mock_embed, mock_store):
        """Test that query page normalizes table names."""
        # This would test the actual web UI code
        # Simplified mock test
        from utils.naming import normalize_table_name_for_pgvector

        # Simulate user selecting table with "data_" prefix
        selected_table = "data_messages_cs500"
        normalized = normalize_table_name_for_pgvector(selected_table)

        assert normalized == "messages_cs500"

        # Verify this is what gets passed to make_vector_store
        # (through S.table in actual implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 7. Documentation Updates Needed

### 7.1 README.md
Add warning about table naming:

```markdown
## Important: Table Naming Convention

⚠️ **Critical:** PGVectorStore automatically prepends `data_` to all table names.

**DO NOT** manually add `data_` prefix to table names:
- ✅ Correct: `PGTABLE=messages_cs500_ov100`
- ❌ Wrong: `PGTABLE=data_messages_cs500_ov100` (creates `data_data_messages_cs500_ov100`)

The system automatically handles the prefix for you.
```

### 7.2 ENVIRONMENT_VARIABLES.md
Update PGTABLE description:

```markdown
### PGTABLE
**Type**: String
**Default**: Auto-generated from document name and config
**Description**: PostgreSQL table name for vector storage

**Important**:
- Do NOT include `data_` prefix (added automatically by PGVectorStore)
- Use only alphanumeric characters and underscores
- Hyphens are allowed but require double-quoting in raw SQL

**Example**:
```bash
PGTABLE=messages_cs700_ov150_bge_260108  # ✅ Correct
PGTABLE=data_messages_cs700              # ❌ Wrong (will create data_data_messages_cs700)
```
```

### 7.3 Add Migration Guide

**File**: `docs/TABLE_NAME_MIGRATION.md`

```markdown
# Table Name Migration Guide

## Background

Prior to this fix, some users may have created tables with double `data_` prefix
due to the PGVectorStore naming bug. This guide helps migrate to the corrected naming.

## Identifying Affected Tables

```sql
-- Find tables with double prefix
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'data\_data\_%';
```

## Migration Options

### Option 1: Rename Table (Preserve Data)

```sql
ALTER TABLE data_data_messages_cs500 RENAME TO data_messages_cs500;
```

### Option 2: Delete and Re-index

```bash
# Delete broken table
PGPASSWORD=your_password psql -h localhost -U your_user -d vector_db \
  -c "DROP TABLE data_data_messages_cs500;"

# Re-index with correct name
PGTABLE=messages_cs500 python rag_low_level_m1_16gb_verbose.py
```

## Verification

```bash
# Check table exists and has data
python -c "
from utils.naming import get_actual_table_name
from rag_low_level_m1_16gb_verbose import verify_table_not_empty

verify_table_not_empty('messages_cs500')
print('✅ Table verified')
"
```
```

---

## 8. Security Considerations

### SQL Injection Prevention

Current code uses `sql.Identifier()` for table names in most places ✅, but some locations use raw strings:

**Vulnerable Code**:
```python
# rag_low_level_m1_16gb_verbose.py line 896
c.execute(f'DROP TABLE IF EXISTS "{S.table}";')  # ❌ Vulnerable
```

**Secure Code**:
```python
from psycopg2 import sql
c.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE').format(sql.Identifier(S.table)))  # ✅ Safe
```

**Recommendation**: Audit all direct SQL queries and replace f-strings with `sql.Identifier()`.

---

## 9. Monitoring & Observability

### 9.1 Add Logging

```python
def make_vector_store() -> PGVectorStore:
    """Create vector store with comprehensive logging."""

    log.info(f"Creating vector store for table: {S.table}")
    log.debug(f"PostgreSQL will create table: data_{S.table}")

    if S.table.startswith("data_"):
        log.error(
            f"⚠️  CRITICAL: Table name '{S.table}' starts with 'data_' prefix. "
            f"This will create 'data_{S.table}' in PostgreSQL, likely causing query failures. "
            f"Use normalize_table_name_for_pgvector() to fix."
        )

    store = PGVectorStore.from_params(...)

    # Verify table was created
    actual_table = f"data_{S.table}"
    log.info(f"✅ Vector store created. PostgreSQL table: {actual_table}")

    return store
```

### 9.2 Add Metrics

Track double-prefix occurrences:

```python
# In web UI
if table_name.startswith("data_"):
    st.warning("⚠️ Table name corrected to prevent double-prefix bug")
    # Log to analytics
    log_metric("table_name_normalization", {
        "original": table_name,
        "normalized": normalized_table,
        "timestamp": datetime.now().isoformat()
    })
```

---

## 10. Testing Checklist

### Unit Tests
- [ ] Test `normalize_table_name_for_pgvector()` with all edge cases
- [ ] Test `get_actual_table_name()` with all edge cases
- [ ] Test round-trip conversions
- [ ] Test empty/invalid input handling
- [ ] Test case sensitivity
- [ ] Test whitespace handling

### Integration Tests
- [ ] Test full indexing pipeline with normalized names
- [ ] Test full query pipeline with normalized names
- [ ] Test table existence verification
- [ ] Test empty table detection
- [ ] Test with actual PostgreSQL database

### End-to-End Tests
- [ ] Index document with "data_" prefixed table name
- [ ] Query the index and verify results
- [ ] Verify no double-prefixed tables created
- [ ] Test web UI indexing flow
- [ ] Test web UI query flow
- [ ] Test CLI tool

### Regression Tests
- [ ] Test existing functionality not broken
- [ ] Test tables without "data_" prefix still work
- [ ] Test table names with hyphens
- [ ] Test auto-generated table names

---

## 11. Deployment Plan

### Phase 1: Add Helper Functions (No Breaking Changes)
1. Add `normalize_table_name_for_pgvector()` to `utils/naming.py`
2. Add `get_actual_table_name()` to `utils/naming.py`
3. Add `verify_table_not_empty()` to main script
4. Deploy and monitor (no behavior changes yet)

### Phase 2: Update Core Functions
1. Update `make_vector_store()` with defensive checks and logging
2. Update all query functions to use normalization
3. Update all indexing functions to use normalization
4. Deploy to staging, run full test suite

### Phase 3: Update Web UIs
1. Update `rag_web.py` query and index functions
2. Update `rag_web_enhanced.py` (already has query fix, add to indexing)
3. Add user warnings for migrating old tables
4. Deploy to production

### Phase 4: Documentation & Migration
1. Update README, ENVIRONMENT_VARIABLES.md
2. Create TABLE_NAME_MIGRATION.md
3. Send migration guide to users
4. Monitor for double-prefixed table creation

---

## 12. Success Criteria

- [ ] Zero new tables created with double "data_" prefix
- [ ] All query functions return correct results
- [ ] All existing tests pass
- [ ] New tests achieve 100% coverage of table name handling
- [ ] No breaking changes to existing valid table names
- [ ] Clear error messages for invalid table names
- [ ] Documentation updated
- [ ] Migration guide provided

---

## 13. Conclusion

### Current Status: ⚠️ CRITICAL BUG PARTIALLY FIXED

**What's Fixed:**
- ✅ Query functionality in `rag_web_enhanced.py`

**What's Still Broken:**
- ❌ Query functionality in `rag_web.py`, `rag_interactive.py`, test scripts
- ❌ Indexing functions still create double-prefixed tables
- ❌ No defensive checks to prevent issue
- ❌ No centralized normalization function

**Impact:**
- Users creating new indexes still hit the bug
- Only one web UI page works correctly
- Potential data loss if users don't realize they have duplicate tables

**Recommendation:** **URGENT FIX REQUIRED**

Implement all P0 fixes immediately:
1. Add helper functions to `utils/naming.py`
2. Update all query functions to use normalization
3. Update all indexing functions to use normalization
4. Add comprehensive tests
5. Deploy with migration guide

**Estimated Effort:** 4-6 hours
**Priority:** P0 (Critical Bug)
**Risk if Not Fixed:** Users continue to experience 0 query results, data duplication, confusion

---

## Appendix A: Files Requiring Changes

### High Priority (P0)
1. `utils/naming.py` - Add helper functions
2. `rag_web_enhanced.py` - Fix indexing function
3. `rag_web.py` - Fix query and indexing functions
4. `rag_low_level_m1_16gb_verbose.py` - Update table name handling, add defensive checks
5. `tests/test_table_name_normalization.py` - New test file
6. `tests/test_table_name_bug_integration.py` - New test file

### Medium Priority (P1)
7. `rag_interactive.py` - Update query functionality
8. `test_retrieval_direct.py` - Update test script
9. `README.md` - Add warning
10. `docs/ENVIRONMENT_VARIABLES.md` - Update PGTABLE docs
11. `docs/TABLE_NAME_MIGRATION.md` - New migration guide

### Low Priority (P2)
12. All SQL queries using f-strings - Replace with `sql.Identifier()`
13. Add monitoring/logging throughout
14. Performance impact testing

---

**Report Generated**: 2026-01-08
**Next Review**: After P0 fixes implemented
**Follow-up**: Monitor for double-prefixed table creation in production
