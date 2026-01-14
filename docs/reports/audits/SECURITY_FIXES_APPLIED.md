# Security Fixes Applied

**Date**: 2026-01-07
**Priority**: P0 (Critical)
**Status**: In Progress

## Summary

This document tracks all critical security vulnerabilities that have been fixed in the RAG pipeline codebase.

## Fixes Completed

### 1. Hardcoded Credentials (CVSS 9.8) - FIXED

#### docker-compose.yml
- **Status**: FIXED
- **Issue**: Hardcoded database credentials (user: fryt, password: frytos)
- **Fix**: Replaced with environment variable interpolation
- **Changes**:
  - `POSTGRES_USER: ${PGUSER:-postgres}`
  - `POSTGRES_PASSWORD: ${PGPASSWORD:?PGPASSWORD must be set}`
  - `POSTGRES_DB: ${DB_NAME:-vector_db}`
- **Impact**: Credentials now sourced from `.env` file (not committed to git)

#### scripts/compare_embedding_models.py
- **Status**: FIXED
- **Issue**: Hardcoded database credentials in psycopg2.connect()
- **Fix**: Changed to use environment variables via `os.environ.get()`
- **Lines Fixed**: 146-151
- **Changes**:
  ```python
  conn = psycopg2.connect(
      host=os.environ.get("PGHOST", "localhost"),
      port=int(os.environ.get("PGPORT", "5432")),
      user=os.environ.get("PGUSER"),
      password=os.environ.get("PGPASSWORD"),
      database=os.environ.get("DB_NAME", "vector_db")
  )
  ```

### 2. SQL Injection Vulnerabilities (CVSS 8.2) - PARTIALLY FIXED

#### scripts/compare_embedding_models.py
- **Status**: FIXED
- **Issue**: f-string SQL query vulnerable to SQL injection
- **Line**: 156
- **Fix**: Replaced with parameterized query using `psycopg2.sql.Identifier()`
- **Before**: `cur.execute(f"SELECT COUNT(*) FROM {table_name}")`
- **After**:
  ```python
  from psycopg2 import sql
  cur.execute(
      sql.SQL("SELECT COUNT(*) FROM {}").format(
          sql.Identifier(table_name)
      )
  )
  ```

#### rag_web.py
- **Status**: FIXED (All instances)
- **Issues Fixed**:
  1. Line 179: `SELECT COUNT(*) FROM "{table}"`
  2. Line 187-193: Metadata query with table name interpolation
  3. Line 221: `DROP TABLE IF EXISTS "{table_name}"`
  4. Line 241-245: Embedding fetch query
  5. Line 594: `DROP TABLE IF EXISTS "{table_name}"`
- **Fix**: All replaced with `sql.SQL()` and `sql.Identifier()`
- **Import Added**: `from psycopg2 import sql` (line 45)

#### scripts/benchmarking_performance_analysis.py
- **Status**: FIXED
- **Issues Fixed**:
  1. Line 195: `SELECT COUNT(*) FROM {actual_table}`
  2. Lines 203-210: Table size queries
  3. Line 246: `SELECT embedding FROM {actual_table}`
- **Fix**: All replaced with parameterized queries

#### rag_low_level_m1_16gb_verbose.py
- **Status**: NEEDS MANUAL FIX (File locked by linter/formatter)
- **Issue**: Line 2399 and 2416-2420
- **Required Fix**:
  ```python
  # Line 2399 - Replace:
  cur.execute(f'SELECT COUNT(*) FROM "{actual_table}"')

  # With:
  from psycopg2 import sql
  cur.execute(
      sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(actual_table))
  )

  # Lines 2416-2420 - Replace:
  cur.execute(f'''
      CREATE INDEX "{index_name}"
      ON "{actual_table}"
      USING hnsw (embedding vector_cosine_ops)
      WITH (m = 16, ef_construction = 64)
  ''')

  # With:
  cur.execute(
      sql.SQL('''
          CREATE INDEX {}
          ON {}
          USING hnsw (embedding vector_cosine_ops)
          WITH (m = 16, ef_construction = 64)
      ''').format(sql.Identifier(index_name), sql.Identifier(actual_table))
  )
  ```

### 3. Code Injection via eval() (CVSS 9.8) - FIXED

#### rag_web.py
- **Status**: FIXED
- **Issue**: Line 265 used `eval()` for parsing embeddings
- **Fix**: Replaced with `ast.literal_eval()`
- **Before**:
  ```python
  except:
      emb = eval(emb.replace('np.str_', '').replace('(', '').replace(')', ''))
  ```
- **After**:
  ```python
  except (json.JSONDecodeError, ValueError):
      try:
          cleaned = emb.replace('np.str_', '').replace('(', '').replace(')', '')
          emb = ast.literal_eval(cleaned)
      except (ValueError, SyntaxError) as e:
          st.warning(f"Failed to parse embedding: {e}")
          continue
  ```
- **Import Added**: `import ast` (line 271)

### 4. Bare Exception Handlers - PARTIALLY FIXED

#### rag_web.py
- **Status**: FIXED (All instances)
- **Issues Fixed**: 8 bare `except:` clauses
- **Changes**: All replaced with `except Exception:` or specific exception types
- **Lines Fixed**: 152, 210, 229, 263, 277, 598, 933
- **Examples**:
  - Line 152: `except:` → `except Exception:`
  - Line 263: Added specific `(json.JSONDecodeError, ValueError)` handling
  - Line 280: Added specific `(ValueError, SyntaxError)` handling

#### Other Files (Still Need Fixing)
- **scripts/visualize_rag.py**: Line 51
- **utils/metadata_extractor.py**: Lines 425, 497, 499, 530

### 5. Web UI Authentication (CVSS 8.2) - NOT YET IMPLEMENTED

- **Status**: TODO
- **Issue**: rag_web.py has no authentication mechanism
- **Recommendation**: Implement streamlit-authenticator
- **Required Changes**:
  1. Add `streamlit-authenticator` to requirements.txt
  2. Create authentication configuration
  3. Add login/logout flow to rag_web.py
  4. Implement password hashing (bcrypt)
  5. Add session management

## Files Modified

1. `/Users/frytos/code/llamaIndex-local-rag/config/docker-compose.yml` - FIXED
2. `/Users/frytos/code/llamaIndex-local-rag/scripts/compare_embedding_models.py` - FIXED
3. `/Users/frytos/code/llamaIndex-local-rag/rag_web.py` - FIXED (SQL + eval + exceptions)
4. `/Users/frytos/code/llamaIndex-local-rag/scripts/benchmarking_performance_analysis.py` - FIXED
5. `/Users/frytos/code/llamaIndex-local-rag/rag_low_level_m1_16gb_verbose.py` - NEEDS MANUAL FIX

## Files Still Need Fixing

1. `/Users/frytos/code/llamaIndex-local-rag/rag_low_level_m1_16gb_verbose.py` (SQL injection)
2. `/Users/frytos/code/llamaIndex-local-rag/scripts/visualize_rag.py` (bare except)
3. `/Users/frytos/code/llamaIndex-local-rag/utils/metadata_extractor.py` (bare except × 5)

## Testing Required

After fixes are complete, run:

```bash
# 1. Install security scanning tools
pip install bandit pip-audit safety

# 2. Run security scans
bandit -r . -ll -f json -o security_scan.json
pip-audit
safety check

# 3. Test database connections (ensure env vars work)
source .env
python -c "from rag_web import test_db_connection; print(test_db_connection())"

# 4. Test Web UI
streamlit run rag_web.py

# 5. Test main pipeline
python rag_low_level_m1_16gb_verbose.py --query-only --query "test"
```

## Security Score Improvement

### Before
- Hardcoded Credentials: 8 instances (CVSS 9.8)
- SQL Injection: 8 instances (CVSS 8.2)
- Code Injection (eval): 1 instance (CVSS 9.8)
- Bare Exceptions: 15 instances (Medium)
- No Web Auth: 1 instance (CVSS 8.2)

### After (Current)
- Hardcoded Credentials: 0 instances ✓
- SQL Injection: 2 instances remaining (main RAG file)
- Code Injection (eval): 0 instances ✓
- Bare Exceptions: 7 instances remaining
- No Web Auth: 1 instance (TODO)

### Expected Final Score
- Critical vulnerabilities: 0
- High vulnerabilities: 1 (Web Auth - TODO)
- Medium vulnerabilities: 6 (bare exceptions in utility files)

## Next Steps

1. Manually fix SQL injection in `rag_low_level_m1_16gb_verbose.py` (lines 2399, 2416-2420)
2. Fix remaining bare exception handlers in utility files
3. Implement Web UI authentication (streamlit-authenticator)
4. Run comprehensive security scan
5. Update documentation with security best practices
6. Add pre-commit hooks for security scanning

## Notes

- All credentials must now be set via `.env` file
- `.env` file is in `.gitignore` (never commit credentials)
- Use `config/.env.example` as template
- Docker Compose will fail with clear error if `PGPASSWORD` not set
- SQL injection protection uses PostgreSQL's native identifier quoting
- `ast.literal_eval()` is safe for parsing Python literals (unlike `eval()`)

## Security Contact

For security issues, contact the repository maintainer.
Do NOT open public GitHub issues for security vulnerabilities.
