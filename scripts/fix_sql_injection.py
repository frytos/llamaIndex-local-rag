#!/usr/bin/env python3
"""
Security Fix Script: SQL Injection in rag_low_level_m1_16gb_verbose.py

This script applies the remaining SQL injection fixes to the main RAG file.

Run with: python scripts/fix_sql_injection.py
"""

import re
from pathlib import Path

def fix_sql_injection():
    """Fix SQL injection vulnerabilities in rag_low_level_m1_16gb_verbose.py."""

    file_path = Path(__file__).parent.parent / "rag_low_level_m1_16gb_verbose.py"

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False

    print(f"Reading {file_path}...")
    content = file_path.read_text()
    original_content = content

    # Fix 1: Line ~2399 - COUNT query
    old_pattern_1 = r"cur\.execute\(f'SELECT COUNT\(\*\) FROM \"{actual_table}\"'\)"
    new_code_1 = """from psycopg2 import sql
            cur.execute(
                sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(actual_table))
            )"""

    if re.search(old_pattern_1, content):
        print("✓ Found SQL injection vulnerability #1 (COUNT query)")
        content = re.sub(
            old_pattern_1,
            new_code_1,
            content
        )
    else:
        print("✗ Pattern #1 not found (may already be fixed)")

    # Fix 2: Lines ~2416-2420 - CREATE INDEX query
    old_pattern_2 = r'''cur\.execute\(f\'\'\'
\s+CREATE INDEX "{index_name}"
\s+ON "{actual_table}"
\s+USING hnsw \(embedding vector_cosine_ops\)
\s+WITH \(m = 16, ef_construction = 64\)
\s+\'\'\'[\s\)]*\)'''

    new_code_2 = """cur.execute(
                sql.SQL('''
                    CREATE INDEX {}
                    ON {}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                ''').format(sql.Identifier(index_name), sql.Identifier(actual_table))
            )"""

    if re.search(old_pattern_2, content, re.MULTILINE):
        print("✓ Found SQL injection vulnerability #2 (CREATE INDEX)")
        content = re.sub(
            old_pattern_2,
            new_code_2,
            content,
            flags=re.MULTILINE
        )
    else:
        print("✗ Pattern #2 not found (may already be fixed)")

    # Check if any changes were made
    if content == original_content:
        print("\n❌ No changes made - vulnerabilities may already be fixed or patterns not matched")
        print("Please check the file manually and apply fixes as documented in SECURITY_FIXES_APPLIED.md")
        return False

    # Create backup
    backup_path = file_path.with_suffix('.py.backup')
    print(f"\nCreating backup: {backup_path}")
    backup_path.write_text(original_content)

    # Write fixed content
    print(f"Writing fixes to {file_path}")
    file_path.write_text(content)

    print("\n✅ SQL injection fixes applied successfully!")
    print("\nNext steps:")
    print("1. Review the changes:")
    print(f"   diff {backup_path} {file_path}")
    print("2. Test the application:")
    print("   python rag_low_level_m1_16gb_verbose.py --query-only --query 'test'")
    print("3. Run security scan:")
    print("   bandit -r . -ll")
    print("4. If everything works, delete the backup:")
    print(f"   rm {backup_path}")

    return True


def verify_import():
    """Verify that psycopg2.sql import is present."""
    file_path = Path(__file__).parent.parent / "rag_low_level_m1_16gb_verbose.py"
    content = file_path.read_text()

    imports = [
        "from psycopg2 import sql",
        "import psycopg2.sql",
    ]

    has_import = any(imp in content for imp in imports)

    if not has_import:
        print("\n⚠️  WARNING: psycopg2.sql import not found!")
        print("Add this import near the top of the file:")
        print("  from psycopg2 import sql")
        return False

    print("✓ psycopg2.sql import present")
    return True


def main():
    """Main entry point."""
    print("="*70)
    print("SQL Injection Fix Script")
    print("="*70)
    print()

    # Verify import
    verify_import()
    print()

    # Apply fixes
    success = fix_sql_injection()

    if success:
        print("\n" + "="*70)
        print("SECURITY FIX COMPLETE")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("MANUAL FIX REQUIRED")
        print("="*70)
        print("\nSee SECURITY_FIXES_APPLIED.md for manual fix instructions")


if __name__ == "__main__":
    main()
