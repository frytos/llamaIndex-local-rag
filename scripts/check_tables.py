#!/usr/bin/env python3
"""Check what tables exist in the database."""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    dbname=os.getenv("DB_NAME", "vector_db")
)

cur = conn.cursor()

# List all tables with 'agathe' in the name
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name LIKE '%agathe%'
    ORDER BY table_name;
""")

tables = cur.fetchall()

print("\n📊 Tables with 'agathe' in name:")
print("=" * 80)
if not tables:
    print("  (none found)")
else:
    for (table_name,) in tables:
        # Count rows
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
        print(f"  • {table_name} ({count:,} rows)")

# Also check what the last index created was
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name LIKE 'data_%'
    AND table_name LIKE '%cs700%'
    ORDER BY table_name DESC
    LIMIT 5;
""")

print("\n📚 Recent indexes (cs700):")
print("=" * 80)
recent = cur.fetchall()
for (table_name,) in recent:
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    print(f"  • {table_name} ({count:,} rows)")

cur.close()
conn.close()
