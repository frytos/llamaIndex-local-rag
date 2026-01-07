#!/usr/bin/env python3
"""
Comprehensive dependency checker for Local RAG Pipeline.

Verifies all required dependencies, versions, and system requirements.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version >= 3.11"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"  ❌ Python {version.major}.{version.minor} detected. Requires Python 3.11+")
        return False
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package.replace("-", "_")
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def check_requirements():
    """Check core requirements"""
    print("\nChecking core dependencies...")

    required_packages = {
        "llama-index": "llama_index",
        "sentence-transformers": "sentence_transformers",
        "psycopg2-binary": "psycopg2",
        "pgvector": "pgvector",
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "streamlit": "streamlit",
        "openai": "openai",
    }

    missing = []
    for package, import_name in required_packages.items():
        if check_package(package, import_name):
            print(f"  ✅ {package}")
        else:
            print(f"  ❌ {package} - missing")
            missing.append(package)

    return len(missing) == 0, missing

def check_optional_dependencies():
    """Check optional dependencies"""
    print("\nChecking optional dependencies...")

    optional_packages = {
        "vllm": "vllm",
        "mlx": "mlx",
        "rank-bm25": "rank_bm25",
    }

    for package, import_name in optional_packages.items():
        if check_package(package, import_name):
            print(f"  ✅ {package}")
        else:
            print(f"  ⚠️  {package} - optional, not installed")

def check_gpu_support():
    """Check GPU availability"""
    print("\nChecking GPU support...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif torch.backends.mps.is_available():
            print("  ✅ Apple Metal (MPS) available")
        else:
            print("  ⚠️  No GPU detected, will use CPU")
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")

def check_database():
    """Check PostgreSQL connection"""
    print("\nChecking database connection...")

    try:
        import psycopg2
        import os
        from dotenv import load_dotenv

        load_dotenv()

        conn_params = {
            "host": os.getenv("PGHOST", "localhost"),
            "port": os.getenv("PGPORT", "5432"),
            "user": os.getenv("PGUSER"),
            "password": os.getenv("PGPASSWORD"),
            "database": os.getenv("DB_NAME", "vector_db"),
        }

        if not conn_params["user"] or not conn_params["password"]:
            print("  ⚠️  Database credentials not set in .env")
            print("     Copy config/.env.example to .env and configure")
            return

        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"  ✅ PostgreSQL connected")
        print(f"     Version: {version.split(',')[0]}")

        # Check pgvector
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("  ✅ pgvector extension installed")
        else:
            print("  ❌ pgvector extension not installed")
            print("     Run: CREATE EXTENSION vector;")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        print("     Make sure PostgreSQL is running:")
        print("     docker-compose -f config/docker-compose.yml up -d")

def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")

    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)

        if free_gb < 5:
            print(f"  ⚠️  Low disk space: {free_gb:.1f} GB free (recommend 5+ GB)")
        else:
            print(f"  ✅ Disk space: {free_gb:.1f} GB free")
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")

def check_model_cache():
    """Check if models are cached"""
    print("\nChecking model cache...")

    cache_dir = Path.home() / ".cache" / "llama_index"
    if cache_dir.exists():
        models = list(cache_dir.glob("models/*.gguf"))
        if models:
            print(f"  ✅ Found {len(models)} cached model(s)")
            for model in models:
                size_mb = model.stat().st_size / (1024**2)
                print(f"     {model.name} ({size_mb:.0f} MB)")
        else:
            print("  ⚠️  No cached models found")
            print("     Models will be downloaded on first run")
    else:
        print("  ⚠️  Cache directory doesn't exist")

def main():
    """Run all checks"""
    print("=" * 60)
    print("Local RAG Pipeline - Dependency Checker")
    print("=" * 60)

    checks_passed = True

    # Required checks
    if not check_python_version():
        checks_passed = False

    requirements_ok, missing = check_requirements()
    if not requirements_ok:
        checks_passed = False
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")

    # Optional/informational checks
    check_optional_dependencies()
    check_gpu_support()
    check_database()
    check_disk_space()
    check_model_cache()

    print("\n" + "=" * 60)
    if checks_passed:
        print("✅ All required dependencies are installed!")
        print("You can now run: python rag_low_level_m1_16gb_verbose.py")
    else:
        print("❌ Some required dependencies are missing")
        print("Please install missing packages and try again")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
