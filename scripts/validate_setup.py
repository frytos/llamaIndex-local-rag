#!/usr/bin/env python3
"""Validate development environment setup."""

import subprocess
import sys
from pathlib import Path
from typing import Tuple


def run_command(cmd: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()


def validate_setup() -> bool:
    """Validate development environment setup."""
    print("Validating development environment setup...\n")

    all_checks_passed = True

    # Check configuration files
    print("1. Checking configuration files...")
    config_files = [
        "pyproject.toml",
        ".pre-commit-config.yaml",
        ".editorconfig",
        ".gitignore",
        "Makefile",
        "DEVELOPMENT.md",
    ]

    for config_file in config_files:
        exists = check_file_exists(config_file)
        status = "✓" if exists else "✗"
        print(f"   {status} {config_file}")
        if not exists:
            all_checks_passed = False

    # Check Python version
    print("\n2. Checking Python version...")
    success, output = run_command("python --version")
    if success:
        version = output.strip()
        print(f"   ✓ {version}")
        if "3.11" not in version and "3.12" not in version:
            print("   ⚠ Warning: Python 3.11+ recommended")
    else:
        print("   ✗ Python not found")
        all_checks_passed = False

    # Check installed tools
    print("\n3. Checking development tools...")
    tools = {
        "black": "black --version",
        "ruff": "ruff --version",
        "mypy": "mypy --version",
        "pytest": "pytest --version",
        "pre-commit": "pre-commit --version",
    }

    for tool_name, tool_cmd in tools.items():
        success, output = run_command(tool_cmd)
        if success:
            version = output.strip().split("\n")[0]
            print(f"   ✓ {tool_name}: {version}")
        else:
            print(f"   ✗ {tool_name}: Not installed")
            all_checks_passed = False

    # Check pre-commit hooks
    print("\n4. Checking pre-commit hooks...")
    if check_file_exists(".git/hooks/pre-commit"):
        print("   ✓ Pre-commit hooks installed")
    else:
        print("   ✗ Pre-commit hooks not installed")
        print("   → Run: pre-commit install")
        all_checks_passed = False

    # Check virtual environment
    print("\n5. Checking virtual environment...")
    success, output = run_command("python -c 'import sys; print(sys.prefix)'")
    if success and ".venv" in output:
        print("   ✓ Virtual environment active")
    else:
        print("   ⚠ Warning: Virtual environment may not be active")
        print("   → Run: source .venv/bin/activate")

    # Final summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ All checks passed! Development environment is ready.")
        print("\nNext steps:")
        print("  1. Run 'make format' to format code")
        print("  2. Run 'make lint' to check code quality")
        print("  3. Run 'make test' to run tests")
        print("  4. Start coding!")
        return True
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nTo install missing tools, run:")
        print("  pip install -e '.[dev]'")
        print("  pre-commit install")
        return False


if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)
