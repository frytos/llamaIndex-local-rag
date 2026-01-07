.PHONY: help install install-dev setup format lint type-check test test-cov clean pre-commit run-interactive run-web

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make setup         - Full setup (install-dev + pre-commit)"
	@echo "  make format        - Format code with black and ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make type-check    - Run type checking with mypy"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make clean         - Clean cache and temporary files"
	@echo "  make pre-commit    - Run pre-commit on all files"
	@echo "  make run-interactive - Run interactive CLI"
	@echo "  make run-web       - Run Streamlit web UI"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

setup: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# Code quality
format:
	black .
	ruff check --fix .
	ruff format .

lint:
	ruff check .

type-check:
	mypy .

# Testing
test:
	pytest

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term-missing

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "Cleaned cache and temporary files"

# Run application
run-interactive:
	python rag_interactive.py

run-web:
	streamlit run rag_web.py

# Database operations
db-test:
	PGPASSWORD=frytos psql -h localhost -U fryt -d vector_db -c "SELECT version();"

# Quick checks (pre-push)
check: format lint type-check test
	@echo "All checks passed!"
