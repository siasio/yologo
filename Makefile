.PHONY: help setup install test test-cov format lint clean clean-all

# Default target
help:
	@echo "YoloGo - Makefile Commands"
	@echo "=========================="
	@echo "setup         : Create virtual environment"
	@echo "install       : Install dependencies"
	@echo "test          : Run tests"
	@echo "test-cov      : Run tests with coverage report"
	@echo "format        : Format code with black"
	@echo "lint          : Run ruff linter"
	@echo "clean         : Remove Python artifacts and caches"
	@echo "clean-all     : Remove everything including venv and outputs"

# Create virtual environment
setup:
	python3 -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

# Format code
format:
	black data/ models/ training/ evaluation/ export/ utils/ scripts/ tests/
	@echo "Code formatted with black"

# Run linter
lint:
	ruff check data/ models/ training/ evaluation/ export/ utils/ scripts/ tests/
	@echo "Linting complete"

# Clean Python artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.py,cover" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -f .coverage
	@echo "Cleaned Python artifacts"

# Clean everything
clean-all: clean
	rm -rf .venv/
	rm -rf logs/
	rm -rf outputs/
	@echo "Cleaned everything including venv and outputs"
