.PHONY: help install install-dev install-jupyter test test-cov lint format clean venv

help:
	@echo "UMAP-DEA Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make venv              Create and configure pyenv virtual environment"
	@echo "  make install           Install core dependencies"
	@echo "  make install-dev       Install core + development dependencies"
	@echo "  make install-jupyter   Install core + Jupyter dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test              Run tests"
	@echo "  make test-cov          Run tests with coverage report"
	@echo "  make lint              Run code linting (black, isort, flake8)"
	@echo "  make format            Format code with black and isort"
	@echo "  make mypy              Run type checking with mypy"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean             Remove build artifacts and caches"
	@echo ""

venv:
	@echo "Setting up pyenv virtual environment..."
	@bash scripts/setup_dev.sh

install:
	@echo "Installing core dependencies..."
	pip install -e .

install-dev:
	@echo "Installing core + development dependencies..."
	pip install -e ".[dev]"

install-jupyter:
	@echo "Installing core + Jupyter dependencies..."
	pip install -e ".[jupyter]"

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=umap_dea --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	@echo "Checking code formatting with black..."
	black --check umap_dea tests
	@echo "Checking import sorting with isort..."
	isort --check-only umap_dea tests
	@echo "Running flake8..."
	flake8 umap_dea tests

format:
	@echo "Formatting code with black..."
	black umap_dea tests
	@echo "Sorting imports with isort..."
	isort umap_dea tests
	@echo "Code formatted!"

mypy:
	@echo "Running type checking with mypy..."
	mypy umap_dea

clean:
	@echo "Removing build artifacts..."
	rm -rf build/ dist/ *.egg-info
	@echo "Removing Python cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Removing pytest cache..."
	rm -rf .pytest_cache htmlcov .coverage
	@echo "Removing mypy cache..."
	rm -rf .mypy_cache
	@echo "Clean complete!"
