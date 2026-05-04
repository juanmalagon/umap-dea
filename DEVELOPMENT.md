# Development Setup Guide

This guide explains how to set up the development environment for UMAP-DEA using pyenv and a virtual environment.

## Prerequisites

Ensure you have pyenv installed:

```bash
# macOS (with Homebrew)
brew install pyenv

# Linux
curl https://pyenv.run | bash

# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

## Setup Steps

### 1. Install the correct Python version

The project specifies Python 3.12.0 in `.python-version`. Install it via pyenv:

```bash
pyenv install 3.12.0
```

### 2. Create a virtual environment

Navigate to the project directory (pyenv will auto-detect `.python-version`):

```bash
cd /home/your-user/repos/umap-dea

# Create a virtualenv specifically for this project
pyenv virtualenv 3.12.0 umap-dea
```

### 3. Activate the virtual environment

```bash
# Auto-activate when entering directory (if you have pyenv-virtualenv)
pyenv local umap-dea

# Or manually activate
pyenv activate umap-dea
```

### 4. Install dependencies

The project uses `pyproject.toml` for dependency management:

```bash
# Install main dependencies + dev dependencies
pip install -e ".[dev]"

# Or just main dependencies (without dev tools)
pip install -e .

# Or with Jupyter support too
pip install -e ".[dev,jupyter]"
```

### 5. Verify installation

```bash
# Check that pytest is available
pytest --version

# Run the test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Alternative: Using Standard venv (if you prefer)

If you prefer the standard Python venv approach instead of pyenv-virtualenv:

```bash
# Create a standard virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"
```

**Note**: With standard venv, you'll need to manually activate/deactivate environments and won't get automatic activation when entering directories.

## Common Commands

```bash
# Activate the environment
pyenv activate umap-dea

# Deactivate the environment
pyenv deactivate

# List all virtual environments
pyenv virtualenvs

# Delete the virtual environment
pyenv virtualenv-delete umap-dea

# Run tests
pytest -v

# Format code with black
black src tests

# Sort imports with isort
isort src tests

# Run linting
flake8 src tests

# Type checking with mypy
mypy src
```

## Project Structure

```
pyproject.toml          # Project metadata and dependencies
.python-version         # Python version (3.12.0)
requirements.txt        # Legacy requirements (can be ignored if using pyproject.toml)
src/                    # Source code
tests/                  # Unit tests
```

## Dependency Groups

The project has three dependency groups:

1. **Core** — Main dependencies for running simulations
2. **dev** — Development tools (pytest, black, isort, flake8, mypy)
3. **jupyter** — Jupyter notebook support

Install subsets with:
```bash
pip install -e .              # Core only
pip install -e ".[dev]"       # Core + dev
pip install -e ".[jupyter]"   # Core + jupyter
pip install -e ".[dev,jupyter]"  # All
```

## Troubleshooting

### pyenv not found
Make sure you've added pyenv to your PATH in your shell profile.

### "pyenv: command not found"
Restart your terminal or run:
```bash
eval "$(pyenv init -)"
```

### Virtual environment not activating
Use `pyenv local umap-dea` in the project directory to auto-activate, or:
```bash
pyenv activate umap-dea
```

### Pip install fails
Ensure the virtual environment is active:
```bash
which python  # Should point to ~/.pyenv/versions/...
```

## Next Steps

- Run tests: `pytest`
- Run grid search: `python run_grid_search.py`
- Start Jupyter: `jupyter notebook`
