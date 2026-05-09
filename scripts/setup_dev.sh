#!/bin/bash
# Setup script for UMAP-DEA development environment

set -e

echo "UMAP-DEA Development Environment Setup"
echo "========================================"
echo ""

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv is not installed. Please install it first:"
    echo "   https://github.com/pyenv/pyenv#installation"
    exit 1
fi

echo "✓ pyenv found"

# Get Python version from .python-version
PYTHON_VERSION=$(cat .python-version)
echo "📌 Using Python version: $PYTHON_VERSION"

# Check if Python version is installed
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "📦 Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
else
    echo "✓ Python $PYTHON_VERSION already installed"
fi

# Create virtual environment (.venv)
VENV_NAME=".venv"
PYTHON_BIN="$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python"

if [[ -d "$VENV_NAME" ]]; then
    echo "✓ Virtual environment '$VENV_NAME' already exists"
else
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "❌ Python binary not found at $PYTHON_BIN"
        exit 1
    fi
    echo "🔨 Creating virtual environment '$VENV_NAME'..."
    "$PYTHON_BIN" -m venv "$VENV_NAME"
fi

echo ""
echo "✓ Virtual environment is ready!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment: source $VENV_NAME/bin/activate"
echo "   2. Install dependencies: pip install -e \".[dev]\""
echo "   3. Run tests: pytest"
echo ""
echo "Or just run:"
echo "   source $VENV_NAME/bin/activate"
echo "   pip install -e \".[dev]\""
echo "   pytest"
