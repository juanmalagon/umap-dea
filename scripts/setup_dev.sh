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

# Create or activate virtual environment
VENV_NAME="umap-dea"
if pyenv virtualenvs | grep -q "$VENV_NAME"; then
    echo "✓ Virtual environment '$VENV_NAME' already exists"
else
    echo "🔨 Creating virtual environment '$VENV_NAME'..."
    pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"
fi

echo "⚙️  Setting local pyenv version..."
pyenv local "$VENV_NAME"

echo ""
echo "✓ Virtual environment is ready!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment: eval \"\$(pyenv init -)\" && pyenv activate $VENV_NAME"
echo "   2. Install dependencies: pip install -e \".[dev]\""
echo "   3. Run tests: pytest"
echo ""
echo "Or just run:"
echo "   pip install -e \".[dev]\""
echo "   pytest"
