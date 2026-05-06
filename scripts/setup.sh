#!/usr/bin/env bash
# setup.sh — Create virtual env and install all dependencies

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== databaseAI Setup ==="
echo "Project: $PROJECT_DIR"

# Create venv if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Upgrade pip silently
pip install --upgrade pip --quiet

# Install project in editable mode with dev extras
echo "Installing dependencies..."
pip install -e ".[dev]" --quiet

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source .venv/bin/activate"
echo "Run demo:       python examples/00_full_demo.py"
echo "Run tests:      pytest tests/ -v"
