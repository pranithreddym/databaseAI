#!/usr/bin/env bash
# run_tests.sh — Run the full test suite with coverage

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

[ -d ".venv" ] && source .venv/bin/activate

echo "=== Running databaseAI Test Suite ==="
pytest tests/ -v --tb=short --cov=src/databaseai --cov-report=term-missing "$@"
