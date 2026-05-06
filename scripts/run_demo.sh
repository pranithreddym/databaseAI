#!/usr/bin/env bash
# run_demo.sh — Run all demos or a specific one

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

[ -d ".venv" ] && source .venv/bin/activate

DEMO="${1:-00}"

case "$DEMO" in
    00) python examples/00_full_demo.py ;;
    01) python examples/01_vector_db_demo.py ;;
    02) python examples/02_relational_db_demo.py ;;
    03) python examples/03_nosql_demo.py ;;
    04) python examples/04_feature_store_demo.py ;;
    05) python examples/05_rag_pipeline_demo.py ;;
    all)
        for i in 01 02 03 04 05; do
            echo ""
            echo "=============================="
            python "examples/0${i}_"*".py"
        done
        ;;
    *)
        echo "Usage: $0 [00|01|02|03|04|05|all]"
        exit 1
        ;;
esac
