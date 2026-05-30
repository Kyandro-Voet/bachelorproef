#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=== Baseline pipeline (Mac): tekst ==="
python pipelines/mac/tekst.py

echo ""
echo "=== Baseline pipeline (Mac): visie ==="
python pipelines/mac/visie.py

echo ""
echo "=== Baseline pipeline (Mac): hybride ==="
python pipelines/mac/hybride.py

echo ""
echo "=== Alle Mac-pipelines klaar ==="
