#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

echo "=== Baseline pipeline: tekst ==="
python pipelines/baseline/tekst.py

echo ""
echo "=== Baseline pipeline: visie ==="
python pipelines/baseline/visie.py

echo ""
echo "=== Baseline pipeline: hybride ==="
python pipelines/baseline/hybride.py

echo ""
echo "=== Alle baseline pipelines klaar ==="
