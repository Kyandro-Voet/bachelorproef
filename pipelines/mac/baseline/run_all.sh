#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Mac Baseline — run alle 3 pipelines
#  Uitvoeren vanuit de project root:
#    bash pipelines/mac/baseline/run_all.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

TEKST_MODEL="qwen3:8b-q4_K_M"
VISIE_MODEL="qwen3-vl:8b"

echo "========================================================"
echo "  Mac Baseline — starten"
echo "========================================================"

# ── Controleer of Ollama draait ────────────────────────────────
if ! ollama list &>/dev/null; then
    echo "FOUT: Ollama reageert niet. Start Ollama eerst."
    exit 1
fi

# ── Tekst model ────────────────────────────────────────────────
echo ""
echo "  Tekst model: $TEKST_MODEL"
if ollama show "$TEKST_MODEL" &>/dev/null; then
    echo "  → Reeds beschikbaar."
else
    echo "  → Downloaden via 'ollama pull'..."
    ollama pull "$TEKST_MODEL"
fi

# ── Visie model ────────────────────────────────────────────────
echo ""
echo "  Visie model: $VISIE_MODEL"
if ollama show "$VISIE_MODEL" &>/dev/null; then
    echo "  → Reeds beschikbaar."
else
    echo "  → Downloaden via 'ollama pull'..."
    ollama pull "$VISIE_MODEL"
fi

# ── Pipelines uitvoeren ────────────────────────────────────────
echo ""
echo "========================================================"
echo "  [1/3] Tekstpipeline"
echo "========================================================"
uv run python pipelines/mac/baseline/tekst.py

echo ""
echo "========================================================"
echo "  [2/3] Visiepipeline"
echo "========================================================"
uv run python pipelines/mac/baseline/visie.py

echo ""
echo "========================================================"
echo "  [3/3] Hybride pipeline"
echo "========================================================"
uv run python pipelines/mac/baseline/hybride.py

echo ""
echo "========================================================"
echo "  Mac Baseline klaar — resultaten in resultaten/mac/baseline/"
echo "========================================================"
