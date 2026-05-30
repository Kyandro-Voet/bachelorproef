#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

TEKST_MODEL="qwen3-8b-finetuned"
MODELFILE_PAD="finetuning/text/Modelfile"
MLX_MODEL_PAD="finetuning/vision/mlx_model"
CONVERT_SCRIPT="finetuning/vision/convert_mlx.py"

# ── Tekst-model (Ollama) ──────────────────────────────────────────────────────
echo "=== Tekst-model controleren ==="
if ollama show "$TEKST_MODEL" > /dev/null 2>&1; then
    echo "    Model '$TEKST_MODEL' beschikbaar in Ollama."
else
    echo "    Model '$TEKST_MODEL' niet gevonden — aanmaken..."
    if [ ! -f "$MODELFILE_PAD" ]; then
        echo "    FOUT: Modelfile niet gevonden: $MODELFILE_PAD"
        exit 1
    fi
    ollama create "$TEKST_MODEL" -f "$MODELFILE_PAD"
    echo "    Model '$TEKST_MODEL' aangemaakt."
fi

# ── Vision-model (MLX) ───────────────────────────────────────────────────────
echo ""
echo "=== Vision-model controleren ==="
if [ -d "$MLX_MODEL_PAD" ]; then
    echo "    MLX model beschikbaar: $MLX_MODEL_PAD"
else
    echo "    MLX model niet gevonden — conversie starten via $CONVERT_SCRIPT..."
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        echo "    FOUT: Conversiescript niet gevonden: $CONVERT_SCRIPT"
        echo "    TIP: Op Ubuntu/Linux gebruik run_all_ubuntu.sh in plaats van dit script."
        exit 1
    fi
    python "$CONVERT_SCRIPT"
    echo "    MLX model aangemaakt."
fi

# ── Pipelines ────────────────────────────────────────────────────────────────
echo ""
echo "=== Finetuned pipeline: tekst ==="
python pipelines/finetuned/tekst.py

echo ""
echo "=== Finetuned pipeline: visie ==="
python pipelines/finetuned/visie.py

echo ""
echo "=== Finetuned pipeline: hybride ==="
python pipelines/finetuned/hybride.py

echo ""
echo "=== Alle finetuned pipelines klaar ==="
