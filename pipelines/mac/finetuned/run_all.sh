#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Mac Finetuned — run alle 3 pipelines
#  Uitvoeren vanuit de project root:
#    bash pipelines/mac/finetuned/run_all.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

TEKST_MODEL="qwen3-8b-finetuned"
TEKST_MODELFILE="finetuning/text/Modelfile"
TEKST_GGUF="finetuning/text/Qwen3-8B.Q4_K_M.gguf"
MLX_MODEL_PAD="finetuning/vision/mlx_model"
MLX_CONVERT_SCRIPT="finetuning/vision/convert_mlx.py"

echo "========================================================"
echo "  Mac Finetuned — starten"
echo "========================================================"

# ── Controleer of Ollama draait ────────────────────────────────
if ! ollama list &>/dev/null; then
    echo "FOUT: Ollama reageert niet. Start Ollama eerst."
    exit 1
fi

# ── Finetuned tekst model (Ollama) ────────────────────────────
echo ""
echo "  Tekst model: $TEKST_MODEL"
if ollama show "$TEKST_MODEL" &>/dev/null; then
    echo "  → Reeds beschikbaar in Ollama."
else
    echo "  → Niet gevonden. Aanmaken..."
    if [ ! -f "$TEKST_GGUF" ]; then
        echo "  FOUT: GGUF niet gevonden: $TEKST_GGUF"
        echo "  Voer eerst de finetuning uit of kopieer het GGUF-bestand."
        exit 1
    fi
    if [ ! -f "$TEKST_MODELFILE" ]; then
        echo "  FOUT: Modelfile niet gevonden: $TEKST_MODELFILE"
        exit 1
    fi
    (cd finetuning/text && ollama create "$TEKST_MODEL" -f Modelfile)
    echo "  → Model '$TEKST_MODEL' aangemaakt."
fi

# ── MLX vision model (Apple Silicon) ─────────────────────────
echo ""
echo "  Vision model (MLX): $MLX_MODEL_PAD"
if [ -d "$MLX_MODEL_PAD" ]; then
    echo "  → MLX model map gevonden."
else
    echo "  → MLX model niet gevonden. Converteren..."
    if [ ! -f "$MLX_CONVERT_SCRIPT" ]; then
        echo "  FOUT: convert_mlx.py niet gevonden: $MLX_CONVERT_SCRIPT"
        exit 1
    fi
    uv run python "$MLX_CONVERT_SCRIPT"
    if [ ! -d "$MLX_MODEL_PAD" ]; then
        echo "  FOUT: MLX model aanmaken mislukt."
        exit 1
    fi
    echo "  → MLX model aangemaakt: $MLX_MODEL_PAD"
fi

# ── Pipelines uitvoeren ────────────────────────────────────────
echo ""
echo "========================================================"
echo "  [1/3] Tekstpipeline (finetuned)"
echo "========================================================"
uv run python pipelines/mac/finetuned/tekst.py

echo ""
echo "========================================================"
echo "  [2/3] Visiepipeline (finetuned, MLX)"
echo "========================================================"
uv run python pipelines/mac/finetuned/visie.py

echo ""
echo "========================================================"
echo "  [3/3] Hybride pipeline (finetuned)"
echo "========================================================"
uv run python pipelines/mac/finetuned/hybride.py

echo ""
echo "========================================================"
echo "  Mac Finetuned klaar — resultaten in resultaten/mac/finetuned/"
echo "========================================================"
