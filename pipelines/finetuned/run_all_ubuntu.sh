#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

# ── NVIDIA GPU environment ────────────────────────────────────────────────────
# Ollama detecteert de RTX 3070 automatisch via CUDA.
# Zorg dat NVIDIA drivers + CUDA toolkit geïnstalleerd zijn op de server.

# ── Configuratie ─────────────────────────────────────────────────────────────
TEKST_MODEL="qwen3-8b-finetuned"
VISION_MODEL="qwen3vl-8b-finetuned"
TEKST_MODELFILE="finetuning/text/Modelfile"
VISION_MODELFILE="finetuning/vision/Modelfile"
TEKST_GGUF="finetuning/text/Qwen3-8B.Q4_K_M.gguf"
VISION_GGUF="finetuning/vision/qwen3vl-8b-q4_k_m.gguf"

# ── GGUF bestanden controleren ────────────────────────────────────────────────
if [ ! -f "$TEKST_GGUF" ]; then
    echo "FOUT: tekst GGUF niet gevonden: $TEKST_GGUF"
    exit 1
fi

if [ ! -f "$VISION_GGUF" ]; then
    echo "FOUT: vision GGUF niet gevonden: $VISION_GGUF"
    exit 1
fi

# ── Tekst-model (Ollama) ──────────────────────────────────────────────────────
echo "=== Tekst-model controleren ==="
if ollama show "$TEKST_MODEL" > /dev/null 2>&1; then
    echo "    Model '$TEKST_MODEL' beschikbaar in Ollama."
else
    echo "    Model '$TEKST_MODEL' niet gevonden — aanmaken..."
    if [ ! -f "$TEKST_MODELFILE" ]; then
        echo "    FOUT: Modelfile niet gevonden: $TEKST_MODELFILE"
        exit 1
    fi
    (cd finetuning/text && ollama create "$TEKST_MODEL" -f Modelfile)
    echo "    Model '$TEKST_MODEL' aangemaakt."
fi

# ── Vision-model (Ollama) ─────────────────────────────────────────────────────
echo ""
echo "=== Vision-model controleren ==="
if ollama show "$VISION_MODEL" > /dev/null 2>&1; then
    echo "    Model '$VISION_MODEL' beschikbaar in Ollama."
else
    echo "    Model '$VISION_MODEL' niet gevonden — aanmaken..."
    if [ ! -f "$VISION_MODELFILE" ]; then
        echo "    FOUT: Modelfile niet gevonden: $VISION_MODELFILE"
        exit 1
    fi
    (cd finetuning/vision && ollama create "$VISION_MODEL" -f Modelfile)
    echo "    Model '$VISION_MODEL' aangemaakt."
fi

# ── Pipelines ─────────────────────────────────────────────────────────────────
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
