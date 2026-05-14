"""
Dataset voorbereiding voor QLoRA fine-tuning.

Verwacht een bestand data/train.jsonl met regels zoals:
    {"input": "<ruwe OCR tekst>", "output": "<correcte JSON string>"}

Uitvoer: Hugging Face Dataset objecten klaar voor SFTTrainer.
"""

import json
from pathlib import Path
from datasets import Dataset


SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "Your sole task is to extract structured information from utility/resource documents "
    "and return it as valid JSON. "
    "Return ONLY a valid JSON object. No explanation, no markdown, no code fences."
)


def _formatteer_als_llama3_chat(ocr_tekst: str, json_output: str) -> str:
    """Formatteer een voorbeeld als Llama 3 chat template.

    Het 'ocr_tekst' veld bevat al de volledige user-prompt inclusief schema-instructie,
    zoals gegenereerd door maak_trainingsdata.py.
    """
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{ocr_tekst}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{json_output}"
        "<|eot_id|>"
    )


def laad_jsonl(pad: Path) -> list[dict]:
    """Lees een JSONL bestand en geef een lijst van dicts terug."""
    voorbeelden = []
    with open(pad, encoding="utf-8") as f:
        for regelnummer, regel in enumerate(f, start=1):
            regel = regel.strip()
            if not regel:
                continue
            try:
                item = json.loads(regel)
                assert "input" in item and "output" in item, \
                    f"Regel {regelnummer}: velden 'input' en 'output' verplicht"
                voorbeelden.append(item)
            except (json.JSONDecodeError, AssertionError) as e:
                print(f"⚠️  Regel {regelnummer} overgeslagen: {e}")
    return voorbeelden


def maak_dataset(jsonl_pad: Path) -> Dataset:
    """Laad JSONL en converteer naar Hugging Face Dataset met 'text' kolom."""
    voorbeelden = laad_jsonl(jsonl_pad)
    teksten = [
        _formatteer_als_llama3_chat(v["input"], v["output"])
        for v in voorbeelden
    ]
    dataset = Dataset.from_dict({"text": teksten})
    print(f"✅ Dataset geladen: {len(dataset)} voorbeelden uit {jsonl_pad}")
    return dataset


def splits_dataset(dataset: Dataset, val_fractie: float = 0.1):
    """Splits in train/validatie sets."""
    splits = dataset.train_test_split(test_size=val_fractie, seed=42)
    print(f"   Train: {len(splits['train'])}  |  Validatie: {len(splits['test'])}")
    return splits["train"], splits["test"]


if __name__ == "__main__":
    data_map = Path("data")
    train_pad = data_map / "train.jsonl"

    if not train_pad.exists():
        print(f"❌ Bestand niet gevonden: {train_pad}")
        print("   Maak data/train.jsonl aan met de structuur:")
        print('   {"input": "<ocr tekst>", "output": "<json string>"}')
        raise SystemExit(1)

    dataset = maak_dataset(train_pad)
    train_ds, val_ds = splits_dataset(dataset)

    # Toon een voorbeeld
    print("\n--- Voorbeeld (eerste 300 tekens) ---")
    print(train_ds[0]["text"][:300])
