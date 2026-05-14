"""
Genereer data/vision_train.jsonl vanuit de PDF+JSON paren in documents/.

Voor elke PDF:
  1. Converteer pagina 1 naar JPEG (200 DPI) → data/vision_images/<categorie>/
  2. Bouw user-prompt op met het categorie-specifieke schema
  3. Schrijf {"image": pad, "input": prompt, "output": ground-truth JSON} naar vision_train.jsonl

Gebruik:
    uv run python maak_trainingsdata_visie.py
"""

import json
import sys
import time
from pathlib import Path

from pdf2image import convert_from_path


# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
DPI = 200
JPEG_KWALITEIT = 90
DOCUMENTS_MAP = Path("documents_training")
DATA_MAP = Path("data")
AFBEELDINGEN_MAP = DATA_MAP / "vision_images"
UITVOER_PAD = DATA_MAP / "vision_train.jsonl"

# Zelfde schemas als vision_pipeline.py
CATEGORIE_SCHEMAS = {
    "electricity": """{
  "consumi": [
    {
      "codice": "<string: POD/contract code, e.g. IT001E...>",
      "consumo": <number: total consumption in kWh>,
      "indirizzo": "<string: full delivery address>",
      "consumo_f1": <number or null: F1 peak consumption>,
      "consumo_f2": <number or null: F2 off-peak consumption>,
      "consumo_f3": <number or null: F3 night consumption>,
      "giorno_inizio": "<YYYY-MM-DD: period start date>",
      "giorno_fine": "<YYYY-MM-DD: period end date>",
      "costo_periodo": <number or null: total cost for the period>
    }
  ]
}""",
    "water": """{
  "consumi": [
    {
      "codice": "<string: meter/contract code>",
      "consumo": <number: total water consumption in m³>,
      "indirizzo": "<string: full delivery address>",
      "consumo_medio": <number or null: average daily consumption>,
      "giorno_inizio": "<YYYY-MM-DD: period start date>",
      "giorno_fine": "<YYYY-MM-DD: period end date>",
      "costo_periodo": <number or null: total cost for the period>
    }
  ]
}""",
    "natural gas": """{
  "consumi": [
    {
      "codice": "<string: PDR/contract code>",
      "consumo": <number: total gas consumption in Sm³>,
      "indirizzo": "<string: full delivery address>",
      "giorno_inizio": "<YYYY-MM-DD: period start date>",
      "giorno_fine": "<YYYY-MM-DD: period end date>",
      "costo_periodo": <number or null: total cost for the period>
    }
  ]
}""",
    "waste": """{
  "rifiuti": [
    {
      "anno": <number: year>,
      "tipo": "<string or null: waste type description>",
      "quantita": <number: quantity in kg>,
      "codice_cer": "<string: European Waste Catalogue code, e.g. 020201>",
      "codice_smaltimento": "<string or null: disposal/recovery code, e.g. R13>"
    }
  ]
}""",
    "fuels": """{
  "fatture": [
    {
      "um": "<string: unit of measure, e.g. L for liters>",
      "codice": "<string: invoice/transaction code>",
      "prezzo": <number: total price>,
      "quantita": <number: quantity purchased>,
      "tipologia": "<string: fuel type, e.g. GASOLIO, EURO 95>",
      "giorno_inizio": "<YYYY-MM-DD: transaction date>",
      "energia_fonte": <number or null: energy content per unit>,
      "energia_unitaria": "<string or null: energy unit>",
      "carbonfootprint_fonte": <number or null: carbon footprint value>,
      "carbonfootprint_unitaria": "<string or null: carbon footprint unit>"
    }
  ]
}""",
}

USER_PROMPT_TEMPLATE = """Extract all data from the document image and return it as a JSON object
that strictly follows this schema:

{schema}

Reply ONLY with valid JSON. No extra text."""


# ──────────────────────────────────────────────
# BEELDVERWERKING
# ──────────────────────────────────────────────
def pdf_naar_jpeg(pdf_pad: Path, uitvoer_pad: Path) -> Path:
    """Converteer pagina 1 van een PDF naar JPEG en sla op. Geeft het uitvoerpad terug."""
    paginas = convert_from_path(str(pdf_pad), dpi=DPI, first_page=1, last_page=1)
    paginas[0].save(str(uitvoer_pad), "JPEG", quality=JPEG_KWALITEIT)
    return uitvoer_pad


# ──────────────────────────────────────────────
# BESTANDSKOPPELING
# ──────────────────────────────────────────────
def zoek_json(pdf_pad: Path) -> Path | None:
    """
    Zoek het bijbehorende JSON bestand voor een PDF.
    Ondersteunt case-insensitieve matching (bv. waste-free-1.pdf → Waste-free-1.json).
    """
    kandidaat = pdf_pad.with_suffix(".json")
    if kandidaat.exists():
        return kandidaat

    stam_lower = pdf_pad.stem.lower()
    for json_bestand in pdf_pad.parent.glob("*.json"):
        if json_bestand.stem.lower() == stam_lower:
            return json_bestand

    return None


# ──────────────────────────────────────────────
# VERWERKING
# ──────────────────────────────────────────────
def verwerk_categorie(categorie_map: Path) -> list[dict]:
    """Verwerk alle PDF+JSON paren in één categoriemap."""
    categorie = categorie_map.name
    schema = CATEGORIE_SCHEMAS.get(categorie)
    if schema is None:
        print(f"  ⚠️  Onbekende categorie '{categorie}', overgeslagen")
        return []

    # Map voor afbeeldingen per categorie
    beeld_map = AFBEELDINGEN_MAP / categorie
    beeld_map.mkdir(parents=True, exist_ok=True)

    paren = []
    for pdf_pad in sorted(categorie_map.glob("*.pdf")):
        json_pad = zoek_json(pdf_pad)
        if json_pad is None:
            print(f"  ⚠️  Geen JSON voor {pdf_pad.name}, overgeslagen")
            continue

        jpeg_pad = beeld_map / (pdf_pad.stem + ".jpg")

        print(f"  Verwerk {pdf_pad.name} ...", end=" ", flush=True)
        start = time.time()
        try:
            pdf_naar_jpeg(pdf_pad, jpeg_pad)

            with open(json_pad, encoding="utf-8") as f:
                ground_truth = json.load(f)

            user_bericht = USER_PROMPT_TEMPLATE.format(schema=schema)
            paren.append({
                # Relatief pad zodat de dataset draagbaar blijft
                "image": str(jpeg_pad),
                "input": user_bericht,
                "output": json.dumps(ground_truth, ensure_ascii=False),
            })
            print(f"OK ({time.time() - start:.1f}s) → {jpeg_pad}")
        except Exception as e:
            print(f"FOUT: {e}")

    return paren


def genereer_trainingsdata() -> list[dict]:
    """Verwerk alle categorieën en geef alle training paren terug."""
    alle_paren = []
    for categorie_map in sorted(DOCUMENTS_MAP.iterdir()):
        if not categorie_map.is_dir():
            continue
        print(f"\nCategorie: {categorie_map.name}")
        paren = verwerk_categorie(categorie_map)
        alle_paren.extend(paren)
        print(f"  → {len(paren)} paren verwerkt")
    return alle_paren


# ──────────────────────────────────────────────
# HOOFDPROGRAMMA
# ──────────────────────────────────────────────
def main():
    DATA_MAP.mkdir(exist_ok=True)
    AFBEELDINGEN_MAP.mkdir(exist_ok=True)

    print(f"Visie-trainingsdata genereren vanuit {DOCUMENTS_MAP}/ ...")
    print(f"Afbeeldingen worden opgeslagen in {AFBEELDINGEN_MAP}/\n")

    paren = genereer_trainingsdata()

    if not paren:
        print("\n❌ Geen paren gevonden. Controleer de documents/ map.")
        sys.exit(1)

    with open(UITVOER_PAD, "w", encoding="utf-8") as f:
        for paar in paren:
            f.write(json.dumps(paar, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(paren)} voorbeelden weggeschreven naar {UITVOER_PAD}")
    print(f"   Afbeeldingen: {AFBEELDINGEN_MAP}/")
    print("\nElk JSONL-record heeft drie velden:")
    print('  "image"  — pad naar de JPEG (pagina 1 van de PDF)')
    print('  "input"  — extractie-prompt met schema')
    print('  "output" — ground-truth JSON string')


if __name__ == "__main__":
    main()
