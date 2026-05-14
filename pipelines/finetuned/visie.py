"""
Visiepipeline met finetuned model: PDF → afbeeldingen → MLX VLM → JSON
PDF-conversie eenmalig, MLX-inferentie 3x voor betrouwbare tijdmeting.

Model: finetuning/vision/mlx_model (MLX 4-bit gekwantiseerd)
Input:  documents_testing/<categorie>/<naam>.pdf
Output: resultaten/visie_ft/<categorie>/<naam>.json
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image
from pymongo import MongoClient


# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
MLX_MODEL_PAD = Path(__file__).resolve().parent.parent.parent / "finetuning/vision/mlx_model"
DPI = 150
MAX_BREEDTE = 1024
PAGINAS_PER_BATCH = 2
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "finetuned/visie"
RUNS = 3

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "bachelorproef"
MONGO_COLLECTION = "resultaten"

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

USER_PROMPT_TEMPLATE = """/no_think
Extract all data from this document and return it as a JSON object
that strictly follows this schema:

{schema}

Answer ONLY with valid JSON. No extra text."""


# ──────────────────────────────────────────────
# MLX VLM
# ──────────────────────────────────────────────
_mlx_model = None
_mlx_processor = None
_mlx_config = None


def ontlaad_mlx_model() -> None:
    """Reset de MLX model cache zodat de volgende run een cold run is."""
    global _mlx_model, _mlx_processor, _mlx_config
    _mlx_model = None
    _mlx_processor = None
    _mlx_config = None


def _laad_mlx_model():
    global _mlx_model, _mlx_processor, _mlx_config
    if _mlx_model is not None:
        return _mlx_model, _mlx_processor, _mlx_config

    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    if not MLX_MODEL_PAD.exists():
        print(f"   FOUT: MLX model niet gevonden: {MLX_MODEL_PAD}")
        print("   Voer eerst finetuning/vision/convert_mlx.py uit.")
        sys.exit(1)

    print(f"   MLX vision model laden: {MLX_MODEL_PAD}...")
    _mlx_model, _mlx_processor = load(str(MLX_MODEL_PAD))
    _mlx_config = load_config(str(MLX_MODEL_PAD))
    print("   Model geladen.")
    return _mlx_model, _mlx_processor, _mlx_config


def mlx_inferentie(afbeelding: Image.Image, prompt_tekst: str, max_tokens: int = 2048) -> tuple[str, float]:
    """Voer MLX VLM inferentie uit. Geeft (output, tijd) terug."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _laad_mlx_model()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        afbeelding.save(f, format="JPEG", quality=85)
        temp_pad = f.name

    try:
        formatted = apply_chat_template(processor, config, prompt_tekst, num_images=1)
        start = time.time()
        output = generate(
            model, processor, formatted,
            image=temp_pad,
            max_tokens=max_tokens,
            temp=0.1,
            verbose=False,
        )
        tijd = time.time() - start
    finally:
        os.unlink(temp_pad)

    tekst = output.text if hasattr(output, "text") else str(output)
    return tekst, round(tijd, 2)


# ──────────────────────────────────────────────
# AFBEELDING HULPFUNCTIES
# ──────────────────────────────────────────────
def schaal_afbeelding(img: Image.Image, max_breedte: int = MAX_BREEDTE) -> Image.Image:
    if img.width <= max_breedte:
        return img
    ratio = max_breedte / img.width
    return img.resize((max_breedte, int(img.height * ratio)), Image.LANCZOS)


def combineer_paginas(afbeeldingen: list[Image.Image]) -> Image.Image:
    """Stapel pagina's verticaal met een grijze scheidingsbalk."""
    geschaald = [schaal_afbeelding(img) for img in afbeeldingen]
    if len(geschaald) == 1:
        return geschaald[0]

    scheiding = 10
    breedte = max(img.width for img in geschaald)
    hoogte = sum(img.height for img in geschaald) + scheiding * (len(geschaald) - 1)
    gecombineerd = Image.new("RGB", (breedte, hoogte), color=(200, 200, 200))

    y = 0
    for img in geschaald:
        gecombineerd.paste(img, (0, y))
        y += img.height + scheiding

    return gecombineerd


def verwerk_in_batches(paginas: list[Image.Image], prompt: str) -> tuple[str, float]:
    """
    Verwerk pagina's in batches van PAGINAS_PER_BATCH.
    Elke batch wordt gecombineerd tot één afbeelding en als aparte MLX-call verwerkt.
    De outputs worden samengevoegd voor de uiteindelijke JSON-parsing.
    """
    alle_outputs = []
    totaal_tijd = 0.0
    batches = [paginas[i:i + PAGINAS_PER_BATCH] for i in range(0, len(paginas), PAGINAS_PER_BATCH)]

    for b, batch in enumerate(batches):
        pagina_nrs = f"{b * PAGINAS_PER_BATCH + 1}-{b * PAGINAS_PER_BATCH + len(batch)}"
        print(f"      Batch {b + 1}/{len(batches)} (pagina's {pagina_nrs})...")
        gecombineerd = combineer_paginas(batch)
        output, tijd = mlx_inferentie(gecombineerd, prompt)
        totaal_tijd += tijd
        alle_outputs.append(output)

    return "\n\n".join(alle_outputs), round(totaal_tijd, 2)


# ──────────────────────────────────────────────
# JSON PARSING
# ──────────────────────────────────────────────
def parse_json(ruwe_output: str) -> dict | None:
    opgeschoond = ruwe_output.strip()
    for prefix in ("```json", "```"):
        if opgeschoond.startswith(prefix):
            opgeschoond = opgeschoond[len(prefix):]
    if opgeschoond.endswith("```"):
        opgeschoond = opgeschoond[:-3]
    opgeschoond = opgeschoond.strip()

    begin = opgeschoond.find("{")
    eind = opgeschoond.rfind("}")
    if begin != -1 and eind != -1:
        opgeschoond = opgeschoond[begin:eind + 1]

    try:
        return json.loads(opgeschoond)
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────
# OPSLAG
# ──────────────────────────────────────────────
def sla_run_op(resultaat: dict) -> None:
    """Sla één run op in MongoDB."""
    document = {
        "bestand": resultaat["bestand"],
        "categorie": resultaat["categorie"],
        "model": MLX_MODEL_PAD.name,
        "pipeline": PIPELINE,
        "run": resultaat["run"],
        "success": resultaat["success"],
        "tijd_totaal": resultaat["tijd_totaal"],
        "extracted": resultaat["extracted"],
        "ruwe_output": resultaat["ruwe_output"],
    }
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client[MONGO_DB][MONGO_COLLECTION].insert_one(document)
        print(f"   MongoDB: {PIPELINE} run{resultaat['run']} — {resultaat['bestand']}")
    except Exception as e:
        print(f"   MongoDB FOUT: {e}")


# ──────────────────────────────────────────────
# VERWERKING (met 3 aparte runs)
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> list[dict]:
    """Verwerk één factuur in RUNS aparte cold runs. Elke run doet PDF-conversie + inferentie opnieuw."""
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 60}")
    print(f"  {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 60}")

    schema = CATEGORIE_SCHEMAS.get(categorie, CATEGORIE_SCHEMAS["electricity"])
    prompt = USER_PROMPT_TEMPLATE.format(schema=schema)

    runs = []
    for run in range(1, RUNS + 1):
        print(f"\n   Run {run}/{RUNS} — cold start (MLX)...")
        ontlaad_mlx_model()

        run_start = time.time()

        # PDF → afbeeldingen per run
        print(f"\n   PDF → afbeeldingen ({DPI} DPI)...")
        paginas = convert_from_path(str(pdf_pad), dpi=DPI)
        print(f"   {len(paginas)} pagina('s) in {time.time() - run_start:.1f}s")

        n_batches = -(-len(paginas) // PAGINAS_PER_BATCH)  # ceiling division
        print(f"   MLX inferentie ({len(paginas)} pagina's, {n_batches} batch(es))...")
        ruwe_output, _ = verwerk_in_batches(paginas, prompt)

        tijd_totaal = round(time.time() - run_start, 2)
        extracted = parse_json(ruwe_output)
        status = "JSON OK" if extracted is not None else "JSON FOUT"
        print(f"   Run {run}: {status} ({tijd_totaal}s totaal incl. PDF-conversie)")

        runs.append({
            "bestand": pdf_pad.name,
            "categorie": categorie,
            "model": MLX_MODEL_PAD.name,
            "pipeline": PIPELINE,
            "run": run,
            "success": extracted is not None,
            "tijd_totaal": tijd_totaal,
            "extracted": extracted,
            "ruwe_output": ruwe_output,
        })

    tijden = [r["tijd_totaal"] for r in runs]
    print(f"\n   Runs klaar — tijden: {tijden}")
    return runs


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) >= 2:
        pdfs = [Path(sys.argv[1])]
    else:
        if not DOCUMENTS_MAP.exists():
            print(f"   Map '{DOCUMENTS_MAP}' niet gevonden.")
            sys.exit(1)
        pdfs = sorted(DOCUMENTS_MAP.rglob("*.pdf"))

    if not pdfs:
        print("   Geen PDF-bestanden gevonden.")
        sys.exit(1)

    print(f"\n   Visiepipeline (finetuned) — MLX model: {MLX_MODEL_PAD.name}")
    print(f"   Facturen: {len(pdfs)}  |  Runs per factuur: {RUNS}")

    alle_runs = []
    for pdf_pad in pdfs:
        if not pdf_pad.exists():
            print(f"   Bestand niet gevonden: {pdf_pad}")
            continue
        runs = verwerk_factuur(pdf_pad)
        for run in runs:
            sla_run_op(run)
        alle_runs.extend(runs)

    print(f"\n\n{'#' * 60}")
    print(f"  SAMENVATTING — {PIPELINE}")
    print(f"{'#' * 60}")
    geslaagd = sum(1 for r in alle_runs if r["success"])
    print(f"\n  Totaal runs: {len(alle_runs)}  ({len(alle_runs) // RUNS} facturen × {RUNS})")
    print(f"  Geslaagd:    {geslaagd}")
    print(f"  Mislukt:     {len(alle_runs) - geslaagd}")
    for r in alle_runs:
        status = "OK" if r["success"] else "FOUT"
        print(f"  [{status}] {r['categorie']}/{r['bestand']} run{r['run']} ({r['tijd_totaal']}s)")


if __name__ == "__main__":
    main()
