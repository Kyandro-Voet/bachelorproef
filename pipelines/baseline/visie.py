"""
Visie-pipeline: PDF → afbeeldingen → vision LLM → JSON
Geen OCR — het vision-model leest de documenten rechtstreeks.
Ondersteunt categorieën: electricity, water, natural gas, waste, fuels
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

import ollama
from pdf2image import convert_from_path
from pymongo import MongoClient

_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pipelines" / "time_limit.py").exists()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from pipelines.time_limit import FACTUUR_TIMEOUT_SECONDEN, FactuurTimeout, factuur_timeout


# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
MODEL = "qwen3-vl:8b"
DPI = 100
MAX_BREEDTE = 1024
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "baseline/visie"
RUNS = 3

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "bachelorproef"
MONGO_COLLECTION = "resultaten"

SYSTEM_PROMPT = """You are a data extraction assistant for documents.
Extract the requested information and return it as valid JSON.
Return ONLY the JSON, no extra text or explanation."""

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

Respond ONLY with valid JSON. No extra text.
"""


# ──────────────────────────────────────────────
# COLD RUN
# ──────────────────────────────────────────────
def _wacht_op_unload(model_naam: str, timeout: float = 30.0) -> None:
    """Stuur unload-request en poll tot het model echt uit Ollama-geheugen is."""
    try:
        actief = {m.model for m in ollama.ps().models}
    except Exception:
        return

    if not any(m.startswith(model_naam) for m in actief):
        return

    try:
        ollama.generate(model=model_naam, prompt="", keep_alive=0)
    except Exception:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            actief = {m.model for m in ollama.ps().models}
            if not any(m.startswith(model_naam) for m in actief):
                return
        except Exception:
            return
        time.sleep(0.5)

    print(f"   Waarschuwing: {model_naam} nog steeds geladen na {timeout:.0f}s")


def ontlaad_model() -> None:
    """Verwijder het model uit het Ollama-geheugen zodat elke run een cold run is."""
    _wacht_op_unload(MODEL)


# ──────────────────────────────────────────────
# OPSLAG
# ──────────────────────────────────────────────
def sla_op_in_mongodb(resultaat: dict) -> None:
    """Sla één run op in MongoDB."""
    document = {
        "bestand": resultaat["bestand"],
        "categorie": resultaat["categorie"],
        "model": MODEL,
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
# PIPELINE STAPPEN
# ──────────────────────────────────────────────
def stap1_pdf_naar_afbeeldingen(pdf_pad: Path) -> list:
    print(f"\n   Stap 1: PDF naar afbeeldingen ({DPI} DPI)...")
    start = time.time()
    afbeeldingen = convert_from_path(str(pdf_pad), dpi=DPI)
    print(f"   {len(afbeeldingen)} pagina('s) in {time.time()-start:.1f}s")
    return afbeeldingen


def schaal_afbeelding(img, max_breedte: int = MAX_BREEDTE):
    """Schaal afbeelding zodat de breedte maximaal max_breedte is."""
    if img.width <= max_breedte:
        return img
    ratio = max_breedte / img.width
    nieuwe_hoogte = int(img.height * ratio)
    return img.resize((max_breedte, nieuwe_hoogte))


def afbeeldingen_naar_base64(afbeeldingen: list) -> list[str]:
    """Schaal en converteer PIL-afbeeldingen naar base64-strings voor Ollama."""
    base64_lijst = []
    for img in afbeeldingen:
        img = schaal_afbeelding(img)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        base64_lijst.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return base64_lijst


def stap2_visie_llm(afbeeldingen: list, categorie: str) -> str:
    """Stuur alle afbeeldingen in één call naar het vision-model."""
    print(f"\n   Stap 2: Afbeeldingen naar {MODEL} sturen (categorie: {categorie})...")
    start = time.time()

    base64_afbeeldingen = afbeeldingen_naar_base64(afbeeldingen)
    schema = CATEGORIE_SCHEMAS.get(categorie, CATEGORIE_SCHEMAS["electricity"])
    user_prompt = USER_PROMPT_TEMPLATE.format(schema=schema)

    print(f"   {len(base64_afbeeldingen)} afbeelding(en) → 1 call...")
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
                "images": base64_afbeeldingen,
            },
        ],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": 16384,
        },
        keep_alive=0,
    )

    print(f"   Antwoord ontvangen in {time.time()-start:.1f}s")
    return response.message.content


def stap3_parse_json(ruwe_output: str) -> dict | None:
    print(f"\n   Stap 3: JSON parsen...")
    opgeschoond = ruwe_output.strip()

    if opgeschoond.startswith("```json"):
        opgeschoond = opgeschoond[7:]
    if opgeschoond.startswith("```"):
        opgeschoond = opgeschoond[3:]
    if opgeschoond.endswith("```"):
        opgeschoond = opgeschoond[:-3]
    opgeschoond = opgeschoond.strip()

    begin = opgeschoond.find("{")
    eind = opgeschoond.rfind("}")
    if begin != -1 and eind != -1:
        opgeschoond = opgeschoond[begin:eind + 1]

    try:
        resultaat = json.loads(opgeschoond)
        print(f"   JSON succesvol geparsed!")
        return resultaat
    except json.JSONDecodeError as e:
        print(f"   JSON parse fout: {e}")
        print(f"   Ruwe output:\n{ruwe_output[:500]}")
        return None


# ──────────────────────────────────────────────
# VERWERKING
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> list[dict]:
    """Verwerk één document in RUNS aparte cold runs."""
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 55}")
    print(f"  {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 55}")

    runs = []
    for run in range(1, RUNS + 1):
        print(f"\n   Run {run}/{RUNS} — cold start ({MODEL})...")
        ontlaad_model()
        run_start = time.time()

        afbeeldingen = stap1_pdf_naar_afbeeldingen(pdf_pad)
        ruwe_output = stap2_visie_llm(afbeeldingen, categorie)
        resultaat = stap3_parse_json(ruwe_output)

        tijd_totaal = round(time.time() - run_start, 2)
        status = "JSON OK" if resultaat is not None else "JSON FOUT"
        print(f"   Run {run}: {status} ({tijd_totaal}s)")

        runs.append({
            "bestand": pdf_pad.name,
            "categorie": categorie,
            "run": run,
            "success": resultaat is not None,
            "tijd_totaal": tijd_totaal,
            "extracted": resultaat,
            "ruwe_output": ruwe_output,
        })

    tijden = [r["tijd_totaal"] for r in runs]
    print(f"\n   Runs klaar — tijden: {tijden}")
    return runs


def timeout_runs(pdf_pad: Path, fout: Exception) -> list[dict]:
    return [{
        "bestand": pdf_pad.name,
        "categorie": pdf_pad.parent.name,
        "run": run,
        "success": False,
        "tijd_totaal": FACTUUR_TIMEOUT_SECONDEN,
        "extracted": None,
        "ruwe_output": f"Timeout: {fout}",
    } for run in range(1, RUNS + 1)]


def main():
    # Bepaal welke PDF's verwerkt moeten worden
    if len(sys.argv) >= 2:
        pdfs = [Path(sys.argv[1])]
    else:
        if not DOCUMENTS_MAP.exists():
            print(f" Map '{DOCUMENTS_MAP}' niet gevonden.")
            print(f"   Gebruik: uv run python vision_pipeline.py <pad/naar/factuur.pdf>")
            print(f"   Of plaats PDF's in categoriesubmappen van '{DOCUMENTS_MAP}/'.")
            sys.exit(1)
        pdfs = sorted(DOCUMENTS_MAP.rglob("*.pdf"))

    if not pdfs:
        print(" Geen PDF-bestanden gevonden.")
        sys.exit(1)

    print(f"\n Visie-pipeline — Model: {MODEL}")
    print(f"   Facturen: {len(pdfs)}  |  Runs per factuur: {RUNS}")
    print(f"   Geen OCR — vision-model leest afbeeldingen rechtstreeks")

    # Verwerk alle facturen
    alle_runs = []
    for pdf_pad in pdfs:
        if not pdf_pad.exists():
            print(f" Bestand niet gevonden: {pdf_pad}")
            continue

        try:
            with factuur_timeout():
                runs = verwerk_factuur(pdf_pad)
        except FactuurTimeout as e:
            print(f"   TIMEOUT na {FACTUUR_TIMEOUT_SECONDEN // 60} min: {pdf_pad.name}")
            runs = timeout_runs(pdf_pad, e)
        for run in runs:
            sla_op_in_mongodb(run)
        alle_runs.extend(runs)

    # ── Samenvatting ──
    print(f"\n\n{'#' * 55}")
    print(f"  SAMENVATTING")
    print(f"{'#' * 55}")

    geslaagd = sum(1 for r in alle_runs if r["success"])
    print(f"\n  Totaal runs: {len(alle_runs)}  ({len(alle_runs) // RUNS} facturen × {RUNS})")
    print(f"  Geslaagd:  {geslaagd}")
    print(f"  Mislukt:   {len(alle_runs) - geslaagd}")

    for r in alle_runs:
        status = "OK" if r["success"] else "FOUT"
        print(f"\n  [{status}] {r['categorie']}/{r['bestand']} run{r['run']} ({r['tijd_totaal']}s)")


if __name__ == "__main__":
    main()
