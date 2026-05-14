"""
Tekstpipeline met finetuned model: PDF → tekst → Qwen3-ft (Ollama) → JSON
Digitale PDF: tekst direct via pdfplumber (geen afbeeldingen nodig).
Gescande PDF: pagina naar afbeelding → Tesseract OCR.
LLM 3x per run voor betrouwbare tijdmeting.

Model: finetuning/text/Qwen3-8B.Q4_K_M.gguf (geladen als Ollama-model 'qwen3-ft:8b')
Input:  documents_testing/<categorie>/<naam>.pdf
Output: resultaten/tekst_ft/<categorie>/<naam>.json
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import ollama
import pdfplumber
from pdf2image import convert_from_path
from pymongo import MongoClient
import pytesseract


# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
MODEL = "qwen3-8b-finetuned"
MODELFILE_PAD = Path(__file__).resolve().parent.parent.parent / "finetuning/text/Modelfile"
OCR_TALEN = "eng+ita"
DPI = 300
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "finetuned/tekst"
RUNS = 3

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "bachelorproef"
MONGO_COLLECTION = "resultaten"

SYSTEM_PROMPT = """You are a precise data extraction assistant.
Your sole task is to extract structured information from utility/resource documents and return it as valid JSON.

Rules you must follow:
- Return ONLY a valid JSON object. No explanation, no markdown, no code fences.
- If a field is not present in the document, set its value to null.
- Do not infer or guess values that are not explicitly stated.
- Normalize all amounts to numbers (e.g. "7.973,12 €" → 7973.12).
- Normalize all dates to ISO 8601 format (YYYY-MM-DD).
- If multiple records exist (e.g. multiple meters or transactions), include all of them as array items."""

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

USER_PROMPT_TEMPLATE = """Extract all data from the document below and return it as a JSON object
that strictly follows this schema:

{schema}

--- DOCUMENT START ---

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
    """Verwijder het model uit het Ollama-geheugen voor een cold run."""
    _wacht_op_unload(MODEL)


# ──────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────
def setup_model() -> None:
    """Maak het Ollama-model aan als het nog niet bestaat."""
    try:
        ollama.show(MODEL)
        print(f"   Model '{MODEL}' beschikbaar in Ollama.")
    except Exception:
        print(f"   Model '{MODEL}' niet gevonden — aanmaken uit {MODELFILE_PAD}...")
        if not MODELFILE_PAD.exists():
            print(f"   FOUT: Modelfile niet gevonden: {MODELFILE_PAD}")
            sys.exit(1)
        result = subprocess.run(
            ["ollama", "create", MODEL, "-f", str(MODELFILE_PAD)],
            cwd=str(MODELFILE_PAD.parent),
            text=True,
        )
        if result.returncode != 0:
            print(f"   FOUT bij aanmaken model (returncode {result.returncode})")
            sys.exit(result.returncode)
        print(f"   Model '{MODEL}' aangemaakt.")


# ──────────────────────────────────────────────
# OPSLAG
# ──────────────────────────────────────────────
def sla_run_op(resultaat: dict) -> None:
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
        "ocr_tekst": resultaat["ocr_tekst"],
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
def stap1_extraheer_tekst(pdf_pad: Path) -> str:
    """Extraheer tekst per pagina.
    Digitale pagina (>= 50 tekens): pdfplumber leest tekst direct.
    Gescande pagina (< 50 tekens): afbeelding → Tesseract OCR.
    """
    print(f"\n   Stap 1: Tekst extraheren uit PDF...")
    start = time.time()
    alle_tekst = []

    with pdfplumber.open(pdf_pad) as pdf:
        for i, pagina in enumerate(pdf.pages):
            digitale_tekst = pagina.extract_text() or ""

            if len(digitale_tekst) >= 50:
                print(f"   Pagina {i + 1}: digitaal ({len(digitale_tekst)} tekens)")
                alle_tekst.append(f"--- Pagina {i + 1} ---\n{digitale_tekst}")
            else:
                print(f"   Pagina {i + 1}: gescand → OCR ({DPI} DPI)...")
                imgs = convert_from_path(str(pdf_pad), dpi=DPI,
                                         first_page=i + 1, last_page=i + 1)
                ocr_tekst = pytesseract.image_to_string(imgs[0], lang=OCR_TALEN)
                print(f"   Pagina {i + 1}: OCR {len(ocr_tekst)} tekens")
                alle_tekst.append(f"--- Pagina {i + 1} (OCR) ---\n{ocr_tekst}")

    volledige_tekst = "\n\n".join(alle_tekst)
    print(f"   Totaal {len(volledige_tekst)} tekens in {time.time() - start:.1f}s")
    return volledige_tekst


def stap2_llm(tekst: str, categorie: str) -> tuple[str, float]:
    """Stuur tekst naar het LLM. Geeft (ruwe_output, tijd) terug."""
    schema = CATEGORIE_SCHEMAS.get(categorie, CATEGORIE_SCHEMAS["electricity"])
    user_prompt = USER_PROMPT_TEMPLATE.format(schema=schema) + tekst

    start = time.time()
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.1, "num_ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        think=False,
    )
    tijd = time.time() - start
    return response.message.content, round(tijd, 2)


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
        resultaat = json.loads(opgeschoond)
        # Afwijzen als het model een foutmelding teruggeeft i.p.v. data
        if "error" in resultaat and len(resultaat) == 1:
            return None
        return resultaat
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────
# VERWERKING (met 3 aparte runs)
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> list[dict]:
    """Verwerk één factuur in RUNS aparte cold runs. Elke run doet OCR + LLM opnieuw."""
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 60}")
    print(f"  {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 60}")

    runs = []
    for run in range(1, RUNS + 1):
        print(f"\n   Run {run}/{RUNS} — cold start ({MODEL})...")
        ontlaad_model()

        run_start = time.time()

        # Tekst extraheren per run (digitaal via pdfplumber, gescand via OCR)
        tekst = stap1_extraheer_tekst(pdf_pad)

        # LLM (model laadt hier opnieuw)
        ruwe_output, _ = stap2_llm(tekst, categorie)

        tijd_totaal = round(time.time() - run_start, 2)
        extracted = parse_json(ruwe_output)
        status = "JSON OK" if extracted is not None else "JSON FOUT"
        print(f"   Run {run}: {status} ({tijd_totaal}s totaal incl. tekstextractie)")

        runs.append({
            "bestand": pdf_pad.name,
            "categorie": categorie,
            "model": MODEL,
            "pipeline": PIPELINE,
            "run": run,
            "success": extracted is not None,
            "tijd_totaal": tijd_totaal,
            "extracted": extracted,
            "ocr_tekst": tekst,
            "ruwe_output": ruwe_output,
        })

    tijden = [r["tijd_totaal"] for r in runs]
    print(f"\n   Runs klaar — tijden: {tijden}")
    return runs


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    setup_model()

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

    print(f"\n   Tekstpipeline (finetuned) — Model: {MODEL}")
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
