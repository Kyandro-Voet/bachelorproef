"""
Minimale tekstpipeline: PDF → tekst → LLM → JSON
Digitale PDF: tekst direct via pdfplumber (geen afbeeldingen nodig).
Gescande PDF: pagina naar afbeelding → Tesseract OCR.
Ondersteunt categorieën: electricity, water, natural gas, waste, fuels
"""

import json
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
MODEL = "qwen3:8b"
OCR_TALEN = "eng+ita"
DPI = 300
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "baseline/tekst"

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
    # Alleen unloaden als het model momenteel geladen is
    try:
        actief = {m.model for m in ollama.ps().models}
    except Exception:
        return

    if not any(m.startswith(model_naam) for m in actief):
        return  # al niet geladen, niets te doen

    try:
        ollama.generate(model=model_naam, prompt="", keep_alive=0)
    except Exception:
        pass

    # Poll tot het model verdwenen is uit /api/ps
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
    """Sla het resultaat op in MongoDB."""
    document = {
        "bestand": resultaat["bestand"],
        "categorie": resultaat["categorie"],
        "model": MODEL,
        "pipeline": PIPELINE,
        "success": resultaat["success"],
        "tijd_totaal": resultaat["tijd_totaal"],
        "extracted": resultaat["extracted"],
        "ocr_tekst": resultaat["ocr_tekst"],
        "ruwe_output": resultaat["ruwe_output"],
    }
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client[MONGO_DB][MONGO_COLLECTION].insert_one(document)
        print(f"   MongoDB: {PIPELINE} — {resultaat['bestand']}")
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


def stap2_llm(tekst: str, categorie: str) -> str:
    print(f"\n   Stap 2: Tekst naar {MODEL} sturen (categorie: {categorie})...")
    start = time.time()

    schema = CATEGORIE_SCHEMAS.get(categorie, CATEGORIE_SCHEMAS["electricity"])
    user_prompt = USER_PROMPT_TEMPLATE.format(schema=schema) + tekst

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": 32768,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
        },
        think=False,
    )
    ruwe_output = response.message.content
    print(f"   Antwoord ontvangen in {time.time()-start:.1f}s")
    return ruwe_output


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
        # Afwijzen als het model een foutmelding teruggeeft i.p.v. data
        if "error" in resultaat and len(resultaat) == 1:
            print(f"   Model-fout: {resultaat}")
            return None
        print(f"   JSON succesvol geparsed!")
        return resultaat
    except json.JSONDecodeError as e:
        print(f"   JSON parse fout: {e}")
        print(f"   Ruwe output:\n{ruwe_output[:500]}")
        return None


# ──────────────────────────────────────────────
# VERWERKING
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> dict:
    """Verwerk één factuur en geef het resultaat terug."""
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 55}")
    print(f"   {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 55}")

    ontlaad_model()
    totaal_start = time.time()

    tekst = stap1_extraheer_tekst(pdf_pad)
    ruwe_output = stap2_llm(tekst, categorie)
    resultaat = stap3_parse_json(ruwe_output)

    totaal_tijd = time.time() - totaal_start

    return {
        "bestand": pdf_pad.name,
        "categorie": categorie,
        "success": resultaat is not None,
        "tijd_totaal": round(totaal_tijd, 2),
        "extracted": resultaat,
        "ocr_tekst": tekst,
        "ruwe_output": ruwe_output,
    }


def main():
    # Bepaal welke PDF's verwerkt moeten worden
    if len(sys.argv) >= 2:
        pdfs = [Path(sys.argv[1])]
    else:
        if not DOCUMENTS_MAP.exists():
            print(f" Map '{DOCUMENTS_MAP}' niet gevonden.")
            print(f"   Gebruik: uv run python main.py <pad/naar/factuur.pdf>")
            print(f"   Of plaats PDF's in categoriesubmappen van '{DOCUMENTS_MAP}/'.")
            sys.exit(1)
        pdfs = sorted(DOCUMENTS_MAP.rglob("*.pdf"))

    if not pdfs:
        print(" Geen PDF-bestanden gevonden.")
        sys.exit(1)

    print(f"\n Tekstpipeline — Model: {MODEL}")
    print(f"   Facturen: {len(pdfs)}")
    print(f"   OCR-talen: {OCR_TALEN}")

    # Verwerk alle facturen
    resultaten = []
    for pdf_pad in pdfs:
        if not pdf_pad.exists():
            print(f" Bestand niet gevonden: {pdf_pad}")
            continue

        resultaat = verwerk_factuur(pdf_pad)
        resultaten.append(resultaat)

        sla_op_in_mongodb(resultaat)

    # ── Samenvatting ──
    print(f"\n\n{'#' * 55}")
    print(f"  SAMENVATTING")
    print(f"{'#' * 55}")

    geslaagd = sum(1 for r in resultaten if r["success"])
    print(f"\n  Totaal:    {len(resultaten)} facturen")
    print(f"  Geslaagd:  {geslaagd}")
    print(f"  Mislukt:   {len(resultaten) - geslaagd}")

    for r in resultaten:
        status = "OK" if r["success"] else "FOUT"
        print(f"\n  [{status}] {r['categorie']}/{r['bestand']} ({r['tijd_totaal']}s)")


if __name__ == "__main__":
    main()
