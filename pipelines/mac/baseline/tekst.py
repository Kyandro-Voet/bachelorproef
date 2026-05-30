"""
Tekstpipeline (Mac): PDF → tekst → LLM → JSON
Digitale PDF: tekst direct via pdfplumber (geen afbeeldingen nodig).
Gescande PDF: pagina naar afbeelding → Tesseract OCR.
Ondersteunt categorieën: electricity, water, natural gas, waste, fuels
"""

import json
import re
import sys
import time
from pathlib import Path

import ollama
from ollama import Client
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

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
MODEL = "qwen3:8b-q4_K_M"
OCR_TALEN = "eng+ita"
DPI = 300
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "mac/baseline/tekst"
RESULTATEN_MAP = Path("resultaten/mac/baseline/tekst")
RUNS = 3
MAX_TEKST_TEKENS = 36000

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
# HULPFUNCTIES
# ──────────────────────────────────────────────
def toon_gpu_cpu_verdeling() -> None:
    try:
        for m in ollama.ps().models:
            if m.model.startswith(MODEL.split(":")[0]):
                vram = m.size_vram
                ram = m.size - vram
                print(f"   GPU: {vram / 1024**3:.2f} GB | CPU: {ram / 1024**3:.2f} GB")
                return
    except Exception:
        pass


def opkuis_tekst(tekst: str) -> str:
    # Elke regel trimmen, decoratieve lijnen en triviale regels verwijderen
    regels = []
    for regel in tekst.splitlines():
        regel = regel.strip()
        if re.match(r'^[-=_.]{4,}$', regel):
            continue
        if 0 < len(regel) < 3:
            continue
        regels.append(regel)
    tekst = "\n".join(regels)
    tekst = re.sub(r'[ \t]+', ' ', tekst)
    tekst = re.sub(r'\n{3,}', '\n\n', tekst)
    tekst = tekst.strip()
    if len(tekst) > MAX_TEKST_TEKENS:
        print(f"   Waarschuwing: tekst ({len(tekst)} tekens) afgekapt op {MAX_TEKST_TEKENS}")
        tekst = tekst[:MAX_TEKST_TEKENS]
    return tekst


def _wacht_op_unload(model_naam: str, timeout: float = 30.0) -> None:
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
    _wacht_op_unload(MODEL)


# ──────────────────────────────────────────────
# OPSLAG
# ──────────────────────────────────────────────
def sla_run_op(resultaat: dict) -> None:
    uitvoer_map = RESULTATEN_MAP / resultaat["categorie"]
    uitvoer_map.mkdir(parents=True, exist_ok=True)
    stem = Path(resultaat["bestand"]).stem
    pad = uitvoer_map / f"{stem}_run{resultaat['run']}.json"
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
    with open(pad, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"   Opgeslagen: {pad}")


# ──────────────────────────────────────────────
# PIPELINE STAPPEN
# ──────────────────────────────────────────────
def stap1_extraheer_tekst(pdf_pad: Path) -> str:
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

    client = Client(timeout=600.0)
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": 16384,
        },
        keep_alive=0,
        think=False,
    )
    ruwe_output = response.message.content
    print(f"   Antwoord ontvangen in {time.time()-start:.1f}s")
    toon_gpu_cpu_verdeling()
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
def verwerk_factuur(pdf_pad: Path) -> list[dict]:
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 60}")
    print(f"  {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 60}")

    runs = []
    for run in range(1, RUNS + 1):
        print(f"\n   Run {run}/{RUNS} — cold start ({MODEL})...")
        ontlaad_model()

        run_start = time.time()

        tekst = opkuis_tekst(stap1_extraheer_tekst(pdf_pad))
        ruwe_output = stap2_llm(tekst, categorie)
        extracted = stap3_parse_json(ruwe_output)

        tijd_totaal = round(time.time() - run_start, 2)
        status = "JSON OK" if extracted is not None else "JSON FOUT"
        print(f"   Run {run}: {status} ({tijd_totaal}s totaal)")

        runs.append({
            "bestand": pdf_pad.name,
            "categorie": categorie,
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


def timeout_runs(pdf_pad: Path, fout: Exception) -> list[dict]:
    categorie = pdf_pad.parent.name
    return [{
        "bestand": pdf_pad.name,
        "categorie": categorie,
        "run": run,
        "success": False,
        "tijd_totaal": FACTUUR_TIMEOUT_SECONDEN,
        "extracted": None,
        "ocr_tekst": "",
        "ruwe_output": f"Timeout: {fout}",
    } for run in range(1, RUNS + 1)]


def main():
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

    print(f"\n   Tekstpipeline (Mac) — Model: {MODEL}")
    print(f"   Facturen: {len(pdfs)}  |  Runs per factuur: {RUNS}")
    print(f"   OCR-talen: {OCR_TALEN}")

    alle_runs = []
    for pdf_pad in pdfs:
        if not pdf_pad.exists():
            print(f"   Bestand niet gevonden: {pdf_pad}")
            continue

        categorie = pdf_pad.parent.name
        alle_runs_bestaan = all(
            (RESULTATEN_MAP / categorie / f"{pdf_pad.stem}_run{r}.json").exists()
            for r in range(1, RUNS + 1)
        )
        if alle_runs_bestaan:
            print(f"   Overgeslagen (alle {RUNS} runs bestaan al): {pdf_pad.name}")
            continue

        try:
            with factuur_timeout():
                runs = verwerk_factuur(pdf_pad)
        except FactuurTimeout as e:
            print(f"   TIMEOUT na {FACTUUR_TIMEOUT_SECONDEN // 60} min: {pdf_pad.name}")
            runs = timeout_runs(pdf_pad, e)
        for run in runs:
            sla_run_op(run)
        alle_runs.extend(runs)

    print(f"\n\n{'#' * 55}")
    print(f"  SAMENVATTING")
    print(f"{'#' * 55}")

    geslaagd = sum(1 for r in alle_runs if r["success"])
    print(f"\n  Totaal runs: {len(alle_runs)}  ({len(alle_runs) // RUNS} facturen × {RUNS})")
    print(f"  Geslaagd:    {geslaagd}")
    print(f"  Mislukt:     {len(alle_runs) - geslaagd}")

    for r in alle_runs:
        status = "OK" if r["success"] else "FOUT"
        print(f"  [{status}] {r['categorie']}/{r['bestand']} run{r['run']} ({r['tijd_totaal']}s)")


if __name__ == "__main__":
    main()
