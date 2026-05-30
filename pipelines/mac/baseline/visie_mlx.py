"""
Visie-pipeline (Mac, baseline MLX): PDF → afbeeldingen → MLX vision LLM → JSON
Geen OCR — het vision-model leest de documenten rechtstreeks.
Ondersteunt categorieën: electricity, water, natural gas, waste, fuels
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

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
MLX_MODEL = os.environ.get("MLX_VISION_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit")
DPI = 100
MAX_BREEDTE = 1024
PAGINAS_PER_BATCH = 5
NUM_CTX = 16384
NUM_PREDICT = 8192   # thinking + uitgebreide JSON output (veel records)
TIMEOUT = 600        # max 10 min per batch (voor grote PDF's)
MLX_TIMEOUT = 600
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "mac/baseline_mlx/visie"
RESULTATEN_MAP = Path("resultaten/mac/baseline_mlx/visie")
RUNS = 3

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

USER_PROMPT_TEMPLATE = """Extract all data from this document and return only a JSON object.

{schema}
"""

# ──────────────────────────────────────────────
# HULPFUNCTIES
# ──────────────────────────────────────────────
def _wacht_op_unload(model_naam: str, timeout: float = 30.0) -> None:
    return


def ontlaad_model() -> None:
    _wacht_op_unload(MLX_MODEL)


def _resterende_runtijd(deadline: float) -> float:
    resterend = deadline - time.time()
    if resterend <= 0:
        raise FactuurTimeout(
            f"Factuurverwerking duurde langer dan {FACTUUR_TIMEOUT_SECONDEN} seconden"
        )
    return resterend


def _is_factuur_timeout_fout(fout: Exception, deadline: float | None = None) -> bool:
    melding = str(fout)
    return (
        isinstance(fout, FactuurTimeout)
        or "Factuurverwerking duurde langer dan" in melding
        or (deadline is not None and time.time() >= deadline)
    )


# ──────────────────────────────────────────────
# OPSLAG
# ──────────────────────────────────────────────
def sla_op_als_json(resultaat: dict) -> None:
    uitvoer_map = RESULTATEN_MAP / resultaat["categorie"]
    uitvoer_map.mkdir(parents=True, exist_ok=True)
    stem = Path(resultaat["bestand"]).stem
    pad = uitvoer_map / f"{stem}_run{resultaat['run']}.json"
    document = {
        "bestand": resultaat["bestand"],
        "categorie": resultaat["categorie"],
        "model": f"MLX {MLX_MODEL}",
        "pipeline": PIPELINE,
        "run": resultaat["run"],
        "success": resultaat["success"],
        "tijd_totaal": resultaat["tijd_totaal"],
        "extracted": resultaat["extracted"],
        "ruwe_output": resultaat["ruwe_output"],
    }
    with open(pad, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"   Opgeslagen: {pad}")


# ──────────────────────────────────────────────
# PIPELINE STAPPEN
# ──────────────────────────────────────────────
def stap1_pdf_naar_afbeeldingen(pdf_pad: Path) -> list:
    print(f"\n   Stap 1: PDF naar afbeeldingen ({DPI} DPI)...")
    start = time.time()
    afbeeldingen = convert_from_path(str(pdf_pad), dpi=DPI)
    print(f"   {len(afbeeldingen)} pagina('s) in {time.time()-start:.1f}s")
    return afbeeldingen


def schaal_afbeelding(img: Image.Image, max_breedte: int = MAX_BREEDTE) -> Image.Image:
    if img.width <= max_breedte:
        return img
    ratio = max_breedte / img.width
    return img.resize((max_breedte, int(img.height * ratio)), Image.LANCZOS)


def combineer_paginas(paginas: list[Image.Image]) -> Image.Image:
    geschaald = [schaal_afbeelding(img) for img in paginas]
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


def mlx_inferentie(afbeelding: Image.Image, prompt_tekst: str, timeout: float) -> str:
    helper = _PROJECT_ROOT / "pipelines/mac/finetuned/mlx_vision_infer.py"
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
        afbeelding.save(img_file, format="JPEG", quality=75)
        temp_img = img_file.name
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as prompt_file:
        prompt_file.write(prompt_tekst)
        temp_prompt = prompt_file.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(helper),
                "--model", MLX_MODEL,
                "--image", temp_img,
                "--prompt", temp_prompt,
                "--max-tokens", str(NUM_PREDICT),
            ],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            fouttekst = (result.stderr or result.stdout).strip()
            regels = [r for r in fouttekst.splitlines() if r.strip()]
            preview = "\n".join(regels[-8:]) if regels else "onbekend"
            raise RuntimeError(f"MLX inferentie faalde (exit {result.returncode}):\n{preview}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired as e:
        raise FactuurTimeout(
            f"Factuurverwerking duurde langer dan {FACTUUR_TIMEOUT_SECONDEN} seconden"
        ) from e
    finally:
        for pad in (temp_img, temp_prompt):
            try:
                os.unlink(pad)
            except OSError:
                pass


def _merge_resultaten(resultaten: list[dict]) -> dict | None:
    """Voeg JSON-dicts van meerdere batches samen door arrays te combineren."""
    if not resultaten:
        return None
    merged: dict = {}
    for r in resultaten:
        for key, value in r.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                merged[key].extend(value)
    return merged or None


def stap2_visie_llm(afbeeldingen: list, categorie: str, deadline: float | None = None) -> dict | None:
    print(f"\n   Stap 2: Afbeeldingen naar MLX {MLX_MODEL} sturen (categorie: {categorie})...")
    start = time.time()

    schema = CATEGORIE_SCHEMAS.get(categorie, CATEGORIE_SCHEMAS["electricity"])
    user_prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(schema=schema)
    batches = [afbeeldingen[i:i + PAGINAS_PER_BATCH]
               for i in range(0, len(afbeeldingen), PAGINAS_PER_BATCH)]

    n = len(afbeeldingen)
    print(f"   {n} pagina('s) → {len(batches)} batch(es) van max {PAGINAS_PER_BATCH} (ctx={NUM_CTX})")

    batch_resultaten = []
    for b, batch in enumerate(batches):
        resterend = _resterende_runtijd(deadline) if deadline is not None else TIMEOUT
        pnrs = f"{b * PAGINAS_PER_BATCH + 1}–{b * PAGINAS_PER_BATCH + len(batch)}"
        print(f"   Batch {b + 1}/{len(batches)} (pagina's {pnrs})...")
        ruwe_tekst = None
        fout_melding = None
        try:
            gecombineerd = combineer_paginas(batch)
            ruwe_tekst = mlx_inferentie(gecombineerd, user_prompt, min(MLX_TIMEOUT, max(1.0, resterend)))
            if deadline is not None:
                _resterende_runtijd(deadline)
            geparsed = _probeer_parse(ruwe_tekst)
        except FactuurTimeout:
            raise
        except Exception as e:
            if _is_factuur_timeout_fout(e, deadline):
                raise FactuurTimeout(
                    f"Factuurverwerking duurde langer dan {FACTUUR_TIMEOUT_SECONDEN} seconden"
                ) from e
            fout_melding = str(e)
            geparsed = None
        if geparsed:
            batch_resultaten.append(geparsed)
        elif fout_melding:
            print(f"   Batch {b + 1}: FOUT — {fout_melding}")
        else:
            preview = repr(ruwe_tekst[:300]) if ruwe_tekst else "(leeg)"
            print(f"   Batch {b + 1}: geen geldige JSON — {preview}")

    print(f"   Alle batches klaar in {time.time() - start:.1f}s")
    return _merge_resultaten(batch_resultaten)


def _probeer_parse(ruwe_output: str) -> dict | None:
    """Intern: parse één batch-output naar dict."""
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
# VERWERKING
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> list[dict]:
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 55}")
    print(f"  {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 55}")

    runs = []
    for run in range(1, RUNS + 1):
        print(f"\n   Run {run}/{RUNS} — cold start...")
        totaal_start = time.time()
        try:
            deadline = time.time() + FACTUUR_TIMEOUT_SECONDEN
            with factuur_timeout():
                ontlaad_model()
                _resterende_runtijd(deadline)
                afbeeldingen = stap1_pdf_naar_afbeeldingen(pdf_pad)
                _resterende_runtijd(deadline)
                resultaat = stap2_visie_llm(afbeeldingen, categorie, deadline)

            totaal_tijd = time.time() - totaal_start
            status = "JSON OK" if resultaat is not None else "FOUT"
            print(f"\n   Stap 3: {status} ({totaal_tijd:.1f}s)")

            runs.append({
                "bestand": pdf_pad.name,
                "categorie": categorie,
                "run": run,
                "success": resultaat is not None,
                "tijd_totaal": round(totaal_tijd, 2),
                "extracted": resultaat,
                "ruwe_output": json.dumps(resultaat, ensure_ascii=False) if resultaat else "",
            })
        except FactuurTimeout as e:
            print(f"   TIMEOUT na {FACTUUR_TIMEOUT_SECONDEN // 60} min: {pdf_pad.name} run{run}")
            runs.append(timeout_run(pdf_pad, run, e))

    return runs


def timeout_run(pdf_pad: Path, run: int, fout: Exception) -> dict:
    categorie = pdf_pad.parent.name
    return {
        "bestand": pdf_pad.name,
        "categorie": categorie,
        "run": run,
        "success": False,
        "tijd_totaal": FACTUUR_TIMEOUT_SECONDEN,
        "extracted": None,
        "ruwe_output": f"Timeout: {fout}",
    }


def timeout_runs(pdf_pad: Path, fout: Exception) -> list[dict]:
    categorie = pdf_pad.parent.name
    return [{
        "bestand": pdf_pad.name,
        "categorie": categorie,
        "run": run,
        "success": False,
        "tijd_totaal": FACTUUR_TIMEOUT_SECONDEN,
        "extracted": None,
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

    print(f"\n   Visie-pipeline (Mac, baseline MLX) — Model: {MLX_MODEL}")
    print(f"   Facturen: {len(pdfs)}  |  Runs per factuur: {RUNS}")
    print(f"   Geen OCR — vision-model leest afbeeldingen rechtstreeks")

    alle_runs = []
    for pdf_pad in pdfs:
        if not pdf_pad.exists():
            print(f"   Bestand niet gevonden: {pdf_pad}")
            continue

        # Sla over als alle runs al bestaan
        categorie = pdf_pad.parent.name
        alle_runs_bestaan = all(
            (RESULTATEN_MAP / categorie / f"{pdf_pad.stem}_run{r}.json").exists()
            for r in range(1, RUNS + 1)
        )
        if alle_runs_bestaan:
            print(f"   Overgeslagen (alle {RUNS} runs bestaan al): {pdf_pad.name}")
            continue

        runs = verwerk_factuur(pdf_pad)
        for run in runs:
            sla_op_als_json(run)
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
        print(f"\n  [{status}] {r['categorie']}/{r['bestand']} run{r['run']} ({r['tijd_totaal']}s)")


if __name__ == "__main__":
    main()
