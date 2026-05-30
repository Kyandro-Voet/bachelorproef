"""
Visiepipeline met finetuned model (Mac): PDF → afbeeldingen → MLX VLM → JSON
Geen OCR — het vision-model leest de documenten rechtstreeks.

Model: finetuning/vision/mlx_model (MLX 4-bit gekwantiseerd)
Input:  data/testing/<categorie>/<naam>.pdf
Output: resultaten/mac/finetuned/visie/<categorie>/<naam>_run<N>.json
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import gc
from pathlib import Path

from pdf2image import convert_from_path, pdfinfo_from_path
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
MLX_MODEL_PAD = Path(__file__).resolve().parent.parent.parent.parent / "finetuning/vision/mlx_model"
DPI = 100
MAX_BREEDTE = 1024
PAGINAS_PER_BATCH = 5
MLX_MAX_TOKENS = 8192
MLX_FALLBACK_MAX_TOKENS = 4096
MLX_TIMEOUT = 600
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "mac/finetuned/visie"
RESULTATEN_MAP = Path("resultaten/mac/finetuned/visie")
RUNS = 3

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
# MLX MODEL CACHE
# ──────────────────────────────────────────────
_mlx_model = None
_mlx_processor = None
_mlx_config = None


def ontlaad_mlx_model() -> None:
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


# ──────────────────────────────────────────────
# INFERENTIE
# ──────────────────────────────────────────────
def mlx_inferentie(afbeelding: Image.Image, prompt_tekst: str, max_tokens: int = MLX_MAX_TOKENS) -> tuple[str, float]:
    helper = Path(__file__).resolve().parent / "mlx_vision_infer.py"
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
        afbeelding.save(img_file, format="JPEG", quality=75)
        temp_img = img_file.name
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as prompt_file:
        prompt_file.write(prompt_tekst)
        temp_prompt = prompt_file.name
    start = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(helper),
                "--model", str(MLX_MODEL_PAD),
                "--image", temp_img,
                "--prompt", temp_prompt,
                "--max-tokens", str(max_tokens),
            ],
            text=True,
            capture_output=True,
            timeout=MLX_TIMEOUT,
        )
        tijd = time.time() - start
        if result.returncode != 0:
            fouttekst = (result.stderr or result.stdout).strip()
            regels = [r for r in fouttekst.splitlines() if r.strip()]
            preview = "\n".join(regels[-8:]) if regels else "onbekend"
            print(f"      MLX batch overgeslagen door fout (exit {result.returncode}):\n{preview}")
            return "", round(tijd, 2)
        return result.stdout.strip(), round(tijd, 2)
    except subprocess.TimeoutExpired:
        tijd = time.time() - start
        print(f"      MLX batch overgeslagen: timeout na {MLX_TIMEOUT}s")
        return "", round(tijd, 2)
    finally:
        for pad in (temp_img, temp_prompt):
            try:
                os.unlink(pad)
            except OSError:
                pass


# ──────────────────────────────────────────────
# AFBEELDING HULPFUNCTIES
# ──────────────────────────────────────────────
def schaal_afbeelding(img: Image.Image, max_breedte: int = MAX_BREEDTE) -> Image.Image:
    if img.width <= max_breedte:
        return img
    ratio = max_breedte / img.width
    return img.resize((max_breedte, int(img.height * ratio)), Image.LANCZOS)


def combineer_paginas(afbeeldingen: list[Image.Image]) -> Image.Image:
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


def verwerk_in_batches(
    paginas: list[Image.Image],
    prompt: str,
    start_index: int = 0,
    paginas_per_batch: int = PAGINAS_PER_BATCH,
    max_tokens: int = MLX_MAX_TOKENS,
) -> tuple[dict | None, str, float]:
    batch_resultaten = []
    ruwe_outputs = []
    totaal_tijd = 0.0
    batches = [paginas[i:i + paginas_per_batch] for i in range(0, len(paginas), paginas_per_batch)]

    for b, batch in enumerate(batches):
        eerste = start_index + b * paginas_per_batch + 1
        laatste = eerste + len(batch) - 1
        pagina_nrs = f"{eerste}-{laatste}"
        print(f"      Batch {b + 1}/{len(batches)} (pagina's {pagina_nrs})...")
        gecombineerd = combineer_paginas(batch)
        output, tijd = mlx_inferentie(gecombineerd, prompt, max_tokens=max_tokens)
        totaal_tijd += tijd
        ruwe_outputs.append(output)

        geparsed = parse_json(output)
        if geparsed:
            batch_resultaten.append(geparsed)
        else:
            preview = repr(output[:300]) if output else "(leeg)"
            print(f"      Batch {b + 1}: geen geldige JSON — {preview}")

    return _merge_resultaten(batch_resultaten), "\n\n".join(ruwe_outputs), round(totaal_tijd, 2)


def verwerk_pdf_in_batches(pdf_pad: Path, prompt: str) -> tuple[dict | None, str, float]:
    info = pdfinfo_from_path(str(pdf_pad))
    aantal_paginas = int(info["Pages"])
    batch_resultaten = []
    ruwe_outputs = []
    totaal_tijd = 0.0
    totaal_batches = -(-aantal_paginas // PAGINAS_PER_BATCH)

    for batch_index, eerste in enumerate(range(1, aantal_paginas + 1, PAGINAS_PER_BATCH), start=1):
        laatste = min(eerste + PAGINAS_PER_BATCH - 1, aantal_paginas)
        print(f"      PDF pagina's {eerste}-{laatste} converteren ({DPI} DPI)...")
        paginas = convert_from_path(str(pdf_pad), dpi=DPI, first_page=eerste, last_page=laatste)
        print(f"      MLX batch {batch_index}/{totaal_batches} ({len(paginas)} pagina's)...")
        extracted, ruwe_output, tijd = verwerk_in_batches(paginas, prompt, start_index=eerste - 1)
        totaal_tijd += tijd
        if extracted:
            batch_resultaten.append(extracted)
            ruwe_outputs.append(ruwe_output)
        elif len(paginas) > 1:
            print("      Batch faalde; retry per pagina met lagere max_tokens...")
            for offset, pagina in enumerate(paginas):
                pagina_nr = eerste + offset
                extracted_page, raw_page, tijd_page = verwerk_in_batches(
                    [pagina],
                    prompt,
                    start_index=pagina_nr - 1,
                    paginas_per_batch=1,
                    max_tokens=MLX_FALLBACK_MAX_TOKENS,
                )
                totaal_tijd += tijd_page
                if extracted_page:
                    batch_resultaten.append(extracted_page)
                ruwe_outputs.append(raw_page)
                gc.collect()
                ontlaad_mlx_model()
        else:
            ruwe_outputs.append(ruwe_output)

        for pagina in paginas:
            pagina.close()
        del paginas
        gc.collect()
        ontlaad_mlx_model()

    return _merge_resultaten(batch_resultaten), "\n\n".join(ruwe_outputs), round(totaal_tijd, 2)


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
    uitvoer_map = RESULTATEN_MAP / resultaat["categorie"]
    uitvoer_map.mkdir(parents=True, exist_ok=True)
    stem = Path(resultaat["bestand"]).stem
    pad = uitvoer_map / f"{stem}_run{resultaat['run']}.json"
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
    with open(pad, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"   Opgeslagen: {pad}")


# ──────────────────────────────────────────────
# VERWERKING (met 3 aparte runs)
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> list[dict]:
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

        print(f"\n   PDF → afbeeldingen per batch ({DPI} DPI)...")
        extracted, ruwe_output, _ = verwerk_pdf_in_batches(pdf_pad, prompt)

        tijd_totaal = round(time.time() - run_start, 2)
        status = "JSON OK" if extracted is not None else "JSON FOUT"
        print(f"   Run {run}: {status} ({tijd_totaal}s totaal incl. PDF-conversie)")

        runs.append({
            "bestand": pdf_pad.name,
            "categorie": categorie,
            "run": run,
            "success": extracted is not None,
            "tijd_totaal": tijd_totaal,
            "extracted": extracted,
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
        "ruwe_output": f"Timeout: {fout}",
    } for run in range(1, RUNS + 1)]


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    if not MLX_MODEL_PAD.exists():
        print(f"   FOUT: MLX model niet gevonden: {MLX_MODEL_PAD}")
        print("   Voer eerst finetuning/vision/convert_mlx.py uit.")
        sys.exit(1)

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

    print(f"\n   Visiepipeline finetuned (Mac) — MLX model: {MLX_MODEL_PAD.name}")
    print(f"   Facturen: {len(pdfs)}  |  Runs per factuur: {RUNS}")

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
