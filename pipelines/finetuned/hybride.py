"""
Hybride pipeline met finetuned modellen:
  PDF → lay-out analyse → routing → JSON

Digitale PDF:
  - pdfplumber extraheert tekst en tabellen direct
  - Ingebedde afbeeldingen → finetuned MLX vision model

Gescande PDF:
  - Tesseract blokdichtheid: tekstblokken → tekst-LLM, visuele blokken → vision model

Gemeenschappelijk:
  - Visuele zones per pagina gecombineerd in één vision-call
  - Definitieve JSON via finetuned tekst-LLM (Ollama)
  - Lay-out analyse eenmalig, LLM calls 3x voor betrouwbare tijdmeting

Tekst model:  finetuning/text/Qwen3-8B.Q4_K_M.gguf (Ollama 'qwen3-ft:8b')
Vision model: finetuning/vision/mlx_model (MLX 4-bit)
Input:  documents_testing/<categorie>/<naam>.pdf
Output: resultaten/hybride_ft/<categorie>/<naam>.json
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ollama
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
from pymongo import MongoClient
import pytesseract


# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
TEKST_MODEL = "qwen3-8b-finetuned"
MODELFILE_PAD = Path(__file__).resolve().parent.parent.parent / "finetuning/text/Modelfile"
MLX_MODEL_PAD = Path(__file__).resolve().parent.parent.parent / "finetuning/vision/mlx_model"
OCR_TALEN = "eng+ita"
DPI = 300
MAX_BREEDTE = 1280
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "finetuned/hybride"
RUNS = 3

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "bachelorproef"
MONGO_COLLECTION = "resultaten"

SYSTEM_PROMPT = """You are a precise data extraction assistant specialized in invoice processing.
Your sole task is to extract structured information from invoice documents and return it as valid JSON.

Rules you must follow:
- Return ONLY a valid JSON object. No explanation, no markdown, no code fences.
- If a field is not present in the document, set its value to null.
- Do not infer or guess values that are not explicitly stated.
- Normalize all amounts to numbers (e.g. "7.973,12 €" → 7973.12).
- Normalize all dates to ISO 8601 format (YYYY-MM-DD).
- If multiple values exist for a field, return an array."""

USER_PROMPT = """
Extract all invoice data from the document below and return it as a JSON object
that strictly follows this schema:

{
  "invoice": {
    "number":        <string>,
    "date":          <ISO date>,
    "due_date":      <ISO date>,
    "type":          <string>
  },
  "supplier": {
    "name":          <string>,
    "vat_number":    <string>,
    "tax_code":      <string>,
    "address": {
      "street":      <string>,
      "postal_code": <string>,
      "city":        <string>,
      "country":     <string>
    },
    "email":         <string>,
    "phone":         <string>,
    "website":       <string>
  },
  "customer": {
    "name":          <string>,
    "vat_number":    <string>,
    "tax_code":      <string>,
    "customer_code": <string>,
    "address": {
      "street":      <string>,
      "postal_code": <string>,
      "city":        <string>,
      "country":     <string>
    }
  },
  "contract": {
    "number":           <string>,
    "product":          <string>,
    "pod_code":         <string>,
    "customer_type":    <string>,
    "period_start":     <ISO date>,
    "period_end":       <ISO date>,
    "delivery_address": <string>
  },
  "line_items": [
    {
      "description": <string>,
      "amount":      <number>
    }
  ],
  "totals": {
    "subtotal":        <number>,
    "tax_base":        <number>,
    "tax_rate_pct":    <number>,
    "tax_amount":      <number>,
    "total_due":       <number>,
    "currency":        <string, ISO 4217>
  },
  "payment": {
    "method":          <string>,
    "bank":            <string>,
    "status":          <string>
  }
}

--- DOCUMENT START ---

"""

VISIE_ZONES_PROMPT = """/no_think
The image contains one or more cropped regions from an invoice (tables, figures, or other elements),
stacked vertically and separated by gray bars.
Extract all relevant data you can read (amounts, dates, names, codes, quantities) as plain text.
Write each piece of data as "label: value" on a separate line.
Skip purely decorative elements (logos without text, background graphics).
Return only readable text, no JSON.
"""


# ──────────────────────────────────────────────
# MLX VLM
# ──────────────────────────────────────────────
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
        sys.exit(1)

    print(f"   MLX vision model laden: {MLX_MODEL_PAD}...")
    _mlx_model, _mlx_processor = load(str(MLX_MODEL_PAD))
    _mlx_config = load_config(str(MLX_MODEL_PAD))
    print("   MLX model geladen.")
    return _mlx_model, _mlx_processor, _mlx_config


def mlx_inferentie(afbeelding: Image.Image, prompt_tekst: str) -> str:
    """Voer MLX VLM inferentie uit op een PIL afbeelding. Geeft tekst terug."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _laad_mlx_model()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        afbeelding.save(f, format="JPEG", quality=85)
        temp_pad = f.name

    try:
        formatted = apply_chat_template(processor, config, prompt_tekst, num_images=1)
        output = generate(
            model, processor, formatted,
            image=temp_pad,
            max_tokens=1024,
            temp=0.1,
            verbose=False,
        )
    finally:
        os.unlink(temp_pad)

    return output.text if hasattr(output, "text") else str(output)


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


def ontlaad_modellen() -> None:
    """Reset beide modellen voor een cold run."""
    global _mlx_model, _mlx_processor, _mlx_config
    _wacht_op_unload(TEKST_MODEL)
    # MLX vision model resetten
    _mlx_model = None
    _mlx_processor = None
    _mlx_config = None


# ──────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────
def setup_model() -> None:
    try:
        ollama.show(TEKST_MODEL)
        print(f"   Model '{TEKST_MODEL}' beschikbaar in Ollama.")
    except Exception:
        print(f"   Model '{TEKST_MODEL}' niet gevonden — aanmaken uit {MODELFILE_PAD}...")
        if not MODELFILE_PAD.exists():
            print(f"   FOUT: Modelfile niet gevonden: {MODELFILE_PAD}")
            sys.exit(1)
        result = subprocess.run(
            ["ollama", "create", TEKST_MODEL, "-f", str(MODELFILE_PAD)],
            cwd=str(MODELFILE_PAD.parent),
            text=True,
        )
        if result.returncode != 0:
            print(f"   FOUT bij aanmaken model (returncode {result.returncode})")
            sys.exit(result.returncode)
        print(f"   Model '{TEKST_MODEL}' aangemaakt.")


# ──────────────────────────────────────────────
# OPSLAG
# ──────────────────────────────────────────────
def sla_run_op(resultaat: dict) -> None:
    """Sla één run op in MongoDB."""
    document = {
        "bestand": resultaat["bestand"],
        "categorie": resultaat.get("categorie", ""),
        "model": f"{TEKST_MODEL} + {MLX_MODEL_PAD.name}",
        "pipeline": PIPELINE,
        "run": resultaat["run"],
        "success": resultaat["success"],
        "tijd_totaal": resultaat["tijd_totaal"],
        "extracted": resultaat["extracted"],
        "ocr_tekst": resultaat["ocr_tekst"],
        "ruwe_output": resultaat["ruwe_output"],
        "tabellen_gevonden": resultaat["tabellen_gevonden"],
        "visuele_zones_gevonden": resultaat["afbeeldingen_gevonden"],
    }
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client[MONGO_DB][MONGO_COLLECTION].insert_one(document)
        print(f"   MongoDB: {PIPELINE} run{resultaat['run']} — {resultaat['bestand']}")
    except Exception as e:
        print(f"   MongoDB FOUT: {e}")


# ──────────────────────────────────────────────
# HULPFUNCTIES
# ──────────────────────────────────────────────
def formatteer_tabel(rijen: list) -> str:
    regels = []
    for rij in rijen:
        cellen = [str(cel or "").strip() for cel in rij]
        if any(cellen):
            regels.append(" | ".join(cellen))
    return "\n".join(regels)


def afbeelding_naar_base64_of_pil(img: Image.Image, max_breedte: int = MAX_BREEDTE) -> Image.Image:
    if img.width > max_breedte:
        ratio = max_breedte / img.width
        img = img.resize((max_breedte, int(img.height * ratio)), Image.LANCZOS)
    return img


def crop_zone(img: Image.Image, bbox: tuple, dpi: int) -> Image.Image:
    schaal = dpi / 72.0
    x0 = max(0, int(bbox[0] * schaal))
    y0 = max(0, int(bbox[1] * schaal))
    x1 = min(img.width, int(bbox[2] * schaal))
    y1 = min(img.height, int(bbox[3] * schaal))
    return img.crop((x0, y0, x1, y1))


def combineer_zones(crops: list[Image.Image]) -> Image.Image:
    if len(crops) == 1:
        return crops[0]
    scheiding = 8
    breedte = max(c.width for c in crops)
    hoogte = sum(c.height for c in crops) + scheiding * (len(crops) - 1)
    gecombineerd = Image.new("RGB", (breedte, hoogte), color=(180, 180, 180))
    y = 0
    for crop in crops:
        gecombineerd.paste(crop, (0, y))
        y += crop.height + scheiding
    return gecombineerd


def splits_gescande_pagina(img: Image.Image) -> tuple[str, list[tuple]]:
    data = pytesseract.image_to_data(img, lang=OCR_TALEN, output_type=pytesseract.Output.DICT)

    blok_bboxen: dict[int, tuple] = {}
    blok_woorden: dict[int, list[str]] = {}

    for i in range(len(data["level"])):
        nr = data["block_num"][i]
        lvl = data["level"][i]

        if lvl == 2:
            blok_bboxen[nr] = (
                data["left"][i], data["top"][i],
                data["left"][i] + data["width"][i],
                data["top"][i] + data["height"][i],
            )
            blok_woorden.setdefault(nr, [])
        elif lvl == 5 and data["conf"][i] >= 30 and data["text"][i].strip():
            blok_woorden.setdefault(nr, []).append(data["text"][i])

    tekst_blokken: list[str] = []
    visuele_bboxen: list[tuple] = []
    schaal = 72.0 / DPI

    for nr, bbox in blok_bboxen.items():
        if (bbox[2] - bbox[0]) < 50 or (bbox[3] - bbox[1]) < 20:
            continue
        woorden = blok_woorden.get(nr, [])
        if woorden:
            tekst_blokken.append(" ".join(woorden))
        else:
            visuele_bboxen.append((
                bbox[0] * schaal, bbox[1] * schaal,
                bbox[2] * schaal, bbox[3] * schaal,
            ))

    return "\n".join(tekst_blokken), visuele_bboxen


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
# PIPELINE STAPPEN (lay-out analyse, eenmalig)
# ──────────────────────────────────────────────
def stap1_layout_analyse(pdf_pad: Path) -> list[dict]:
    print(f"\n   Stap 1: Lay-out analyse (pdfplumber)...")
    pagina_data = []

    with pdfplumber.open(pdf_pad) as pdf:
        for i, pagina in enumerate(pdf.pages):
            tabellen = pagina.find_tables()
            tabel_bboxen = [t.bbox for t in tabellen]

            if tabel_bboxen:
                def buiten_tabellen(obj):
                    for bbox in tabel_bboxen:
                        if (obj.get("x0", 0) >= bbox[0] - 2 and
                                obj.get("top", 0) >= bbox[1] - 2 and
                                obj.get("x1", 0) <= bbox[2] + 2 and
                                obj.get("bottom", 0) <= bbox[3] + 2):
                            return False
                    return True
                tekst = pagina.filter(buiten_tabellen).extract_text() or ""
            else:
                tekst = pagina.extract_text() or ""

            gescand = len(tekst) < 50

            if gescand:
                pagina_data.append({
                    "pagina_nr": i, "tekst": "", "tabel_teksten": [],
                    "visuele_bboxen": [], "afbeelding_bboxen": [], "gescand": True,
                })
                print(f"   Pagina {i + 1}: gescand")
                continue

            tabel_teksten = []
            for tabel in tabellen:
                rijen = tabel.extract()
                if rijen:
                    tabel_teksten.append(formatteer_tabel(rijen))

            afb_bboxen = [
                (obj["x0"], obj["top"], obj["x1"], obj["bottom"])
                for obj in pagina.images
                if obj["width"] > 30 and obj["height"] > 30
            ]

            pagina_data.append({
                "pagina_nr": i, "tekst": tekst.strip(),
                "tabel_teksten": tabel_teksten, "visuele_bboxen": [],
                "afbeelding_bboxen": afb_bboxen, "gescand": False,
            })
            print(f"   Pagina {i + 1}: {len(tekst)} tekens, "
                  f"{len(tabel_teksten)} tabel(len), {len(afb_bboxen)} afbeelding(en)")

    return pagina_data


def stap2_converteer_naar_afbeeldingen(pdf_pad: Path, pagina_data: list[dict]) -> dict[int, Image.Image]:
    benodigde = {p["pagina_nr"] for p in pagina_data if p["gescand"] or p["afbeelding_bboxen"]}
    if not benodigde:
        print(f"\n   Stap 2: Geen afbeeldingen nodig.")
        return {}

    print(f"\n   Stap 2: PDF → afbeeldingen ({DPI} DPI) voor pagina('s) "
          f"{sorted(p + 1 for p in benodigde)}...")
    start = time.time()
    afbeeldingen: dict[int, Image.Image] = {}
    for nr in sorted(benodigde):
        imgs = convert_from_path(str(pdf_pad), dpi=DPI, first_page=nr + 1, last_page=nr + 1)
        afbeeldingen[nr] = imgs[0]
    print(f"   {len(afbeeldingen)} pagina('s) omgezet in {time.time() - start:.1f}s")
    return afbeeldingen


def stap3_verwerk_gescande_paginas(pagina_data: list[dict], afbeeldingen: dict[int, Image.Image]) -> None:
    gescande = [p for p in pagina_data if p["gescand"]]
    if not gescande:
        return
    print(f"\n   Stap 3: Tesseract analyse voor {len(gescande)} gescande pagina('s)...")
    for pdata in gescande:
        nr = pdata["pagina_nr"]
        tekst, visuele_bboxen = splits_gescande_pagina(afbeeldingen[nr])
        pdata["tekst"] = tekst
        pdata["visuele_bboxen"] = visuele_bboxen
        print(f"   Pagina {nr + 1}: {len(tekst)} OCR-tekens, {len(visuele_bboxen)} visuele zone(s)")


def stap4_verwerk_visuele_zones(pagina_data: list[dict], afbeeldingen: dict[int, Image.Image]) -> list[str]:
    """Stuur visuele zones naar het MLX vision model."""
    resultaten: list[str] = []
    totaal_zones = sum(
        len(p["visuele_bboxen"]) + len(p["afbeelding_bboxen"])
        for p in pagina_data
    )

    if totaal_zones == 0:
        print(f"\n   Stap 4: Geen visuele zones.")
        return []

    print(f"\n   Stap 4: {totaal_zones} visuele zone(s) → MLX vision model...")
    start = time.time()

    for pdata in pagina_data:
        pnr = pdata["pagina_nr"]
        img = afbeeldingen.get(pnr)
        if img is None:
            continue

        alle_bboxen = pdata["visuele_bboxen"] + pdata["afbeelding_bboxen"]
        if not alle_bboxen:
            continue

        crops = [
            crop_zone(img, bbox, DPI)
            for bbox in alle_bboxen
            if crop_zone(img, bbox, DPI).width >= 50 and crop_zone(img, bbox, DPI).height >= 20
        ]
        if not crops:
            continue

        gecombineerd = afbeelding_naar_base64_of_pil(combineer_zones(crops))
        print(f"   Pagina {pnr + 1}: {len(crops)} zone(s) → MLX...")
        tekst = mlx_inferentie(gecombineerd, VISIE_ZONES_PROMPT)
        if tekst:
            resultaten.append(f"[Visuele zones — pagina {pnr + 1}]\n{tekst}")

    print(f"   Visuele zones verwerkt in {time.time() - start:.1f}s")
    return resultaten


def stap5_bouw_context(pagina_data: list[dict]) -> str:
    tekst_delen = []
    for pdata in pagina_data:
        label = "gescand/OCR" if pdata["gescand"] else "tekst"
        deel = f"--- Pagina {pdata['pagina_nr'] + 1} ({label}) ---\n{pdata['tekst']}"
        for j, tabel_tekst in enumerate(pdata.get("tabel_teksten", [])):
            deel += f"\n\n[Tabel {j + 1}]\n{tabel_tekst}"
        tekst_delen.append(deel)
    return "\n\n".join(tekst_delen)


def stap6_llm(tekst_context: str, visuele_resultaten: list[str]) -> tuple[str, float]:
    """Stuur context naar het finetuned tekst-LLM. Geeft (ruwe_output, tijd) terug."""
    context = tekst_context
    if visuele_resultaten:
        context += (
            "\n\n--- VISUELE ZONES (uitgelezen door vision-model) ---\n"
            + "\n\n".join(visuele_resultaten)
        )

    start = time.time()
    response = ollama.chat(
        model=TEKST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + context},
        ],
        format="json",
        options={"temperature": 0.1, "num_ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        think=False,
    )
    tijd = time.time() - start
    return response.message.content, round(tijd, 2)


# ──────────────────────────────────────────────
# VERWERKING (met 3 aparte runs)
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> list[dict]:
    """Verwerk één factuur in RUNS aparte cold runs.
    Stap 1 (pdfplumber lay-out) loopt eenmalig — geen model, deterministisch.
    Stap 2-6 (afbeeldingen, OCR, visie, LLM) lopen per run opnieuw.
    """
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 60}")
    print(f"  {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 60}")

    # Lay-out analyse eenmalig (pdfplumber, geen model)
    pagina_data_basis = stap1_layout_analyse(pdf_pad)

    totaal_tabellen = (
        sum(len(p["tabel_teksten"]) for p in pagina_data_basis if not p["gescand"])
        + sum(len(p["visuele_bboxen"]) for p in pagina_data_basis if p["gescand"])
    )
    totaal_afbeeldingen = sum(len(p["afbeelding_bboxen"]) for p in pagina_data_basis)

    runs = []
    for run in range(1, RUNS + 1):
        print(f"\n   Run {run}/{RUNS} — cold start...")
        ontlaad_modellen()

        run_start = time.time()

        # Deep copy zodat stap3 visuele zones en OCR-tekst opnieuw kan invullen
        pagina_data = copy.deepcopy(pagina_data_basis)

        afbeeldingen = stap2_converteer_naar_afbeeldingen(pdf_pad, pagina_data)
        stap3_verwerk_gescande_paginas(pagina_data, afbeeldingen)
        visuele_resultaten = stap4_verwerk_visuele_zones(pagina_data, afbeeldingen)
        tekst_context = stap5_bouw_context(pagina_data)
        ruwe_output, _ = stap6_llm(tekst_context, visuele_resultaten)

        tijd_totaal = round(time.time() - run_start, 2)
        extracted = parse_json(ruwe_output)
        status = "JSON OK" if extracted is not None else "JSON FOUT"
        print(f"   Run {run}: {status} ({tijd_totaal}s totaal incl. OCR + visie)")

        runs.append({
            "bestand": pdf_pad.name,
            "categorie": categorie,
            "model": f"{TEKST_MODEL} + MLX vision",
            "pipeline": PIPELINE,
            "run": run,
            "success": extracted is not None,
            "tijd_totaal": tijd_totaal,
            "extracted": extracted,
            "ocr_tekst": tekst_context,
            "ruwe_output": ruwe_output,
            "tabellen_gevonden": totaal_tabellen,
            "afbeeldingen_gevonden": totaal_afbeeldingen,
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

    print(f"\n   Hybride pipeline (finetuned)")
    print(f"   Tekst: {TEKST_MODEL}  |  Vision: MLX {MLX_MODEL_PAD.name}")
    print(f"   Facturen: {len(pdfs)}  |  LLM runs per factuur: {RUNS}")

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
        print(f"  [{status}] {r['bestand']} run{r['run']} ({r['tijd_totaal']}s)")


if __name__ == "__main__":
    main()
