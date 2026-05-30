"""
Hybride pipeline (Mac): PDF → lay-out analyse → routing → JSON

Digitale PDF:
  - pdfplumber extraheert tekst en tabellen direct (geen vision inference voor tabellen)
  - Ingebedde afbeeldingen → vision model (qwen3-vl:8b)

Gescande PDF:
  - Tesseract blokdichtheid: blokken zonder herkende woorden → visuele zones
  - Visuele zones → vision model, tekstblokken → tekst-LLM

Gemeenschappelijk:
  - Alle visuele zones per pagina gecombineerd in één vision-call
  - Definitieve JSON via tekst-LLM (qwen3:8b)
"""

import base64
import copy
import io
import json
import re
import sys
import time
from pathlib import Path

import ollama
from ollama import Client
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
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
TEKST_MODEL = "qwen3:8b-q4_K_M"
VISIE_MODEL = "qwen3-vl:8b"
OCR_TALEN = "eng+ita"
DPI = 300
MAX_BREEDTE = 1280
DOCUMENTS_MAP = Path("data/testing")
PIPELINE = "mac/baseline/hybride"
RESULTATEN_MAP = Path("resultaten/mac/baseline/hybride")
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

VISIE_ZONES_PROMPT = """/no_think
The images are cropped regions from an invoice (tables, figures, or other elements).
Extract all relevant data you can read (amounts, dates, names, codes, quantities) as plain text.
Write each piece of data as "label: value" on a separate line.
Skip purely decorative elements (logos without text, background graphics).
Return only readable text, no JSON.
"""


# ──────────────────────────────────────────────
# HULPFUNCTIES
# ──────────────────────────────────────────────
def toon_gpu_cpu_verdeling(model: str) -> None:
    try:
        for m in ollama.ps().models:
            if m.model.startswith(model.split(":")[0]):
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


def ontlaad_modellen() -> None:
    for model in (TEKST_MODEL, VISIE_MODEL):
        _wacht_op_unload(model)


def ontlaad_tekstmodel() -> None:
    _wacht_op_unload(TEKST_MODEL)


def ontlaad_visiemodel() -> None:
    _wacht_op_unload(VISIE_MODEL)


def formatteer_tabel(rijen: list) -> str:
    regels = []
    for rij in rijen:
        cellen = [str(cel or "").strip() for cel in rij]
        if any(cellen):
            regels.append(" | ".join(cellen))
    return "\n".join(regels)


def afbeelding_naar_base64(img: Image.Image, max_breedte: int = MAX_BREEDTE) -> str:
    if img.width > max_breedte:
        ratio = max_breedte / img.width
        img = img.resize((max_breedte, int(img.height * ratio)), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def crop_tabel(img: Image.Image, bbox: tuple, dpi: int) -> Image.Image:
    schaal = dpi / 72.0
    x0 = max(0, int(bbox[0] * schaal))
    y0 = max(0, int(bbox[1] * schaal))
    x1 = min(img.width, int(bbox[2] * schaal))
    y1 = min(img.height, int(bbox[3] * schaal))
    return img.crop((x0, y0, x1, y1))


def is_relevante_pdf_afbeelding(obj: dict) -> bool:
    """Filter embedded PDF-afbeeldingen: behoud grafieken, sla header-logo's over."""
    w = obj.get("width", 0)
    h = obj.get("height", 0)
    if not ((w * h > 10000) and (w > 20) and (h > 20)):
        return False

    linksboven_header = obj.get("x0", 0) < 220 and obj.get("top", 0) < 120
    logo_formaat = w < 250 and h < 140
    if linksboven_header and logo_formaat:
        return False

    return True


def combineer_zones_naar_afbeelding(crops: list[Image.Image]) -> Image.Image:
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
    data = pytesseract.image_to_data(
        img, lang=OCR_TALEN, output_type=pytesseract.Output.DICT
    )

    blok_bboxen: dict[int, tuple] = {}
    blok_woorden: dict[int, list[str]] = {}

    for i in range(len(data["level"])):
        nr = data["block_num"][i]
        lvl = data["level"][i]

        if lvl == 2:
            blok_bboxen[nr] = (
                data["left"][i],
                data["top"][i],
                data["left"][i] + data["width"][i],
                data["top"][i] + data["height"][i],
            )
            blok_woorden.setdefault(nr, [])
        elif lvl == 5 and data["conf"][i] >= 30 and data["text"][i].strip():
            blok_woorden.setdefault(nr, []).append(data["text"][i])

    tekst_blokken: list[str] = []
    schaal = 72.0 / DPI
    heeft_content = False

    for nr, bbox in blok_bboxen.items():
        if (bbox[2] - bbox[0]) < 50 or (bbox[3] - bbox[1]) < 20:
            continue
        woorden = blok_woorden.get(nr, [])
        if woorden:
            tekst_blokken.append(" ".join(woorden))
            heeft_content = True
        else:
            heeft_content = True

    visuele_bboxen = []
    if heeft_content:
        # Als de pagina content bevat, sturen we de volledige pagina als één zone naar het visuele model
        visuele_bboxen.append((0.0, 0.0, img.width * schaal, img.height * schaal))

    return "\n".join(tekst_blokken), visuele_bboxen


def parse_json(ruwe_output: str) -> dict | None:
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
        return json.loads(opgeschoond)
    except json.JSONDecodeError as e:
        print(f"   JSON parse fout: {e}")
        return None


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
        "model": f"{TEKST_MODEL} + {VISIE_MODEL}",
        "pipeline": PIPELINE,
        "run": resultaat["run"],
        "success": resultaat["success"],
        "tijd_totaal": resultaat["tijd_totaal"],
        "extracted": resultaat["extracted"],
        "ocr_tekst": resultaat["ocr_tekst"],
        "ruwe_output": resultaat["ruwe_output"],
        "tabellen_gevonden": resultaat["tabellen_gevonden"],
        "visuele_zones_gevonden": resultaat["visuele_zones_gevonden"],
    }
    with open(pad, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"   Opgeslagen: {pad}")


# ──────────────────────────────────────────────
# PIPELINE STAPPEN
# ──────────────────────────────────────────────
def stap1_layout_analyse(pdf_pad: Path) -> list[dict]:
    print(f"\n   Stap 1: Lay-out analyse met pdfplumber...")
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
                    "pagina_nr": i,
                    "tekst": "",
                    "tabel_teksten": [],
                    "visuele_bboxen": [],
                    "afbeelding_bboxen": [],
                    "gescand": True,
                })
                print(f"   Pagina {i+1}: gescand — Tesseract-analyse volgt in stap 3")
                continue

            tabel_teksten = []
            for tabel in tabellen:
                rijen = tabel.extract()
                if rijen:
                    tabel_teksten.append(formatteer_tabel(rijen))

            alle_afb = [obj for obj in pagina.images if obj.get("width", 0) > 20 and obj.get("height", 0) > 20]
            afb_bboxen = [
                (obj["x0"], obj["top"], obj["x1"], obj["bottom"])
                for obj in alle_afb
                if is_relevante_pdf_afbeelding(obj)
            ]

            pagina_data.append({
                "pagina_nr": i,
                "tekst": tekst.strip(),
                "tabel_teksten": tabel_teksten,
                "visuele_bboxen": [],
                "afbeelding_bboxen": afb_bboxen,
                "gescand": False,
            })

            status_delen = []
            if tabel_teksten:
                status_delen.append(f"{len(tabel_teksten)} tabel(len) via pdfplumber")
            if afb_bboxen:
                status_delen.append(f"{len(afb_bboxen)} afbeelding(en) → vision model")
            overgeslagen_afb = len(alle_afb) - len(afb_bboxen)
            if overgeslagen_afb > 0:
                status_delen.append(f"{overgeslagen_afb} afb. overgeslagen (logo/decoratief)")
            status = ", ".join(status_delen) if status_delen else "geen visuele zones"
            print(f"   Pagina {i+1}: {len(tekst)} tekst-tekens, {status}")

    return pagina_data


def stap2_converteer_naar_afbeeldingen(pdf_pad: Path, pagina_data: list[dict]) -> dict[int, Image.Image]:
    benodigde_paginas = {
        p["pagina_nr"] for p in pagina_data
        if p["gescand"] or p["afbeelding_bboxen"]
    }

    if not benodigde_paginas:
        print(f"\n   Stap 2: Geen afbeeldingen nodig — digitale PDF zonder visuele zones")
        return {}

    print(f"\n   Stap 2: PDF naar afbeeldingen ({DPI} DPI) voor pagina('s) "
          f"{sorted(p + 1 for p in benodigde_paginas)}...")
    start = time.time()

    afbeeldingen: dict[int, Image.Image] = {}
    for pagina_nr in sorted(benodigde_paginas):
        imgs = convert_from_path(
            str(pdf_pad), dpi=DPI,
            first_page=pagina_nr + 1,
            last_page=pagina_nr + 1,
        )
        afbeeldingen[pagina_nr] = imgs[0]

    print(f"   {len(afbeeldingen)} pagina('s) omgezet in {time.time() - start:.1f}s")
    return afbeeldingen


def stap3_verwerk_gescande_paginas(pagina_data: list[dict], afbeeldingen: dict[int, Image.Image]) -> None:
    gescande_paginas = [p for p in pagina_data if p["gescand"]]

    if not gescande_paginas:
        print(f"\n   Stap 3: Geen gescande pagina's — stap overgeslagen")
        return

    print(f"\n   Stap 3: Tesseract blokanalyse voor {len(gescande_paginas)} gescande pagina('s)...")
    start = time.time()

    for pdata in gescande_paginas:
        nr = pdata["pagina_nr"]
        tekst, visuele_bboxen = splits_gescande_pagina(afbeeldingen[nr])
        pdata["tekst"] = tekst
        pdata["visuele_bboxen"] = visuele_bboxen
        print(f"   Pagina {nr + 1}: {len(tekst)} OCR-tekens, "
              f"{len(visuele_bboxen)} visuele zone(s) → vision model")

    print(f"   Gescande pagina's verwerkt in {time.time() - start:.1f}s")


def stap4_verwerk_visuele_zones(pagina_data: list[dict], afbeeldingen: dict[int, Image.Image]) -> list[str]:
    totaal_zones = sum(
        len(p["visuele_bboxen"]) + len(p["afbeelding_bboxen"])
        for p in pagina_data
    )

    ontlaad_tekstmodel()

    if totaal_zones == 0:
        print(f"\n   Stap 4: Geen visuele zones — stap overgeslagen")
        return []

    print(f"\n   Stap 4: {totaal_zones} visuele zone(s) → {VISIE_MODEL} (één call totaal)...")
    start = time.time()

    alle_crops: list[Image.Image] = []
    for pdata in pagina_data:
        pnr = pdata["pagina_nr"]
        img = afbeeldingen.get(pnr)
        if img is None:
            continue

        alle_bboxen = pdata["visuele_bboxen"] + pdata["afbeelding_bboxen"]
        if len(alle_bboxen) > 3:
            schaal = 72.0 / DPI
            alle_bboxen = [(0.0, 0.0, img.width * schaal, img.height * schaal)]
            print(f"   Pagina {pnr + 1}: Meer dan 3 zones gedetecteerd → gecombineerd tot volledige pagina om Ollama-timeout te voorkomen")

        for bbox in alle_bboxen:
            crop = crop_tabel(img, bbox, DPI)
            if crop.width >= 50 and crop.height >= 20:
                alle_crops.append(crop)

    if not alle_crops:
        return []

    gecombineerd = combineer_zones_naar_afbeelding(alle_crops)
    b64 = afbeelding_naar_base64(gecombineerd)

    print(f"   {len(alle_crops)} zone(s) → 1 vision-call met 1 gecombineerde afbeelding...")
    client = Client(timeout=600.0)
    response = client.chat(
        model=VISIE_MODEL,
        messages=[{"role": "user", "content": VISIE_ZONES_PROMPT, "images": [b64]}],
        options={"temperature": 0.1, "num_ctx": 4096},
        keep_alive=0,
    )
    tekst = response.message.content.strip()
    ontlaad_visiemodel()

    print(f"   Visuele zones verwerkt in {time.time() - start:.1f}s")
    return [tekst] if tekst else []


def stap5_bouw_context(pagina_data: list[dict]) -> str:
    print(f"\n   Stap 5: Tekst samenstellen...")
    tekst_delen = []

    for pdata in pagina_data:
        label = "gescand/OCR" if pdata["gescand"] else "tekst"
        deel = f"--- Pagina {pdata['pagina_nr'] + 1} ({label}) ---\n{pdata['tekst']}"

        for j, tabel_tekst in enumerate(pdata.get("tabel_teksten", [])):
            deel += f"\n\n[Tabel {j + 1}]\n{tabel_tekst}"

        tekst_delen.append(deel)

    return opkuis_tekst("\n\n".join(tekst_delen))


def stap6_llm(tekst_context: str, visuele_resultaten: list[str], categorie: str) -> str:
    print(f"\n   Stap 6: Context naar {TEKST_MODEL} sturen...")
    ontlaad_visiemodel()
    start = time.time()

    context = tekst_context
    if visuele_resultaten:
        context += (
            "\n\n--- VISUELE ZONES (uitgelezen door vision-model) ---\n"
            + "\n\n".join(visuele_resultaten)
        )

    schema = CATEGORIE_SCHEMAS.get(categorie, CATEGORIE_SCHEMAS["electricity"])
    user_prompt = USER_PROMPT_TEMPLATE.format(schema=schema) + context

    client = Client(timeout=600.0)
    response = client.chat(
        model=TEKST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.1, "num_ctx": 16384, "cache_type_k": "q4_0", "cache_type_v": "q4_0"},
        keep_alive=0,
        think=False,
    )
    print(f"   Antwoord ontvangen in {time.time() - start:.1f}s")
    toon_gpu_cpu_verdeling(TEKST_MODEL)
    return response.message.content


def stap7_parse_json(ruwe_output: str) -> dict | None:
    print(f"\n   Stap 7: JSON parsen...")
    resultaat = parse_json(ruwe_output)
    if resultaat:
        print(f"   JSON succesvol geparsed!")
    else:
        print(f"   Ruwe output:\n{ruwe_output[:500]}")
    return resultaat


# ──────────────────────────────────────────────
# VERWERKING (met 3 aparte runs)
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> list[dict]:
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 55}")
    print(f"  {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 55}")

    pagina_data_basis = stap1_layout_analyse(pdf_pad)

    totaal_tabellen = (
        sum(len(p["tabel_teksten"]) for p in pagina_data_basis if not p["gescand"])
        + sum(len(p["visuele_bboxen"]) for p in pagina_data_basis if p["gescand"])
    )
    totaal_visuele_zones = sum(len(p["afbeelding_bboxen"]) for p in pagina_data_basis)

    runs = []
    for run in range(1, RUNS + 1):
        print(f"\n   Run {run}/{RUNS} — cold start...")
        run_start = time.time()
        try:
            ontlaad_modellen()

            pagina_data = copy.deepcopy(pagina_data_basis)

            afbeeldingen = stap2_converteer_naar_afbeeldingen(pdf_pad, pagina_data)
            stap3_verwerk_gescande_paginas(pagina_data, afbeeldingen)
            visuele_resultaten = stap4_verwerk_visuele_zones(pagina_data, afbeeldingen)
            tekst_context = stap5_bouw_context(pagina_data)
            ruwe_output = stap6_llm(tekst_context, visuele_resultaten, categorie)
            resultaat = stap7_parse_json(ruwe_output)

            tijd_totaal = round(time.time() - run_start, 2)
            status = "JSON OK" if resultaat is not None else "JSON FOUT"
            print(f"   Run {run}: {status} ({tijd_totaal}s totaal)")

            runs.append({
                "bestand": pdf_pad.name,
                "categorie": categorie,
                "run": run,
                "success": resultaat is not None,
                "tijd_totaal": tijd_totaal,
                "extracted": resultaat,
                "ocr_tekst": tekst_context,
                "ruwe_output": ruwe_output,
                "tabellen_gevonden": totaal_tabellen,
                "visuele_zones_gevonden": totaal_visuele_zones,
            })
        except Exception as e:
            tijd_totaal = round(time.time() - run_start, 2)
            print(f"   Run {run} mislukt door fout: {e}")
            runs.append({
                "bestand": pdf_pad.name,
                "categorie": categorie,
                "run": run,
                "success": False,
                "tijd_totaal": tijd_totaal,
                "extracted": None,
                "ocr_tekst": "",
                "ruwe_output": f"Fout: {e}",
                "tabellen_gevonden": totaal_tabellen,
                "visuele_zones_gevonden": totaal_visuele_zones,
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
        "tabellen_gevonden": 0,
        "visuele_zones_gevonden": 0,
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

    print(f"\n   Hybride pipeline (Mac) — Tekst: {TEKST_MODEL} | Visie: {VISIE_MODEL}")
    print(f"   Facturen: {len(pdfs)}  |  LLM runs per factuur: {RUNS}")

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
            for run in runs:
                sla_op_als_json(run)
            alle_runs.extend(runs)
        except FactuurTimeout as e:
            print(f"\n   TIMEOUT na {FACTUUR_TIMEOUT_SECONDEN // 60} min bij {pdf_pad.name}")
            runs = timeout_runs(pdf_pad, e)
            for run in runs:
                sla_op_als_json(run)
            alle_runs.extend(runs)
        except Exception as e:
            print(f"\n   FOUT bij verwerken van {pdf_pad.name}: {e}")
            for r in range(1, RUNS + 1):
                alle_runs.append({
                    "bestand": pdf_pad.name,
                    "categorie": categorie,
                    "run": r,
                    "success": False,
                    "tijd_totaal": 0.0,
                    "extracted": None,
                    "ocr_tekst": "",
                    "ruwe_output": f"Kritieke fout: {e}",
                    "tabellen_gevonden": 0,
                    "visuele_zones_gevonden": 0,
                })

    print(f"\n\n{'#' * 55}")
    print(f"  SAMENVATTING — {PIPELINE}")
    print(f"{'#' * 55}")

    geslaagd = sum(1 for r in alle_runs if r["success"])
    print(f"\n  Totaal runs: {len(alle_runs)}  ({len(alle_runs) // RUNS if alle_runs else 0} facturen × {RUNS})")
    print(f"  Geslaagd:    {geslaagd}")
    print(f"  Mislukt:     {len(alle_runs) - geslaagd}")

    for r in alle_runs:
        status = "OK" if r["success"] else "FOUT"
        print(f"  [{status}] {r['categorie']}/{r['bestand']} run{r['run']} ({r['tijd_totaal']}s)")


if __name__ == "__main__":
    main()
