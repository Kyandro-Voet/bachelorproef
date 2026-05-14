"""
Hybride pipeline: PDF → lay-out analyse → routing → JSON

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
import io
import json
import sys
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
TEKST_MODEL = "qwen3:8b"
VISIE_MODEL = "qwen3-vl:8b"
OCR_TALEN = "eng+ita"
DPI = 300
MAX_BREEDTE = 1280
DOCUMENTS_MAP = Path("data/training")
PIPELINE = "baseline/hybride"

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

VISIE_ZONES_PROMPT = """/no_think
The image contains one or more cropped regions from an invoice (tables, figures, or other elements),
stacked vertically and separated by gray bars.
Extract all relevant data you can read (amounts, dates, names, codes, quantities) as plain text.
Write each piece of data as "label: value" on a separate line.
Skip purely decorative elements (logos without text, background graphics).
Return only readable text, no JSON.
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


def ontlaad_modellen() -> None:
    """Verwijder beide modellen uit het Ollama-geheugen zodat elke run een cold run is."""
    for model in (TEKST_MODEL, VISIE_MODEL):
        _wacht_op_unload(model)


# ──────────────────────────────────────────────
# OPSLAG
# ──────────────────────────────────────────────
def sla_op_in_mongodb(resultaat: dict) -> None:
    """Sla het resultaat op in MongoDB."""
    document = {
        "bestand": resultaat["bestand"],
        "categorie": resultaat["categorie"],
        "model": f"{TEKST_MODEL} + {VISIE_MODEL}",
        "pipeline": PIPELINE,
        "success": resultaat["success"],
        "tijd_totaal": resultaat["tijd_totaal"],
        "extracted": resultaat["extracted"],
        "ocr_tekst": resultaat["ocr_tekst"],
        "ruwe_output": resultaat["ruwe_output"],
        "tabellen_gevonden": resultaat["tabellen_gevonden"],
        "visuele_zones_gevonden": resultaat["visuele_zones_gevonden"],
    }
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client[MONGO_DB][MONGO_COLLECTION].insert_one(document)
        print(f"   MongoDB: {PIPELINE} — {resultaat['bestand']}")
    except Exception as e:
        print(f"   MongoDB FOUT: {e}")


# ──────────────────────────────────────────────
# HULPFUNCTIES
# ──────────────────────────────────────────────
def formatteer_tabel(rijen: list) -> str:
    """Formatteer een pdfplumber-tabel (2D lijst) als leesbare platte tekst."""
    regels = []
    for rij in rijen:
        cellen = [str(cel or "").strip() for cel in rij]
        if any(cellen):
            regels.append(" | ".join(cellen))
    return "\n".join(regels)


def afbeelding_naar_base64(img: Image.Image, max_breedte: int = MAX_BREEDTE) -> str:
    """
    Schaal op basis van de breedte (max MAX_BREEDTE px) en codeer naar base64.
    Gebruikt LANCZOS voor betere kwaliteit bij downscaling.
    """
    if img.width > max_breedte:
        ratio = max_breedte / img.width
        img = img.resize((max_breedte, int(img.height * ratio)), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def crop_tabel(img: Image.Image, bbox: tuple, dpi: int) -> Image.Image:
    """
    Crop een zone uit een PIL-afbeelding op basis van pdfplumber-coördinaten.
    pdfplumber geeft (x0, top, x1, bottom) in punten (1 punt = 1/72 inch).
    """
    schaal = dpi / 72.0
    x0 = max(0, int(bbox[0] * schaal))
    y0 = max(0, int(bbox[1] * schaal))
    x1 = min(img.width, int(bbox[2] * schaal))
    y1 = min(img.height, int(bbox[3] * schaal))
    return img.crop((x0, y0, x1, y1))


def combineer_zones_naar_afbeelding(crops: list[Image.Image]) -> Image.Image:
    """
    Stapel meerdere crops verticaal met een grijze scheidingsbalk.
    Resultaat: één afbeelding voor de vision model in één request.
    """
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
    """
    Analyseer een gescande pagina via Tesseract blok-niveau output.

    Classificatie per blok (level=2):
      - Blokken met herkende woorden (level=5, confidence ≥ 30) → OCR-tekst
      - Blokken zonder herkende woorden maar voldoende groot → visuele zone
        (tabel, afbeelding of grafiek die Tesseract niet als tekst herkent)

    Visuele zones worden teruggegeven als bboxen in pdfplumber-puntformaat
    zodat crop_tabel ze zonder aanpassing kan verwerken.
    """
    data = pytesseract.image_to_data(
        img, lang=OCR_TALEN, output_type=pytesseract.Output.DICT
    )

    # Verzamel blok-bboxen (level=2) en woorden per blok (level=5)
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
    visuele_bboxen: list[tuple] = []
    schaal = 72.0 / DPI

    for nr, bbox in blok_bboxen.items():
        breedte = bbox[2] - bbox[0]
        hoogte = bbox[3] - bbox[1]

        # Filter te kleine zones (ruis)
        if breedte < 50 or hoogte < 20:
            continue

        woorden = blok_woorden.get(nr, [])

        if woorden:
            # Blok heeft herkende tekst → naar tekst-LLM
            tekst_blokken.append(" ".join(woorden))
        else:
            # Geen herkende woorden → visuele zone → vision model
            visuele_bboxen.append((
                bbox[0] * schaal,
                bbox[1] * schaal,
                bbox[2] * schaal,
                bbox[3] * schaal,
            ))

    ocr_tekst = "\n".join(tekst_blokken)
    return ocr_tekst, visuele_bboxen


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
        print(f"   ❌ JSON parse fout: {e}")
        return None


# ──────────────────────────────────────────────
# PIPELINE STAPPEN
# ──────────────────────────────────────────────
def stap1_layout_analyse(pdf_pad: Path) -> list[dict]:
    """
    Lay-out analyse via pdfplumber — geen afbeeldingen nodig.

    Digitale pagina's:
      - Tekst buiten tabellen direct uit PDF-objecten
      - Tabellen direct uitgelezen via .extract_table() → opgeslagen als tekst
        (geen vision inference nodig voor digitale tabellen)
      - Ingebedde afbeeldingen geregistreerd als bboxen voor vision model

    Gescande pagina's (< 50 tekens digitale tekst):
      - Gemarkeerd als gescand; tekst en visuele zones volgen in stap 3
    """
    print(f"\n🗺️  Stap 1: Lay-out analyse met pdfplumber...")
    pagina_data = []

    with pdfplumber.open(pdf_pad) as pdf:
        for i, pagina in enumerate(pdf.pages):
            tabellen = pagina.find_tables()
            tabel_bboxen = [t.bbox for t in tabellen]

            # Tekst buiten tabelgebieden
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
                # Tekst en visuele zones worden bepaald in stap 3
                pagina_data.append({
                    "pagina_nr": i,
                    "tekst": "",
                    "tabel_teksten": [],
                    "visuele_bboxen": [],
                    "afbeelding_bboxen": [],
                    "gescand": True,
                })
                print(f"   📃 Pagina {i+1}: gescand — Tesseract-analyse volgt in stap 3")
                continue

            # Digitale pagina: extraheer tabellen direct via pdfplumber
            tabel_teksten = []
            for tabel in tabellen:
                rijen = tabel.extract()
                if rijen:
                    tabel_teksten.append(formatteer_tabel(rijen))

            # Ingebedde afbeeldingen (voor vision model)
            afb_bboxen = [
                (obj["x0"], obj["top"], obj["x1"], obj["bottom"])
                for obj in pagina.images
                if obj["width"] > 30 and obj["height"] > 30
            ]

            pagina_data.append({
                "pagina_nr": i,
                "tekst": tekst.strip(),
                "tabel_teksten": tabel_teksten,
                "visuele_bboxen": [],           # alleen voor gescande pagina's
                "afbeelding_bboxen": afb_bboxen,
                "gescand": False,
            })

            status_delen = []
            if tabel_teksten:
                status_delen.append(f"{len(tabel_teksten)} tabel(len) via pdfplumber")
            if afb_bboxen:
                status_delen.append(f"{len(afb_bboxen)} afbeelding(en) → vision model")
            status = ", ".join(status_delen) if status_delen else "geen visuele zones"
            print(f"   📃 Pagina {i+1}: {len(tekst)} tekst-tekens, {status}")

    return pagina_data


def stap2_converteer_naar_afbeeldingen(pdf_pad: Path, pagina_data: list[dict]) -> dict[int, Image.Image]:
    """
    Zet alleen pagina's om die afbeeldingen nodig hebben:
      - Gescande pagina's (stap 3: Tesseract + vision model)
      - Digitale pagina's met ingebedde afbeeldingen (stap 4: vision model)

    Digitale pagina's waarvan tabellen al via pdfplumber zijn uitgelezen
    én die geen ingebedde afbeeldingen hebben, worden overgeslagen.
    """
    benodigde_paginas = {
        p["pagina_nr"] for p in pagina_data
        if p["gescand"] or p["afbeelding_bboxen"]
    }

    if not benodigde_paginas:
        print(f"\n📄 Stap 2: Geen afbeeldingen nodig — digitale PDF zonder visuele zones")
        return {}

    print(f"\n📄 Stap 2: PDF naar afbeeldingen ({DPI} DPI) voor pagina('s) "
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

    print(f"   ✅ {len(afbeeldingen)} pagina('s) omgezet in {time.time() - start:.1f}s")
    return afbeeldingen


def stap3_verwerk_gescande_paginas(pagina_data: list[dict], afbeeldingen: dict[int, Image.Image]) -> None:
    """
    Tesseract blokanalyse voor gescande pagina's.
    Werkt pagina_data bij (in-place):
      - tekst: OCR-tekst uit blokken met herkende woorden
      - visuele_bboxen: blokken zonder woorden (tabellen, figuren) → vision model
    """
    gescande_paginas = [p for p in pagina_data if p["gescand"]]

    if not gescande_paginas:
        print(f"\n🔍 Stap 3: Geen gescande pagina's — stap overgeslagen")
        return

    print(f"\n🔍 Stap 3: Tesseract blokanalyse voor {len(gescande_paginas)} gescande pagina('s)...")
    start = time.time()

    for pdata in gescande_paginas:
        nr = pdata["pagina_nr"]
        tekst, visuele_bboxen = splits_gescande_pagina(afbeeldingen[nr])
        pdata["tekst"] = tekst
        pdata["visuele_bboxen"] = visuele_bboxen
        print(f"   📃 Pagina {nr + 1}: {len(tekst)} OCR-tekens, "
              f"{len(visuele_bboxen)} visuele zone(s) → vision model")

    print(f"   ✅ Gescande pagina's verwerkt in {time.time() - start:.1f}s")


def stap4_verwerk_visuele_zones(pagina_data: list[dict], afbeeldingen: dict[int, Image.Image]) -> list[str]:
    """
    Stuur visuele zones naar het vision model — één call per pagina.

    Verwerkte zones:
      - Digitale pagina's: ingebedde afbeeldingen (afbeelding_bboxen)
      - Gescande pagina's: niet-tekst blokken (visuele_bboxen)

    Alle zones van één pagina worden gecombineerd in één afbeelding (gestapeld)
    en in één Ollama-request gestuurd. Dit reduceert het aantal inference-calls.
    """
    resultaten: list[str] = []
    totaal_zones = sum(
        len(p["visuele_bboxen"]) + len(p["afbeelding_bboxen"])
        for p in pagina_data
    )

    if totaal_zones == 0:
        print(f"\n📊 Stap 4: Geen visuele zones — stap overgeslagen")
        return []

    print(f"\n📊 Stap 4: {totaal_zones} visuele zone(s) → {VISIE_MODEL} (één call per pagina)...")
    start = time.time()

    for pdata in pagina_data:
        pnr = pdata["pagina_nr"]
        img = afbeeldingen.get(pnr)
        if img is None:
            continue

        alle_bboxen = pdata["visuele_bboxen"] + pdata["afbeelding_bboxen"]
        if not alle_bboxen:
            continue

        # Knip alle zones bij en filter te kleine crops
        crops = []
        for bbox in alle_bboxen:
            crop = crop_tabel(img, bbox, DPI)
            if crop.width >= 50 and crop.height >= 20:
                crops.append(crop)

        if not crops:
            continue

        # Combineer alle crops in één afbeelding → één vision-call per pagina
        gecombineerd = combineer_zones_naar_afbeelding(crops)
        b64 = afbeelding_naar_base64(gecombineerd)

        print(f"   📋 Pagina {pnr + 1}: {len(crops)} zone(s) gecombineerd → vision model...")
        response = ollama.chat(
            model=VISIE_MODEL,
            messages=[{"role": "user", "content": VISIE_ZONES_PROMPT, "images": [b64]}],
            options={"temperature": 0.1, "num_ctx": 4096},
            think=False,
        )
        tekst = response.message.content.strip()
        if tekst:
            resultaten.append(f"[Visuele zones — pagina {pnr + 1}]\n{tekst}")

    print(f"   ✅ Visuele zones verwerkt in {time.time() - start:.1f}s")
    return resultaten


def stap5_bouw_context(pagina_data: list[dict]) -> str:
    """
    Stel de volledige tekst-context samen voor het tekst-LLM.

    Per pagina:
      - Digitale pagina: pdfplumber-tekst + pdfplumber-tabeltekst (als structured text)
      - Gescande pagina: Tesseract OCR-tekst (ingevuld door stap 3)

    Geen afbeeldingen nodig — alle tekst zit al in pagina_data.
    """
    print(f"\n📝 Stap 5: Tekst samenstellen...")
    tekst_delen = []

    for pdata in pagina_data:
        label = "gescand/OCR" if pdata["gescand"] else "tekst"
        deel = f"--- Pagina {pdata['pagina_nr'] + 1} ({label}) ---\n{pdata['tekst']}"

        # Digitale tabellen direct opnemen (pdfplumber-extractie, geen vision nodig)
        for j, tabel_tekst in enumerate(pdata.get("tabel_teksten", [])):
            deel += f"\n\n[Tabel {j + 1}]\n{tabel_tekst}"

        tekst_delen.append(deel)

    return "\n\n".join(tekst_delen)


def stap6_llm(tekst_context: str, visuele_resultaten: list[str], categorie: str) -> str:
    """Combineer tekst en visuele output en stuur naar het tekst-LLM voor definitieve JSON."""
    print(f"\n🤖 Stap 6: Context naar {TEKST_MODEL} sturen...")
    start = time.time()

    context = tekst_context
    if visuele_resultaten:
        context += (
            "\n\n--- VISUELE ZONES (uitgelezen door vision-model) ---\n"
            + "\n\n".join(visuele_resultaten)
        )

    schema = CATEGORIE_SCHEMAS.get(categorie, CATEGORIE_SCHEMAS["electricity"])
    user_prompt = USER_PROMPT_TEMPLATE.format(schema=schema) + context

    response = ollama.chat(
        model=TEKST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.1, "num_ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        think=False,
    )
    print(f"   ✅ Antwoord ontvangen in {time.time() - start:.1f}s")
    return response.message.content


def stap7_parse_json(ruwe_output: str) -> dict | None:
    print(f"\n🔧 Stap 7: JSON parsen...")
    resultaat = parse_json(ruwe_output)
    if resultaat:
        print(f"   ✅ JSON succesvol geparsed!")
    else:
        print(f"   Ruwe output:\n{ruwe_output[:500]}")
    return resultaat


# ──────────────────────────────────────────────
# VERWERKING
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> dict:
    """Verwerk één factuur met de hybride pipeline."""
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 55}")
    print(f"  📄 {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 55}")

    ontlaad_modellen()
    totaal_start = time.time()

    pagina_data = stap1_layout_analyse(pdf_pad)
    afbeeldingen = stap2_converteer_naar_afbeeldingen(pdf_pad, pagina_data)
    stap3_verwerk_gescande_paginas(pagina_data, afbeeldingen)
    visuele_resultaten = stap4_verwerk_visuele_zones(pagina_data, afbeeldingen)
    tekst_context = stap5_bouw_context(pagina_data)
    ruwe_output = stap6_llm(tekst_context, visuele_resultaten, categorie)
    resultaat = stap7_parse_json(ruwe_output)

    totaal_tabellen = (
        sum(len(p["tabel_teksten"]) for p in pagina_data if not p["gescand"])
        + sum(len(p["visuele_bboxen"]) for p in pagina_data if p["gescand"])
    )
    totaal_visuele_zones = sum(len(p["afbeelding_bboxen"]) for p in pagina_data)

    return {
        "bestand": pdf_pad.name,
        "categorie": categorie,
        "success": resultaat is not None,
        "tijd_totaal": round(time.time() - totaal_start, 2),
        "extracted": resultaat,
        "ocr_tekst": tekst_context,
        "ruwe_output": ruwe_output,
        "tabellen_gevonden": totaal_tabellen,
        "visuele_zones_gevonden": totaal_visuele_zones,
    }


def main():
    if len(sys.argv) >= 2:
        pdfs = [Path(sys.argv[1])]
    else:
        if not DOCUMENTS_MAP.exists():
            print(f"❌ Map '{DOCUMENTS_MAP}' niet gevonden.")
            print(f"   Gebruik: uv run python hybrid_pipeline.py <factuur.pdf>")
            sys.exit(1)
        pdfs = sorted(DOCUMENTS_MAP.rglob("*.pdf"))

    if not pdfs:
        print("❌ Geen PDF-bestanden gevonden.")
        sys.exit(1)

    print(f"\n🚀 Hybride pipeline — Tekst: {TEKST_MODEL} | Visie: {VISIE_MODEL}")
    print(f"   Facturen: {len(pdfs)}")

    resultaten = []
    for pdf_pad in pdfs:
        if not pdf_pad.exists():
            print(f"❌ Bestand niet gevonden: {pdf_pad}")
            continue

        resultaat = verwerk_factuur(pdf_pad)
        resultaten.append(resultaat)
        sla_op_in_mongodb(resultaat)

    print(f"\n\n{'#' * 55}")
    print(f"  SAMENVATTING")
    print(f"{'#' * 55}")

    geslaagd = sum(1 for r in resultaten if r["success"])
    print(f"\n  Totaal:    {len(resultaten)} facturen")
    print(f"  Geslaagd:  {geslaagd} ✅")
    print(f"  Mislukt:   {len(resultaten) - geslaagd} ❌")

    for r in resultaten:
        status = "✅" if r["success"] else "❌"
        print(f"\n  {status} {r['categorie']}/{r['bestand']} ({r['tijd_totaal']}s, "
              f"{r['tabellen_gevonden']} tabellen, {r['visuele_zones_gevonden']} visuele zones)")


if __name__ == "__main__":
    main()
