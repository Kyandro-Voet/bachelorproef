"""
Inferentie met het QLoRA fine-tuned model.

Laadt het base model + adapter-gewichten en verwerkt documenten
op dezelfde manier als main.py maar met het fine-tuned model.
Ondersteunt categorieën: electricity, water, natural gas, waste, fuels

Gebruik:
    python inference_qlora.py                                  # alle PDF's
    python inference_qlora.py electricity/factuur.pdf          # één bestand
"""

import json
import sys
import time
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pdf2image import convert_from_path
import pytesseract


# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PAD = Path("output/qlora-factuur")
OCR_TALEN = "nld+eng+ita"
DPI = 300
DOCUMENTS_MAP = Path("documents")
RESULTATEN_MAP = Path("resultaten")
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "Extract the requested information from the document and return it as valid JSON. "
    "Return ONLY the JSON, no extra text or explanation."
)

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

USER_PROMPT_TEMPLATE = (
    "Extract all data from the document below and return it as a JSON object "
    "that strictly follows this schema:\n\n"
    "{schema}\n\n"
    "--- DOCUMENT START ---\n"
    "{ocr_tekst}"
)


# ──────────────────────────────────────────────
# MODEL LADEN (éénmalig)
# ──────────────────────────────────────────────
_model = None
_tokenizer = None


def _laad_model():
    """Laad het base model + QLoRA adapters (éénmalig in geheugen)."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    if not ADAPTER_PAD.exists():
        raise FileNotFoundError(
            f"Adapter niet gevonden: {ADAPTER_PAD}\n"
            "Voer eerst train_qlora.py uit om het model te fine-tunen."
        )

    print(f"   Base model laden: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"   Adapters laden: {ADAPTER_PAD}")
    _model = PeftModel.from_pretrained(base, str(ADAPTER_PAD))
    _model.eval()

    _tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PAD))
    _tokenizer.pad_token = _tokenizer.eos_token

    print("   Model met adapters geladen\n")
    return _model, _tokenizer


# ──────────────────────────────────────────────
# PIPELINE STAPPEN
# ──────────────────────────────────────────────
def stap1_pdf_naar_afbeeldingen(pdf_pad: Path) -> list:
    print(f"\n   Stap 1: PDF naar afbeeldingen ({DPI} DPI)...")
    start = time.time()
    afbeeldingen = convert_from_path(str(pdf_pad), dpi=DPI)
    print(f"   {len(afbeeldingen)} pagina('s) in {time.time()-start:.1f}s")
    return afbeeldingen


def stap2_ocr(afbeeldingen: list) -> str:
    print(f"\n   Stap 2: OCR met Tesseract (talen: {OCR_TALEN})...")
    start = time.time()
    alle_tekst = []
    for i, img in enumerate(afbeeldingen):
        tekst = pytesseract.image_to_string(img, lang=OCR_TALEN)
        alle_tekst.append(f"--- Pagina {i+1} ---\n{tekst}")
        print(f"   Pagina {i+1}: {len(tekst)} karakters")
    volledige_tekst = "\n\n".join(alle_tekst)
    print(f"   Totaal {len(volledige_tekst)} karakters in {time.time()-start:.1f}s")
    return volledige_tekst


def stap3_qlora_llm(ocr_tekst: str, categorie: str) -> str:
    """Stuur OCR-tekst naar het fine-tuned model en ontvang JSON."""
    print(f"\n   Stap 3: Tekst naar QLoRA model sturen (categorie: {categorie})...")
    start = time.time()

    model, tokenizer = _laad_model()

    schema = CATEGORIE_SCHEMAS.get(categorie, CATEGORIE_SCHEMAS["electricity"])
    user_content = USER_PROMPT_TEMPLATE.format(schema=schema, ocr_tekst=ocr_tekst)

    berichten = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        berichten,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gegenereerd = output_ids[0][inputs["input_ids"].shape[1]:]
    ruwe_output = tokenizer.decode(gegenereerd, skip_special_tokens=True)

    print(f"   Antwoord ontvangen in {time.time()-start:.1f}s")
    return ruwe_output


def stap4_parse_json(ruwe_output: str) -> dict | None:
    print(f"\n   Stap 4: JSON parsen...")
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
# OPSLAG
# ──────────────────────────────────────────────
def sla_op_als_json(resultaat: dict):
    """Sla een resultaat op als JSON-bestand in resultaten/qlora/<categorie>/."""
    categorie = resultaat["categorie"]
    map_pad = RESULTATEN_MAP / "qlora" / categorie
    map_pad.mkdir(parents=True, exist_ok=True)

    bestandsnaam = Path(resultaat["bestand"]).stem + ".json"
    uitvoer_pad = map_pad / bestandsnaam

    document = {
        "bestand": resultaat["bestand"],
        "categorie": categorie,
        "model": f"qlora:{BASE_MODEL}",
        "pipeline": "qlora",
        "success": resultaat["success"],
        "tijd_totaal": resultaat["tijd_totaal"],
        "extracted": resultaat["extracted"],
        "ocr_tekst": resultaat["ocr_tekst"],
        "ruwe_output": resultaat["ruwe_output"],
    }

    with open(uitvoer_pad, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"   Opgeslagen als {uitvoer_pad}")


# ──────────────────────────────────────────────
# HOOFDVERWERKING
# ──────────────────────────────────────────────
def verwerk_factuur(pdf_pad: Path) -> dict:
    categorie = pdf_pad.parent.name

    print(f"\n{'=' * 55}")
    print(f"  {pdf_pad.name}  [{categorie}]")
    print(f"{'=' * 55}")

    totaal_start = time.time()
    afbeeldingen = stap1_pdf_naar_afbeeldingen(pdf_pad)
    ocr_tekst = stap2_ocr(afbeeldingen)
    ruwe_output = stap3_qlora_llm(ocr_tekst, categorie)
    resultaat = stap4_parse_json(ruwe_output)

    return {
        "bestand": pdf_pad.name,
        "categorie": categorie,
        "success": resultaat is not None,
        "tijd_totaal": round(time.time() - totaal_start, 2),
        "extracted": resultaat,
        "ocr_tekst": ocr_tekst,
        "ruwe_output": ruwe_output,
    }


def main():
    if len(sys.argv) >= 2:
        pdfs = [Path(sys.argv[1])]
    else:
        if not DOCUMENTS_MAP.exists():
            print(f" Map '{DOCUMENTS_MAP}' niet gevonden.")
            sys.exit(1)
        pdfs = sorted(DOCUMENTS_MAP.rglob("*.pdf"))

    if not pdfs:
        print(" Geen PDF-bestanden gevonden.")
        sys.exit(1)

    print(f"\n QLoRA Pipeline — Adapters: {ADAPTER_PAD}")
    print(f"   Facturen: {len(pdfs)}")

    resultaten = []
    for pdf_pad in pdfs:
        if not pdf_pad.exists():
            print(f" Bestand niet gevonden: {pdf_pad}")
            continue
        resultaat = verwerk_factuur(pdf_pad)
        resultaten.append(resultaat)
        sla_op_als_json(resultaat)

    # Samenvatting
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
