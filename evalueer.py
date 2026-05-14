"""
Evalueer pipeline-output door te vergelijken met de verwachte resultaten.
De verwachte JSON staat naast het PDF-bestand in documents/<categorie>/<naam>.json.
Resultaten worden gelezen uit resultaten/<pipeline>/<categorie>/<naam>.json.

Gebruik:
    uv run python evalueer.py                                          # alle resultaten, alle pipelines
    uv run python evalueer.py --pipeline tekst                         # alle resultaten, één pipeline
    uv run python evalueer.py electricity/factuur1.pdf                 # één bestand, eerste gevonden pipeline
    uv run python evalueer.py electricity/factuur1.pdf --pipeline tekst
"""

import argparse
import json
from pathlib import Path

from pymongo import MongoClient

DOCUMENTS_MAP = Path("data/training")
TOLERANTIE = 0.02

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "bachelorproef"
MONGO_COLLECTION = "resultaten"


def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client[MONGO_DB][MONGO_COLLECTION]

# Velden per categorie om te vergelijken, met hun type
CATEGORIE_VELDEN = {
    "electricity": {
        "sleutel": "consumi",
        "match_veld": "codice",
        "numeriek": {"consumo", "consumo_f1", "consumo_f2", "consumo_f3", "costo_periodo"},
        "tekst": {"codice", "indirizzo"},
        "datum": {"giorno_inizio", "giorno_fine"},
    },
    "water": {
        "sleutel": "consumi",
        "match_veld": "codice",
        "numeriek": {"consumo", "consumo_medio", "costo_periodo"},
        "tekst": {"codice", "indirizzo"},
        "datum": {"giorno_inizio", "giorno_fine"},
    },
    "natural gas": {
        "sleutel": "consumi",
        "match_veld": "codice",
        "numeriek": {"consumo", "costo_periodo"},
        "tekst": {"codice", "indirizzo"},
        "datum": {"giorno_inizio", "giorno_fine"},
    },
    "waste": {
        "sleutel": "rifiuti",
        "match_veld": "codice_cer",
        "numeriek": {"anno", "quantita"},
        "tekst": {"tipo", "codice_cer", "codice_smaltimento"},
        "datum": set(),
    },
    "fuels": {
        "sleutel": "fatture",
        "match_veld": "codice",
        "numeriek": {"prezzo", "quantita", "energia_fonte", "carbonfootprint_fonte"},
        "tekst": {"um", "codice", "tipologia", "energia_unitaria", "carbonfootprint_unitaria"},
        "datum": {"giorno_inizio"},
    },
}


# ──────────────────────────────────────────────
# VERGELIJKING
# ──────────────────────────────────────────────
def vergelijk_waarde(veld: str, geext, verwacht, config: dict) -> tuple[bool, str]:
    """Vergelijk één veldwaarde. Geeft (correct, toelichting) terug."""
    if geext is None and verwacht is None:
        return True, "beide null"
    if geext is None:
        return False, f"niet geëxtraheerd (verwacht: {verwacht!r})"
    if verwacht is None:
        # Verwacht null maar iets geëxtraheerd — tellen als fout
        return False, f"verwacht null maar geëxtraheerd: {geext!r}"

    if veld in config["numeriek"]:
        try:
            e = float(geext)
            v = float(verwacht)
            correct = abs(e - v) <= TOLERANTIE
            return correct, f"{e} vs {v}"
        except (TypeError, ValueError):
            return False, f"niet als getal te vergelijken: {geext!r} vs {verwacht!r}"

    # Tekst- en datumvelden: case-insensitief, witruimte genormaliseerd
    e_str = " ".join(str(geext).strip().lower().split())
    v_str = " ".join(str(verwacht).strip().lower().split())
    return e_str == v_str, f"{geext!r} vs {verwacht!r}"


def haal_alle_velden(config: dict) -> list[str]:
    """Geef alle veldnamen terug voor een categorie."""
    return sorted(config["numeriek"] | config["tekst"] | config["datum"])


def vergelijk_record(geext_rec: dict, verw_rec: dict, config: dict) -> tuple[int, int, int, list]:
    """Vergelijk één record (item in array). Geeft (tp, fp, fn, details) terug."""
    tp = fp = fn = 0
    details = []
    velden = haal_alle_velden(config)

    for veld in velden:
        geext_waarde = geext_rec.get(veld) if geext_rec else None
        verw_waarde = verw_rec.get(veld) if verw_rec else None
        correct, toelichting = vergelijk_waarde(veld, geext_waarde, verw_waarde, config)

        if correct:
            tp += 1
        else:
            if geext_waarde is not None:
                fp += 1
            if verw_waarde is not None:
                fn += 1

        details.append((correct, veld, toelichting))

    return tp, fp, fn, details


def vergelijk_arrays(geext_lijst: list, verw_lijst: list, config: dict) -> tuple[int, int, int, list]:
    """Vergelijk twee arrays van records. Match op sleutelveld, daarna op index."""
    if not verw_lijst:
        return 0, 0, 0, []

    match_veld = config["match_veld"]
    totaal_tp = totaal_fp = totaal_fn = 0
    alle_details = []

    # Bouw opzoektabel op basis van sleutelveld
    geext_op_sleutel = {}
    if geext_lijst:
        for rec in geext_lijst:
            sleutel = rec.get(match_veld) if rec else None
            if sleutel:
                geext_op_sleutel[str(sleutel).strip()] = rec

    for i, verw_rec in enumerate(verw_lijst):
        verw_sleutel = verw_rec.get(match_veld)

        # Probeer te matchen op sleutelveld, anders op index
        geext_rec = None
        if verw_sleutel and str(verw_sleutel).strip() in geext_op_sleutel:
            geext_rec = geext_op_sleutel[str(verw_sleutel).strip()]
        elif geext_lijst and i < len(geext_lijst):
            geext_rec = geext_lijst[i]

        tp, fp, fn, details = vergelijk_record(geext_rec, verw_rec, config)
        totaal_tp += tp
        totaal_fp += fp
        totaal_fn += fn
        alle_details.append((i, verw_sleutel, details))

    return totaal_tp, totaal_fp, totaal_fn, alle_details


def bereken_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ──────────────────────────────────────────────
# EVALUATIE VAN ÉÉN DOCUMENT
# ──────────────────────────────────────────────
def evalueer_document(geext: dict, verwacht: dict, categorie: str, pipeline: str, bestand: str) -> tuple[int, int, int]:
    """Druk evaluatie af en geef (tp, fp, fn) terug."""
    config = CATEGORIE_VELDEN.get(categorie)
    if config is None:
        print(f"  Onbekende categorie: {categorie}")
        return 0, 0, 0

    sleutel = config["sleutel"]
    verw_lijst = verwacht.get(sleutel, [])
    geext_lijst = geext.get(sleutel, []) if geext else []

    if not isinstance(verw_lijst, list):
        verw_lijst = []
    if not isinstance(geext_lijst, list):
        geext_lijst = []

    tp, fp, fn, alle_details = vergelijk_arrays(geext_lijst, verw_lijst, config)
    precision, recall, f1 = bereken_f1(tp, fp, fn)
    totaal_velden = tp + fn

    print(f"\n{'=' * 65}")
    print(f"  Evaluatie: {categorie}/{bestand}  |  Pipeline: {pipeline}")
    print(f"{'=' * 65}")
    print(f"  Records verwacht: {len(verw_lijst)}  |  Records geëxtraheerd: {len(geext_lijst)}")

    for i, sleutel_waarde, details in alle_details:
        label = f"[{i}] {sleutel_waarde or '?'}"
        print(f"\n  Record {label}:")
        print(f"  {'Veld':<30} {'Correct':<10} Details")
        print(f"  {'-' * 58}")
        for correct, veld, toelichting in details:
            vinkje = "OK" if correct else "FOUT"
            print(f"  [{vinkje}] {veld:<28} {toelichting}")

    goed = tp
    print(f"\n  Score: {goed}/{totaal_velden} velden correct"
          + (f" ({100*goed//totaal_velden}%)" if totaal_velden > 0 else ""))
    print(f"  Precision: {precision:.2f}  |  Recall: {recall:.2f}  |  F1: {f1:.2f}")
    print(f"{'=' * 65}\n")

    return tp, fp, fn


# ──────────────────────────────────────────────
# OPHALEN UIT MONGODB
# ──────────────────────────────────────────────
def vind_resultaat(categorie: str, bestandsnaam: str, pipeline: str | None) -> tuple[dict | None, str | None]:
    """Zoek het resultaat in MongoDB voor een gegeven document en pipeline.
    Voor finetuned pipelines wordt run=1 gebruikt."""
    pipelines = [pipeline] if pipeline else [
        "baseline/tekst", "baseline/visie", "baseline/hybride",
        "finetuned/tekst", "finetuned/visie", "finetuned/hybride",
    ]

    try:
        col = get_collection()
    except Exception as e:
        print(f"   MongoDB FOUT: {e}")
        return None, None

    for p in pipelines:
        query = {"bestand": bestandsnaam, "categorie": categorie, "pipeline": p}
        # Finetuned pipelines hebben meerdere runs — neem run 1
        if p.startswith("finetuned/"):
            query["run"] = 1
        doc = col.find_one(query, {"_id": 0})
        if doc:
            return doc, p
    return None, None


def vind_alle_resultaten(pipeline: str | None) -> list[tuple[str, str, str]]:
    """Geef lijst van unieke (categorie, bestandsnaam, pipeline) uit MongoDB."""
    pipelines = [pipeline] if pipeline else [
        "baseline/tekst", "baseline/visie", "baseline/hybride",
        "finetuned/tekst", "finetuned/visie", "finetuned/hybride",
    ]

    try:
        col = get_collection()
    except Exception as e:
        print(f"   MongoDB FOUT: {e}")
        return []

    gevonden = []
    gezien = set()

    for p in pipelines:
        query = {"pipeline": p}
        # Finetuned: alleen run 1 om duplicaten te vermijden
        if p.startswith("finetuned/"):
            query["run"] = 1
        for doc in col.find(query, {"bestand": 1, "categorie": 1, "pipeline": 1, "_id": 0}):
            key = (doc.get("categorie", ""), doc.get("bestand", ""), p)
            if key not in gezien:
                gezien.add(key)
                gevonden.append(key)

    return gevonden


def vind_verwacht_pad(categorie: str, bestandsnaam: str) -> Path:
    """Zoek het verwachte resultaat in data/training/ of data/testing/."""
    for root in (Path("data/training"), Path("data/testing")):
        pad = root / categorie / (Path(bestandsnaam).stem + ".json")
        if pad.exists():
            return pad
    return DOCUMENTS_MAP / categorie / (Path(bestandsnaam).stem + ".json")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evalueer pipeline-output ten opzichte van verwachte resultaten."
    )
    parser.add_argument(
        "bestand", nargs="?",
        help="Relatief pad naar de factuur (bv. electricity/factuur1.pdf)"
    )
    parser.add_argument(
        "--pipeline",
        choices=[
            "baseline/tekst", "baseline/visie", "baseline/hybride",
            "finetuned/tekst", "finetuned/visie", "finetuned/hybride",
        ],
        help="Welke pipeline te evalueren (standaard: alle beschikbare)"
    )
    args = parser.parse_args()

    if args.bestand:
        # Evalueer één document
        pad = Path(args.bestand)
        if len(pad.parts) < 2:
            print(f" Geef het relatieve pad inclusief categorie op,")
            print(f"   bv: electricity/factuur1.pdf")
            raise SystemExit(1)

        categorie = pad.parts[-2]
        bestandsnaam = pad.name

        verwacht_pad = vind_verwacht_pad(categorie, bestandsnaam)
        if not verwacht_pad.exists():
            print(f" Geen verwacht resultaat gevonden: {verwacht_pad}")
            raise SystemExit(1)

        with open(verwacht_pad, encoding="utf-8") as f:
            verwacht_raw = json.load(f)

        doc, gevonden_pipeline = vind_resultaat(categorie, bestandsnaam, args.pipeline)
        if doc is None:
            print(f" Geen resultaat gevonden voor '{categorie}/{bestandsnaam}'"
                  + (f" (pipeline: {args.pipeline})" if args.pipeline else ""))
            print("   Voer eerst een pipeline uit:")
            print(f"     uv run python pipelines/baseline/tekst.py data/training/{categorie}/{bestandsnaam}")
            raise SystemExit(1)

        if not doc.get("success") or doc.get("extracted") is None:
            print(f" De extractie was niet succesvol (pipeline: {gevonden_pipeline})")
            print(f"   Ruwe output: {doc.get('ruwe_output', '')[:300]}")
            raise SystemExit(1)

        evalueer_document(doc["extracted"], verwacht_raw, categorie, gevonden_pipeline, bestandsnaam)

    else:
        # Batch-modus: evalueer alle beschikbare resultaten
        docs = vind_alle_resultaten(args.pipeline)
        if not docs:
            print(" Geen resultaten gevonden in resultaten/.")
            print("   Voer eerst een pipeline uit: uv run python main.py")
            raise SystemExit(1)

        totaal_tp = totaal_fp = totaal_fn = 0
        per_categorie: dict[str, tuple[int, int, int]] = {}
        per_pipeline: dict[str, tuple[int, int, int]] = {}
        fouten = []

        for categorie, bestandsnaam, pipeline in docs:
            verwacht_pad = vind_verwacht_pad(categorie, bestandsnaam)
            if not verwacht_pad.exists():
                fouten.append(f"Geen verwacht resultaat: {verwacht_pad}")
                continue

            with open(verwacht_pad, encoding="utf-8") as f:
                verwacht_raw = json.load(f)

            doc, _ = vind_resultaat(categorie, bestandsnaam, pipeline)
            if doc is None or not doc.get("success") or doc.get("extracted") is None:
                fouten.append(f"Extractie mislukt: {pipeline}/{categorie}/{bestandsnaam}")
                continue

            tp, fp, fn = evalueer_document(
                doc["extracted"], verwacht_raw, categorie, pipeline, bestandsnaam
            )

            totaal_tp += tp
            totaal_fp += fp
            totaal_fn += fn

            cat_tp, cat_fp, cat_fn = per_categorie.get(categorie, (0, 0, 0))
            per_categorie[categorie] = (cat_tp + tp, cat_fp + fp, cat_fn + fn)

            pip_tp, pip_fp, pip_fn = per_pipeline.get(pipeline, (0, 0, 0))
            per_pipeline[pipeline] = (pip_tp + tp, pip_fp + fp, pip_fn + fn)

        # Samenvattingstabel
        print(f"\n{'#' * 65}")
        print(f"  TOTAALOVERZICHT")
        print(f"{'#' * 65}")

        if per_pipeline:
            print(f"\n  Per pipeline:")
            print(f"  {'Pipeline':<12} {'Prec':>6} {'Rec':>6} {'F1':>6}")
            print(f"  {'-'*34}")
            for pip, (tp, fp, fn) in sorted(per_pipeline.items()):
                p, r, f = bereken_f1(tp, fp, fn)
                print(f"  {pip:<12} {p:>6.2f} {r:>6.2f} {f:>6.2f}")

        if per_categorie:
            print(f"\n  Per categorie:")
            print(f"  {'Categorie':<16} {'Prec':>6} {'Rec':>6} {'F1':>6}")
            print(f"  {'-'*38}")
            for cat, (tp, fp, fn) in sorted(per_categorie.items()):
                p, r, f = bereken_f1(tp, fp, fn)
                print(f"  {cat:<16} {p:>6.2f} {r:>6.2f} {f:>6.2f}")

        overall_p, overall_r, overall_f = bereken_f1(totaal_tp, totaal_fp, totaal_fn)
        print(f"\n  Totaal:  Precision={overall_p:.2f}  Recall={overall_r:.2f}  F1={overall_f:.2f}")
        print(f"  Documenten geëvalueerd: {len(docs) - len(fouten)}/{len(docs)}")

        if fouten:
            print(f"\n  Waarschuwingen ({len(fouten)}):")
            for f in fouten:
                print(f"    - {f}")

        print()


if __name__ == "__main__":
    main()
