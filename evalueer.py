#!/usr/bin/env python3
"""
Evaluatiescript – bachelorproef pipelines
==========================================
Vergelijkt pipeline-output veld voor veld met de ground truth.

Metrieken (per thesis §2.10):
  • JSON-compliance  : % van bestanden met geldige, niet-lege JSON-output
  • Exact Match (EM) : % correct geëxtraheerde veldwaarden
                       (2% relatieve tolerantie voor numerieke velden)
  • Field-level F1   : harmonisch gemiddelde van precisie en recall
  • CER              : gemiddelde Levenshtein-afwijking t.o.v. ground truth
  • Verwerkingstijd  : gemiddelde tijd per factuur in seconden

Gebruik:
  uv run python evalueer.py                             # alle mac-pipelines
  uv run python evalueer.py --pipeline baseline/tekst   # één pipeline
  uv run python evalueer.py --categorie electricity     # één categorie
  uv run python evalueer.py --detail                    # per-record detail
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
GROUND_TRUTH_MAP = Path("data/testing")
RESULTATEN_MAP = Path("resultaten/mac")
NUMERIEKE_TOLERANTIE = 0.02   # 2% relatieve tolerantie
RUNS = 3                       # aantal runs voor finetuned-pipelines

PIPELINES = [
    "baseline/tekst",
    "baseline/visie",
    "baseline/hybride",
    "finetuned/tekst",
    "finetuned/visie",
    "finetuned/hybride",
]

# Configuratie per categorie: welke lijst-key, match-veld, en veldtypes
CATEGORIE_CONFIG = {
    "electricity": {
        "lijst_key": "consumi",
        "match_veld": "giorno_inizio",       # één record per maandperiode
        "numeriek": {"consumo", "consumo_f1", "consumo_f2", "consumo_f3", "costo_periodo"},
        "tekst":    {"codice", "indirizzo"},
        "datum":    {"giorno_inizio", "giorno_fine"},
        "enkel_huidig": True,   # factuur = maandfactuur; historische tabel is extra info
    },
    "water": {
        "lijst_key": "consumi",
        "match_veld": "giorno_inizio",
        "numeriek": {"consumo", "consumo_medio", "costo_periodo"},
        "tekst":    {"codice", "indirizzo"},
        "datum":    {"giorno_inizio", "giorno_fine"},
        "enkel_huidig": True,   # idem
    },
    "natural gas": {
        "lijst_key": "consumi",
        "match_veld": "giorno_inizio",
        "numeriek": {"consumo", "costo_periodo"},
        "tekst":    {"codice", "indirizzo"},
        "datum":    {"giorno_inizio", "giorno_fine"},
        "enkel_huidig": True,   # idem
    },
    "waste": {
        "lijst_key": "rifiuti",
        "match_veld": "codice_cer",
        "numeriek": {"anno", "quantita"},
        "tekst":    {"tipo", "codice_cer", "codice_smaltimento"},
        "datum":    set(),
        "enkel_huidig": False,
    },
    "fuels": {
        "lijst_key": "fatture",
        "alt_lijst_keys": ["acquisti"],      # sommige GT-bestanden gebruiken "acquisti"
        # codice alleen is niet uniek; samengestelde sleutel
        "match_veld": "_sleutel",            # zie _maak_sleutel()
        "numeriek": {"prezzo", "quantita", "energia_fonte", "carbonfootprint_fonte"},
        "tekst":    {"um", "codice", "tipologia", "energia_unitaria", "carbonfootprint_unitaria"},
        "datum":    {"giorno_inizio"},
        "enkel_huidig": False,
    },
}


# ──────────────────────────────────────────────
# HULPFUNCTIES
# ──────────────────────────────────────────────
def _haal_lijst(data: dict, config: dict) -> list:
    """Haal de recordlijst op uit een dict, met fallback naar alternatieve keys."""
    for key in [config["lijst_key"]] + config.get("alt_lijst_keys", []):
        waarde = data.get(key)
        if isinstance(waarde, list) and waarde:
            return waarde
    return []


def _parse_datum(s) -> str:
    """Normaliseer datum naar YYYY-MM-DD voor sortering (ondersteunt ISO en DD.MM.YYYY)."""
    if s is None:
        return ""
    s = str(s).strip()
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', s)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return s


def _filter_huidig_record(records: list) -> list:
    """Geef alleen het meest recente record terug (op basis van giorno_fine of giorno_inizio)."""
    if not records:
        return records
    def sort_key(r):
        return _parse_datum(r.get("giorno_fine") or r.get("giorno_inizio") or "")
    return [sorted(records, key=sort_key, reverse=True)[0]]


def _maak_sleutel(record: dict, categorie: str) -> str:
    """Maak een unieke sleutel voor een record."""
    if categorie == "fuels":
        codice = str(record.get("codice", "")).strip()
        datum  = str(record.get("giorno_inizio", "")).strip()
        return f"{codice}|{datum}"
    match_veld = CATEGORIE_CONFIG[categorie]["match_veld"]
    return str(record.get(match_veld, "")).strip().lower()


def _normaliseer(waarde) -> str:
    """Normaliseer tekst: lowercase, witruimte samenvoegen."""
    return re.sub(r"\s+", " ", str(waarde).lower().strip())


def _levenshtein(a: str, b: str) -> int:
    """Berekent de Levenshtein-afstand tussen twee strings."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _cer(geext, verwacht) -> float:
    """Character Error Rate: Levenshtein / len(verwacht). Geclipped op [0, 1]."""
    g_str = _normaliseer(geext) if geext is not None else ""
    v_str = _normaliseer(verwacht) if verwacht is not None else ""
    if not v_str:
        return 0.0 if not g_str else 1.0
    return min(_levenshtein(g_str, v_str) / len(v_str), 1.0)


def _vergelijk_waarde(veld: str, geext, verwacht, config: dict) -> tuple[bool, float]:
    """
    Vergelijk één veldwaarde.
    Geeft (is_correct, cer) terug.

    Null-logica:
      - beiden null  → correct, CER=0
      - verwacht null maar iets geëxtraheerd → fout (hallucinatie)
      - verwacht waarde maar null geëxtraheerd → fout (gemist)
    """
    if verwacht is None and geext is None:
        return True, 0.0
    if verwacht is None:
        return False, 1.0   # hallucinatie
    if geext is None:
        return False, 1.0   # gemist

    if veld in config["numeriek"]:
        try:
            g = float(geext)
            v = float(verwacht)
            if v == 0:
                correct = g == 0
            else:
                correct = abs(g - v) / abs(v) <= NUMERIEKE_TOLERANTIE
            cer = 0.0 if correct else (0.0 if v == 0 else min(abs(g - v) / abs(v), 1.0))
            return correct, cer
        except (TypeError, ValueError):
            pass

    g_str = _normaliseer(geext)
    v_str = _normaliseer(verwacht)
    correct = g_str == v_str
    cer = _cer(geext, verwacht)
    return correct, cer


# ──────────────────────────────────────────────
# RECORD-VERGELIJKING
# ──────────────────────────────────────────────
def _alle_velden(config: dict) -> list[str]:
    return sorted(config["numeriek"] | config["tekst"] | config["datum"])


def _vergelijk_record(geext_rec: Optional[dict], verw_rec: Optional[dict], config: dict):
    """
    Vergelijk één recordpaar.
    Geeft dict met per-veld (correct, cer, gt_heeft_waarde, ext_heeft_waarde) terug.
    Beide-null wordt overgeslagen (geen informatiewaarde).
    """
    velden = _alle_velden(config)
    resultaten = {}
    for veld in velden:
        g = geext_rec.get(veld) if geext_rec else None
        v = verw_rec.get(veld) if verw_rec else None
        gt_val = v is not None
        ext_val = g is not None
        if not gt_val and not ext_val:
            continue   # beide null → sla over
        correct, cer = _vergelijk_waarde(veld, g, v, config)
        resultaten[veld] = (correct, cer, gt_val, ext_val)
    return resultaten


def _vergelijk_arrays(geext_lijst: list, verw_lijst: list, categorie: str, config: dict):
    """
    Vergelijk twee arrays van records.
    Match op sleutelveld (order-onafhankelijk), daarna op index als fallback.
    Geeft lijst van per-record-dict {veld: (correct, cer)} terug.
    Sluit ook unmatched extracted-records in (hallucinaties).
    """
    resultaten = []

    # Bouw opzoektabel voor geëxtraheerde records
    geext_index: dict[str, dict] = {}
    geext_gebruikt: set[str] = set()
    for rec in (geext_lijst or []):
        key = _maak_sleutel(rec, categorie)
        geext_index[key] = rec   # bij duplicaat: laatste wint

    for i, verw_rec in enumerate(verw_lijst or []):
        verw_key = _maak_sleutel(verw_rec, categorie)
        if verw_key in geext_index:
            geext_rec = geext_index[verw_key]
            geext_gebruikt.add(verw_key)
        elif (geext_lijst or []) and i < len(geext_lijst):
            geext_rec = geext_lijst[i]
        else:
            geext_rec = None
        resultaten.append(_vergelijk_record(geext_rec, verw_rec, config))

    # Unmatched extracted records (hallucinaties) tellen als FP per veld
    gt_sleutels = {_maak_sleutel(v, categorie) for v in (verw_lijst or [])}
    for rec in (geext_lijst or []):
        key = _maak_sleutel(rec, categorie)
        if key not in geext_gebruikt and key not in gt_sleutels:
            extra = {}
            for veld in _alle_velden(config):
                if rec.get(veld) is not None:
                    extra[veld] = (False, 1.0, False, True)   # FP: gt=null, ext=waarde
            if extra:
                resultaten.append(extra)

    return resultaten


def _aggregeer_records(record_resultaten: list) -> dict[str, dict]:
    """
    Aggregeer per-record-resultaten naar per-veld statistieken.
    Telt TP, FP, FN apart voor correcte F1-berekening:
      - TP: GT=waarde, extracted=correct
      - FP: extracted=waarde maar GT=null of fout
      - FN: GT=waarde maar extracted=null of fout
    Beide-null wordt overgeslagen.
    """
    stats: dict[str, dict] = {}
    for rec_dict in record_resultaten:
        for veld, (correct, cer, gt_val, ext_val) in rec_dict.items():
            if veld not in stats:
                stats[veld] = {"tp": 0, "fp": 0, "fn": 0, "cer_som": 0.0}
            if correct:
                stats[veld]["tp"] += 1
            else:
                if ext_val:
                    stats[veld]["fp"] += 1   # geëxtraheerd maar fout/hallucinatie
                if gt_val:
                    stats[veld]["fn"] += 1   # verwacht maar gemist/fout
            stats[veld]["cer_som"] += cer
    return stats


# ──────────────────────────────────────────────
# DOCUMENT LADEN
# ──────────────────────────────────────────────
def _laad_ground_truth(categorie: str, bestandsnaam: str) -> Optional[dict]:
    stem = Path(bestandsnaam).stem
    pad = GROUND_TRUTH_MAP / categorie / f"{stem}.json"
    if not pad.exists():
        return None
    with open(pad, encoding="utf-8") as f:
        return json.load(f)


def _laad_resultaat(pipeline: str, categorie: str, bestandsnaam: str) -> list[dict]:
    """
    Laad alle runs voor één document als afzonderlijke resultaten.
    Geeft een lege lijst terug als er geen bestanden zijn.
    """
    stem = Path(bestandsnaam).stem
    basis = RESULTATEN_MAP / pipeline / categorie
    runs = []
    for r in range(1, RUNS + 1):
        pad = basis / f"{stem}_run{r}.json"
        if pad.exists():
            with open(pad, encoding="utf-8") as f:
                data = json.load(f)
                runs.append({
                    "success": data.get("success", False),
                    "extracted": data.get("extracted"),
                    "tijd_totaal": data.get("tijd_totaal", 0),
                })
    return runs


# ──────────────────────────────────────────────
# KERN-EVALUATIE
# ──────────────────────────────────────────────
def evalueer_document(resultaat: dict, ground_truth: dict, categorie: str) -> dict:
    """
    Evalueer één document. Geeft {compliance, veld_stats, tijd} terug.
    veld_stats: {veld: {correct, totaal, cer_som}}
    """
    config = CATEGORIE_CONFIG.get(categorie)
    if config is None:
        return {"compliance": False, "veld_stats": {}, "tijd": 0.0}

    compliance = bool(resultaat.get("success") and resultaat.get("extracted"))
    tijd = resultaat.get("tijd_totaal", 0.0)

    if not compliance:
        # Alle velden als gemist tellen
        verw_lijst = _haal_lijst(ground_truth, config)
        if config.get("enkel_huidig") and verw_lijst:
            verw_lijst = _filter_huidig_record(verw_lijst)
        nul_records = [_vergelijk_record(None, v, config) for v in verw_lijst]
        return {
            "compliance": False,
            "veld_stats": _aggregeer_records(nul_records),
            "tijd": tijd,
        }

    geext_lijst = _haal_lijst(resultaat["extracted"], config)
    verw_lijst = _haal_lijst(ground_truth, config)

    if config.get("enkel_huidig") and verw_lijst:
        verw_lijst = _filter_huidig_record(verw_lijst)

    record_resultaten = _vergelijk_arrays(geext_lijst, verw_lijst, categorie, config)
    veld_stats = _aggregeer_records(record_resultaten)
    return {"compliance": True, "veld_stats": veld_stats, "tijd": tijd}


def _f1_uit_stats(stats: dict) -> float:
    """Micro-gemiddelde F1 over alle velden: 2*TP / (2*TP + FP + FN)."""
    tp = sum(s["tp"] for s in stats.values())
    fp = sum(s["fp"] for s in stats.values())
    fn = sum(s["fn"] for s in stats.values())
    deler = 2 * tp + fp + fn
    return (2 * tp / deler) if deler > 0 else 0.0


def _gem_em(stats: dict) -> float:
    """Gemiddeld Exact Match per veld: TP / (TP + FN) per veld, daarna gemiddeld."""
    if not stats:
        return 0.0
    ems = []
    for s in stats.values():
        deler = s["tp"] + s["fn"]
        if deler > 0:
            ems.append(s["tp"] / deler)
    return sum(ems) / len(ems) if ems else 0.0


def _gem_cer(stats: dict) -> float:
    """Gemiddelde CER per veld."""
    if not stats:
        return 0.0
    totalen = [s["tp"] + s["fp"] + s["fn"] for s in stats.values()]
    cers = [s["cer_som"] / t for s, t in zip(stats.values(), totalen) if t > 0]
    return sum(cers) / len(cers) if cers else 0.0


def _combineer_stats(a: dict, b: dict) -> dict:
    """Voeg twee veld_stats-dicts samen."""
    result = {veld: dict(s) for veld, s in a.items()}
    for veld, s in b.items():
        if veld in result:
            result[veld]["tp"]      += s["tp"]
            result[veld]["fp"]      += s["fp"]
            result[veld]["fn"]      += s["fn"]
            result[veld]["cer_som"] += s["cer_som"]
        else:
            result[veld] = dict(s)
    return result


# ──────────────────────────────────────────────
# BATCH-EVALUATIE
# ──────────────────────────────────────────────
def evalueer_pipeline(pipeline: str, categorie_filter: Optional[str] = None,
                      detail: bool = False, include_per_doc: bool = False) -> dict:
    """
    Evalueer alle documenten voor één pipeline.
    Geeft {pipeline, per_cat: {cat: {compliance, em, cer, f1, tijd}}, overall} terug.
    Met include_per_doc=True ook per_doc: {bestandsnaam: {cat, eval}} toegevoegd.
    """
    gt_bestanden = sorted(GROUND_TRUTH_MAP.rglob("*.json"))
    resultaten_per_cat: dict[str, list[dict]] = {}
    per_doc: dict[str, dict] = {}

    for gt_pad in gt_bestanden:
        categorie = gt_pad.parent.name
        bestandsnaam = gt_pad.stem + ".pdf"

        if categorie_filter and categorie != categorie_filter:
            continue
        if categorie not in CATEGORIE_CONFIG:
            continue

        with open(gt_pad, encoding="utf-8") as f:
            ground_truth = json.load(f)

        runs = _laad_resultaat(pipeline, categorie, bestandsnaam)
        if not runs:
            continue   # nog niet uitgevoerd

        # Evalueer elke run afzonderlijk en combineer statistieken
        run_evals = [evalueer_document(r, ground_truth, categorie) for r in runs]
        n_success = sum(1 for e in run_evals if e["compliance"])
        gecomb_veld_stats: dict = {}
        for e in run_evals:
            gecomb_veld_stats = _combineer_stats(gecomb_veld_stats, e["veld_stats"])
        doc_eval = {
            "compliance": n_success / len(run_evals),   # fractie geslaagde runs
            "veld_stats": gecomb_veld_stats,
            "tijd": sum(e["tijd"] for e in run_evals) / len(run_evals),
            "n_runs": len(run_evals),
            "n_success": n_success,
        }

        if detail:
            succesvolle = [r for r in runs if r.get("success") and r.get("extracted")]
            detail_run = succesvolle[0] if succesvolle else runs[0]
            _print_detail(pipeline, categorie, gt_pad.stem, detail_run, ground_truth, doc_eval)

        if include_per_doc:
            per_doc[gt_pad.stem] = {"categorie": categorie, "eval": doc_eval}

        if categorie not in resultaten_per_cat:
            resultaten_per_cat[categorie] = []
        resultaten_per_cat[categorie].append(doc_eval)

    # Aggregeer per categorie
    per_cat = {}
    for cat, evals in resultaten_per_cat.items():
        n_totaal = len(evals)
        # compliance is nu een fractie per document (bijv. 2/3 runs geslaagd = 0.667)
        n_compliant = sum(e["compliance"] for e in evals)
        gecomb_stats = {}
        for e in evals:
            gecomb_stats = _combineer_stats(gecomb_stats, e["veld_stats"])
        avg_tijd = sum(e["tijd"] for e in evals) / n_totaal if n_totaal else 0

        per_cat[cat] = {
            "n": n_totaal,
            "compliance": n_compliant / n_totaal if n_totaal else 0.0,
            "em": _gem_em(gecomb_stats),
            "cer": _gem_cer(gecomb_stats),
            "f1": _f1_uit_stats(gecomb_stats),
            "tijd": avg_tijd,
            "veld_stats": gecomb_stats,
        }

    # Overall
    all_evals = [e for evals in resultaten_per_cat.values() for e in evals]
    n_totaal = len(all_evals)
    n_compliant = sum(e["compliance"] for e in all_evals)
    all_stats = {}
    for e in all_evals:
        all_stats = _combineer_stats(all_stats, e["veld_stats"])
    avg_tijd = sum(e["tijd"] for e in all_evals) / n_totaal if n_totaal else 0

    overall = {
        "n": n_totaal,
        "compliance": n_compliant / n_totaal if n_totaal else 0.0,
        "em": _gem_em(all_stats),
        "cer": _gem_cer(all_stats),
        "f1": _f1_uit_stats(all_stats),
        "tijd": avg_tijd,
    }
    result = {"pipeline": pipeline, "per_cat": per_cat, "overall": overall}
    if include_per_doc:
        result["per_doc"] = per_doc
    return result


def _print_detail(pipeline, categorie, stem, resultaat, ground_truth, doc_eval):
    print(f"\n  {'─'*62}")
    print(f"  {pipeline}  |  {categorie}/{stem}.pdf")
    config = CATEGORIE_CONFIG.get(categorie)
    if not config:
        return
    geext_lijst = _haal_lijst(resultaat.get("extracted") or {}, config)
    verw_lijst = _haal_lijst(ground_truth, config)
    n_s = doc_eval.get("n_success", "?")
    n_r = doc_eval.get("n_runs", "?")
    print(f"  Records GT: {len(verw_lijst)}  |  Extracted: {len(geext_lijst)}"
          f"  |  Compliant: {n_s}/{n_r} runs ({100*doc_eval['compliance']:.0f}%)")
    for veld, s in sorted(doc_eval["veld_stats"].items()):
        deler = s["tp"] + s["fn"]
        em_pct = 100 * s["tp"] / deler if deler else 0
        totaal = s["tp"] + s["fp"] + s["fn"]
        avg_cer = s["cer_som"] / totaal if totaal else 0
        print(f"    {veld:<30} EM={em_pct:5.1f}%  CER={avg_cer:.3f}"
              f"  TP={s['tp']} FP={s['fp']} FN={s['fn']}")


# ──────────────────────────────────────────────
# UITVOER
# ──────────────────────────────────────────────
CATEGORIEEN = ["electricity", "natural gas", "water", "fuels", "waste"]


def _print_tabel(pipeline_resultaten: list[dict], per_cat_tabel: bool = True):
    breed = 90
    print()
    print("=" * breed)
    print(f"  {'EVALUATIEOVERZICHT — MAC PIPELINES':^{breed-4}}")
    print("=" * breed)

    # Overzichtstabel pipelines
    header = f"  {'Pipeline':<22}  {'Comp%':>6}  {'EM%':>6}  {'F1%':>6}  {'CER':>6}  {'Tijd(s)':>8}  {'#':>4}"
    print()
    print(header)
    print("  " + "─" * (breed - 2))

    for pr in pipeline_resultaten:
        o = pr["overall"]
        if o["n"] == 0:
            print(f"  {pr['pipeline']:<22}  {'(geen resultaten)':>40}")
            continue
        print(f"  {pr['pipeline']:<22}  "
              f"{100*o['compliance']:>5.1f}%  "
              f"{100*o['em']:>5.1f}%  "
              f"{100*o['f1']:>5.1f}%  "
              f"{o['cer']:>6.3f}  "
              f"{o['tijd']:>8.1f}  "
              f"{o['n']:>4}")

    if not per_cat_tabel:
        print()
        return

    # Per-categorie tabel
    print()
    print("  " + "─" * (breed - 2))
    print(f"  {'PER CATEGORIE':^{breed-4}}")
    print("  " + "─" * (breed - 2))

    for cat in CATEGORIEEN:
        has_data = any(cat in pr["per_cat"] for pr in pipeline_resultaten)
        if not has_data:
            continue
        print(f"\n  [{cat.upper()}]")
        print(f"  {'Pipeline':<22}  {'Comp%':>6}  {'EM%':>6}  {'F1%':>6}  {'CER':>6}  {'Tijd(s)':>8}  {'#':>4}")
        for pr in pipeline_resultaten:
            d = pr["per_cat"].get(cat)
            if d is None or d["n"] == 0:
                continue
            print(f"  {pr['pipeline']:<22}  "
                  f"{100*d['compliance']:>5.1f}%  "
                  f"{100*d['em']:>5.1f}%  "
                  f"{100*d['f1']:>5.1f}%  "
                  f"{d['cer']:>6.3f}  "
                  f"{d['tijd']:>8.1f}  "
                  f"{d['n']:>4}")

    print()
    print("=" * breed)
    print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evalueer mac-pipeline output ten opzichte van de ground truth."
    )
    parser.add_argument(
        "--pipeline",
        choices=PIPELINES,
        help="Evalueer alleen deze pipeline (standaard: alle)"
    )
    parser.add_argument(
        "--categorie",
        choices=list(CATEGORIE_CONFIG.keys()),
        help="Evalueer alleen deze categorie"
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Toon per-document per-veld detail"
    )
    parser.add_argument(
        "--geen-cat-tabel", action="store_true",
        help="Toon alleen de samenvattingstabel, niet per categorie"
    )
    args = parser.parse_args()

    te_evalueren = [args.pipeline] if args.pipeline else PIPELINES

    pipeline_resultaten = []
    for pipeline in te_evalueren:
        print(f"  Evalueer {pipeline}...", end="\r")
        pr = evalueer_pipeline(pipeline, args.categorie, args.detail)
        pipeline_resultaten.append(pr)

    _print_tabel(pipeline_resultaten, per_cat_tabel=not args.geen_cat_tabel)


if __name__ == "__main__":
    main()
