#!/usr/bin/env python3
"""
Analyse: baseline/hybride vs finetuned/hybride.

Genereert grafieken en een kort rapport voor hoofdstuk 6.
"""

from pathlib import Path
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pdfplumber

from evalueer import evalueer_pipeline, _combineer_stats, _f1_uit_stats, _gem_cer, _gem_em


BASELINE = "baseline/hybride"
FINETUNED = "finetuned/hybride"
UITVOER_MAP = Path("resultaten/analyse/baseline_vs_finetuned/hybride")
DATA_MAP = Path("data/testing")

CATEGORIEEN = ["electricity", "natural gas", "water", "fuels", "waste"]
CAT_LABELS = ["Electricity", "Natural gas", "Water", "Fuels", "Waste"]

KLEUR_BASELINE = "#4C72B0"
KLEUR_FINETUNED = "#DD8452"


def _stijl():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def _metrics(pr: dict, cat: str | None = None) -> dict:
    if cat:
        return pr["per_cat"].get(cat) or {}
    return pr["overall"]


def _veld_em(pr: dict) -> dict[str, float]:
    stats = {}
    for cat_data in pr["per_cat"].values():
        stats = _combineer_stats(stats, cat_data.get("veld_stats", {}))

    result = {}
    for veld, s in stats.items():
        deler = s["tp"] + s["fn"]
        result[veld] = s["tp"] / deler if deler else 0.0
    return result


def _is_digitaal(bestandsnaam: str, categorie: str) -> bool:
    pad = DATA_MAP / categorie / bestandsnaam
    if not pad.exists():
        return True
    try:
        with pdfplumber.open(pad) as pdf:
            return any(p.extract_text() for p in pdf.pages[:2])
    except Exception:
        return True


def _classificeer_pdfs() -> dict[str, bool]:
    classificatie = {}
    for cat_dir in DATA_MAP.iterdir():
        if not cat_dir.is_dir():
            continue
        for pdf in cat_dir.glob("*.pdf"):
            classificatie[pdf.stem] = _is_digitaal(pdf.name, cat_dir.name)
    return classificatie


def _aggregeer_doc_groep(per_doc: dict, stems: list[str]) -> dict:
    evals = [per_doc[s]["eval"] for s in stems if s in per_doc]
    if not evals:
        return {
            "n": 0,
            "compliance": None,
            "em": None,
            "f1": None,
            "cer": None,
            "tijd": None,
            "tijd_mediaan": None,
        }
    stats = {}
    for e in evals:
        stats = _combineer_stats(stats, e["veld_stats"])
    tijden = [e["tijd"] for e in evals]
    return {
        "n": len(evals),
        "compliance": sum(e["compliance"] for e in evals) / len(evals),
        "em": _gem_em(stats),
        "f1": _f1_uit_stats(stats),
        "cer": _gem_cer(stats),
        "tijd": sum(tijden) / len(tijden),
        "tijd_mediaan": statistics.median(tijden),
    }


def _save(fig, pad: Path):
    fig.tight_layout()
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_overall(bl: dict, ft: dict, uitvoer: Path):
    labels = ["Compliance", "Exact Match", "F1"]
    keys = ["compliance", "em", "f1"]
    bl_vals = [_metrics(bl)[k] * 100 for k in keys]
    ft_vals = [_metrics(ft)[k] * 100 for k in keys]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars_bl = ax.bar(x - w / 2, bl_vals, w, label="Baseline", color=KLEUR_BASELINE)
    bars_ft = ax.bar(x + w / 2, ft_vals, w, label="Finetuned", color=KLEUR_FINETUNED)

    for bar in [*bars_bl, *bars_ft]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Overall metrieken - baseline vs finetuned (hybride)")
    ax.legend()
    _save(fig, uitvoer / "1_overall_metrics.png")


def fig_cer_overall(bl: dict, ft: dict, uitvoer: Path):
    vals = [_metrics(bl)["cer"], _metrics(ft)["cer"]]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["Baseline", "Finetuned"], vals,
                  color=[KLEUR_BASELINE, KLEUR_FINETUNED], width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CER (lager = beter)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Character Error Rate - hybride")
    _save(fig, uitvoer / "2_cer_overall.png")


def _run_tijden(pipeline: str) -> list[float]:
    basis = Path("resultaten/mac") / pipeline
    tijden = []
    for pad in sorted(basis.rglob("*_run*.json")):
        with open(pad, encoding="utf-8") as f:
            data = json.load(f)
        tijden.append(float(data.get("tijd_totaal") or 0))
    return tijden


def fig_tijd_overall(uitvoer: Path):
    bl_tijden = _run_tijden(BASELINE)
    ft_tijden = _run_tijden(FINETUNED)
    gemiddelden = [sum(bl_tijden) / len(bl_tijden), sum(ft_tijden) / len(ft_tijden)]
    medianen = [statistics.median(bl_tijden), statistics.median(ft_tijden)]

    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars_avg = ax.bar(x - w / 2, gemiddelden, w, label="Gemiddelde",
                      color=[KLEUR_BASELINE, KLEUR_FINETUNED])
    bars_med = ax.bar(x + w / 2, medianen, w, label="Mediaan",
                      color=[KLEUR_BASELINE, KLEUR_FINETUNED], alpha=0.55,
                      hatch="//", edgecolor="white")

    for bar in [*bars_avg, *bars_med]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Finetuned"])
    ax.set_ylabel("Verwerkingstijd (s)")
    ax.set_ylim(0, max(gemiddelden + medianen) * 1.25)
    ax.set_title("Overall verwerkingstijd - hybride")
    ax.legend()
    _save(fig, uitvoer / "10_tijd_overall.png")


def fig_per_cat(bl: dict, ft: dict, key: str, label: str, uitvoer: Path,
                bestandsnaam: str, pct: bool = True):
    bl_vals = []
    ft_vals = []
    labels = []
    factor = 100 if pct else 1
    for cat, cat_label in zip(CATEGORIEEN, CAT_LABELS):
        bl_vals.append((_metrics(bl, cat).get(key) or 0) * factor)
        ft_vals.append((_metrics(ft, cat).get(key) or 0) * factor)
        labels.append(cat_label)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_bl = ax.bar(x - w / 2, bl_vals, w, label="Baseline", color=KLEUR_BASELINE)
    bars_ft = ax.bar(x + w / 2, ft_vals, w, label="Finetuned", color=KLEUR_FINETUNED)

    suffix = "%" if pct else ""
    offset = 0.7 if pct else 0.01
    for bar in [*bars_bl, *bars_ft]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{bar.get_height():.1f}{suffix}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{label}{' (%)' if pct else ''}")
    ax.set_ylim(0, (max([*bl_vals, *ft_vals]) or 1) * 1.22)
    ax.set_title(f"{label} per categorie - baseline vs finetuned (hybride)")
    ax.legend()
    _save(fig, uitvoer / bestandsnaam)


def fig_tijd(bl: dict, ft: dict, uitvoer: Path):
    fig_per_cat(bl, ft, "tijd", "Gemiddelde verwerkingstijd (s)",
                uitvoer, "7_tijd_per_cat.png", pct=False)


def fig_veld_em(bl: dict, ft: dict, uitvoer: Path):
    bl_veld = _veld_em(bl)
    ft_veld = _veld_em(ft)
    velden = sorted(set(bl_veld) | set(ft_veld))
    bl_vals = [bl_veld.get(v, 0) * 100 for v in velden]
    ft_vals = [ft_veld.get(v, 0) * 100 for v in velden]

    y = np.arange(len(velden))
    h = 0.35
    fig, ax = plt.subplots(figsize=(9, max(5, len(velden) * 0.5)))
    ax.barh(y + h / 2, bl_vals, h, label="Baseline", color=KLEUR_BASELINE)
    ax.barh(y - h / 2, ft_vals, h, label="Finetuned", color=KLEUR_FINETUNED)
    ax.set_yticks(y)
    ax.set_yticklabels(velden, fontsize=8.5)
    ax.set_xlabel("Exact Match (%)")
    ax.set_xlim(0, 115)
    ax.set_title("Exact Match per veld - hybride")
    ax.legend()
    _save(fig, uitvoer / "8_veld_em.png")


def fig_delta_veld(bl: dict, ft: dict, uitvoer: Path):
    bl_veld = _veld_em(bl)
    ft_veld = _veld_em(ft)
    deltas = [(v, (ft_veld.get(v, 0) - bl_veld.get(v, 0)) * 100)
              for v in sorted(set(bl_veld) | set(ft_veld))]
    deltas.sort(key=lambda item: item[1])

    velden = [d[0] for d in deltas]
    waarden = [d[1] for d in deltas]
    kleuren = [KLEUR_FINETUNED if w >= 0 else KLEUR_BASELINE for w in waarden]

    fig, ax = plt.subplots(figsize=(8, max(4, len(velden) * 0.5)))
    ax.barh(range(len(velden)), waarden, color=kleuren, edgecolor="white")
    ax.set_yticks(range(len(velden)))
    ax.set_yticklabels(velden, fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta EM (finetuned - baseline, procentpunten)")
    ax.set_title("Verschil per veld door fine-tuning - hybride")
    ax.legend(handles=[
        mpatches.Patch(color=KLEUR_FINETUNED, label="Finetuned beter"),
        mpatches.Patch(color=KLEUR_BASELINE, label="Baseline beter"),
    ], fontsize=8)
    _save(fig, uitvoer / "9_delta_veld.png")


def fig_compliance_digitaal_gescand(bl_pd: dict, ft_pd: dict,
                                    classificatie: dict, uitvoer: Path):
    digitaal_stems = [s for s, digitaal in classificatie.items() if digitaal]
    gescand_stems = [s for s, digitaal in classificatie.items() if not digitaal]

    bl_dig = _aggregeer_doc_groep(bl_pd, digitaal_stems)
    ft_dig = _aggregeer_doc_groep(ft_pd, digitaal_stems)
    bl_scan = _aggregeer_doc_groep(bl_pd, gescand_stems)
    ft_scan = _aggregeer_doc_groep(ft_pd, gescand_stems)

    groepen = ["Digitaal\nBaseline", "Digitaal\nFinetuned",
               "Gescand\nBaseline", "Gescand\nFinetuned"]
    waarden = [
        (bl_dig["compliance"] or 0) * 100,
        (ft_dig["compliance"] or 0) * 100,
        (bl_scan["compliance"] or 0) * 100,
        (ft_scan["compliance"] or 0) * 100,
    ]
    kleuren = [KLEUR_BASELINE, KLEUR_FINETUNED, KLEUR_BASELINE, KLEUR_FINETUNED]
    hatches = ["", "", "//", "//"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(groepen, waarden, color=kleuren, hatch=hatches,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, waarden):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Compliance (%)")
    ax.set_ylim(0, 115)
    ax.set_title("JSON Compliance - digitaal vs gescand (hybride)")
    ax.legend(handles=[
        mpatches.Patch(color=KLEUR_BASELINE, label="Baseline"),
        mpatches.Patch(color=KLEUR_FINETUNED, label="Finetuned"),
    ], fontsize=9)
    _save(fig, uitvoer / "11_compliance_digitaal_gescand.png")


def fig_tijd_digitaal_gescand(bl_pd: dict, ft_pd: dict,
                              classificatie: dict, uitvoer: Path):
    digitaal_stems = [s for s, digitaal in classificatie.items() if digitaal]
    gescand_stems = [s for s, digitaal in classificatie.items() if not digitaal]

    bl_dig = _aggregeer_doc_groep(bl_pd, digitaal_stems)
    ft_dig = _aggregeer_doc_groep(ft_pd, digitaal_stems)
    bl_scan = _aggregeer_doc_groep(bl_pd, gescand_stems)
    ft_scan = _aggregeer_doc_groep(ft_pd, gescand_stems)

    groepen = ["Digitaal\nBaseline", "Digitaal\nFinetuned",
               "Gescand\nBaseline", "Gescand\nFinetuned"]
    gemiddelden = [
        bl_dig["tijd"] or 0,
        ft_dig["tijd"] or 0,
        bl_scan["tijd"] or 0,
        ft_scan["tijd"] or 0,
    ]
    medianen = [
        bl_dig["tijd_mediaan"] or 0,
        ft_dig["tijd_mediaan"] or 0,
        bl_scan["tijd_mediaan"] or 0,
        ft_scan["tijd_mediaan"] or 0,
    ]
    kleuren = [KLEUR_BASELINE, KLEUR_FINETUNED, KLEUR_BASELINE, KLEUR_FINETUNED]
    hatches = ["", "", "//", "//"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(groepen))
    w = 0.34
    bars_avg = ax.bar(x - w / 2, gemiddelden, w, color=kleuren, hatch=hatches,
                      edgecolor="white", label="Gemiddelde")
    bars_med = ax.bar(x + w / 2, medianen, w, color=kleuren, hatch=hatches,
                      edgecolor="white", alpha=0.55, label="Mediaan")
    for bar in [*bars_avg, *bars_med]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(groepen)
    ax.set_ylabel("Gemiddelde verwerkingstijd (s)")
    ax.set_ylim(0, max(gemiddelden + medianen or [1]) * 1.25)
    ax.set_title("Verwerkingstijd - digitaal vs gescand (hybride)")
    ax.legend(fontsize=9)
    _save(fig, uitvoer / "11_tijd_digitaal_gescand.png")


def fig_digitaal_vs_gescand(bl_pd: dict, ft_pd: dict,
                            classificatie: dict, uitvoer: Path):
    digitaal_stems = [s for s, digitaal in classificatie.items() if digitaal]
    gescand_stems = [s for s, digitaal in classificatie.items() if not digitaal]

    bl_dig = _aggregeer_doc_groep(bl_pd, digitaal_stems)
    ft_dig = _aggregeer_doc_groep(ft_pd, digitaal_stems)
    bl_scan = _aggregeer_doc_groep(bl_pd, gescand_stems)
    ft_scan = _aggregeer_doc_groep(ft_pd, gescand_stems)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    groepen = ["Digitaal\nBaseline", "Digitaal\nFinetuned",
               "Gescand\nBaseline", "Gescand\nFinetuned"]
    kleuren = [KLEUR_BASELINE, KLEUR_FINETUNED, KLEUR_BASELINE, KLEUR_FINETUNED]
    hatches = ["", "", "//", "//"]

    for ax, metriek, titel in [
        (axes[0], "em", "Exact Match (%)"),
        (axes[1], "f1", "F1 (%)"),
    ]:
        waarden = [
            (bl_dig[metriek] or 0) * 100,
            (ft_dig[metriek] or 0) * 100,
            (bl_scan[metriek] or 0) * 100,
            (ft_scan[metriek] or 0) * 100,
        ]
        bars = ax.bar(groepen, waarden, color=kleuren, hatch=hatches,
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, waarden):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)
        ax.set_ylabel(titel)
        ax.set_ylim(0, max(waarden or [1]) * 1.25 + 5)
        ax.set_title(titel + " - digitaal vs gescand")
        ax.legend(handles=[
            mpatches.Patch(color=KLEUR_BASELINE, label="Baseline"),
            mpatches.Patch(color=KLEUR_FINETUNED, label="Finetuned"),
        ], fontsize=8)

    fig.suptitle(
        f"Digitaal vs gescand - hybride (digitaal n={bl_dig['n']}, gescand n={bl_scan['n']})",
        fontsize=11, y=1.02,
    )
    _save(fig, uitvoer / "12_digitaal_vs_gescand.png")


def druk_rapport(bl: dict, ft: dict):
    print("\n" + "=" * 74)
    print("  RAPPORT - baseline/hybride vs finetuned/hybride")
    print("=" * 74)
    print(f"\n  {'Metriek':<18} {'Baseline':>12} {'Finetuned':>12} {'Delta':>12}")
    print("  " + "-" * 58)
    for key, label, pct in [
        ("compliance", "Compliance", True),
        ("em", "Exact Match", True),
        ("f1", "F1", True),
        ("cer", "CER", False),
        ("tijd", "Tijd (s)", False),
    ]:
        bv = _metrics(bl)[key]
        fv = _metrics(ft)[key]
        factor = 100 if pct else 1
        suffix = "%" if pct else ""
        delta = (fv - bv) * factor
        print(f"  {label:<18} {bv * factor:>10.1f}{suffix} "
              f"{fv * factor:>10.1f}{suffix} {delta:>+10.1f}{suffix}")

    print(f"\n  {'Categorie':<14} {'Comp bl':>8} {'Comp ft':>8} "
          f"{'EM bl':>8} {'EM ft':>8} {'F1 bl':>8} {'F1 ft':>8} "
          f"{'CER bl':>8} {'CER ft':>8} {'Tijd bl':>8} {'Tijd ft':>8}")
    print("  " + "-" * 104)
    for cat, label in zip(CATEGORIEEN, CAT_LABELS):
        bm = _metrics(bl, cat)
        fm = _metrics(ft, cat)
        print(f"  {label:<14} {bm['compliance']*100:>7.1f}% {fm['compliance']*100:>7.1f}% "
              f"{bm['em']*100:>7.1f}% {fm['em']*100:>7.1f}% "
              f"{bm['f1']*100:>7.1f}% {fm['f1']*100:>7.1f}% "
              f"{bm['cer']:>8.3f} {fm['cer']:>8.3f} "
              f"{bm['tijd']:>7.1f}s {fm['tijd']:>7.1f}s")
    print()


def main():
    UITVOER_MAP.mkdir(parents=True, exist_ok=True)
    print("  Evalueer baseline/hybride...")
    bl = evalueer_pipeline(BASELINE, include_per_doc=True)
    print("  Evalueer finetuned/hybride...")
    ft = evalueer_pipeline(FINETUNED, include_per_doc=True)
    classificatie = _classificeer_pdfs()

    druk_rapport(bl, ft)
    _stijl()
    print("  Grafieken genereren...")
    fig_overall(bl, ft, UITVOER_MAP)
    fig_cer_overall(bl, ft, UITVOER_MAP)
    fig_tijd_overall(UITVOER_MAP)
    fig_per_cat(bl, ft, "compliance", "JSON Compliance", UITVOER_MAP, "3_compliance_per_cat.png")
    fig_per_cat(bl, ft, "em", "Exact Match", UITVOER_MAP, "4_em_per_cat.png")
    fig_per_cat(bl, ft, "f1", "F1", UITVOER_MAP, "5_f1_per_cat.png")
    fig_per_cat(bl, ft, "cer", "CER", UITVOER_MAP, "6_cer_per_cat.png", pct=False)
    fig_tijd(bl, ft, UITVOER_MAP)
    fig_veld_em(bl, ft, UITVOER_MAP)
    fig_delta_veld(bl, ft, UITVOER_MAP)
    fig_compliance_digitaal_gescand(bl["per_doc"], ft["per_doc"], classificatie, UITVOER_MAP)
    fig_tijd_digitaal_gescand(bl["per_doc"], ft["per_doc"], classificatie, UITVOER_MAP)
    fig_digitaal_vs_gescand(bl["per_doc"], ft["per_doc"], classificatie, UITVOER_MAP)
    print(f"\n  Alle grafieken opgeslagen in: {UITVOER_MAP}/")


if __name__ == "__main__":
    main()
