#!/usr/bin/env python3
"""
Analyse §6.2: Effect van fine-tuning — baseline/tekst vs finetuned/tekst
=========================================================================
Genereert grafieken en een tekstueel rapport voor de thesistekst.

Gebruik:
  uv run python analyseer_finetuning.py
  uv run python analyseer_finetuning.py --uitvoer resultaten/analyse
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pdfplumber

from evalueer import (
    CATEGORIE_CONFIG,
    PIPELINES,
    evalueer_pipeline,
    _combineer_stats,
    _f1_uit_stats,
    _gem_em,
    _gem_cer,
    evalueer_document,
)

DATA_MAP = Path("data/testing")

# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
BASELINE  = "baseline/tekst"
FINETUNED = "finetuned/tekst"

CATEGORIEEN = ["electricity", "natural gas", "water", "fuels", "waste"]
CAT_LABELS  = ["Electricity", "Natural gas", "Water", "Fuels", "Waste"]

KLEUR_BASELINE  = "#4C72B0"
KLEUR_FINETUNED = "#DD8452"
KLEUR_DIGITAAL  = "#2ca02c"
KLEUR_GESCAND   = "#d62728"

UITVOER_MAP = Path("resultaten/analyse/baseline_vs_finetuned/tekst")


# ──────────────────────────────────────────────
# HULPFUNCTIES
# ──────────────────────────────────────────────
def _metrics(pr: dict, cat: str | None = None) -> dict:
    """Haal compliance/em/f1/cer/tijd op voor overall of één categorie."""
    bron = pr["per_cat"].get(cat) if cat else pr["overall"]
    if bron is None or bron.get("n", 0) == 0:
        return {"compliance": None, "em": None, "f1": None, "cer": None, "tijd": None}
    return bron


def _veld_em(pr: dict, cat: str | None = None) -> dict[str, float]:
    """EM per veld (voor overall of één categorie)."""
    if cat:
        evals = pr["per_cat"].get(cat, {}).get("veld_stats", {})
    else:
        # aggregeer alle categorieën
        evals = {}
        for c_data in pr["per_cat"].values():
            evals = _combineer_stats(evals, c_data.get("veld_stats", {}))

    result = {}
    for veld, s in evals.items():
        deler = s["tp"] + s["fn"]
        result[veld] = s["tp"] / deler if deler > 0 else 0.0
    return result


def _is_digitaal(bestandsnaam: str, categorie: str) -> bool:
    """Controleer of een PDF digitaal is (heeft extraheerbare tekst) of gescand."""
    pad = DATA_MAP / categorie / bestandsnaam
    if not pad.exists():
        return True  # fallback: behandel als digitaal
    try:
        with pdfplumber.open(pad) as pdf:
            return any(p.extract_text() for p in pdf.pages[:2])
    except Exception:
        return True


def _classificeer_pdfs() -> dict[str, bool]:
    """Geeft {bestandsnaam_stem: is_digitaal} voor alle test-PDFs."""
    classificatie = {}
    for cat_dir in DATA_MAP.iterdir():
        if not cat_dir.is_dir():
            continue
        for pdf in cat_dir.glob("*.pdf"):
            classificatie[pdf.stem] = _is_digitaal(pdf.name, cat_dir.name)
    return classificatie


def _aggregeer_doc_groep(per_doc: dict, stems: list[str]) -> dict:
    """Aggregeer per_doc evaluaties voor een set document-stems."""
    evals = [per_doc[s]["eval"] for s in stems if s in per_doc]
    if not evals:
        return {"n": 0, "compliance": None, "em": None, "f1": None,
                "cer": None, "tijd": None}
    n = len(evals)
    n_compliant = sum(1 for e in evals if e["compliance"])
    stats = {}
    for e in evals:
        stats = _combineer_stats(stats, e["veld_stats"])
    avg_tijd = sum(e["tijd"] for e in evals) / n
    return {
        "n": n,
        "compliance": n_compliant / n,
        "em": _gem_em(stats),
        "f1": _f1_uit_stats(stats),
        "cer": _gem_cer(stats),
        "tijd": avg_tijd,
    }


def _stel_stijl_in():
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "font.size":       10,
        "axes.titlesize":  11,
        "axes.labelsize":  10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":      150,
        "savefig.dpi":     150,
        "savefig.bbox":    "tight",
    })


# ──────────────────────────────────────────────
# GRAFIEKEN
# ──────────────────────────────────────────────
def fig_overall_metrics(bl: dict, ft: dict, uitvoer: Path):
    """Staafdiagram: 4 metrieken overall naast elkaar."""
    metrieken = ["compliance", "em", "f1"]
    labels    = ["Compliance", "Exact Match", "F1"]

    bl_vals = [_metrics(bl)[m] * 100 for m in metrieken]
    ft_vals = [_metrics(ft)[m] * 100 for m in metrieken]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))

    bars_bl = ax.bar(x - w/2, bl_vals, w, label="Baseline", color=KLEUR_BASELINE)
    bars_ft = ax.bar(x + w/2, ft_vals, w, label="Finetuned", color=KLEUR_FINETUNED)

    for bar in list(bars_bl) + list(bars_ft):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Overall metrieken — baseline vs finetuned (tekst)")
    ax.legend()
    fig.tight_layout()
    pad = uitvoer / "1_overall_metrics.png"
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_cer_overall(bl: dict, ft: dict, uitvoer: Path):
    """Staafdiagram: CER overall (lager = beter)."""
    bl_cer = _metrics(bl)["cer"]
    ft_cer = _metrics(ft)["cer"]

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(["Baseline", "Finetuned"],
                  [bl_cer, ft_cer],
                  color=[KLEUR_BASELINE, KLEUR_FINETUNED], width=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CER (lager = beter)")
    ax.set_ylim(0, max(bl_cer, ft_cer) * 1.25)
    ax.set_title("Character Error Rate — overall")
    fig.tight_layout()
    pad = uitvoer / "2_cer_overall.png"
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_per_categorie(bl: dict, ft: dict, metriek: str, label: str,
                      uitvoer: Path, bestandsnaam: str, pct: bool = True):
    """Gegroepeerd staafdiagram per categorie voor één metriek."""
    bl_vals, ft_vals, cats_met_data = [], [], []
    for cat, cat_label in zip(CATEGORIEEN, CAT_LABELS):
        bv = _metrics(bl, cat)[metriek]
        fv = _metrics(ft, cat)[metriek]
        if bv is None and fv is None:
            continue
        bl_vals.append((bv or 0) * (100 if pct else 1))
        ft_vals.append((fv or 0) * (100 if pct else 1))
        cats_met_data.append(cat_label)

    x = np.arange(len(cats_met_data))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars_bl = ax.bar(x - w/2, bl_vals, w, label="Baseline", color=KLEUR_BASELINE)
    bars_ft = ax.bar(x + w/2, ft_vals, w, label="Finetuned", color=KLEUR_FINETUNED)

    suffix = "%" if pct else ""
    for bar in list(bars_bl) + list(bars_ft):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.5 if pct else 0.005),
                f"{v:.1f}{suffix}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cats_met_data)
    ylabel = f"{label} ({'%' if pct else ''})"
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, (max(bl_vals + ft_vals) or 1) * 1.2)
    ax.set_title(f"{label} per categorie — baseline vs finetuned")
    ax.legend()
    fig.tight_layout()
    pad = uitvoer / bestandsnaam
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_veld_em(bl: dict, ft: dict, uitvoer: Path):
    """Horizontaal staafdiagram: EM per veld voor beide modellen."""
    # Verzamel alle velden
    bl_veld = _veld_em(bl)
    ft_veld = _veld_em(ft)
    alle_velden = sorted(set(bl_veld) | set(ft_veld))

    bl_vals = [bl_veld.get(v, 0) * 100 for v in alle_velden]
    ft_vals = [ft_veld.get(v, 0) * 100 for v in alle_velden]

    y = np.arange(len(alle_velden))
    h = 0.35
    fig, ax = plt.subplots(figsize=(9, max(5, len(alle_velden) * 0.5)))

    ax.barh(y + h/2, bl_vals, h, label="Baseline", color=KLEUR_BASELINE)
    ax.barh(y - h/2, ft_vals, h, label="Finetuned", color=KLEUR_FINETUNED)

    ax.set_yticks(y)
    ax.set_yticklabels(alle_velden, fontsize=8.5)
    ax.set_xlabel("Exact Match (%)")
    ax.set_xlim(0, 120)
    ax.set_title("Exact Match per veld — baseline vs finetuned")
    ax.legend()
    fig.tight_layout()
    pad = uitvoer / "5_veld_em.png"
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_delta_veld(bl: dict, ft: dict, uitvoer: Path):
    """Delta EM per veld: finetuned − baseline (positief = finetuned beter)."""
    bl_veld = _veld_em(bl)
    ft_veld = _veld_em(ft)
    alle_velden = sorted(set(bl_veld) | set(ft_veld))

    deltas = [(v, (ft_veld.get(v, 0) - bl_veld.get(v, 0)) * 100) for v in alle_velden]
    deltas.sort(key=lambda x: x[1])

    velden  = [d[0] for d in deltas]
    waarden = [d[1] for d in deltas]
    kleuren = [KLEUR_FINETUNED if w >= 0 else KLEUR_BASELINE for w in waarden]

    fig, ax = plt.subplots(figsize=(8, max(4, len(velden) * 0.5)))
    ax.barh(range(len(velden)), waarden, color=kleuren, edgecolor="white")
    ax.set_yticks(range(len(velden)))
    ax.set_yticklabels(velden, fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔEM (finetuned − baseline, procentpunten)")
    ax.set_title("Verbetering/verslechtering per veld door fine-tuning")

    patch_ft = mpatches.Patch(color=KLEUR_FINETUNED, label="Finetuned beter")
    patch_bl = mpatches.Patch(color=KLEUR_BASELINE,  label="Baseline beter")
    ax.legend(handles=[patch_ft, patch_bl], fontsize=8)
    fig.tight_layout()
    pad = uitvoer / "6_delta_veld.png"
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_tijd(bl: dict, ft: dict, uitvoer: Path):
    """Verwerkingstijd per categorie."""
    bl_tijden, ft_tijden, cats_met_data = [], [], []
    for cat, cat_label in zip(CATEGORIEEN, CAT_LABELS):
        bt = _metrics(bl, cat)["tijd"]
        ft_t = _metrics(ft, cat)["tijd"]
        if bt is None and ft_t is None:
            continue
        bl_tijden.append(bt or 0)
        ft_tijden.append(ft_t or 0)
        cats_met_data.append(cat_label)

    x = np.arange(len(cats_met_data))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))

    bars_bl = ax.bar(x - w/2, bl_tijden, w, label="Baseline", color=KLEUR_BASELINE)
    bars_ft = ax.bar(x + w/2, ft_tijden, w, label="Finetuned", color=KLEUR_FINETUNED)

    for bar in list(bars_bl) + list(bars_ft):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cats_met_data)
    ax.set_ylabel("Gemiddelde verwerkingstijd (s)")
    ax.set_title("Verwerkingstijd per categorie — baseline vs finetuned")
    ax.legend()
    fig.tight_layout()
    pad = uitvoer / "7_verwerkingstijd.png"
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_digitaal_vs_gescand(bl_pd: dict, ft_pd: dict,
                            classificatie: dict, uitvoer: Path):
    """
    Vergelijk baseline vs finetuned gesplitst op digitaal vs gescand.
    2 subplots: EM en F1, elk met 4 balken (bl_dig, ft_dig, bl_scan, ft_scan).
    """
    digitaal_stems = [s for s, d in classificatie.items() if d]
    gescand_stems  = [s for s, d in classificatie.items() if not d]

    bl_dig  = _aggregeer_doc_groep(bl_pd, digitaal_stems)
    bl_scan = _aggregeer_doc_groep(bl_pd, gescand_stems)
    ft_dig  = _aggregeer_doc_groep(ft_pd, digitaal_stems)
    ft_scan = _aggregeer_doc_groep(ft_pd, gescand_stems)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metriek, titel in [
        (axes[0], "em", "Exact Match (%)"),
        (axes[1], "f1", "F1 (%)"),
    ]:
        groepen = ["Digitaal\nBaseline", "Digitaal\nFinetuned",
                   "Gescand\nBaseline",  "Gescand\nFinetuned"]
        waarden = [
            (bl_dig[metriek] or 0) * 100,
            (ft_dig[metriek] or 0) * 100,
            (bl_scan[metriek] or 0) * 100,
            (ft_scan[metriek] or 0) * 100,
        ]
        kleuren = [KLEUR_BASELINE, KLEUR_FINETUNED, KLEUR_BASELINE, KLEUR_FINETUNED]
        hatches = ["", "", "//", "//"]

        bars = ax.bar(groepen, waarden, color=kleuren, hatch=hatches,
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, waarden):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)

        ax.set_ylabel(titel)
        ax.set_ylim(0, max(waarden or [1]) * 1.25 + 5)
        ax.set_title(titel + " — digitaal vs gescand")

        patch_bl  = mpatches.Patch(color=KLEUR_BASELINE,  label="Baseline")
        patch_ft  = mpatches.Patch(color=KLEUR_FINETUNED, label="Finetuned")
        ax.legend(handles=[patch_bl, patch_ft], fontsize=8)

    # N-annotatie in titels
    n_dig_bl  = bl_dig["n"]
    n_scan_bl = bl_scan["n"]
    fig.suptitle(
        f"Digitaal vs gescand — baseline vs finetuned\n"
        f"(digitaal n={n_dig_bl}, gescand n={n_scan_bl})",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    pad = uitvoer / "9_digitaal_vs_gescand.png"
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_tijd_analyse(bl_pd: dict, ft_pd: dict,
                     classificatie: dict, uitvoer: Path):
    """
    Verwerkingstijd: 2 subplots.
    Links: overall + digitaal + gescand voor baseline en finetuned.
    Rechts: gemiddelde tijd per categorie, baseline vs finetuned.
    """
    digitaal_stems = [s for s, d in classificatie.items() if d]
    gescand_stems  = [s for s, d in classificatie.items() if not d]

    # Groepen
    bl_all  = _aggregeer_doc_groep(bl_pd, list(classificatie.keys()))
    ft_all  = _aggregeer_doc_groep(ft_pd, list(classificatie.keys()))
    bl_dig  = _aggregeer_doc_groep(bl_pd, digitaal_stems)
    ft_dig  = _aggregeer_doc_groep(ft_pd, digitaal_stems)
    bl_scan = _aggregeer_doc_groep(bl_pd, gescand_stems)
    ft_scan = _aggregeer_doc_groep(ft_pd, gescand_stems)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Links: overall / digitaal / gescand
    ax = axes[0]
    x = np.arange(3)
    w = 0.35
    bl_tijden = [bl_all["tijd"] or 0, bl_dig["tijd"] or 0, bl_scan["tijd"] or 0]
    ft_tijden = [ft_all["tijd"] or 0, ft_dig["tijd"] or 0, ft_scan["tijd"] or 0]

    bars_bl = ax.bar(x - w/2, bl_tijden, w, label="Baseline", color=KLEUR_BASELINE)
    bars_ft = ax.bar(x + w/2, ft_tijden, w, label="Finetuned", color=KLEUR_FINETUNED)
    for bar in list(bars_bl) + list(bars_ft):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Overall", "Digitaal", "Gescand"])
    ax.set_ylabel("Gemiddelde verwerkingstijd (s)")
    ax.set_title("Verwerkingstijd — digitaal vs gescand")
    ax.legend()

    # Rechts: per categorie
    ax2 = axes[1]
    cat_tijden_bl, cat_tijden_ft, cat_labels_plot = [], [], []
    for cat, label in zip(CATEGORIEEN, CAT_LABELS):
        # aggregeer per_doc voor deze categorie
        stems_cat = [s for s, d in bl_pd.items() if d["categorie"] == cat]
        bl_cat = _aggregeer_doc_groep(bl_pd, stems_cat)
        ft_cat = _aggregeer_doc_groep(ft_pd, stems_cat)
        if bl_cat["tijd"] is None and ft_cat["tijd"] is None:
            continue
        cat_tijden_bl.append(bl_cat["tijd"] or 0)
        cat_tijden_ft.append(ft_cat["tijd"] or 0)
        cat_labels_plot.append(label)

    x2 = np.arange(len(cat_labels_plot))
    bars_bl2 = ax2.bar(x2 - w/2, cat_tijden_bl, w, label="Baseline", color=KLEUR_BASELINE)
    bars_ft2 = ax2.bar(x2 + w/2, cat_tijden_ft, w, label="Finetuned", color=KLEUR_FINETUNED)
    for bar in list(bars_bl2) + list(bars_ft2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cat_labels_plot)
    ax2.set_ylabel("Gemiddelde verwerkingstijd (s)")
    ax2.set_title("Verwerkingstijd per categorie")
    ax2.legend()

    fig.tight_layout()
    pad = uitvoer / "10_tijd_analyse.png"
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


def fig_compliance_digitaal_gescand(bl_pd: dict, ft_pd: dict,
                                     classificatie: dict, uitvoer: Path):
    """Compliance rate: digitaal vs gescand, baseline vs finetuned."""
    digitaal_stems = [s for s, d in classificatie.items() if d]
    gescand_stems  = [s for s, d in classificatie.items() if not d]

    bl_dig  = _aggregeer_doc_groep(bl_pd, digitaal_stems)
    bl_scan = _aggregeer_doc_groep(bl_pd, gescand_stems)
    ft_dig  = _aggregeer_doc_groep(ft_pd, digitaal_stems)
    ft_scan = _aggregeer_doc_groep(ft_pd, gescand_stems)

    groepen  = ["Digitaal\nBaseline", "Digitaal\nFinetuned",
                "Gescand\nBaseline",  "Gescand\nFinetuned"]
    waarden  = [
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
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Compliance (%)")
    ax.set_ylim(0, 115)
    ax.set_title(
        f"JSON Compliance — digitaal vs gescand\n"
        f"(digitaal n={bl_dig['n']}, gescand n={bl_scan['n']})"
    )
    patch_bl = mpatches.Patch(color=KLEUR_BASELINE,  label="Baseline")
    patch_ft = mpatches.Patch(color=KLEUR_FINETUNED, label="Finetuned")
    ax.legend(handles=[patch_bl, patch_ft], fontsize=9)
    fig.tight_layout()
    pad = uitvoer / "11_compliance_digitaal_gescand.png"
    fig.savefig(pad)
    plt.close(fig)
    print(f"  Opgeslagen: {pad}")


# ──────────────────────────────────────────────
# TEKSTRAPPORT
# ──────────────────────────────────────────────
def druk_rapport(bl: dict, ft: dict):
    print("\n" + "=" * 68)
    print("  RAPPORT §6.2 — Effect van fine-tuning (tekst-pipeline)")
    print("=" * 68)

    # Overall tabel
    print(f"\n  {'Metriek':<20} {'Baseline':>12} {'Finetuned':>12} {'Δ':>8}")
    print("  " + "─" * 54)
    for metriek, label, pct in [
        ("compliance", "Compliance",  True),
        ("em",         "Exact Match", True),
        ("f1",         "F1",          True),
        ("cer",        "CER",         False),
        ("tijd",       "Tijd (s)",    False),
    ]:
        bv = _metrics(bl)[metriek]
        fv = _metrics(ft)[metriek]
        if bv is None or fv is None:
            continue
        factor = 100 if pct else 1
        suffix = "%" if pct else ""
        delta  = (fv - bv) * factor
        teken  = "+" if delta > 0 else ""
        print(f"  {label:<20} {bv*factor:>10.1f}{suffix}  {fv*factor:>10.1f}{suffix}  "
              f"{teken}{delta:>5.1f}{suffix}")

    # Per categorie
    print(f"\n  {'':─<68}")
    print(f"  {'Categorie':<16} {'':>4}  "
          f"{'EM bl':>7} {'EM ft':>7} {'ΔEM':>6}  "
          f"{'F1 bl':>7} {'F1 ft':>7} {'ΔF1':>6}  "
          f"{'CER bl':>7} {'CER ft':>7}")
    print("  " + "─" * 68)
    for cat, label in zip(CATEGORIEEN, CAT_LABELS):
        bm = _metrics(bl, cat)
        fm = _metrics(ft, cat)
        if bm["em"] is None and fm["em"] is None:
            continue
        be, fe = (bm["em"] or 0)*100, (fm["em"] or 0)*100
        bf, ff = (bm["f1"] or 0)*100, (fm["f1"] or 0)*100
        bc, fc = (bm["cer"] or 0),    (fm["cer"] or 0)
        teken_e = "+" if fe-be > 0 else ""
        teken_f = "+" if ff-bf > 0 else ""
        print(f"  {label:<16} {'':>4}  "
              f"{be:>6.1f}% {fe:>6.1f}% {teken_e}{fe-be:>4.1f}%  "
              f"{bf:>6.1f}% {ff:>6.1f}% {teken_f}{ff-bf:>4.1f}%  "
              f"{bc:>7.3f} {fc:>7.3f}")

    # Samenvatting
    print(f"\n  {'':─<68}")
    bl_em = _metrics(bl)["em"] or 0
    ft_em = _metrics(ft)["em"] or 0
    conclusie = "FINETUNED BETER" if ft_em > bl_em + 0.01 else \
                "BASELINE BETER"  if bl_em > ft_em + 0.01 else \
                "GEEN SIGNIFICANT VERSCHIL"
    print(f"\n  Tussentijdse conclusie: {conclusie}")
    print(f"  Overall ΔEM = {(ft_em - bl_em)*100:+.1f} procentpunten")
    print()


def druk_digitaal_gescand_rapport(bl_pd: dict, ft_pd: dict, classificatie: dict):
    """Tekstrapport voor digitaal vs gescand analyse."""
    digitaal_stems = [s for s, d in classificatie.items() if d]
    gescand_stems  = [s for s, d in classificatie.items() if not d]

    bl_dig  = _aggregeer_doc_groep(bl_pd, digitaal_stems)
    bl_scan = _aggregeer_doc_groep(bl_pd, gescand_stems)
    ft_dig  = _aggregeer_doc_groep(ft_pd, digitaal_stems)
    ft_scan = _aggregeer_doc_groep(ft_pd, gescand_stems)

    print("\n" + "=" * 68)
    print("  DIGITAAL vs GESCAND — effect fine-tuning")
    print("=" * 68)

    for groep_label, bl_g, ft_g, n in [
        ("Digitaal", bl_dig, ft_dig, bl_dig["n"]),
        ("Gescand",  bl_scan, ft_scan, bl_scan["n"]),
    ]:
        print(f"\n  {groep_label} (n={n})")
        print(f"  {'Metriek':<16} {'Baseline':>10} {'Finetuned':>10} {'Δ':>8}")
        print("  " + "─" * 46)
        for metriek, label, pct in [
            ("compliance", "Compliance",  True),
            ("em",         "Exact Match", True),
            ("f1",         "F1",          True),
            ("cer",        "CER",         False),
            ("tijd",       "Tijd (s)",    False),
        ]:
            bv = bl_g.get(metriek)
            fv = ft_g.get(metriek)
            if bv is None or fv is None:
                continue
            factor = 100 if pct else 1
            suffix = "%" if pct else ""
            delta = (fv - bv) * factor
            teken = "+" if delta > 0 else ""
            print(f"  {label:<16} {bv*factor:>8.1f}{suffix}  {fv*factor:>8.1f}{suffix}  "
                  f"{teken}{delta:>5.1f}{suffix}")

    # Samenvatting digitaal vs gescand
    print(f"\n  {'':─<68}")
    print(f"  {'':16} {'Digitaal EM':>12} {'Gescand EM':>12} {'Verschil':>10}")
    print("  " + "─" * 54)
    for label, bl_g, ft_g in [("Baseline", bl_dig, bl_scan), ("Finetuned", ft_dig, ft_scan)]:
        dig_em  = (bl_g["em"] or 0) * 100
        scan_em = (ft_g["em"] or 0) * 100
        diff = dig_em - scan_em
        teken = "+" if diff > 0 else ""
        print(f"  {label:<16} {dig_em:>10.1f}%  {scan_em:>10.1f}%  {teken}{diff:>7.1f}pp")
    print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uitvoer", default=str(UITVOER_MAP))
    args = parser.parse_args()

    uitvoer = Path(args.uitvoer)
    uitvoer.mkdir(parents=True, exist_ok=True)

    print("  Evalueer baseline/tekst...")
    bl = evalueer_pipeline(BASELINE, include_per_doc=True)
    print("  Evalueer finetuned/tekst...")
    ft = evalueer_pipeline(FINETUNED, include_per_doc=True)

    bl_pd = bl["per_doc"]
    ft_pd = ft["per_doc"]

    print("  PDF-types classificeren (digitaal/gescand)...")
    classificatie = _classificeer_pdfs()
    n_dig   = sum(1 for v in classificatie.values() if v)
    n_scan  = sum(1 for v in classificatie.values() if not v)
    print(f"    Digitaal: {n_dig}   Gescand: {n_scan}")

    druk_rapport(bl, ft)
    druk_digitaal_gescand_rapport(bl_pd, ft_pd, classificatie)

    _stel_stijl_in()
    print("  Grafieken genereren...")

    fig_overall_metrics(bl, ft, uitvoer)
    fig_cer_overall(bl, ft, uitvoer)
    fig_per_categorie(bl, ft, "em",  "Exact Match", uitvoer, "3_em_per_cat.png")
    fig_per_categorie(bl, ft, "f1",  "F1",          uitvoer, "4_f1_per_cat.png")
    fig_veld_em(bl, ft, uitvoer)
    fig_delta_veld(bl, ft, uitvoer)
    fig_per_categorie(bl, ft, "cer", "CER", uitvoer, "8_cer_per_cat.png", pct=False)
    fig_tijd(bl, ft, uitvoer)
    fig_digitaal_vs_gescand(bl_pd, ft_pd, classificatie, uitvoer)
    fig_tijd_analyse(bl_pd, ft_pd, classificatie, uitvoer)
    fig_compliance_digitaal_gescand(bl_pd, ft_pd, classificatie, uitvoer)

    print(f"\n  Alle grafieken opgeslagen in: {uitvoer}/")


if __name__ == "__main__":
    main()
