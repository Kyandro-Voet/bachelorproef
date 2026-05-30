#!/usr/bin/env python3
"""
Analyse: baseline/visie vs finetuned/visie.

Genereert grafieken voor hoofdstuk 6 in dezelfde structuur als de tekst- en
hybrideanalyse.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evalueer import evalueer_pipeline


BASELINE = "baseline/visie"
FINETUNED = "finetuned/visie"
UITVOER_MAP = Path("resultaten/analyse/baseline_vs_finetuned/visie")

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
    ax.set_title("Overall metrieken - baseline vs finetuned (visie)")
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
    ax.set_title("Character Error Rate - visie")
    _save(fig, uitvoer / "2_cer_overall.png")


def fig_tijd_overall(bl: dict, ft: dict, uitvoer: Path):
    vals = [_metrics(bl)["tijd"], _metrics(ft)["tijd"]]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["Baseline", "Finetuned"], vals,
                  color=[KLEUR_BASELINE, KLEUR_FINETUNED], width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Verwerkingstijd (s)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Gemiddelde verwerkingstijd - visie")
    _save(fig, uitvoer / "8_tijd_overall.png")


def fig_per_cat(bl: dict, ft: dict, key: str, label: str, uitvoer: Path,
                bestandsnaam: str, pct: bool = True):
    bl_vals = []
    ft_vals = []
    factor = 100 if pct else 1
    for cat in CATEGORIEEN:
        bl_vals.append((_metrics(bl, cat).get(key) or 0) * factor)
        ft_vals.append((_metrics(ft, cat).get(key) or 0) * factor)

    x = np.arange(len(CAT_LABELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_bl = ax.bar(x - w / 2, bl_vals, w, label="Baseline", color=KLEUR_BASELINE)
    bars_ft = ax.bar(x + w / 2, ft_vals, w, label="Finetuned", color=KLEUR_FINETUNED)

    suffix = "%" if pct else ("s" if key == "tijd" else "")
    offset = 0.7 if pct else max([*bl_vals, *ft_vals, 1]) * 0.015
    for bar in [*bars_bl, *bars_ft]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{bar.get_height():.1f}{suffix}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS)
    ax.set_ylabel(f"{label}{' (%)' if pct else ''}")
    ax.set_ylim(0, (max([*bl_vals, *ft_vals]) or 1) * 1.22)
    ax.set_title(f"{label} per categorie - baseline vs finetuned (visie)")
    ax.legend()
    _save(fig, uitvoer / bestandsnaam)


def main():
    _stijl()
    UITVOER_MAP.mkdir(parents=True, exist_ok=True)

    print("Analyse baseline/visie vs finetuned/visie")
    bl = evalueer_pipeline(BASELINE)
    ft = evalueer_pipeline(FINETUNED)

    fig_overall(bl, ft, UITVOER_MAP)
    fig_cer_overall(bl, ft, UITVOER_MAP)
    fig_per_cat(bl, ft, "compliance", "JSON Compliance", UITVOER_MAP, "3_compliance_per_cat.png")
    fig_per_cat(bl, ft, "em", "Exact Match", UITVOER_MAP, "4_em_per_cat.png")
    fig_per_cat(bl, ft, "f1", "F1-score", UITVOER_MAP, "5_f1_per_cat.png")
    fig_per_cat(bl, ft, "cer", "Character Error Rate", UITVOER_MAP, "6_cer_per_cat.png", pct=False)
    fig_per_cat(bl, ft, "tijd", "Gemiddelde verwerkingstijd (s)", UITVOER_MAP, "7_tijd_per_cat.png", pct=False)
    fig_tijd_overall(bl, ft, UITVOER_MAP)


if __name__ == "__main__":
    main()
