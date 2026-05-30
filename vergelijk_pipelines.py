#!/usr/bin/env python3
"""
Vergelijk de drie pipeline-aanpakken: tekst, visie en hybride.

Gebruikt dezelfde evaluatielogica als evalueer.py, inclusief de drie-run
aggregatie per factuur.

Voorbeelden:
  uv run python vergelijk_pipelines.py
  uv run python vergelijk_pipelines.py --variant finetuned
  uv run python vergelijk_pipelines.py --categorie electricity
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evalueer import CATEGORIE_CONFIG, GROUND_TRUTH_MAP, evalueer_pipeline


AANPAKKEN = [
    ("tekst", "Tekst"),
    ("visie", "Visie"),
    ("hybride", "Hybride"),
]
CATEGORIEEN = ["electricity", "natural gas", "water", "fuels", "waste"]
UITVOER_MAP = Path("resultaten/analyse/pipeline_vergelijking")


def _verwacht_aantal_docs(categorie: str | None = None) -> int:
    bestanden = [
        p for p in GROUND_TRUTH_MAP.rglob("*.json")
        if p.parent.name in CATEGORIE_CONFIG
    ]
    if categorie:
        bestanden = [p for p in bestanden if p.parent.name == categorie]
    return len(bestanden)


def _pipeline_naam(variant: str, aanpak: str) -> str:
    return f"{variant}/{aanpak}"


def _pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{100 * v:.1f}%"


def _num(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _label(pr: dict) -> str:
    aanpak = pr["pipeline"].split("/")[-1]
    return dict(AANPAKKEN).get(aanpak, aanpak)


def verzamel_resultaten(variant: str, categorie: str | None) -> list[dict]:
    resultaten = []
    for aanpak, _ in AANPAKKEN:
        pipeline = _pipeline_naam(variant, aanpak)
        print(f"  Evalueer {pipeline}...", end="\r")
        resultaten.append(evalueer_pipeline(pipeline, categorie_filter=categorie))
    print(" " * 80, end="\r")
    return resultaten


def print_overzicht(resultaten: list[dict], verwacht_docs: int) -> None:
    print()
    print("=" * 88)
    print("  PIPELINEVERGELIJKING")
    print("=" * 88)
    print(f"  {'Pipeline':<10} {'Comp':>8} {'EM':>8} {'F1':>8} {'CER':>8} {'Tijd(s)':>10} {'Docs':>6}")
    print("  " + "-" * 84)

    for pr in resultaten:
        o = pr["overall"]
        n = o["n"]
        status = "" if n == verwacht_docs else f" / {verwacht_docs}"
        print(
            f"  {_label(pr):<10} "
            f"{_pct(o['compliance']):>8} "
            f"{_pct(o['em']):>8} "
            f"{_pct(o['f1']):>8} "
            f"{_num(o['cer']):>8} "
            f"{o['tijd']:>10.1f} "
            f"{str(n) + status:>6}"
        )

    print()
    _print_beste(resultaten)
    print()


def _print_beste(resultaten: list[dict]) -> None:
    geldig = [pr for pr in resultaten if pr["overall"]["n"] > 0]
    if not geldig:
        print("  Geen resultaten gevonden.")
        return

    beste_em = max(geldig, key=lambda pr: pr["overall"]["em"])
    beste_f1 = max(geldig, key=lambda pr: pr["overall"]["f1"])
    beste_cer = min(geldig, key=lambda pr: pr["overall"]["cer"])
    snelste = min(geldig, key=lambda pr: pr["overall"]["tijd"])

    print(f"  Beste Exact Match: {_label(beste_em)} ({_pct(beste_em['overall']['em'])})")
    print(f"  Beste F1:          {_label(beste_f1)} ({_pct(beste_f1['overall']['f1'])})")
    print(f"  Beste CER:         {_label(beste_cer)} ({_num(beste_cer['overall']['cer'])})")
    print(f"  Snelste:           {_label(snelste)} ({snelste['overall']['tijd']:.1f}s)")


def print_per_categorie(resultaten: list[dict]) -> None:
    print("=" * 88)
    print("  PER CATEGORIE")
    print("=" * 88)

    for cat in CATEGORIEEN:
        if not any(cat in pr["per_cat"] for pr in resultaten):
            continue
        print(f"\n  [{cat}]")
        print(f"  {'Pipeline':<10} {'Comp':>8} {'EM':>8} {'F1':>8} {'CER':>8} {'Tijd(s)':>10} {'Docs':>6}")
        for pr in resultaten:
            d = pr["per_cat"].get(cat)
            if not d:
                continue
            print(
                f"  {_label(pr):<10} "
                f"{_pct(d['compliance']):>8} "
                f"{_pct(d['em']):>8} "
                f"{_pct(d['f1']):>8} "
                f"{_num(d['cer']):>8} "
                f"{d['tijd']:>10.1f} "
                f"{d['n']:>6}"
            )
    print()


def schrijf_csvs(resultaten: list[dict], uitvoer: Path) -> None:
    uitvoer.mkdir(parents=True, exist_ok=True)

    overall_pad = uitvoer / "overall.csv"
    with open(overall_pad, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "n", "compliance", "exact_match", "f1", "cer", "tijd"])
        for pr in resultaten:
            o = pr["overall"]
            writer.writerow([
                pr["pipeline"], o["n"], o["compliance"], o["em"],
                o["f1"], o["cer"], o["tijd"],
            ])

    per_cat_pad = uitvoer / "per_categorie.csv"
    with open(per_cat_pad, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "categorie", "n", "compliance", "exact_match", "f1", "cer", "tijd"])
        for pr in resultaten:
            for cat in CATEGORIEEN:
                d = pr["per_cat"].get(cat)
                if not d:
                    continue
                writer.writerow([
                    pr["pipeline"], cat, d["n"], d["compliance"], d["em"],
                    d["f1"], d["cer"], d["tijd"],
                ])

    print(f"  CSV opgeslagen: {overall_pad}")
    print(f"  CSV opgeslagen: {per_cat_pad}")


def maak_grafieken(resultaten: list[dict], uitvoer: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    uitvoer.mkdir(parents=True, exist_ok=True)
    labels = [_label(pr) for pr in resultaten]
    x = np.arange(len(labels))

    metrics = [
        ("compliance", "JSON Compliance (%)", True),
        ("em", "Exact Match (%)", True),
        ("f1", "Field-level F1 (%)", True),
        ("cer", "CER", False),
        ("tijd", "Verwerkingstijd (s)", False),
    ]
    waarden = {
        key: [pr["overall"][key] * 100 if pct else pr["overall"][key] for pr in resultaten]
        for key, _, pct in metrics
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, key, titel in zip(axes, ["compliance", "em", "f1"], ["Compliance", "Exact Match", "F1"]):
        bars = ax.bar(x, waarden[key], color=["#4C72B0", "#55A868", "#DD8452"])
        ax.set_title(titel)
        ax.set_ylabel("%")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 110)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    pad = uitvoer / "1_overall_metrics.png"
    fig.savefig(pad, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, key, titel, ylabel in [
        (axes[0], "cer", "Character Error Rate", "CER"),
        (axes[1], "tijd", "Verwerkingstijd", "seconden"),
    ]:
        bars = ax.bar(x, waarden[key], color=["#4C72B0", "#55A868", "#DD8452"])
        ax.set_title(titel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.2f}" if key == "cer" else f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    pad2 = uitvoer / "2_cer_tijd.png"
    fig.savefig(pad2, dpi=150)
    plt.close(fig)

    cat_labels = ["Electricity", "Natural gas", "Water", "Fuels", "Waste"]
    cat_x = np.arange(len(CATEGORIEEN))
    breedte = 0.25
    kleuren = ["#4C72B0", "#55A868", "#DD8452"]

    def _per_cat_fig(key: str, titel: str, ylabel: str, bestandsnaam: str, pct: bool = True) -> Path:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for i, pr in enumerate(resultaten):
            vals = []
            for cat in CATEGORIEEN:
                d = pr["per_cat"].get(cat, {})
                v = d.get(key, 0)
                vals.append(v * 100 if pct else v)
            offset = (i - (len(resultaten) - 1) / 2) * breedte
            bars = ax.bar(cat_x + offset, vals, breedte, label=_label(pr), color=kleuren[i])
            for bar in bars:
                waarde = bar.get_height()
                label = f"{waarde:.1f}%" if pct else (f"{waarde:.3f}" if key == "cer" else f"{waarde:.1f}s")
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    waarde + (0.7 if pct else max(vals + [1]) * 0.015),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90 if pct else 0,
                )

        ax.set_title(titel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(cat_x)
        ax.set_xticklabels(cat_labels)
        ax.legend()
        ax.set_ylim(0, (max([p.get_height() for p in ax.patches]) or 1) * 1.25)
        fig.tight_layout()
        pad_cat = uitvoer / bestandsnaam
        fig.savefig(pad_cat, dpi=150)
        plt.close(fig)
        return pad_cat

    per_cat_paden = [
        _per_cat_fig("compliance", "JSON Compliance per categorie", "Compliance (%)", "3_compliance_per_cat.png"),
        _per_cat_fig("em", "Exact Match per categorie", "Exact Match (%)", "4_em_per_cat.png"),
        _per_cat_fig("f1", "F1-score per categorie", "F1 (%)", "5_f1_per_cat.png"),
        _per_cat_fig("cer", "Character Error Rate per categorie", "CER", "6_cer_per_cat.png", pct=False),
        _per_cat_fig("tijd", "Verwerkingstijd per categorie", "Gemiddelde verwerkingstijd (s)", "7_tijd_per_cat.png", pct=False),
    ]

    print(f"  Grafiek opgeslagen: {pad}")
    print(f"  Grafiek opgeslagen: {pad2}")
    for p in per_cat_paden:
        print(f"  Grafiek opgeslagen: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vergelijk tekst-, visie- en hybride pipeline op dezelfde evaluatiemetrieken."
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "finetuned"],
        default="baseline",
        help="Welke variant vergelijken: baseline of finetuned.",
    )
    parser.add_argument(
        "--categorie",
        choices=list(CATEGORIE_CONFIG.keys()),
        help="Beperk de vergelijking tot één categorie.",
    )
    parser.add_argument(
        "--uitvoer",
        default=str(UITVOER_MAP),
        help="Map voor CSV- en grafiekoutput.",
    )
    parser.add_argument(
        "--geen-grafieken",
        action="store_true",
        help="Schrijf alleen CSV en console-output.",
    )
    args = parser.parse_args()

    resultaten = verzamel_resultaten(args.variant, args.categorie)
    verwacht_docs = _verwacht_aantal_docs(args.categorie)
    uitvoer = Path(args.uitvoer) / args.variant

    print_overzicht(resultaten, verwacht_docs)
    print_per_categorie(resultaten)
    schrijf_csvs(resultaten, uitvoer)
    if not args.geen_grafieken:
        maak_grafieken(resultaten, uitvoer)


if __name__ == "__main__":
    main()
