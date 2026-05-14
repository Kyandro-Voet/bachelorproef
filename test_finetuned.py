"""
Test alle 3 finetuned pipelines voor alle facturen in data/testing/.

Volgorde:
  1. finetuned/tekst   (Ollama — qwen3-8b-finetuned)
  2. finetuned/visie   (MLX vision model)
  3. finetuned/hybride (Ollama + MLX vision)

Gebruik:
    uv run python test_finetuned.py
    uv run python test_finetuned.py --pipeline tekst       # alleen tekst
    uv run python test_finetuned.py --pipeline visie
    uv run python test_finetuned.py --pipeline hybride
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PIPELINES = [
    ("tekst",   "pipelines/finetuned/tekst.py"),
    ("visie",   "pipelines/finetuned/visie.py"),
    ("hybride", "pipelines/finetuned/hybride.py"),
]

ROOT = Path(__file__).resolve().parent


def run_pipeline(naam: str, script: Path) -> bool:
    print(f"\n{'#' * 65}")
    print(f"  PIPELINE: finetuned/{naam}")
    print(f"{'#' * 65}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    duur = time.time() - start

    ok = result.returncode == 0
    status = "KLAAR" if ok else f"FOUT (returncode {result.returncode})"
    print(f"\n  finetuned/{naam}: {status} — {duur:.0f}s")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test alle finetuned pipelines voor data/testing/."
    )
    parser.add_argument(
        "--pipeline",
        choices=["tekst", "visie", "hybride"],
        help="Voer alleen deze pipeline uit (standaard: alle drie)",
    )
    args = parser.parse_args()

    te_draaien = [
        (naam, ROOT / script)
        for naam, script in PIPELINES
        if args.pipeline is None or naam == args.pipeline
    ]

    totaal_start = time.time()
    resultaten: list[tuple[str, bool]] = []

    for naam, script in te_draaien:
        if not script.exists():
            print(f"  FOUT: script niet gevonden: {script}")
            resultaten.append((naam, False))
            continue
        ok = run_pipeline(naam, script)
        resultaten.append((naam, ok))

    totaal_duur = time.time() - totaal_start

    print(f"\n\n{'#' * 65}")
    print(f"  EINDOVERZICHT — finetuned pipelines")
    print(f"{'#' * 65}")
    for naam, ok in resultaten:
        status = "OK" if ok else "FOUT"
        print(f"  [{status}] finetuned/{naam}")
    print(f"\n  Totale tijd: {totaal_duur:.0f}s")
    print()

    if not all(ok for _, ok in resultaten):
        sys.exit(1)


if __name__ == "__main__":
    main()
