import subprocess
import sys
from pathlib import Path


def main() -> None:
    hf_path = Path.home() / "Documents/school/Bachelorproef/finetuning/vision/merged_model"
    mlx_path = Path.home() / "Documents/school/Bachelorproef/finetuning/vision/mlx_model"

    # Venv Python: .venv zit twee niveaus boven dit script (Bachelorproef/.venv)
    project_root = Path(__file__).resolve().parent.parent.parent
    venv_python = project_root / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        print(f"FOUT: venv Python niet gevonden op {venv_python}")
        sys.exit(1)

    if not hf_path.exists():
        print(f"FOUT: bronmap niet gevonden: {hf_path}")
        sys.exit(1)

    # Stap 1 — installeer mlx-vlm via uv
    print("\n>>> Stap 1 — mlx-vlm installeren")
    result = subprocess.run(["uv", "pip", "install", "mlx-vlm"], text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Stap 2 — converteer naar MLX formaat (4-bit gekwantiseerd)
    print("\n>>> Stap 2 — converteren naar MLX formaat (4-bit, ~5 minuten)\n")
    cmd = [
        str(venv_python), "-m", "mlx_vlm.convert",
        "--hf-path", str(hf_path),
        "--mlx-path", str(mlx_path),
        "--quantize",
        "--q-bits", "4",
    ]
    print(f"    {' '.join(cmd)}\n")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"FOUT: conversie mislukt (returncode {result.returncode})")
        sys.exit(result.returncode)

    print(f"\nKlaar! MLX-model staat in: {mlx_path}")


if __name__ == "__main__":
    main()
