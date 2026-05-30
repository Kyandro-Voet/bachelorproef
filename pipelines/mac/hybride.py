"""
Compatibiliteits-entrypoint voor de Mac baseline hybride pipeline.

De eigenlijke implementatie staat in pipelines/mac/baseline/hybride.py.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "baseline" / "hybride.py"
    runpy.run_path(str(script), run_name="__main__")
