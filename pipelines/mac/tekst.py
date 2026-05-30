"""
Compatibiliteits-entrypoint voor de Mac baseline tekstpipeline.

De eigenlijke implementatie staat in pipelines/mac/baseline/tekst.py.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "baseline" / "tekst.py"
    runpy.run_path(str(script), run_name="__main__")
