"""Download the five Azure PdM CSVs (about 80 MB) into data/raw/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdm.data import RAW_DIR, download  # noqa: E402

if __name__ == "__main__":
    download()
    print(f"done: {RAW_DIR}")
