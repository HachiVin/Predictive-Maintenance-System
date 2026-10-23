"""Loading the Azure PdM CSVs (full dataset for training, small replay set for the demo)."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPLAY_DIR = ROOT / "data" / "replay"
TABLES = ("telemetry", "errors", "maint", "failures", "machines")

# The Microsoft Azure AI Gallery copy, with a GitHub mirror of the same files as fallback.
SOURCES = (
    "https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_{name}.csv",
    "https://raw.githubusercontent.com/ashishpatel26/"
    "Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_{name}.csv",
)


@dataclass
class Tables:
    telemetry: pd.DataFrame
    errors: pd.DataFrame
    maint: pd.DataFrame
    failures: pd.DataFrame
    machines: pd.DataFrame


def _read(path: Path, dated: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if dated:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame


def download(directory: Path = RAW_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        target = directory / f"PdM_{name}.csv"
        if target.exists():
            continue
        for source in SOURCES:
            url = source.format(name=name)
            try:
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, target)
                break
            except OSError as exc:
                print(f"  failed: {exc}")
                target.unlink(missing_ok=True)
        else:
            raise SystemExit(f"Could not download PdM_{name}.csv; put it in {directory}.")


def load_raw(directory: Path = RAW_DIR) -> Tables:
    return Tables(
        *(_read(directory / f"PdM_{n}.csv", n != "machines") for n in TABLES)  # type: ignore[arg-type]
    )


def load_replay(directory: Path = REPLAY_DIR) -> Tables:
    """The demo subset written by scripts/train.py (test period only, a few machines)."""
    files = {
        n: directory / f"{n}.csv.gz" if n == "telemetry" else directory / f"{n}.csv" for n in TABLES
    }
    return Tables(*(_read(files[n], n != "machines") for n in TABLES))  # type: ignore[arg-type]
