"""Replay the demo data against a running API, hour by hour, like a plant gateway would.

uvicorn app.main:app --port 8000          # or: docker compose up api
python scripts/sensor_client.py --machine 13 --start "2015-09-21 12:00" --hours 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdm.data import load_replay  # noqa: E402
from pdm.replay import payload  # noqa: E402


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--machine", type=int, default=13)
    parser.add_argument("--start", default="2015-09-21 12:00")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between requests")
    args = parser.parse_args()

    tables = load_replay()
    failures = tables.failures[tables.failures["machineID"] == args.machine]
    print(
        f"machine {args.machine}, recorded failures: "
        f"{', '.join(f'{r.datetime:%m-%d %H:%M} {r.failure}' for r in failures.itertuples())}"
    )
    for hour in pd.date_range(args.start, periods=args.hours, freq="h"):
        try:
            result = post(f"{args.api}/predict", payload(tables, args.machine, hour))
        except urllib.error.HTTPError as exc:
            print(f"{hour:%Y-%m-%d %H:%M}  HTTP {exc.code}: {exc.read().decode()[:200]}")
            continue
        except urllib.error.URLError as exc:
            raise SystemExit(f"API not reachable at {args.api}: {exc.reason}") from exc
        signals = "; ".join(result["signals"])
        print(f"{hour:%Y-%m-%d %H:%M}  {result['status']:<5}  risk {result['risk']:.3f}  {signals}")
        time.sleep(args.delay)

    with urllib.request.urlopen(f"{args.api}/metrics", timeout=10) as response:
        print(json.dumps(json.loads(response.read()), indent=2))


if __name__ == "__main__":
    main()
