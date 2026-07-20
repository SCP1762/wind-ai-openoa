#!/usr/bin/env python3
"""Run a smoke check for Python, OpenOA and the example data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    if not (3, 9) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError(
            f"当前 Python 为 {sys.version.split()[0]}；OpenOA 3.2 支持 Python 3.9–3.12。"
        )

    import openoa

    from wind_ai.openoa_loader import load_plant

    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenOA: {openoa.__version__}")

    plant = load_plant()
    scada = plant.scada.reset_index()

    print(f"SCADA rows: {len(scada):,}")
    print(f"Turbines: {scada['asset_id'].nunique()}")
    print(f"Time range: {scada['time'].min()} -> {scada['time'].max()}")
    print(f"SCADA columns: {list(plant.scada.columns)}")
    print("PlantData validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
