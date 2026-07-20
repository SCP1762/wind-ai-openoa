#!/usr/bin/env python3
"""Generate report figures from data and experiment outputs."""

from __future__ import annotations

import generate_report


def main() -> int:
    inputs = generate_report.load_inputs()
    figures = generate_report.generate_figures(inputs)

    print(f"Figures written: {generate_report.FIGURES_DIR}")
    for figure in figures.values():
        if figure.is_file():
            print(f"- {figure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
