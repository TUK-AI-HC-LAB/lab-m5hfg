"""Convert a WinCLIP pipe table in log.txt into a reviewable CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[list[str]] = []
    for raw in args.log.read_text(encoding="utf-8").splitlines():
        if "|" not in raw:
            continue
        cells = [cell.strip() for cell in raw[raw.find("|") :].strip().strip("|").split("|")]
        if cells and not all(set(cell) <= {":", "-"} for cell in cells):
            rows.append(cells)
    if len(rows) < 2:
        raise ValueError("No WinCLIP results table found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)


if __name__ == "__main__":
    main()
