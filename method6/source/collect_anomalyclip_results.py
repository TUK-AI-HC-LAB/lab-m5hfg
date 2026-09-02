"""Convert the official AnomalyCLIP evaluation table in log.txt to a CSV file.

The upstream test script prints a Markdown-style table through its logger.  This
helper preserves those reported values in a repository-friendly CSV without
recalculating any metric.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_last_pipe_table(log_text: str) -> list[list[str]]:
    """Return the final contiguous ``| ... |`` table from the official log."""
    tables: list[list[str]] = []
    current: list[str] = []

    for raw_line in log_text.splitlines():
        # Logger prefixes each row with a timestamp, so retain only its table part.
        line = raw_line[raw_line.find("|") :] if "|" in raw_line else ""
        if line.startswith("|"):
            current.append(line)
        elif current:
            tables = current
            current = []
    if current:
        tables = current

    rows: list[list[str]] = []
    for line in tables:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        # The second line consists only of alignment markers such as :---.
        if cells and all(set(cell) <= {":", "-"} for cell in cells):
            continue
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        raise ValueError("No AnomalyCLIP result table found in the supplied log.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True, help="Official test.py log.txt")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path")
    args = parser.parse_args()

    rows = parse_last_pipe_table(args.log.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)


if __name__ == "__main__":
    main()
