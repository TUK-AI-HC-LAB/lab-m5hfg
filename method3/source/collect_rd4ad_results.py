"""Collect one final MVTec AD row from each RD4AD category run."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path


CATEGORIES = (
    "carpet", "bottle", "hazelnut", "leather", "cable", "capsule", "grid", "pill",
    "transistor", "metal_nut", "screw", "toothbrush", "zipper", "tile", "wood",
)
SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = SOURCE_DIR / "results" / "rd4ad_runs"
DEFAULT_OUTPUT = SOURCE_DIR / "results" / "RD4AD_MVTecAD_WR50_results.csv"


def read_category(category: str, run_root: Path) -> dict[str, float | str]:
    path = run_root / f"{category}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or rows[0]["category"] != category:
        raise RuntimeError(f"Incomplete RD4AD result: {path}")
    row = rows[0]
    return {
        "category": category,
        "image_auroc": float(row["image_auroc"]),
        "pixel_auroc": float(row["pixel_auroc"]),
        "pro_auroc": float(row["pro_auroc"]),
        "raw_result": path.relative_to(SOURCE_DIR).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = [read_category(category, args.run_root) for category in CATEGORIES]
    records.append({
        "category": "mean",
        "image_auroc": sum(float(row["image_auroc"]) for row in records) / len(records),
        "pixel_auroc": sum(float(row["pixel_auroc"]) for row in records) / len(records),
        "pro_auroc": sum(float(row["pro_auroc"]) for row in records) / len(records),
        "raw_result": "mean of 15 category rows",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(args.output)


if __name__ == "__main__":
    main()
