"""Collect one final MVTec AD result CSV from each local SimpleNet run."""

from __future__ import annotations

import csv
from pathlib import Path


CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)

SOURCE_DIR = Path(__file__).resolve().parent
RUN_ROOT = SOURCE_DIR / "simplenet" / "results" / "MVTecAD_Results" / "simplenet_mvtec"
OUTPUT = SOURCE_DIR / "results" / "SimpleNet_MVTecAD_WR50_results.csv"


def read_category_result(category: str) -> dict[str, float | str]:
    result_path = RUN_ROOT / f"{category}_official_default_workers2" / "results.csv"
    if not result_path.is_file():
        raise RuntimeError(f"Expected workers=2 result for {category}: {result_path}")

    with result_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if row["Row Names"] == f"mvtec_{category}")
    return {
        "category": category,
        "image_auroc": float(row["instance_auroc"]),
        "pixel_auroc": float(row["full_pixel_auroc"]),
        "pro_auroc": float(row["anomaly_pixel_auroc"]),
        "raw_result": str(result_path.relative_to(SOURCE_DIR)).replace("\\", "/"),
    }


def main() -> None:
    records = [read_category_result(category) for category in CATEGORIES]
    mean = {
        "category": "mean",
        "image_auroc": sum(float(row["image_auroc"]) for row in records) / len(records),
        "pixel_auroc": sum(float(row["pixel_auroc"]) for row in records) / len(records),
        "pro_auroc": sum(float(row["pro_auroc"]) for row in records) / len(records),
        "raw_result": "mean of 15 category rows",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mean.keys())
        writer.writeheader()
        writer.writerows(records + [mean])
    print(OUTPUT)


if __name__ == "__main__":
    main()
