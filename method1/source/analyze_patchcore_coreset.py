"""PatchCore의 10%/1% coreset 결과를 원본 CSV에서 비교 표로 만든다."""

from __future__ import annotations

import csv
from pathlib import Path


CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

ROOT = Path(__file__).resolve().parent
SUMMARY_10 = ROOT / "results" / "PatchCore_MVTecAD_IM224_WR50_baseline.csv"
PATCHCORE_RESULTS = ROOT / "patchcore-inspection" / "results"
OUTPUT = ROOT / "results" / "PatchCore_MVTecAD_IM224_WR50_coreset_comparison.csv"


def read_baseline() -> dict[str, tuple[float, float]]:
    with SUMMARY_10.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return {
            row["category"]: (float(row["image_auroc"]), float(row["full_pixel_auroc"]))
            for row in rows
            if row["category"] != "mean"
        }


def read_one_percent(category: str) -> tuple[float, float]:
    candidates = sorted(
        PATCHCORE_RESULTS.glob(
            f"IM224_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S0_{category}/project/group*/results.csv"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"1% 결과를 찾을 수 없음: {category}")
    with candidates[0].open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    return float(row["instance_auroc"]), float(row["full_pixel_auroc"])


def main() -> None:
    baseline = read_baseline()
    rows: list[dict[str, float | str]] = []
    for category in CATEGORIES:
        image_10, pixel_10 = baseline[category]
        image_1, pixel_1 = read_one_percent(category)
        rows.append(
            {
                "category": category,
                "image_auroc_10pct": image_10,
                "image_auroc_1pct": image_1,
                "image_auroc_change": image_1 - image_10,
                "pixel_auroc_10pct": pixel_10,
                "pixel_auroc_1pct": pixel_1,
                "pixel_auroc_change": pixel_1 - pixel_10,
            }
        )

    mean_row: dict[str, float | str] = {"category": "mean"}
    for metric in rows[0]:
        if metric != "category":
            mean_row[metric] = sum(float(row[metric]) for row in rows) / len(CATEGORIES)
    rows.append(mean_row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
