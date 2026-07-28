"""PatchCore의 10%/1% coreset 결과를 원본 CSV에서 비교 표로 만든다."""

from __future__ import annotations

import csv
from pathlib import Path


# 실험한 MVTec AD 15개 종류를 고정된 순서로 적는다.
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

# 이 Python 파일이 있는 method1/source 폴더를 기준 경로로 사용한다.
ROOT = Path(__file__).resolve().parent
# 이미 계산해 둔 10% baseline 결과표를 읽는다.
SUMMARY_10 = ROOT / "results" / "PatchCore_MVTecAD_IM224_WR50_baseline.csv"
# 1% 실행의 원본 results.csv가 저장된 로컬 PatchCore 결과 폴더다.
PATCHCORE_RESULTS = ROOT / "patchcore-inspection" / "results"
# 이 스크립트가 새로 만드는 10% 대 1% 비교표다.
OUTPUT = ROOT / "results" / "PatchCore_MVTecAD_IM224_WR50_coreset_comparison.csv"


def read_baseline() -> dict[str, tuple[float, float]]:
    """10% 결과 CSV를 {종류: (image AUROC, pixel AUROC)} 형태로 바꾼다."""
    with SUMMARY_10.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        # mean 행은 종류별 비교에 쓰지 않으므로 제외한다.
        return {
            row["category"]: (float(row["image_auroc"]), float(row["full_pixel_auroc"]))
            for row in rows
            if row["category"] != "mean"
        }


def read_one_percent(category: str) -> tuple[float, float]:
    """한 종류의 1% 실행 결과 CSV를 찾아 두 AUROC 값을 읽는다."""
    # group, group_0처럼 실행마다 폴더 이름이 조금 달라질 수 있어 group*을 사용한다.
    # 수정 시간이 가장 최근인 결과를 선택한다.
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
        # 각 category results.csv에는 한 행의 결과만 들어 있다.
        row = next(csv.DictReader(file))
    return float(row["instance_auroc"]), float(row["full_pixel_auroc"])


def main() -> None:
    # 먼저 기존 10% baseline 전체를 메모리로 읽는다.
    baseline = read_baseline()
    rows: list[dict[str, float | str]] = []
    for category in CATEGORIES:
        # 같은 종류의 10%와 1% 점수를 나란히 가져온다.
        image_10, pixel_10 = baseline[category]
        image_1, pixel_1 = read_one_percent(category)

        # change가 음수면 1%가 10%보다 낮아졌다는 뜻이다.
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

    # 15개 종류의 평균도 마지막 행에 추가한다.
    mean_row: dict[str, float | str] = {"category": "mean"}
    for metric in rows[0]:
        if metric != "category":
            mean_row[metric] = sum(float(row[metric]) for row in rows) / len(CATEGORIES)
    rows.append(mean_row)

    # results 폴더가 없더라도 만들고, 계산한 표를 CSV로 저장한다.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
