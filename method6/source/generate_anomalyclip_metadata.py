"""Create AnomalyCLIP-compatible metadata from existing MVTec AD and VisA folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MVTEC_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def write_json(root: Path, metadata: dict[str, dict[str, list[dict[str, object]]]]) -> None:
    (root / "meta.json").write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")


def build_mvtec(root: Path) -> None:
    metadata: dict[str, dict[str, list[dict[str, object]]]] = {"train": {}, "test": {}}
    for class_name in MVTEC_CLASSES:
        for split in ("train", "test"):
            entries: list[dict[str, object]] = []
            for defect_type in sorted((root / class_name / split).iterdir()):
                is_anomaly = defect_type.name != "good"
                images = sorted(defect_type.iterdir())
                masks = sorted((root / class_name / "ground_truth" / defect_type.name).iterdir()) if is_anomaly else []
                for index, image_path in enumerate(images):
                    entries.append(
                        {
                            "img_path": image_path.relative_to(root).as_posix(),
                            "mask_path": masks[index].relative_to(root).as_posix() if is_anomaly else "",
                            "cls_name": class_name,
                            "specie_name": defect_type.name,
                            "anomaly": int(is_anomaly),
                        }
                    )
            metadata[split][class_name] = entries
    write_json(root, metadata)


def build_visa(root: Path) -> None:
    split_csv = root / "split_csv" / "1cls.csv"
    dataframe = pd.read_csv(split_csv)
    metadata: dict[str, dict[str, list[dict[str, object]]]] = {"train": {}, "test": {}}
    for class_name in VISA_CLASSES:
        class_rows = dataframe[dataframe["object"] == class_name]
        for split in ("train", "test"):
            entries: list[dict[str, object]] = []
            for _, row in class_rows[class_rows["split"] == split].iterrows():
                is_anomaly = row["label"] == "anomaly"
                entries.append(
                    {
                        "img_path": row["image"],
                        "mask_path": row["mask"] if is_anomaly else "",
                        "cls_name": class_name,
                        "specie_name": "",
                        "anomaly": int(is_anomaly),
                    }
                )
            metadata[split][class_name] = entries
    write_json(root, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-root", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    args = parser.parse_args()
    build_mvtec(args.mvtec_root)
    build_visa(args.visa_root)


if __name__ == "__main__":
    main()
