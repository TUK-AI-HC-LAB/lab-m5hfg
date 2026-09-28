"""VisA 직사각형 범주 전체에서 AnomalyCLIP mask 좌표 변환 영향을 평가한다.

각 범주에서 이미지와 model prediction은 공식 AnomalyCLIP 전처리
``Resize(short edge=518) -> CenterCrop(518)``로 고정한다. 동일 map에 다음 두
nearest-neighbor 정답 mask를 대입한다.

- official: ``Resize(short edge=518) -> CenterCrop(518) -> ToTensor``
- common: ``Resize((518, 518)) -> ToTensor``

따라서 두 조건의 Image AUROC는 같아야 하며, Pixel AUROC 차이는 mask 공간 좌표
변환의 영향만 나타낸다. 한 범주씩 예측해 GPU/RAM 사용량을 제한하고, 매 범주가 끝날
때마다 JSON을 저장해 중단돼도 완료 결과를 보존한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VisaCategoryDataset(Dataset):
    """한 VisA test category를 공통 AnomalyCLIP sample 형식으로 제공한다."""

    def __init__(self, root: Path, table: pd.DataFrame, image_size: int) -> None:
        self.root = root
        self.rows = table.reset_index(drop=True)
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        # 이번 비교에서는 보간을 nearest로 통일하고 공간 변환 순서만 다르게 둔다.
        self.official_mask_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        self.common_mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _mask(self, row: pd.Series, transform) -> torch.Tensor:
        if row.label == "normal":
            return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        with Image.open(self.root / row["mask"]) as mask:
            output = transform(mask.convert("L"))
        # VisA PNG에는 foreground가 1로 저장된 mask가 있다. ToTensor() 뒤에는
        # 1/255가 되므로, 공통 BaseDataset과 동일하게 nonzero를 binary defect=1로
        # 바꿔야 Pixel AUROC가 정상/결함 두 클래스를 인식한다.
        output[output != 0] = 1.0
        return output

    def common_masks(self) -> np.ndarray:
        return np.asarray(
            [self._mask(row, self.common_mask_transform).squeeze(0).numpy() for _, row in self.rows.iterrows()]
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index]
        with Image.open(self.root / row.image) as image:
            image_tensor = self.image_transform(image.convert("RGB"))
        return {
            "image": image_tensor,
            "mask": self._mask(row, self.official_mask_transform),
            "is_anomaly": torch.tensor(int(row.label == "anomaly"), dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="*", help="기본값은 VisA test 전체 범주")
    return parser.parse_args()


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cli = parse_args()
    framework_root = cli.framework_root.resolve()
    sys.path.insert(0, str(framework_root))
    from main import resolve_args  # noqa: PLC0415
    from main_net import net  # noqa: PLC0415

    args = resolve_args(["--config", str(cli.config.resolve())])
    args.anomalyclip_checkpoint_path = str(cli.checkpoint.resolve())
    args.num_workers = 0
    args.batch_size = 1
    visa_root = cli.visa_root.resolve()
    table = pd.read_csv(visa_root / "split_csv" / "1cls.csv")
    test_table = table[table["split"] == "test"].copy()
    categories = cli.categories or sorted(test_table["object"].unique().tolist())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = net(args)[1]((3, args.imagesize, args.imagesize), device)[0]
    record = {
        "dataset": "VisA 1cls test",
        "categories_requested": categories,
        "checkpoint": str(cli.checkpoint.resolve()),
        "device": str(device),
        "image_preprocess": "fixed: shortest-edge Resize(518) -> CenterCrop(518)",
        "official_mask_preprocess": "Resize(short edge=518, nearest) -> CenterCrop(518) -> ToTensor",
        "common_mask_preprocess": "Resize((518, 518), nearest) -> ToTensor",
        "results": {},
    }

    for category in categories:
        category_table = test_table[test_table["object"] == category]
        if category_table.empty:
            raise ValueError(f"No VisA test rows found for category '{category}'")
        dataset = VisaCategoryDataset(visa_root, category_table, args.imagesize)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        scores, maps, _, labels, official_masks = trainer.predict(loader)
        common_masks = dataset.common_masks()
        official_metrics = trainer._compute_metrics(scores, maps, labels, official_masks)
        common_metrics = trainer._compute_metrics(scores, maps, labels, common_masks)
        official_masks = np.asarray(official_masks)
        record["results"][category] = {
            "sample_count": len(dataset),
            "original_image_size": list(Image.open(visa_root / category_table.iloc[0].image).size),
            "official": official_metrics,
            "common_mask": common_metrics,
            "delta_common_minus_official": {
                key: common_metrics[key] - official_metrics[key] for key in official_metrics
            },
            "different_ground_truth_mask_pixels": int(np.count_nonzero(official_masks != common_masks)),
            "ground_truth_mask_pixel_count": int(official_masks.size),
            "different_mask_pixel_ratio": float(np.count_nonzero(official_masks != common_masks) / official_masks.size),
        }
        save_record(cli.output, record)
        print(f"completed {category}: {record['results'][category]}", flush=True)


if __name__ == "__main__":
    main()
