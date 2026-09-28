"""직사각형 VisA candle에서 AnomalyCLIP mask 공간 변환의 평가 영향을 확인한다.

모델 입력 이미지는 항상 공식 AnomalyCLIP 전처리
``Resize(short edge=518) -> CenterCrop(518)``를 사용한다. 같은 예측 anomaly map을
두 정답 mask와 각각 평가한다.

- 공식 mask: ``Resize(short edge=518) -> CenterCrop(518) -> ToTensor``
- 공통 mask: ``Resize((518, 518)) -> ToTensor``

따라서 Image AUROC, image score, anomaly map은 평가 조건과 무관하게 동일하며,
Pixel AUROC만 mask 좌표 불일치의 영향을 받는다.
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


class VisaCandleTestDataset(Dataset):
    """VisA split CSV를 AnomalyCLIP adapter의 공통 sample 형식으로 제공한다."""

    def __init__(self, root: Path, split_csv: Path, image_size: int) -> None:
        table = pd.read_csv(split_csv)
        self.rows = table[(table["object"] == "candle") & (table["split"] == "test")]
        self.rows = self.rows.reset_index(drop=True)
        self.root = root
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.official_mask_transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()]
        )
        self.common_mask_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _mask(self, row, transform) -> torch.Tensor:
        if row.label == "normal":
            return torch.zeros((1, 518, 518), dtype=torch.float32)
        with Image.open(self.root / row["mask"]) as mask:
            output = transform(mask.convert("L"))
        # 공통 BaseDataset.__getitem__과 같이 fractional boundary를 positive로 이진화한다.
        output[output != 0] = 1.0
        return output

    def common_masks(self) -> np.ndarray:
        return np.asarray([self._mask(row, self.common_mask_transform).squeeze(0).numpy() for _, row in self.rows.iterrows()])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index]
        with Image.open(self.root / row.image) as image:
            transformed_image = self.image_transform(image.convert("RGB"))
        return {
            "image": transformed_image,
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
    return parser.parse_args()


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
    dataset = VisaCandleTestDataset(visa_root, visa_root / "split_csv" / "1cls.csv", args.imagesize)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = net(args)[1]((3, args.imagesize, args.imagesize), device)[0]
    scores, maps, _, labels, official_masks = trainer.predict(loader)
    official = trainer._compute_metrics(scores, maps, labels, official_masks)
    common_masks = dataset.common_masks()
    common = trainer._compute_metrics(scores, maps, labels, common_masks)

    official_masks = np.asarray(official_masks)
    changed_pixels = int(np.count_nonzero(official_masks != common_masks))
    record = {
        "dataset": "VisA candle test",
        "sample_count": len(dataset),
        "original_image_size": [1284, 1168],
        "checkpoint": str(cli.checkpoint.resolve()),
        "image_preprocess": "unchanged: shortest-edge Resize(518) -> CenterCrop(518)",
        "official_mask_preprocess": "shortest-edge Resize(518) -> CenterCrop(518) -> ToTensor -> nonzero binarization",
        "common_mask_preprocess": "Resize((518, 518)) -> ToTensor -> nonzero binarization",
        "official": official,
        "common_mask": common,
        "delta_common_minus_official": {key: common[key] - official[key] for key in official},
        "different_ground_truth_mask_pixels": changed_pixels,
        "ground_truth_mask_pixel_count": int(official_masks.size),
        "different_mask_pixel_ratio": changed_pixels / official_masks.size,
        "interpretation": "The model was executed once. Only the evaluation mask was changed, so image scores and anomaly maps are fixed.",
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
