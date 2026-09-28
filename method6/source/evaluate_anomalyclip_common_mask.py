"""AnomalyCLIP의 공식 mask 처리와 공통 mask 처리를 같은 예측 map으로 비교한다.

이미지 전처리와 AnomalyCLIP prompt checkpoint는 바꾸지 않는다. 비교하는 대상은
정답 ground-truth mask의 공간 변환뿐이다.

조건 A (공식): shortest-edge Resize -> CenterCrop(518) -> Tensor
조건 B (공통): Resize(518, 518) -> Tensor

따라서 Image AUROC와 anomaly score/map은 두 조건에서 같아야 한다. Pixel AUROC의
차이는 모델 성능 차이가 아니라, 평가에 쓰인 mask 좌표 변환 차이로 해석해야 한다.

예시:
  wsl -d Ubuntu -- /home/test/miniforge3/envs/patchcore-gpu/bin/python \\
    /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method6/source/evaluate_anomalyclip_common_mask.py \\
    --framework-root /mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase \\
    --config /mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase/experiment_anomalyclip.yaml \\
    --output /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method6/source/results/w40_anomalyclip_mask_transform_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(trainer, loader) -> dict[str, object]:
    """공통 Trainer의 실제 predict/metric 경로로 한 조건을 평가한다."""

    scores, maps, _, labels, masks = trainer.predict(loader)
    metrics = trainer._compute_metrics(scores, maps, labels, masks)
    return {
        "metrics": metrics,
        "scores": np.asarray(scores, dtype=np.float64),
        "maps": np.asarray(maps, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "masks": np.asarray(masks, dtype=np.float32),
    }


def main() -> None:
    cli = parse_args()
    framework_root = cli.framework_root.resolve()
    sys.path.insert(0, str(framework_root))

    from main import resolve_args  # noqa: PLC0415
    from main_dataset import dataset  # noqa: PLC0415
    from main_net import net  # noqa: PLC0415

    args = resolve_args(["--config", str(cli.config.resolve())])
    args.num_workers = 0
    args.batch_size = 1

    # A: registry의 기본 AnomalyCLIP 전용 Dataset variant.
    get_dataloaders = dataset(args)[1]
    official_loaders = get_dataloaders(0)[0]

    # B: AnomalyCLIP Dataset variant를 그대로 한 번 더 만든다. 따라서 image는
    # 두 조건 모두 shortest-edge Resize -> CenterCrop을 거친다. 여기서는 mask
    # transform만 공통 Dataset의 Resize((H, W)) 방식으로 바꾼다.
    common_loaders = dataset(args)[1](0)[0]
    common_test_dataset = common_loaders["testing"].dataset
    common_test_dataset.transform_mask = transforms.Compose(
        [
            transforms.Resize((args.masksize, args.masksize)),
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = net(args)[1](official_loaders["training"].dataset.imagesize, device)[0]
    if not trainer.checkpoint_loaded:
        raise RuntimeError("AnomalyCLIP checkpoint was not loaded.")

    official = evaluate(trainer, official_loaders["testing"])
    common = evaluate(trainer, common_loaders["testing"])

    scores_equal = bool(np.array_equal(official["scores"], common["scores"]))
    maps_equal = bool(np.array_equal(official["maps"], common["maps"]))
    mask_difference_pixels = int(np.count_nonzero(official["masks"] != common["masks"]))
    pixel_count = int(official["masks"].size)

    record = {
        "category": args.subdatasets,
        "checkpoint": str(args.anomalyclip_checkpoint_path),
        "device": str(device),
        "image_preprocess": "official AnomalyCLIP unchanged: shortest-edge Resize(518) -> CenterCrop(518)",
        "official_mask_preprocess": "shortest-edge Resize(518) -> CenterCrop(518) -> ToTensor",
        "common_mask_preprocess": "Resize((518, 518)) -> ToTensor",
        "official": official["metrics"],
        "common_mask": common["metrics"],
        "delta_common_minus_official": {
            key: common["metrics"][key] - official["metrics"][key]
            for key in official["metrics"]
        },
        "prediction_scores_bitwise_equal": scores_equal,
        "prediction_maps_bitwise_equal": maps_equal,
        "different_ground_truth_mask_pixels": mask_difference_pixels,
        "ground_truth_mask_pixel_count": pixel_count,
        "different_mask_pixel_ratio": mask_difference_pixels / pixel_count,
        "interpretation": (
            "Image preprocessing, prompt checkpoint, model prediction scores, and anomaly maps are unchanged. "
            "Any Pixel AUROC difference therefore comes only from the ground-truth mask transform."
        ),
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
