"""공통 Dataset/factory/lifecycle으로 QueCo를 소수 이미지에서 확인합니다.

전체 test 평가가 아닌 동작 검사입니다. 정상 3장과 결함 3장을 test bank에
넣어 실제 특징 추출, 패치 비교, 7개 metric 계산과 CSV 저장을 검증합니다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from main import ROOT, resolve_args
from main_dataset import dataset
from main_net import net
from main_run import train_and_collect_metrics


def main():
    """QueCo의 공통 실행 경로를 확인하고 검사 결과를 저장합니다."""
    args = resolve_args([
        "--config", str(ROOT / "experiment.yaml"), "--method", "queco",
        "--dataset", "mvtec", "--category", "bottle",
        "--data-path", "/home/test/data/mvtec", "--seed", "0",
        "--num-workers", "1", "--batch-size", "1",
    ])
    loaders = dataset(args)[1](0)[0]
    testing = loaders["testing"].dataset
    rows = testing.data_to_iterate
    good = [i for i, row in enumerate(rows) if row[1] == "good"][:3]
    defect = [i for i, row in enumerate(rows) if row[1] != "good"][:3]
    if len(good) != 3 or len(defect) != 3:
        raise ValueError("Smoke check requires 3 normal and 3 anomalous images.")
    loaders["testing"] = DataLoader(
        Subset(testing, good + defect), batch_size=1, shuffle=False, num_workers=0
    )
    args.support_m = 3
    device = torch.device("cuda:0")
    trainer = net(args)[1](testing.imagesize, device)[0]
    metrics = train_and_collect_metrics(trainer, loaders, "mvtec_bottle_smoke")
    required = (
        "image_auroc_mean", "image_ap_mean", "image_f1_mean",
        "pixel_auroc_mean", "pixel_ap_mean", "pixel_f1_mean", "pixel_aupro_mean",
    )
    if not all(np.isfinite(metrics[key]) and 0 <= metrics[key] <= 1 for key in required):
        raise ValueError(f"Invalid metric result: {metrics}")
    if trainer.__class__.get_evaluation_metrics.__qualname__ != "Trainer.get_evaluation_metrics":
        raise AssertionError("QueCo must inherit the common metric getter.")
    result = {
        "scope": "smoke_only_not_full_test", "num_images": 6,
        "image_paths": [rows[i][2] for i in good + defect],
        "metrics": metrics,
        "input_shape": list(next(iter(loaders["testing"]))["image"].shape),
        "map_shape": list(torch.cat(trainer._last_results["anomaly_maps"]).shape),
    }
    destination = ROOT / "results" / "queco_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "smoke.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame([{"dataset_name": "mvtec_bottle_smoke", **metrics}]).to_csv(
        destination / "metrics.csv", index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
