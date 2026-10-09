"""공통 anomaly-detection framework의 한 이미지 입출력을 저장한다.

이 스크립트는 W40 시연용이다. 공통 framework의 실제 Dataset/DataLoader와
AnomalyCLIP adapter를 그대로 사용해, 지정한 MVTec test 이미지를 다음 순서로
저장한다.

원본 파일 -> ImageNet 정규화 batch -> CLIP 정규화 model input
-> image/text similarity -> anomaly map -> 최종 image anomaly score

사용 예시는 ``related_work/markdown/W40_공통_프레임워크_한_이미지_시연방법.txt``에 있다.
데이터셋, checkpoint, 외부 framework 자체는 이 저장소에 복사하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Subset

# WSL에서도 시연 panel의 한글을 그대로 표시한다. 없으면 matplotlib 기본 글꼴을 쓴다.
try:
    from matplotlib import font_manager

    _MALGUN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
    if _MALGUN_FONT.is_file():
        font_manager.fontManager.addfont(str(_MALGUN_FONT))
        plt.rcParams["font.family"] = "Malgun Gothic"
except Exception:  # 폰트 문제로 실제 모델 실행이 중단되면 안 된다.
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework-root",
        type=Path,
        required=True,
        help="main.py가 있는 공통 framework 경로",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="AnomalyCLIP experiment YAML 경로",
    )
    parser.add_argument(
        "--image-suffix",
        default="bottle/test/broken_large/000.png",
        help="MVTec root 뒤의 대상 test 이미지 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="PNG와 JSON을 저장할 폴더",
    )
    return parser.parse_args()


def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    """tensor의 shape와 숫자 범위를 JSON으로 바꾼다."""

    detached = tensor.detach().float().cpu()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "min": float(detached.min()),
        "max": float(detached.max()),
        "mean": float(detached.mean()),
        "std": float(detached.std()),
    }


def inverse_normalize(image: torch.Tensor, mean, std) -> np.ndarray:
    """정규화 tensor를 사람이 볼 수 있는 RGB 이미지로 되돌린다."""

    mean = image.new_tensor(mean)[None, :, None, None]
    std = image.new_tensor(std)[None, :, None, None]
    rgb = (image * std + mean).clamp(0, 1)
    return rgb[0].permute(1, 2, 0).detach().cpu().numpy()


def main() -> None:
    cli = parse_args()
    framework_root = cli.framework_root.resolve()
    config = cli.config.resolve()
    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 공통 framework의 실제 설정 병합·Dataset factory·model factory를 사용한다.
    sys.path.insert(0, str(framework_root))
    from main import resolve_args  # noqa: PLC0415
    from main_dataset import dataset  # noqa: PLC0415
    from main_net import net  # noqa: PLC0415

    config_args = resolve_args(["--config", str(config)])
    config_args.num_workers = 0  # 시연에서는 한 process로 한 이미지의 순서를 명확히 한다.
    config_args.batch_size = 1

    # Dataset factory가 test split 전체를 만든 뒤, 지정한 파일 하나만 Subset으로 고른다.
    get_dataloaders = dataset(config_args)[1]
    test_dataset = get_dataloaders(0)[0]["testing"].dataset
    wanted = cli.image_suffix.replace("\\", "/")
    matching_indices = [
        index
        for index, item in enumerate(test_dataset.data_to_iterate)
        if str(item[2]).replace("\\", "/").endswith(wanted)
    ]
    if len(matching_indices) != 1:
        raise RuntimeError(
            f"Expected exactly one image ending with '{wanted}', found {len(matching_indices)}"
        )
    one_loader = DataLoader(Subset(test_dataset, matching_indices), batch_size=1)
    batch = next(iter(one_loader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_models = net(config_args)[1]
    trainer = get_models(test_dataset.imagesize, device)[0]

    # trainer.predict()은 production runner가 사용하는 바로 그 최종 score/map 경로다.
    scores, maps, _, labels, masks = trainer.predict(one_loader)

    # 아래는 predict() 내부의 중간 상태를 읽기 위해 같은 연산을 한 번 더 펼친 것이다.
    # score/map은 위의 production 경로 결과를 보고하며, 이 중간 값은 설명용 evidence다.
    raw_image = batch["image"].to(device, dtype=torch.float32)
    clip_image = trainer._to_clip_normalized(raw_image)
    gt_mask = batch["mask"].to(device, dtype=torch.float32)
    with torch.no_grad():
        text_features = trainer._text_features()
        image_features, patch_features = trainer.model.encode_image(
            clip_image,
            list(config_args.anomalyclip_features_list),
            DPAM_layer=config_args.anomalyclip_dapm_layer,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_logits = image_features @ text_features.T
        manual_score = (image_logits / 0.07).softmax(dim=-1)[:, 1]

        layer_maps = []
        for layer_index, patch_feature in enumerate(patch_features):
            if layer_index < config_args.anomalyclip_feature_map_start:
                continue
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity, _ = trainer._anomalyclip_lib.compute_similarity(
                patch_feature, text_features
            )
            similarity_map = trainer._anomalyclip_lib.get_similarity_map(
                similarity[:, 1:, :], config_args.imagesize
            )
            layer_maps.append(
                (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
            )
        raw_map = torch.stack(layer_maps, dim=0).sum(dim=0)
        resized_map = F.interpolate(
            raw_map.unsqueeze(1),
            size=gt_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        smoothed_map = torch.stack(
            [
                torch.from_numpy(
                    gaussian_filter(item.detach().cpu().numpy(), sigma=config_args.anomalyclip_sigma)
                )
                for item in resized_map
            ]
        )

    # ImageNet input과 CLIP model input은 정규화 상수만 다르다. 역정규화하면 같은 RGB다.
    rgb = inverse_normalize(raw_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    clip_rgb = inverse_normalize(
        clip_image, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    )
    map_array = np.asarray(maps)[0]
    mask_array = np.asarray(masks)[0]

    # 실제 시연 때 한 화면에 보여 줄 panel이다.
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. 원본 RGB 이미지")
    axes[0, 1].imshow(clip_rgb)
    axes[0, 1].set_title("2. 전처리 후 모델 입력\n(CLIP 역정규화 표시)")
    axes[0, 2].imshow(mask_array, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"3. 정답 mask (label={int(labels[0])})")
    heat = axes[1, 0].imshow(map_array, cmap="turbo")
    axes[1, 0].set_title("4. 최종 anomaly map")
    figure.colorbar(heat, ax=axes[1, 0], fraction=0.046)
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(map_array, cmap="turbo", alpha=0.55)
    axes[1, 1].set_title("5. 원본 위 anomaly map")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.03,
        0.90,
        "6. Image-level output\n\n"
        f"normal similarity: {float(image_logits[0, 0]):.4f}\n"
        f"abnormal similarity: {float(image_logits[0, 1]):.4f}\n"
        f"final anomaly score: {float(scores[0]):.6f}\n\n"
        "score는 확률이 아니라\n같은 설정 안에서 비교하는\n상대적 이상 점수임.",
        va="top",
        fontsize=14,
    )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(output_dir / "single_image_io_panel.png", dpi=180)
    plt.close(figure)

    # 개별 artifact도 남겨 발표 자료에서 필요한 것만 따로 쓸 수 있게 한다.
    plt.imsave(output_dir / "01_original_rgb.png", rgb)
    plt.imsave(output_dir / "02_model_input_after_inverse_normalization.png", clip_rgb)
    plt.imsave(output_dir / "03_ground_truth_mask.png", mask_array, cmap="gray", vmin=0, vmax=1)
    plt.imsave(output_dir / "04_final_anomaly_map.png", map_array, cmap="turbo")
    map_range = float(np.ptp(map_array)) or 1.0
    heat_rgb = plt.get_cmap("turbo")(
        (map_array - map_array.min()) / map_range
    )[..., :3]
    plt.imsave(
        output_dir / "05_anomaly_map_overlay.png",
        np.clip(0.45 * rgb + 0.55 * heat_rgb, 0, 1),
    )

    record = {
        "image_path": str(batch["image_path"][0]),
        "image_name": str(batch["image_name"][0]),
        "label": int(labels[0]),
        "framework": str(framework_root),
        "config": str(config),
        "device": str(device),
        "raw_dataset_image_imagenet_normalized": tensor_summary(raw_image),
        "clip_model_input": tensor_summary(clip_image),
        "ground_truth_mask": tensor_summary(gt_mask),
        "text_features": tensor_summary(text_features),
        "global_image_features": tensor_summary(image_features),
        "patch_feature_shapes": [tensor_summary(item) for item in patch_features],
        "raw_patch_score_map": tensor_summary(raw_map),
        "resized_map_before_smoothing": tensor_summary(resized_map),
        "final_anomaly_map": tensor_summary(smoothed_map),
        "normal_text_similarity_logit": float(image_logits[0, 0]),
        "abnormal_text_similarity_logit": float(image_logits[0, 1]),
        "manual_score": float(manual_score[0]),
        "production_trainer_predict_score": float(scores[0]),
        "production_vs_manual_absolute_difference": float(abs(float(scores[0]) - float(manual_score[0]))),
        "final_map_min": float(map_array.min()),
        "final_map_max": float(map_array.max()),
    }
    (output_dir / "single_image_io_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
