"""Profile one official AA-CLIP text-adapter training step by phase.

This script does not load or overwrite a training checkpoint. It recreates the
official full-shot VisA text-step with the same model and hyperparameters, then
writes phase timings and parameter-gradient status as JSON.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

AA_CLIP_ROOT = Path("/home/test/Project/AA-CLIP")
OUTPUT = Path(
    "/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method4/source/results/"
    "AA_CLIP_text_step_profile_tf32_fp32.json"
)

sys.path.insert(0, str(AA_CLIP_ROOT))

from dataset import get_dataset  # noqa: E402
from forward_utils import (  # noqa: E402
    calculate_seg_loss,
    calculate_similarity_map,
    get_adapted_single_class_text_embedding,
)
from model.adapter import AdaptedCLIP  # noqa: E402
from model.clip import create_model  # noqa: E402
from utils import setup_seed  # noqa: E402


def elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return round(start.elapsed_time(end), 3)


def main() -> None:
    os.chdir(AA_CLIP_ROOT)
    setup_seed(111)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    clip_surgery = create_model(
        model_name="ViT-L-14-336",
        img_size=518,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=20)

    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=518,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=0.1,
        image_adapt_weight=0.1,
        text_adapt_until=3,
        image_adapt_until=6,
        relu=False,
    ).to(device)
    model.eval()
    optimizer = torch.optim.Adam(model.text_adapter.parameters(), lr=0.00001, betas=(0.5, 0.999))

    logger = logging.getLogger("aaclip-profile")
    text_dataset, _ = get_dataset("VisA", 518, "full_shot", -1, "train", logger)
    loader = torch.utils.data.DataLoader(
        text_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    loader_iter = iter(loader)
    loader_start = time.perf_counter()
    next(loader_iter)  # Worker-process startup and first prefetch.
    loader_first_batch_ms = round((time.perf_counter() - loader_start) * 1000, 3)
    loader_start = time.perf_counter()
    input_data = next(loader_iter)
    loader_steady_batch_ms = round((time.perf_counter() - loader_start) * 1000, 3)
    image = input_data["image"].to(device)
    mask = input_data["mask"].to(device)
    class_names = input_data["class_name"]

    timing = {}
    torch.cuda.reset_peak_memory_stats(device)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    epoch_text_feature_dict = {}
    for class_name in set(class_names):
        epoch_text_feature_dict[class_name] = get_adapted_single_class_text_embedding(
            model, "VisA", class_name, device
        )
    epoch_text_feature = torch.stack(
        [epoch_text_feature_dict[class_name] for class_name in class_names], dim=0
    )
    end.record()
    timing["text_embedding_ms"] = elapsed_ms(start, end)

    start.record()
    with torch.no_grad():
        _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
    end.record()
    timing["clip_surgery_encode_image_ms"] = elapsed_ms(start, end)

    start.record()
    with torch.no_grad():
        cls_token, _ = model.clipmodel.encode_image(image, [])
    end.record()
    timing["adapted_clip_encode_image_ms"] = elapsed_ms(start, end)

    start.record()
    with torch.no_grad():
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        patch_features = [clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features]
        patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
        patch_features = [t / t.norm(dim=-1, keepdim=True) for t in patch_features]
        patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]
    end.record()
    timing["image_feature_postprocess_ms"] = elapsed_ms(start, end)

    start.record()
    for feature in patch_features:
        patch_preds = calculate_similarity_map(feature, epoch_text_feature, 518)
        loss = calculate_seg_loss(patch_preds, mask)
        orthogonal_loss = (
            (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1]).sum(1).mean()
        ) ** 2
        loss += orthogonal_loss * 0.1
    end.record()
    timing["similarity_and_loss_ms"] = elapsed_ms(start, end)

    start.record()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    end.record()
    timing["backward_and_optimizer_ms"] = elapsed_ms(start, end)

    total_ms = round(sum(timing.values()), 3)
    clip_parameters = list(model.clipmodel.parameters())
    result = {
        "settings": {"tf32": True, "amp": False, "batch_size": 16, "img_size": 518},
        "timing_ms": timing,
        "dataloader_ms": {
            "first_batch_with_worker_startup": loader_first_batch_ms,
            "second_batch_steady_state": loader_steady_batch_ms,
        },
        "total_profiled_ms": total_ms,
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "clip_parameter_count": sum(parameter.numel() for parameter in clip_parameters),
        "clip_parameters_requires_grad": sum(parameter.requires_grad for parameter in clip_parameters),
        "adapter_parameter_count": sum(parameter.numel() for parameter in model.text_adapter.parameters()),
        "loss": float(loss.detach().cpu()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
