"""Save a few interpretable AnomalyCLIP predictions after the official evaluation.

This is separate from upstream ``test.py``: it uses the same trained prompt,
model, preprocessing, anomaly-map computation, and Gaussian smoothing, but
selects three abnormal MVTec AD examples to avoid storing the whole dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

CODE_ROOT = Path("/home/test/anomalyclip")
sys.path.insert(0, str(CODE_ROOT))

import AnomalyCLIP_lib  # noqa: E402
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD  # noqa: E402
from AnomalyCLIP_lib.transform import image_transform  # noqa: E402
from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402


DATA_ROOT = Path("/home/test/data/mvtec")
CHECKPOINT = Path("/home/test/anomalyclip_checkpoints/visa_to_mvtec/epoch_15.pth")
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "anomalyclip_visa_to_mvtec" / "visualizations"
EXAMPLES = [("bottle", "broken_large"), ("pill", "color"), ("transistor", "bent_lead")]


def first_test_image(category: str, defect: str) -> Path:
    files = sorted((DATA_ROOT / category / "test" / defect).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No MVTec AD test image for {category}/{defect}")
    return files[0]


def make_preprocess() -> object:
    transform = image_transform(518, is_train=False, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    # The official code replaces its first two transforms with resize and crop.
    import torchvision.transforms as transforms

    transform.transforms[0] = transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC)
    transform.transforms[1] = transforms.CenterCrop(518)
    return transform


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    design = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=design)
    model.eval()
    learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    learner.load_state_dict(torch.load(CHECKPOINT, map_location="cpu")["prompt_learner"])
    learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized, compound = learner(cls_id=None)
    text = model.encode_text_learn(prompts, tokenized, compound).float()
    text = torch.stack(torch.chunk(text, chunks=2, dim=0), dim=1)
    text = text / text.norm(dim=-1, keepdim=True)
    preprocess = make_preprocess()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for category, defect in EXAMPLES:
        image_path = first_test_image(category, defect)
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            _, patch_features = model.encode_image(image, [24], DPAM_layer=20)
            maps = []
            for patch in patch_features:
                patch = patch / patch.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch, text[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], 518)
                maps.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
            anomaly_map = torch.stack(maps).sum(dim=0)[0].detach().cpu().numpy()
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        score = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

        original = cv2.resize(cv2.imread(str(image_path)), (518, 518))
        heatmap = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)
        cv2.imwrite(str(OUTPUT_ROOT / f"{category}_{defect}_input.png"), original)
        cv2.imwrite(str(OUTPUT_ROOT / f"{category}_{defect}_anomaly_map.png"), heatmap)
        cv2.imwrite(str(OUTPUT_ROOT / f"{category}_{defect}_overlay.png"), overlay)


if __name__ == "__main__":
    main()
