"""Save the RD4AD anomaly map for the common MVTec AD transistor example.

This script is a diagnostic artifact for W39. It only runs inference with the
paper-protocol checkpoint; it neither trains nor changes the reproduction score.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

CODE_ROOT = Path("/home/test/RD4AD")
DATA_ROOT = Path("/home/test/data/mvtec")
CHECKPOINT = Path("/home/test/rd4ad_paper_protocol_checkpoints/wres50_transistor.pth")
OUTPUT = Path(__file__).resolve().parent / "results" / "w39_visualizations"

sys.path.insert(0, str(CODE_ROOT))
from dataset import get_data_transforms  # noqa: E402
from de_resnet import de_wide_resnet50_2  # noqa: E402
from resnet import wide_resnet50_2  # noqa: E402


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform, _ = get_data_transforms(256, 256)
    image_path = DATA_ROOT / "transistor" / "test" / "bent_lead" / "000.png"
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    encoder, bottleneck = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    encoder, bottleneck, decoder = encoder.to(device).eval(), bottleneck.to(device).eval(), decoder.to(device).eval()
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    for key in list(checkpoint["bn"]):
        if "memory" in key:
            checkpoint["bn"].pop(key)
    bottleneck.load_state_dict(checkpoint["bn"])
    decoder.load_state_dict(checkpoint["decoder"])

    with torch.no_grad():
        teacher = encoder(image)
        student = decoder(bottleneck(teacher))
        score = np.zeros((256, 256), dtype=np.float32)
        for teacher_feature, student_feature in zip(teacher, student):
            distance = 1 - torch.nn.functional.cosine_similarity(teacher_feature, student_feature)
            distance = torch.nn.functional.interpolate(distance.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=True)
            score += distance[0, 0].detach().cpu().numpy()
    score = gaussian_filter(score, sigma=4)
    score = (score - score.min()) / (score.max() - score.min() + 1e-12)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT / "transistor_bent_lead_000_rd4ad_anomaly_map.png"), cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET))


if __name__ == "__main__":
    main()
