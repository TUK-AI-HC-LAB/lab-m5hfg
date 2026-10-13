"""QueCoのCLIPおよびDINOv2を通常のcacheに準備します。"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import torch
from QueCo_lib.model_load import _download, _MODELS

if __name__ == "__main__":
    clip_path = Path(_download(_MODELS["ViT-L/14@336px"]))
    print("CLIP:", clip_path)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    print("DINOv2 ready:", type(model).__name__)
    dino_path = Path(torch.hub.get_dir()) / "checkpoints/dinov2_vitb14_reg4_pretrain.pth"
    manifest = {
        "weights": {
            p.name: {"bytes": p.stat().st_size, "sha256": hashlib.file_digest(
                p.open("rb"), "sha256"
            ).hexdigest()} for p in (clip_path, dino_path)
        },
        "packages": {
            name: version(name) for name in (
                "torch", "torchvision", "numpy", "scipy", "scikit-learn",
                "scikit-image", "torchmetrics", "pandas", "matplotlib",
                "ftfy", "regex", "setuptools"
            )
        },
    }
    output = Path(__file__).resolve().parent / "results/queco_setup"
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
