"""Create a reproducible 10,000-image AnomalyVFM training subset using symlinks.

The official post-paper archive contains more images than the paper's 10,000-image
training set. This utility picks 5,000 normal and 5,000 anomalous images with
seed 12 and links their masks, so the source archive is never copied.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Unexpected existing path: {target}")
    target.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for label in ("ok", "bad"):
        images = sorted((args.source / "train" / label).glob("*.png"))
        if len(images) < args.per_class:
            raise ValueError(f"{label}: expected at least {args.per_class}, got {len(images)}")
        selected = sorted(rng.sample(images, args.per_class))
        for image in selected:
            link(image, args.output / "train" / label / image.name)
            if label == "bad":
                mask = args.source / "ground_truth" / label / image.name
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                link(mask, args.output / "ground_truth" / label / image.name)

    manifest = args.output / "manifest.txt"
    manifest.write_text(
        f"seed={args.seed}\nnormal={args.per_class}\nanomalous={args.per_class}\ntotal={args.per_class * 2}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
