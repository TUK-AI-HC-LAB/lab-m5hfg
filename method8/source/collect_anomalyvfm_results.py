"""Collect AnomalyVFM's official evaluator CSV into a reviewable result table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    source.insert(0, "protocol", "CVPR26 paper-disclosed settings")
    source.insert(1, "evaluation_scope", "MVTec AD + VisA available on this PC")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
