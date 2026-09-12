"""Collect AnomalyVFM's official evaluator CSV into a reviewable result table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--protocol",
        default="CVPR26 numerical settings + BF16/TF32 optimized runtime",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    source.insert(0, "protocol", args.protocol)
    source.insert(1, "evaluation_scope", "MVTec AD + VisA available on this PC")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output, index=False)

    if args.summary_output:
        metrics = [
            "AUROC-IMG",
            "F1-IMG",
            "AP-IMG",
            "AUROC-PIXEL",
            "F1-PIXEL",
            "AP-PIXEL",
            "AUPRO-0.3",
        ]
        summary = source.groupby("Dataset", as_index=False)[metrics].mean()
        overall = pd.DataFrame(
            [{"Dataset": "mvtec_ad+visa (category macro average)", **source[metrics].mean().to_dict()}]
        )
        summary = pd.concat([summary, overall], ignore_index=True)
        summary.insert(0, "protocol", args.protocol)
        summary.insert(1, "evaluation_scope", "MVTec AD (15) + VisA (12); each category has equal weight")
        summary.insert(3, "category_count", [15, 12, len(source)])
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)


if __name__ == "__main__":
    main()
