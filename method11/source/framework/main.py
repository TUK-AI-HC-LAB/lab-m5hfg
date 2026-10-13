"""Command-line entry point for shared anomaly-detection experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from loguru import logger

from component_registry import get_method_spec, list_datasets, list_methods
from main_dataset import dataset
from main_net import net
from main_run import run


ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return values


def load_method_config(method: str) -> dict:
    config_name = get_method_spec(method).config_name
    if not config_name:
        return {}
    path = CONFIGS_DIR / f"{config_name}.yaml"
    return load_yaml(path) if path.exists() else {}


def _normalise_experiment(values: dict) -> dict:
    values = dict(values)
    aliases = {
        "method": "mainmodel",
        "category": "subdatasets",
        "categories": "subdatasets",
        "seeds": "seed_list",
    }
    for source, target in aliases.items():
        if source in values:
            if target in values:
                raise ValueError(f"Use only one of '{source}' and '{target}'")
            values[target] = values.pop(source)

    categories = values.get("subdatasets")
    if isinstance(categories, str):
        values["subdatasets"] = [categories]
    seeds = values.get("seed_list")
    if isinstance(seeds, int):
        values["seed_list"] = [seeds]
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one method/dataset experiment. Edit experiment.yaml or pass "
            "--method, --dataset, --category, and --data-path."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "experiment.yaml")

    selection = parser.add_argument_group("experiment selection")
    selection.add_argument(
        "--method", "--mainmodel", dest="mainmodel", choices=list_methods()
    )
    selection.add_argument("--dataset", choices=list_datasets())
    selection.add_argument("--category", dest="subdatasets", nargs="+")
    selection.add_argument("--data-path", dest="data_path")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--results-path", dest="results_path")
    runtime.add_argument("--gpu", type=int, nargs="+")
    runtime.add_argument("--seed", dest="seed_list", type=int, nargs="+")
    runtime.add_argument("--batch-size", dest="batch_size", type=int)
    runtime.add_argument("--num-workers", dest="num_workers", type=int)
    runtime.add_argument(
        "--enable-wandb", dest="wan", action="store_true", default=None
    )
    runtime.add_argument(
        "--save-segmentation-images", action="store_true", default=None
    )

    overrides = parser.add_argument_group("common model overrides")
    overrides.add_argument("--resize", type=int)
    overrides.add_argument("--imagesize", type=int)
    overrides.add_argument("--masksize", type=int)
    overrides.add_argument("--meta-epochs", dest="meta_epochs", type=int)
    overrides.add_argument("--gan-epochs", dest="gan_epochs", type=int)
    overrides.add_argument("--total-iter", dest="total_iter", type=int)
    overrides.add_argument("--coreset-ratio", dest="coreset_ratio", type=float)
    return parser


DEFAULTS = {
    "mainmodel": "simple",
    "dataset": "mvtec",
    "subdatasets": ["screw"],
    "data_path": "/datasets/mvtec",
    "results_path": "results",
    "gpu": [0],
    "seed_list": [0],
    "log_group": "",
    "log_project": "dinomaly",
    "run_name": "experiment",
    "csv_save_name": "",
    "wan": False,
    "save_segmentation_images": False,
    "save_fault_images": 0,
    "num_workers": 1,
    "batch_size": 1,
    "train_val_split": 1.0,
    "resize": 224,
    "imagesize": 224,
    "masksize": 224,
    "crop": None,
    "rotate_degrees": 0,
    "translate": 0.0,
    "scale": 0.0,
    "brightness": 0.0,
    "contrast": 0.0,
    "saturation": 0.0,
    "gray": 0.0,
    "hflip": 0.0,
    "vflip": 0.0,
    "augment": False,
    "subtest": 0,
    "subtrain": 0,
    "fg_mask": 0,
    "realiad_jsons_path": "",
    "anomaly_source_path": "/datasets/dtd/images",
    "backbone_names": ["wideresnet50"],
    "layers_to_extract_from": ["layer2", "layer3"],
    "pretrain_embed_dimension": 1536,
    "target_embed_dimension": 1536,
    "patchsize": 3,
    "meta_epochs": 40,
    "aed_meta_epochs": 1,
    "gan_epochs": 4,
    "total_iter": 5000,
    "noise_std": 0.1,
    "dsc_layers": 2,
    "dsc_hidden": 1024,
    "dsc_margin": 0.5,
    "dsc_lr": 0.0002,
    "auto_noise": 0,
    "train_backbone": False,
    "cos_lr": False,
    "pre_proj": 2,
    "proj_layer_type": 0,
    "mix_noise": 1,
    "onnx": "no",
    "coreset_ratio": 0.1,
    "patchcore_coreset_num": 10000,
    "rd_decoder_norm": "batchnorm",
    "epoch_test_mode": 0,
    "last_score_blur_sigma": 1.0,
    "batch_size_tr": 8,
    "skip_ea": 1,
    "pixel_auroc": -1,
    "few_cluster_alg": "coreset",
    "plot_tsne": 0,
}


def resolve_args(argv=None):
    parser = build_parser()
    parsed = parser.parse_args(argv)
    cli_values = {
        key: value for key, value in vars(parsed).items() if value is not None
    }

    config_path = cli_values.pop("config")
    experiment = (
        _normalise_experiment(load_yaml(config_path))
        if config_path.exists()
        else {}
    )
    method = cli_values.get(
        "mainmodel", experiment.get("mainmodel", DEFAULTS["mainmodel"])
    )
    if method not in list_methods():
        raise ValueError(
            f"Unknown method '{method}'. Available: {', '.join(list_methods())}"
        )

    values = dict(DEFAULTS)
    values.update(load_method_config(method))
    values.update(experiment)
    values.update(cli_values)
    values["mainmodel"] = method

    if values.get("dataset") not in list_datasets():
        raise ValueError(
            f"Unknown dataset '{values.get('dataset')}'. "
            f"Available: {', '.join(list_datasets())}"
        )
    if not values.get("subdatasets"):
        raise ValueError("At least one category is required")
    if not values.get("data_path"):
        raise ValueError("data_path is required")
    return SimpleNamespace(**values)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = resolve_args(argv)
    logger.info(
        "method={} dataset={} categories={} data_path={}",
        args.mainmodel,
        args.dataset,
        args.subdatasets,
        args.data_path,
    )
    if args.wan:
        import wandb

        args.wandb_run = wandb.init(
            project=args.log_project, config=vars(args)
        )
    run([dataset(args), net(args)], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
