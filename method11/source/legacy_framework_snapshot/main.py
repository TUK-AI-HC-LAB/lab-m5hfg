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
    # 역할: `load_yaml`에 해당하는 작업을 수행.
    # 매개변수: path.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `path.open`, `isinstance`, `ValueError`, `yaml.safe_load`입니다.
    # 제어 흐름: 조건 분기 1개, 자원/문맥 관리 1개, 예외 발생 경로 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return values


def load_method_config(method: str) -> dict:
    # 역할: `load_method_config`에 해당하는 작업을 수행.
    # 매개변수: method.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `get_method_spec`, `path.exists`, `load_yaml`입니다.
    # 제어 흐름: 조건 분기 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
    config_name = get_method_spec(method).config_name
    if not config_name:
        return {}
    path = CONFIGS_DIR / f"{config_name}.yaml"
    return load_yaml(path) if path.exists() else {}


def _normalise_experiment(values: dict) -> dict:
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: values.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `dict`, `aliases.items`, `values.get`, `isinstance`, `values.pop`입니다.
    # 제어 흐름: 반복문 1개, 조건 분기 4개, 예외 발생 경로 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
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
    # 역할: `build_parser`에 해당하는 작업을 수행.
    # 매개변수: 없음.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_argument_group`, `selection.add_argument`, `runtime.add_argument`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
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
    # 정상 train 이미지의 90%는 학습, 10%는 설정 선택용 validation으로 사용합니다.
    # 1.0을 지정하면 기존처럼 validation 분할을 사용하지 않습니다.
    "train_val_split": 0.9,
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
    # 역할: `resolve_args`에 해당하는 작업을 수행.
    # 매개변수: argv.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `build_parser`, `parser.parse_args`, `cli_values.pop`, `cli_values.get`, `dict`입니다.
    # 제어 흐름: 조건 분기 4개, 예외 발생 경로 4개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    parser = build_parser()
    # CLI에서 실제로 전달된 값만 따로 모아, 설정 파일의 기본값을 덮어쓸 준비를 합니다.
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

    # 우선순위는 공통 기본값 → 방법별 YAML → experiment.yaml → CLI입니다.
    # 따라서 사용자가 CLI에서 직접 지정한 값이 항상 가장 마지막에 적용됩니다.
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
    # 역할: `main`에 해당하는 작업을 수행.
    # 매개변수: argv.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `logging.basicConfig`, `resolve_args`, `logger.info`, `run`, `wandb.init`입니다.
    # 제어 흐름: 조건 분기 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
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
    # Dataset hook과 network hook을 만든 뒤 공통 runner에 전달합니다.
    # 실제 학습/평가 반복은 main_run.py의 run()이 담당합니다.
    run([dataset(args), net(args)], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
