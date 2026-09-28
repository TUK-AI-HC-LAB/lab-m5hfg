"""Dataset selection and DataLoader construction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from component_registry import (
    DATASET_REGISTRY,
    get_dataset_spec,
    resolve_dataset_name,
    validate_dataset_instance,
)

# Backward-compatible name for scripts that inspect the old registry.
_DATASETS = DATASET_REGISTRY

@dataclass(frozen=True)
class DatasetPlan:
    registry_name: str
    train_path: str
    validation_path: str
    test_path: str
    train_categories: tuple[str, ...]
    validation_categories: tuple[str, ...]
    test_categories: tuple[str, ...]


def resolve_dataset_plan(args, categories):
    # 역할: `resolve_dataset_plan`에 해당하는 작업을 수행.
    # 매개변수: args, categories.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `DatasetPlan`, `ValueError`입니다.
    # 제어 흐름: 조건 분기 1개, 예외 발생 경로 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    """Resolve one explicit dataset selection without mutating ``args``."""

    family = args.dataset
    path = args.data_path
    if not path:
        raise ValueError(f"Dataset '{family}' requires data_path")
    return DatasetPlan(
        registry_name=family,
        train_path=path,
        validation_path=path,
        test_path=path,
        train_categories=categories,
        validation_categories=categories,
        test_categories=categories,
    )


def print_stat(dataset, name):
    # 역할: `print_stat`에 해당하는 작업을 수행.
    # 매개변수: dataset, name.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `torch.tensor.unique`, `zip`, `int`, `logger.warning`, `logger.info`입니다.
    # 제어 흐름: 반복문 1개, 조건 분기 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    targets = [int(data[1] != "good") for data in dataset.data_to_iterate]
    if not targets:
        logger.warning(f"{name} is empty")
        return
    classes, counts = torch.tensor(targets).unique(return_counts=True)
    for class_id, count in zip(classes, counts):
        logger.info(f"{name} class {class_id}: {count}")


def _common_dataset_kwargs(args):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: args.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    return {
        "resize": args.resize,
        "imagesize": args.imagesize,
        "args": args,
    }


def _build_path_datasets(args, spec, plan, seed):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: args, spec, plan, seed.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `spec.load_class`, `spec.load_split`, `_common_dataset_kwargs`, `dataset_class`, `getattr`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    dataset_class = spec.load_class()
    split = spec.load_split()
    common = _common_dataset_kwargs(args)
    anomaly_source_path = (
        getattr(args, "anomaly_source_path", "") or "/datasets/dtd/images"
    )
    train_dataset = dataset_class(
        plan.train_path,
        classname=list(plan.train_categories),
        split=split.TRAIN,
        train_val_split=args.train_val_split,
        seed=seed,
        rotate_degrees=args.rotate_degrees,
        translate=args.translate,
        brightness_factor=args.brightness,
        contrast_factor=args.contrast,
        saturation_factor=args.saturation,
        gray_p=args.gray,
        h_flip_p=args.hflip,
        v_flip_p=args.vflip,
        scale=args.scale,
        augment=args.augment,
        subtest=args.subtest,
        anomaly_source_path=anomaly_source_path,
        **common,
    )
    validation_dataset = dataset_class(
        plan.validation_path,
        classname=list(plan.validation_categories),
        split=split.VAL,
        # validation은 train/good에서 분리한 정상 이미지 전체를 사용합니다.
        # test 폴더의 1%를 임시로 가져오지 않습니다.
        train_val_split=args.train_val_split,
        seed=seed,
        subtest=False,
        anomaly_source_path=anomaly_source_path,
        **common,
    )
    test_dataset = dataset_class(
        plan.test_path,
        classname=list(plan.test_categories),
        split=split.TEST,
        seed=seed,
        anomaly_source_path=anomaly_source_path,
        **common,
    )
    return train_dataset, validation_dataset, test_dataset


def _realiad_meta_files(args, categories, data_path):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: args, categories, data_path.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `getattr`, `os.path.join`, `sorted`, `os.path.dirname`, `str`입니다.
    # 제어 흐름: 조건 분기 2개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
    jsons_dir = getattr(args, "realiad_jsons_path", "")
    if not jsons_dir:
        jsons_dir = os.path.join(os.path.dirname(data_path), "realiad_jsons")
    if categories == ("realiad",):
        return sorted(str(path) for path in Path(jsons_dir).glob("*.json"))
    return [os.path.join(jsons_dir, f"{category}.json") for category in categories]


def _build_metadata_datasets(args, spec, plan, seed):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: args, spec, plan, seed.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `spec.load_class`, `spec.load_split`, `_realiad_meta_files`, `_common_dataset_kwargs`, `dataset_class`입니다.
    # 제어 흐름: 조건 분기 1개, 예외 발생 경로 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    del seed
    dataset_class = spec.load_class()
    split = spec.load_split()
    meta_files = _realiad_meta_files(
        args, plan.train_categories, plan.train_path
    )
    if not meta_files:
        raise ValueError("No Real-IAD metadata JSON files were found")
    common = _common_dataset_kwargs(args)
    train_dataset = dataset_class(
        meta_file=meta_files,
        image_dir=plan.train_path,
        split=split.TRAIN,
        train_val_split=args.train_val_split,
        rotate_degrees=args.rotate_degrees,
        translate=args.translate,
        brightness_factor=args.brightness,
        contrast_factor=args.contrast,
        saturation_factor=args.saturation,
        gray_p=args.gray,
        h_flip_p=args.hflip,
        v_flip_p=args.vflip,
        scale=args.scale,
        subtest=args.subtest,
        **common,
    )
    validation_dataset = dataset_class(
        meta_file=meta_files,
        image_dir=plan.validation_path,
        split=split.VAL,
        subtest=True,
        **common,
    )
    test_dataset = dataset_class(
        meta_file=meta_files,
        image_dir=plan.test_path,
        split=split.TEST,
        **common,
    )
    return train_dataset, validation_dataset, test_dataset


_DATASET_BUILDERS = {
    "path": _build_path_datasets,
    "metadata": _build_metadata_datasets,
}


def _make_loader(dataset, batch_size, num_workers, shuffle, drop_last=False):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: dataset, batch_size, num_workers, shuffle, drop_last.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `torch.utils.data.DataLoader`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        drop_last=drop_last,
    )


def dataset(args):
    # 역할: `dataset`에 해당하는 작업을 수행.
    # 매개변수: args.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `tuple`, `resolve_dataset_plan`, `resolve_dataset_name`, `get_dataset_spec`, `builder`입니다.
    # 제어 흐름: 반복문 2개, 조건 분기 1개, 예외 발생 경로 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
    """Return the legacy runner hook backed by the dataset registry."""

    def get_dataloaders(seed):
        # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
        # 매개변수: seed.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `tuple`, `resolve_dataset_plan`, `resolve_dataset_name`, `get_dataset_spec`, `builder`입니다.
        # 제어 흐름: 반복문 2개, 조건 분기 1개, 예외 발생 경로 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        dataloaders = []
        # category 목록의 각 항목은 하나의 실험 단위입니다.
        # 예: "bottle+cable"은 두 클래스를 같은 DataLoader에 합칩니다.
        requested_groups = args.subdatasets
        for raw_group in requested_groups:
            categories = tuple(raw_group.split("+"))
            plan = resolve_dataset_plan(args, categories)
            resolved_name = resolve_dataset_name(
                plan.registry_name, args.mainmodel
            )
            # method에 따라 같은 dataset key도 다른 구현을 사용할 수 있습니다.
            # 대표적으로 GLASS는 전용 MVTec transform 구현을 선택합니다.
            spec = get_dataset_spec(plan.registry_name, args.mainmodel)
            try:
                builder = _DATASET_BUILDERS[spec.constructor_style]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown dataset constructor style '{spec.constructor_style}'"
                ) from exc
            train_dataset, validation_dataset, test_dataset = builder(
                args, spec, plan, seed
            )

            for split_name, split_dataset in (
                ("train", train_dataset),
                ("validation", validation_dataset),
                ("test", test_dataset),
            ):
                validate_dataset_instance(split_dataset, check_sample=False)
                print_stat(split_dataset, f"{categories}_{split_name}")

            # 마지막 train batch가 한 장만 남으면 BatchNorm 등의 학습이 불안정할 수 있어 버립니다.
            drop_last = (
                len(train_dataset) > args.batch_size
                and len(train_dataset) % args.batch_size < 2
            )
            train_loader = _make_loader(
                train_dataset,
                args.batch_size,
                args.num_workers,
                shuffle=True,
                drop_last=drop_last,
            )
            # validation 데이터가 없으면 None을 유지해 빈 loader를 만들지 않습니다.
            validation_loader = None
            if validation_dataset.data_to_iterate:
                validation_loader = _make_loader(
                    validation_dataset,
                    1,
                    args.num_workers,
                    shuffle=False,
                )
            test_loader = _make_loader(
                test_dataset, 1, args.num_workers, shuffle=False
            )

            # runner가 결과 CSV와 로그에서 구분할 수 있도록 사람이 읽는 이름을 붙입니다.
            train_loader.name = f"{resolved_name}_{'+'.join(categories)}"
            dataloaders.append(
                {
                    "training": train_loader,
                    "validation": validation_loader,
                    "testing": test_loader,
                }
            )
            logger.info(
                f"Dataset {train_loader.name}: "
                f"train={len(train_dataset)} test={len(test_dataset)}"
            )
        return dataloaders

    return "get_dataloaders", get_dataloaders
# 한국어 코드 안내: 이 파일은 이미지 데이터셋의 파일 탐색과 sample 생성을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.


