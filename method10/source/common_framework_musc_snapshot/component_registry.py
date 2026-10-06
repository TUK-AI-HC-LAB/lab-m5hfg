"""Lazy registries and runtime contracts for methods and datasets.

The training pipeline imports this module without importing optional model
dependencies.  A component is imported only when its registry key is selected.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Iterable, Mapping, MutableMapping, Type

TRAINER_METHODS = (
    "load",
    "train",
    "predict",
    "get_evaluation_metrics",
    "set_model_dir",
)
DATASET_ATTRIBUTES = ("imagesize", "data_to_iterate")
DATASET_METHODS = ("having_mask",)
SAMPLE_KEYS = ("image", "mask", "is_anomaly")


class ComponentContractError(TypeError):
    """Raised when a registered component does not satisfy the public API."""


@dataclass(frozen=True)
class MethodSpec:
    """How to import and construct one anomaly-detection method."""

    module: str
    class_name: str
    config_name: str | None = None
    backbone_module: str = "backbones"
    constructor_uses_context: bool = False
    # AnomalyCLIP처럼 자체 CLIP backbone을 만드는 방법은 공통 backbone loader를 사용하지 않습니다.
    uses_backbone: bool = True

    def load_class(self) -> Type[Any]:
        # 역할: `load_class`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `importlib.import_module`, `getattr`, `validate_trainer_class`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        module = importlib.import_module(self.module)
        trainer_class = getattr(module, self.class_name)
        validate_trainer_class(trainer_class)
        return trainer_class

    def create(self, device: Any, context: Mapping[str, Any]) -> Any:
        # 역할: `create`에 해당하는 작업을 수행.
        # 매개변수: device, context.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `self.load_class`, `trainer_class`, `dict`입니다.
        # 제어 흐름: 조건 분기 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        trainer_class = self.load_class()
        if self.constructor_uses_context:
            return trainer_class(device, dict(context))
        return trainer_class(device)

    def load_backbone(self, backbone_name: str) -> Any:
        # 역할: `load_backbone`에 해당하는 작업을 수행.
        # 매개변수: backbone_name.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `importlib.import_module`, `module.load`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        module = importlib.import_module(self.backbone_module)
        return module.load(backbone_name)


@dataclass(frozen=True)
class DatasetSpec:
    """How to import one dataset implementation."""

    module: str
    class_name: str
    constructor_style: str = "path"

    def load_module(self) -> ModuleType:
        # 역할: `load_module`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `importlib.import_module`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return importlib.import_module(self.module)

    def load_class(self) -> Type[Any]:
        # 역할: `load_class`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `self.load_module`, `getattr`, `isinstance`, `ComponentContractError`입니다.
        # 제어 흐름: 조건 분기 1개, 예외 발생 경로 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        module = self.load_module()
        dataset_class = getattr(module, self.class_name)
        if not isinstance(dataset_class, type):
            raise ComponentContractError(
                f"{self.module}.{self.class_name} is not a class"
            )
        return dataset_class

    def load_split(self) -> Type[Any]:
        # 역할: `load_split`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `self.load_module`, `ComponentContractError`입니다.
        # 제어 흐름: 예외 발생 경로 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        module = self.load_module()
        try:
            return module.DatasetSplit
        except AttributeError as exc:
            raise ComponentContractError(
                f"{self.module} must expose DatasetSplit"
            ) from exc


METHOD_REGISTRY: MutableMapping[str, MethodSpec] = {
    "musc": MethodSpec(
        "trainer.trainer_musc", "Trainer_MuSc", "musc", uses_backbone=False
    ),
    "simple": MethodSpec(
        "trainer.trainer_simplenet", "Trainer_SimpleNet", "simple"
    ),
    "patchcore": MethodSpec(
        "trainer.trainer_patchcore", "Trainer_PatchCore", "patchcore"
    ),
    "promptad": MethodSpec(
        "trainer.trainer_promptad", "Trainer_PromptAD", "promptad"
    ),
    "winclip": MethodSpec(
        "trainer.trainer_winclip", "Trainer_WinCLIP", "winclip"
    ),
    "glass": MethodSpec(
        "trainer.trainer_glass", "Trainer_GLASS", "glass"
    ),
    "rd": MethodSpec("trainer.trainer_rd", "Trainer_RD", "rd"),
    "rd_orig": MethodSpec(
        "trainer.trainer_rd_orig", "Trainer_RD_Orig", "rd_orig"
    ),
    "uniad": MethodSpec(
        "trainer.trainer_uniad", "Trainer_UniAD", "uniad"
    ),
    "padim": MethodSpec("trainer.trainer_padim", "Trainer_PaDiM"),
    "dinomaly": MethodSpec(
        "trainer.trainer_dinomaly", "Trainer_Dinomaly", "dinomaly"
    ),
    "coad": MethodSpec(
        "COAD.COAD",
        "COADNetwork",
        "coad",
        backbone_module="COAD.encoder",
        constructor_uses_context=True,
    ),
    "anomalyclip": MethodSpec(
        "trainer.trainer_anomalyclip",
        "Trainer_AnomalyCLIP",
        "anomalyclip",
        uses_backbone=False,
    ),
}


DATASET_REGISTRY: MutableMapping[str, DatasetSpec] = {
    "mvtec": DatasetSpec("datasets.mvtec", "MVTecDataset"),
    "mvtec_anomalyclip": DatasetSpec(
        "datasets.anomalyclip_mvtec", "MVTecDataset"
    ),
    "mvtec_glass": DatasetSpec("trainer.GLASS_lib.mvtec", "MVTecDataset"),
    "multi_glass": DatasetSpec("trainer.GLASS_lib.mvtec", "MVTecDataset"),
    "multi": DatasetSpec("datasets.mvtec", "MVTecDataset"),
    "medical": DatasetSpec("datasets.mvtec", "MVTecDataset"),
    "visa": DatasetSpec("datasets.mvtec", "MVTecDataset"),
    "lg": DatasetSpec("datasets.lgdata", "LGDataset"),
    "custom": DatasetSpec("datasets.mvtec", "MVTecDataset"),
    "realiad": DatasetSpec(
        "datasets.explicit_realiad", "ExplicitDataset", "metadata"
    ),
}


DATASET_METHOD_VARIANTS = {
    ("mvtec", "glass"): "mvtec_glass",
    ("multi", "glass"): "multi_glass",
    # 공식 AnomalyCLIP의 Resize(short edge) → CenterCrop 전처리를 사용합니다.
    ("mvtec", "anomalyclip"): "mvtec_anomalyclip",
}


def _get_spec(
    registry: Mapping[str, Any], key: str, component_name: str
) -> Any:
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: registry, key, component_name.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `.join`, `ValueError`, `sorted`입니다.
    # 제어 흐름: 예외 발생 경로 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    try:
        return registry[key]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown {component_name} '{key}'. Available: {available}"
        ) from exc


def get_method_spec(name: str) -> MethodSpec:
    # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
    # 매개변수: name.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `_get_spec`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    return _get_spec(METHOD_REGISTRY, name, "method")


def get_dataset_spec(name: str, method_name: str | None = None) -> DatasetSpec:
    # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
    # 매개변수: name, method_name.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `DATASET_METHOD_VARIANTS.get`, `_get_spec`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    resolved_name = DATASET_METHOD_VARIANTS.get((name, method_name), name)
    return _get_spec(DATASET_REGISTRY, resolved_name, "dataset")


def resolve_dataset_name(name: str, method_name: str | None = None) -> str:
    # 역할: `resolve_dataset_name`에 해당하는 작업을 수행.
    # 매개변수: name, method_name.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `DATASET_METHOD_VARIANTS.get`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    return DATASET_METHOD_VARIANTS.get((name, method_name), name)


def list_methods() -> tuple[str, ...]:
    # 역할: `list_methods`에 해당하는 작업을 수행.
    # 매개변수: 없음.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `tuple`, `sorted`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    return tuple(sorted(METHOD_REGISTRY))


def list_datasets() -> tuple[str, ...]:
    # 역할: `list_datasets`에 해당하는 작업을 수행.
    # 매개변수: 없음.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `tuple`, `sorted`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    return tuple(sorted(DATASET_REGISTRY))


def register_method(name: str, spec: MethodSpec) -> None:
    # 역할: `register_method`에 해당하는 작업을 수행.
    # 매개변수: name, spec.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `_register`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
    _register(METHOD_REGISTRY, name, spec, MethodSpec)


def register_dataset(name: str, spec: DatasetSpec) -> None:
    # 역할: `register_dataset`에 해당하는 작업을 수행.
    # 매개변수: name, spec.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `_register`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
    _register(DATASET_REGISTRY, name, spec, DatasetSpec)


def _register(
    registry: MutableMapping[str, Any],
    name: str,
    spec: Any,
    expected_type: Type[Any],
) -> None:
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: registry, name, spec, expected_type.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `ValueError`, `isinstance`, `TypeError`, `name.strip`입니다.
    # 제어 흐름: 조건 분기 3개, 예외 발생 경로 3개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
    if not name or not name.strip():
        raise ValueError("Component name must be non-empty")
    if name in registry:
        raise ValueError(f"Component '{name}' is already registered")
    if not isinstance(spec, expected_type):
        raise TypeError(f"spec must be {expected_type.__name__}")
    registry[name] = spec


def validate_trainer_class(trainer_class: Type[Any]) -> None:
    # 역할: `validate_trainer_class`에 해당하는 작업을 수행.
    # 매개변수: trainer_class.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `isinstance`, `ComponentContractError`, `callable`, `getattr`, `.join`입니다.
    # 제어 흐름: 조건 분기 2개, 예외 발생 경로 2개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
    if not isinstance(trainer_class, type):
        raise ComponentContractError(f"{trainer_class!r} is not a class")
    missing = [name for name in TRAINER_METHODS if not callable(getattr(trainer_class, name, None))]
    if missing:
        raise ComponentContractError(
            f"{trainer_class.__module__}.{trainer_class.__name__} is missing "
            f"Trainer methods: {', '.join(missing)}"
        )


def validate_metrics(metrics: Any) -> Mapping[str, Any]:
    # 역할: `validate_metrics`에 해당하는 작업을 수행.
    # 매개변수: metrics.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `isinstance`, `ComponentContractError`입니다.
    # 제어 흐름: 조건 분기 2개, 예외 발생 경로 2개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    if not isinstance(metrics, Mapping):
        raise ComponentContractError(
            "get_evaluation_metrics() must return a mapping"
        )
    invalid = [key for key in metrics if not isinstance(key, str)]
    if invalid:
        raise ComponentContractError("metric keys must be strings")
    return metrics


def run_trainer_lifecycle(
    trainer: Any,
    training_data: Any,
    validation_data: Any,
    test_data: Any,
    dataset_name: str,
    collect_metrics: bool = True,
) -> Mapping[str, Any]:
    # 역할: `run_trainer_lifecycle`에 해당하는 작업을 수행.
    # 매개변수: trainer, training_data, validation_data, test_data, dataset_name, collect_metrics.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `trainer.train`, `validate_metrics`, `trainer.get_evaluation_metrics`입니다.
    # 제어 흐름: 조건 분기 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
    """Execute the method-independent lifecycle used by the runner."""

    trainer.train(training_data, validation_data, test_data, dataset_name)
    if not collect_metrics:
        return {}
    return validate_metrics(trainer.get_evaluation_metrics())


def validate_dataset_instance(dataset: Any, check_sample: bool = True) -> None:
    # 역할: `validate_dataset_instance`에 해당하는 작업을 수행.
    # 매개변수: dataset, check_sample.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `missing.extend`, `ComponentContractError`, `len`, `isinstance`, `hasattr`입니다.
    # 제어 흐름: 조건 분기 4개, 예외 발생 경로 3개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
    missing = [name for name in DATASET_ATTRIBUTES if not hasattr(dataset, name)]
    missing.extend(
        name for name in DATASET_METHODS if not callable(getattr(dataset, name, None))
    )
    if missing:
        raise ComponentContractError(
            f"{type(dataset).__module__}.{type(dataset).__name__} is missing "
            f"dataset members: {', '.join(missing)}"
        )

    if check_sample and len(dataset):
        sample = dataset[0]
        if not isinstance(sample, Mapping):
            raise ComponentContractError("dataset sample must be a mapping")
        missing_keys = [key for key in SAMPLE_KEYS if key not in sample]
        if missing_keys:
            raise ComponentContractError(
                f"dataset sample is missing keys: {', '.join(missing_keys)}"
            )


def validate_registry_keys(keys: Iterable[str], registry: Mapping[str, Any]) -> None:
    # 역할: `validate_registry_keys`에 해당하는 작업을 수행.
    # 매개변수: keys, registry.
    # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
    # 상세 흐름: 주요 호출은 `list`, `ValueError`, `key_list.count`, `.join`, `sorted`입니다.
    # 제어 흐름: 조건 분기 2개, 예외 발생 경로 2개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
    key_list = list(keys)
    duplicates = {key for key in key_list if key_list.count(key) > 1}
    if duplicates:
        raise ValueError(f"Duplicate keys: {', '.join(sorted(duplicates))}")
    if not registry:
        raise ValueError("Registry must not be empty")
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
