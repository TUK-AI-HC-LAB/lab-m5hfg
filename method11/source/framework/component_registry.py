"""Lazy registries and runtime contracts for methods and datasets.

The training pipeline imports this module without importing optional model
dependencies.  A component is imported only when its registry key is selected.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
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
    uses_backbone: bool = True

    def load_class(self) -> Type[Any]:
        module = importlib.import_module(self.module)
        trainer_class = getattr(module, self.class_name)
        validate_trainer_class(trainer_class)
        return trainer_class

    def create(self, device: Any, context: Mapping[str, Any]) -> Any:
        trainer_class = self.load_class()
        if self.constructor_uses_context:
            return trainer_class(device, dict(context))
        return trainer_class(device)

    def load_backbone(self, backbone_name: str) -> Any:
        if not self.uses_backbone:
            # 실제 모델이 아니라 runner가 조회하는 이름·seed만 담습니다.
            # CLIP과 DINO는 QueCo Trainer에서 직접 불러옵니다.
            return SimpleNamespace(name=backbone_name, seed=None)

        # 기존 방법은 이전처럼 공통 backbone을 불러옵니다.
        module = importlib.import_module(self.backbone_module)
        return module.load(backbone_name)    


@dataclass(frozen=True)
class DatasetSpec:
    """How to import one dataset implementation."""

    module: str
    class_name: str
    constructor_style: str = "path"

    def load_module(self) -> ModuleType:
        return importlib.import_module(self.module)

    def load_class(self) -> Type[Any]:
        module = self.load_module()
        dataset_class = getattr(module, self.class_name)
        if not isinstance(dataset_class, type):
            raise ComponentContractError(
                f"{self.module}.{self.class_name} is not a class"
            )
        return dataset_class

    def load_split(self) -> Type[Any]:
        module = self.load_module()
        try:
            return module.DatasetSplit
        except AttributeError as exc:
            raise ComponentContractError(
                f"{self.module} must expose DatasetSplit"
            ) from exc


METHOD_REGISTRY: MutableMapping[str, MethodSpec] = {
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
    "queco": MethodSpec(
        "trainer.trainer_ourzad",  # 불러올 Python 파일
        "Trainer_OurZAD",          # 그 파일 안의 클래스 이름
        "queco",                  # configs/queco.yaml 설정 사용
        uses_backbone=False,
    ),
}


DATASET_REGISTRY: MutableMapping[str, DatasetSpec] = {
    "mvtec": DatasetSpec("datasets.mvtec", "MVTecDataset"),
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
}


def _get_spec(
    registry: Mapping[str, Any], key: str, component_name: str
) -> Any:
    try:
        return registry[key]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown {component_name} '{key}'. Available: {available}"
        ) from exc


def get_method_spec(name: str) -> MethodSpec:
    return _get_spec(METHOD_REGISTRY, name, "method")


def get_dataset_spec(name: str, method_name: str | None = None) -> DatasetSpec:
    resolved_name = DATASET_METHOD_VARIANTS.get((name, method_name), name)
    return _get_spec(DATASET_REGISTRY, resolved_name, "dataset")


def resolve_dataset_name(name: str, method_name: str | None = None) -> str:
    return DATASET_METHOD_VARIANTS.get((name, method_name), name)


def list_methods() -> tuple[str, ...]:
    return tuple(sorted(METHOD_REGISTRY))


def list_datasets() -> tuple[str, ...]:
    return tuple(sorted(DATASET_REGISTRY))


def register_method(name: str, spec: MethodSpec) -> None:
    _register(METHOD_REGISTRY, name, spec, MethodSpec)


def register_dataset(name: str, spec: DatasetSpec) -> None:
    _register(DATASET_REGISTRY, name, spec, DatasetSpec)


def _register(
    registry: MutableMapping[str, Any],
    name: str,
    spec: Any,
    expected_type: Type[Any],
) -> None:
    if not name or not name.strip():
        raise ValueError("Component name must be non-empty")
    if name in registry:
        raise ValueError(f"Component '{name}' is already registered")
    if not isinstance(spec, expected_type):
        raise TypeError(f"spec must be {expected_type.__name__}")
    registry[name] = spec


def validate_trainer_class(trainer_class: Type[Any]) -> None:
    if not isinstance(trainer_class, type):
        raise ComponentContractError(f"{trainer_class!r} is not a class")
    missing = [name for name in TRAINER_METHODS if not callable(getattr(trainer_class, name, None))]
    if missing:
        raise ComponentContractError(
            f"{trainer_class.__module__}.{trainer_class.__name__} is missing "
            f"Trainer methods: {', '.join(missing)}"
        )


def validate_metrics(metrics: Any) -> Mapping[str, Any]:
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
    """Execute the method-independent lifecycle used by the runner."""

    trainer.train(training_data, validation_data, test_data, dataset_name)
    if not collect_metrics:
        return {}
    return validate_metrics(trainer.get_evaluation_metrics())


def validate_dataset_instance(dataset: Any, check_sample: bool = True) -> None:
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
    key_list = list(keys)
    duplicates = {key for key in key_list if key_list.count(key) > 1}
    if duplicates:
        raise ValueError(f"Duplicate keys: {', '.join(sorted(duplicates))}")
    if not registry:
        raise ValueError("Registry must not be empty")
