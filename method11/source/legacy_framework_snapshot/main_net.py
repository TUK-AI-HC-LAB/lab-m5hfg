"""Method factory used by the experiment runner."""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

from component_registry import METHOD_REGISTRY, get_method_spec

# Backward-compatible name for scripts that inspect the old registry.
TRAINER_REGISTRY = METHOD_REGISTRY


def _get_trainer_class(mainmodel: str):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: mainmodel.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `get_method_spec.load_class`, `get_method_spec`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    """Resolve one trainer lazily and validate its public contract."""

    return get_method_spec(mainmodel).load_class()


def load_backbone(args: Any, backbone_name: str):
    # 역할: `load_backbone`에 해당하는 작업을 수행.
    # 매개변수: args, backbone_name.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `get_method_spec.load_backbone`, `get_method_spec`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    """Load a backbone according to the selected method specification."""

    return get_method_spec(args.mainmodel).load_backbone(backbone_name)


def _group_layers(backbone_names, layers_to_extract_from):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: backbone_names, layers_to_extract_from.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `len`, `layer.partition`, `int`, `.append`, `list`입니다.
    # 제어 흐름: 반복문 1개, 조건 분기 3개, 예외 발생 경로 2개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
    if len(backbone_names) == 1:
        return [list(layers_to_extract_from)]

    grouped = [[] for _ in backbone_names]
    for layer in layers_to_extract_from:
        prefix, separator, layer_name = layer.partition(".")
        if not separator or not prefix.isdigit():
            raise ValueError(
                "With multiple backbones, layer names must use '<index>.<layer>', "
                f"got '{layer}'"
            )
        index = int(prefix)
        if index >= len(backbone_names):
            raise ValueError(
                f"Layer '{layer}' refers to missing backbone index {index}"
            )
        grouped[index].append(layer_name)
    return grouped


def _parse_backbone_name(backbone_name: str):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: backbone_name.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `backbone_name.rsplit`, `int`, `ValueError`입니다.
    # 제어 흐름: 조건 분기 1개, 예외 발생 경로 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
    marker = ".seed-"
    if marker not in backbone_name:
        return backbone_name, None
    name, seed_text = backbone_name.rsplit(marker, 1)
    try:
        return name, int(seed_text)
    except ValueError as exc:
        raise ValueError(
            f"Backbone seed must be an integer in '{backbone_name}'"
        ) from exc


def net(args):
    # 역할: `net`에 해당하는 작업을 수행.
    # 매개변수: args.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `list`, `_group_layers`, `get_method_spec`, `zip`, `_parse_backbone_name`입니다.
    # 제어 흐름: 반복문 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
    """Return the legacy runner hook backed by the method registry."""

    backbone_names = list(args.backbone_names)
    layers_by_backbone = _group_layers(
        backbone_names, args.layers_to_extract_from
    )
    method_spec = get_method_spec(args.mainmodel)

    def get_simplenet(input_shape, device):
        # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
        # 매개변수: input_shape, device.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `zip`, `_parse_backbone_name`, `method_spec.load_backbone`, `method_spec.create`, `trainer.load`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        trainers = []
        for raw_backbone_name, layers in zip(backbone_names, layers_by_backbone):
            backbone_name, backbone_seed = _parse_backbone_name(raw_backbone_name)
            # 대부분의 방법은 framework backbone을 사용합니다.
            # AnomalyCLIP은 공식 ViT-L/14 CLIP을 adapter 안에서 직접 만들므로,
            # runner가 요구하는 name/seed만 가진 빈 backbone 정보를 전달합니다.
            backbone = (
                method_spec.load_backbone(backbone_name)
                if method_spec.uses_backbone
                else SimpleNamespace()
            )
            backbone.name = backbone_name
            backbone.seed = backbone_seed

            context = {
                "args": args,
                "input_shape": input_shape,
                "backbone_name": backbone_name,
                "layers_to_extract_from": layers,
            }
            trainer = method_spec.create(device, context)
            trainer.load(
                backbone=backbone,
                layers_to_extract_from=layers,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=args.pretrain_embed_dimension,
                target_embed_dimension=args.target_embed_dimension,
                patchsize=args.patchsize,
                meta_epochs=args.meta_epochs,
                aed_meta_epochs=args.aed_meta_epochs,
                gan_epochs=args.gan_epochs,
                noise_std=args.noise_std,
                dsc_layers=args.dsc_layers,
                dsc_hidden=args.dsc_hidden,
                dsc_margin=args.dsc_margin,
                dsc_lr=args.dsc_lr,
                auto_noise=args.auto_noise,
                train_backbone=args.train_backbone,
                cos_lr=args.cos_lr,
                pre_proj=args.pre_proj,
                proj_layer_type=args.proj_layer_type,
                mix_noise=args.mix_noise,
                onnx=args.onnx,
                args=args,
            )
            trainers.append(trainer)
        return trainers

    return "get_simplenet", get_simplenet
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
