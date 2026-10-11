"""Method factory used by the experiment runner."""

from __future__ import annotations

from typing import Any

from component_registry import METHOD_REGISTRY, get_method_spec

# Backward-compatible name for scripts that inspect the old registry.
TRAINER_REGISTRY = METHOD_REGISTRY


def _get_trainer_class(mainmodel: str):
    """Resolve one trainer lazily and validate its public contract."""

    return get_method_spec(mainmodel).load_class()


def load_backbone(args: Any, backbone_name: str):
    """Load a backbone according to the selected method specification."""

    return get_method_spec(args.mainmodel).load_backbone(backbone_name)


def _group_layers(backbone_names, layers_to_extract_from):
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
    """Return the legacy runner hook backed by the method registry."""

    backbone_names = list(args.backbone_names)
    layers_by_backbone = _group_layers(
        backbone_names, args.layers_to_extract_from
    )
    method_spec = get_method_spec(args.mainmodel)

    def get_simplenet(input_shape, device):
        trainers = []
        for raw_backbone_name, layers in zip(backbone_names, layers_by_backbone):
            backbone_name, backbone_seed = _parse_backbone_name(raw_backbone_name)
            backbone = method_spec.load_backbone(backbone_name)
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
