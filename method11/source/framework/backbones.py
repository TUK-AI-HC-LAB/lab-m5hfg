import torchvision.models as models  # noqa

_BACKBONES = {
    "wideresnet50": lambda: models.wide_resnet50_2(weights="DEFAULT"),
}


def load(name):
    try:
        factory = _BACKBONES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKBONES))
        raise ValueError(f"Unknown backbone '{name}'. Available: {available}") from exc
    return factory()
