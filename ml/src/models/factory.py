"""Model factory used by training, evaluation, export and inference."""

from __future__ import annotations

from typing import Any

from ml.src.models.deeplabv3 import build_deeplabv3plus
from ml.src.models.unet import build_unet

ARCHITECTURES = ("unet", "deeplabv3plus")

DEFAULT_ENCODERS = {
    "unet": "resnet34",
    "deeplabv3plus": "resnet50",
}


def build_model(
    architecture: str,
    *,
    num_classes: int = 4,
    encoder_name: str | None = None,
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
) -> Any:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose {ARCHITECTURES}")
    encoder = encoder_name or DEFAULT_ENCODERS[architecture]
    if architecture == "unet":
        return build_unet(
            num_classes=num_classes,
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
        )
    return build_deeplabv3plus(
        num_classes=num_classes,
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
    )


def architecture_config(config: dict[str, Any], architecture: str) -> dict[str, Any]:
    entry = config.get("model", {}).get("architectures", {}).get(architecture, {})
    return {
        "architecture": architecture,
        "encoder": entry.get("encoder", DEFAULT_ENCODERS[architecture]),
        "encoder_weights": entry.get("encoder_weights", "imagenet"),
        "in_channels": int(config.get("model", {}).get("in_channels", 3)),
    }


def describe_model(model: Any) -> dict[str, Any]:
    return {
        "class": type(model).__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
