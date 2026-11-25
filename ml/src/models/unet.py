"""U-Net with a ResNet encoder (segmentation-models-pytorch)."""

from __future__ import annotations

from typing import Any

DEFAULT_ENCODER = "resnet34"


def build_unet(
    *,
    num_classes: int = 4,
    encoder_name: str = DEFAULT_ENCODER,
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
) -> Any:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "segmentation-models-pytorch is required. Install ml/requirements.txt."
        ) from exc
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
