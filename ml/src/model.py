"""SegFormer MiT-B2 model construction and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

DEFAULT_PRETRAINED = "nvidia/segformer-b2-finetuned-ade-512-512"


class SegFormerB2(nn.Module):
    """Thin wrapper around Hugging Face SegFormer MiT-B2.

    The pretrained ADE-20k encoder and decoder are preserved; only the final
    1x1 classification head is replaced (``ignore_mismatched_sizes=True``).
    SegFormer returns logits at 1/4 input resolution; ``forward`` optionally
    upsamples them to a requested size for the loss or for inference.
    """

    def __init__(
        self,
        *,
        num_labels: int = 3,
        pretrained: str = DEFAULT_PRETRAINED,
        ignore_mismatched_sizes: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            local_files_only=local_files_only,
        )
        self.num_labels = num_labels
        self.pretrained_name = pretrained

    @property
    def encoder(self) -> nn.Module:
        return self.model.segformer

    @property
    def decode_head(self) -> nn.Module:
        return self.model.decode_head

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        logits = self.model(pixel_values=pixel_values).logits
        if output_size is not None and tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(
                logits,
                size=tuple(output_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        id2label: dict[int, str] | None = None,
    ) -> None:
        if id2label is not None:
            self.model.config.id2label = {int(k): v for k, v in id2label.items()}
            self.model.config.label2id = {
                v: int(k) for k, v in id2label.items()
            }
            self.model.config.num_labels = len(id2label)
        self.model.save_pretrained(str(directory))


def build_model(
    config: dict[str, Any],
    *,
    local_files_only: bool = False,
) -> SegFormerB2:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    num_labels = int(model_cfg.get("num_labels", data_cfg.get("num_classes", 3)))
    pretrained = str(model_cfg.get("pretrained", DEFAULT_PRETRAINED))
    return SegFormerB2(
        num_labels=num_labels,
        pretrained=pretrained,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )


def parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """AdamW groups: no weight decay for bias and LayerNorm parameters."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "LayerNorm" in name
            or "layer_norm" in name
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: dict[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[SegFormerB2, dict[str, Any]]:
    """Load a trained three-class checkpoint for inference.

    The ADE-20k model is only the *training* initializer. At inference we build
    from the trained ``best_hf`` copy next to the checkpoint (when present) and
    load every tensor from the checkpoint with ``strict=True``. Each loaded
    tensor is then compared with the checkpoint payload, so a partially-random
    model cannot be returned silently.
    """
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model_state_dict", payload))
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{checkpoint_path} has no usable model state")
    num_labels = int(payload.get("num_labels", config.get("data", {}).get("num_classes", 3)))
    hf_dir = checkpoint_path.parent / "best_hf"
    local_files_only = hf_dir.is_dir()
    pretrained = str(hf_dir) if local_files_only else str(
        payload.get(
            "pretrained",
            config.get("model", {}).get("pretrained", DEFAULT_PRETRAINED),
        )
    )
    model = SegFormerB2(
        num_labels=num_labels,
        pretrained=pretrained,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    mismatched = [key for key, value in state.items() if not torch.equal(value, loaded[key])]
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} checkpoint tensor(s) were not loaded exactly; "
            f"first: {mismatched[:3]}"
        )
    model.to(device)
    model.eval()
    return model, payload


def trainable_parameter_count(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
