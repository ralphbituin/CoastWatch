"""Gradient accumulation with all-ignore batches must still flush and step."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from ml.scripts.train import _train_epoch  # noqa: E402


class _TinySegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3, 1, 1))

    def forward(self, pixel_values: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return self.logits.view(1, 3, 1, 1).expand(pixel_values.shape[0], 3, *output_size)


def _batch(label: int | None) -> dict[str, torch.Tensor]:
    labels = (
        torch.full((1, 4, 4), 255, dtype=torch.long)
        if label is None
        else torch.full((1, 4, 4), label, dtype=torch.long)
    )
    return {"pixel_values": torch.zeros(1, 3, 4, 4), "labels": labels}


def _run(batches: list[dict[str, torch.Tensor]], accum: int = 2):
    model = _TinySegmenter()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    config = {
        "training": {
            "gradient_accumulation_steps": accum,
            "precision": "fp32",
            "gradient_clip_norm": 0.0,
            "log_every": 0,
        }
    }
    stats = _train_epoch(
        model,
        batches,
        optimizer,
        scheduler,
        scaler,
        nn.CrossEntropyLoss(ignore_index=255),
        torch.device("cpu"),
        config,
        epoch=1,
        num_epochs=1,
    )
    return stats, scheduler, model


def test_final_all_ignore_batch_flushes_pending_gradients() -> None:
    # The all-ignore batch lands on the accumulation boundary and is also the
    # final batch, so the pending gradient from batch 1 must still be applied.
    stats, scheduler, model = _run([_batch(0), _batch(None)], accum=2)
    assert stats["train_batches"] == 1
    assert scheduler.last_epoch == 1
    assert model.logits.grad is None


def test_all_ignore_batch_still_closes_the_accumulation_window() -> None:
    # Two windows: (supervised, all-ignore) and (supervised, end of epoch).
    # The scheduler must advance twice, not once, despite the ignored batch.
    stats, scheduler, _ = _run([_batch(0), _batch(None), _batch(1)], accum=2)
    assert stats["train_batches"] == 2
    assert scheduler.last_epoch == 2
