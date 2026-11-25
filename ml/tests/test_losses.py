"""Loss guards: fully-ignored crops and untrained classes must not produce NaN."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.src.training.losses import build_loss  # noqa: E402


def test_all_ignore_batch_returns_finite_zero_loss() -> None:
    loss_fn = build_loss(
        num_classes=4, ignore_index=255, class_weights=np.array([0, 1, 1, 0], np.float32)
    )
    logits = torch.zeros(2, 4, 16, 16, requires_grad=True)
    target = torch.full((2, 16, 16), 255, dtype=torch.long)
    loss, parts = loss_fn(logits, target)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert parts["loss"] == 0.0
    loss.backward()  # must not raise


def test_untrained_class_gets_no_ce_weight() -> None:
    weights = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    loss_fn = build_loss(num_classes=4, ignore_index=255, class_weights=weights)
    logits = torch.zeros(1, 4, 8, 8, requires_grad=True)
    target = torch.full((1, 8, 8), 1, dtype=torch.long)
    _, parts = loss_fn(logits, target)
    assert np.isfinite(parts["loss"])
    assert loss_fn.class_weights.tolist() == weights.tolist()


def test_class_weights_length_is_validated() -> None:
    with pytest.raises(ValueError):
        build_loss(num_classes=4, ignore_index=255, class_weights=np.ones(3, np.float32))
