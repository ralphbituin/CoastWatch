"""Frame preprocessing (Phase 3/4).

Planned: frame sampling, resizing, noise reduction, contrast enhancement,
image stabilization and region-of-interest selection. Also frame-quality
checks (black frames, frozen frames, severe blur, obstruction, timestamp
inconsistencies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FrameQuality:
    score: float
    is_black: bool = False
    is_frozen: bool = False
    is_blurry: bool = False
    is_obstructed: bool = False
    flags: list[str] = field(default_factory=list)


def preprocess(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Vision preprocessing is implemented in Phase 3/4.")


def assess_quality(*args: Any, **kwargs: Any) -> FrameQuality:
    raise NotImplementedError("Frame-quality assessment is implemented in Phase 3/4.")
