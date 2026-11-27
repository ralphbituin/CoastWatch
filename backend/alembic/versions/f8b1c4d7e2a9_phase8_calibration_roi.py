"""phase 8: calibration monitoring outline (ROI polygon)

Revision ID: f8b1c4d7e2a9
Revises: e5a9c3d7f2b4
Create Date: 2026-11-27

Additive and nullable only. Existing calibration profiles keep full-frame
behaviour (``roi_polygon IS NULL``); no rows are rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f8b1c4d7e2a9"
down_revision: str | None = "e5a9c3d7f2b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "camera_calibrations",
        sa.Column(
            "roi_polygon",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("camera_calibrations", "roi_polygon")
