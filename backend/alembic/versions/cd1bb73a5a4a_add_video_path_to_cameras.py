"""add_video_path_to_cameras

Revision ID: cd1bb73a5a4a
Revises: 7bcefb290c3d
Create Date: 2026-11-19 10:31:54.870150
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cd1bb73a5a4a"
down_revision: str | None = "7bcefb290c3d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Virtual video cameras (Phase 4 dev tooling): path to a video file,
    # relative to DATA_DIR/videos/. NULL for all other transports.
    op.add_column("cameras", sa.Column("video_path", sa.String(length=500), nullable=True))


def downgrade() -> None:
    op.drop_column("cameras", "video_path")
