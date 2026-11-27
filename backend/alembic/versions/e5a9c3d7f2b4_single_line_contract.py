"""Add single-line shoreline diagnostics.

Revision ID: e5a9c3d7f2b4
Revises: d4e8f2a6c1b9
Create Date: 2026-11-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e5a9c3d7f2b4"
down_revision: str | None = "d4e8f2a6c1b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "shoreline_observations",
        sa.Column("shoreline_axis", sa.String(length=12), nullable=True),
    )
    op.add_column(
        "shoreline_observations",
        sa.Column(
            "interpolated_ranges",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "shoreline_observations",
        sa.Column("measured_coverage", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("shoreline_observations", "measured_coverage")
    op.drop_column("shoreline_observations", "interpolated_ranges")
    op.drop_column("shoreline_observations", "shoreline_axis")
