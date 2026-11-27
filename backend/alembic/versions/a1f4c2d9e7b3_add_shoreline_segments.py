"""add ordered shoreline segments to observations

Revision ID: a1f4c2d9e7b3
Revises: cd1bb73a5a4a
Create Date: 2026-11-24 00:00:00.000000

The legacy ``shoreline_geometry`` column is kept for backward compatibility
and holds the concatenation of the ordered segments. ``shoreline_segments``
adds the segment breaks so the frontend never draws a line between two
disjoint shorelines. Historical rows remain NULL; nothing is rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a1f4c2d9e7b3"
down_revision: str | None = "cd1bb73a5a4a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "shoreline_observations",
        sa.Column(
            "shoreline_segments",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("shoreline_observations", "shoreline_segments")
