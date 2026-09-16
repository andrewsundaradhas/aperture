"""add episode_frames.action and episode_frames.state

The commanded action and observed robot state per timestep, as JSON float vectors.

Without these a dataset export carries only diagnostic scalars (action confidence, contact
force, subgoal), which cannot be fine-tuned on — a policy learns from (observation, action)
pairs. Both nullable, so existing rows and every episode ingested before this migration stay
valid and simply have no action to export.

Revision ID: b2c3d4e5f6a7
Revises: a1f2c3d4e5f6
Create Date: 2026-09-16 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = 'b2c3d4e5f6a7'
down_revision = 'a1f2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('episode_frames', sa.Column('action', sa.JSON(), nullable=True))
    op.add_column('episode_frames', sa.Column('state', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('episode_frames', 'state')
    op.drop_column('episode_frames', 'action')
