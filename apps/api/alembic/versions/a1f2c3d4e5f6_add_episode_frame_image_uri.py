"""add episode_frames.image_uri

Adds an optional per-frame image blob uri, used by the learned-model path (policy / attention
rollout / failure classifier). Nullable, so existing rows and the heuristic path are unaffected.

Revision ID: a1f2c3d4e5f6
Revises: 86b04106e66e
Create Date: 2026-08-18 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = 'a1f2c3d4e5f6'
down_revision = '86b04106e66e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('episode_frames', sa.Column('image_uri', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('episode_frames', 'image_uri')
