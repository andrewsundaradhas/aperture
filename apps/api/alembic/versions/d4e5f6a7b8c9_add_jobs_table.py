"""add jobs table

Background work the API owes and its own worker drains: cluster recompute and episode ingestion,
both of which were running inline in the request and would time out on a real fleet's volume.

Separate from `attribution_jobs`, which is a pull contract for external GPU workers with a
different lifecycle (remote claim, HTTP completion) and a different trust boundary.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-09-16 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = 'd4e5f6a7b8c9'
down_revision = 'c3d4e5f6a7b8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'jobs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='queued'),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error', sa.String(), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_jobs_org_id', 'jobs', ['org_id'])
    op.create_index('ix_jobs_kind', 'jobs', ['kind'])
    op.create_index('ix_jobs_status', 'jobs', ['status'])
    op.create_index('ix_jobs_created_at', 'jobs', ['created_at'])
    # The worker's hot path: oldest queued job of any kind.
    op.create_index('ix_jobs_status_created', 'jobs', ['status', 'created_at'])


def downgrade() -> None:
    for name in ('ix_jobs_status_created', 'ix_jobs_created_at', 'ix_jobs_status',
                 'ix_jobs_kind', 'ix_jobs_org_id'):
        op.drop_index(name, table_name='jobs')
    op.drop_table('jobs')
