"""add api_keys table

Per-organization API keys, stored as SHA-256 hashes. Replaces the in-process settings string,
where a key issued on one replica was invisible to the others and every key was lost on restart.

Only the hash is persisted; `prefix` keeps the first few characters in the clear so a human can
tell two keys apart without the secret being recoverable from a database backup.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-09-16 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = 'c3d4e5f6a7b8'
down_revision = 'b2c3d4e5f6a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'api_keys',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('key_hash', sa.String(), nullable=False),
        sa.Column('prefix', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False, server_default='default'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_api_keys_org_id', 'api_keys', ['org_id'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_api_keys_key_hash', table_name='api_keys')
    op.drop_index('ix_api_keys_org_id', table_name='api_keys')
    op.drop_table('api_keys')
