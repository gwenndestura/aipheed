"""add dismissed to alert_records

Revision ID: a1b2c3d4e5f6
Revises: 68355a016610
Create Date: 2026-05-03

"""
from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = '68355a016610'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'alert_records',
        sa.Column('dismissed', sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column('alert_records', 'dismissed')
