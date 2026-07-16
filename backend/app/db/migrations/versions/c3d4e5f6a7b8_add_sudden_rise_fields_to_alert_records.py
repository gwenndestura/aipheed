"""add sudden-rise detection fields to alert_records

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-05-12

Adds three columns that support the sudden-rise alert detection logic
introduced in Predictor.detect_alerts():

    prev_risk_probability  — the previous quarter's risk probability
    risk_delta             — current minus previous (e.g. +0.18)
    alert_reason           — reason code, currently always "SUDDEN_RISE"

All three are nullable so existing alert rows are unaffected.
"""
from alembic import op
import sqlalchemy as sa

revision = 'c3d4e5f6a7b8'
down_revision = 'b2c3d4e5f6a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'alert_records',
        sa.Column('prev_risk_probability', sa.Float(), nullable=True),
    )
    op.add_column(
        'alert_records',
        sa.Column('risk_delta', sa.Float(), nullable=True),
    )
    op.add_column(
        'alert_records',
        sa.Column('alert_reason', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('alert_records', 'alert_reason')
    op.drop_column('alert_records', 'risk_delta')
    op.drop_column('alert_records', 'prev_risk_probability')
