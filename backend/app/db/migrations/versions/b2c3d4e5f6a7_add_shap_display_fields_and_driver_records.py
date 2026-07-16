"""add shap display fields and driver_records table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-12
"""

from alembic import op
import sqlalchemy as sa

revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Add display fields to shap_records ──────────────────────────────
    op.add_column("shap_records", sa.Column("display_name",  sa.String(),  nullable=True))
    op.add_column("shap_records", sa.Column("feature_group", sa.String(),  nullable=True))
    op.add_column("shap_records", sa.Column("feature_value", sa.Float(),   nullable=True))
    op.add_column("shap_records", sa.Column("unit",          sa.String(),  nullable=True))
    op.add_column("shap_records", sa.Column("baseline",      sa.Float(),   nullable=True))
    op.add_column("shap_records", sa.Column("final_rfii",    sa.Float(),   nullable=True))

    # ── Create driver_records table ──────────────────────────────────────
    op.create_table(
        "driver_records",
        sa.Column("id",                 sa.Integer(),  primary_key=True, autoincrement=True),
        sa.Column("quarter",            sa.String(),   nullable=False),
        sa.Column("province_code",      sa.String(),   nullable=False),
        sa.Column("driver_group",       sa.String(),   nullable=False),
        sa.Column("driver_label",       sa.String(),   nullable=False),
        sa.Column("group_shap",         sa.Float(),    nullable=False),
        sa.Column("direction",          sa.String(),   nullable=False),
        sa.Column("display_pct",        sa.Float(),    nullable=False),
        sa.Column("trigger_proportion", sa.Float(),    nullable=True),
        sa.Column("article_count",      sa.Integer(),  nullable=True),
    )
    op.create_index(
        "ix_driver_quarter_province",
        "driver_records",
        ["quarter", "province_code"],
    )


def downgrade() -> None:
    op.drop_index("ix_driver_quarter_province", table_name="driver_records")
    op.drop_table("driver_records")

    for col in ["display_name", "feature_group", "feature_value", "unit", "baseline", "final_rfii"]:
        op.drop_column("shap_records", col)
