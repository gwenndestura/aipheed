from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Index, JSON, Text, UniqueConstraint
)
from sqlalchemy.sql import func
from app.db.database import Base


class ForecastRecord(Base):
    """
    Stores province-level food insecurity forecast.
    One row per province per quarter.
    5 CALABARZON provinces.
    """
    __tablename__ = "forecast_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    quarter = Column(String, nullable=False)         # format: "2025-Q4"
    province_code = Column(String, nullable=False)   # PSGC e.g. "PH-BTG"
    province_name = Column(String, nullable=False)   # e.g. "Batangas"
    risk_probability = Column(Float, nullable=False) # 0.0 to 1.0
    risk_label = Column(String, nullable=False)      # "HIGH" or "LOW"
    data_sufficiency_flag = Column(String, nullable=True)  # "LIMITED_SIGNAL" or None
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_forecast_quarter_province", "quarter", "province_code"),
    )


class MunicipalForecastRecord(Base):
    """
    Stores municipal/city-level disaggregated forecast.
    One row per LGU per quarter.
    142 cities and municipalities of CALABARZON.
    Derived from province forecast via:
    y_m = y_province * (0.6 * poverty_m + 0.4 * density_m)
    Every row MUST have disaggregation_label per Backend Guide Rule 8.
    """
    __tablename__ = "municipal_forecast_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    quarter = Column(String, nullable=False)
    province_code = Column(String, nullable=False)   # parent province
    lgu_code = Column(String, nullable=False)        # PSGC municipal code
    municipality_name = Column(String, nullable=False)
    risk_index = Column(Float, nullable=False)       # 0.0 to 1.0
    disaggregation_label = Column(String, nullable=False)  # NEVER null
    data_sufficiency_flag = Column(String, nullable=True)  # inherited from province
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_municipal_quarter_lgu", "quarter", "lgu_code"),
        Index("ix_municipal_quarter_province", "quarter", "province_code"),
    )


class SHAPRecord(Base):
    """
    Stores SHAP feature importance values.
    One row per feature per province per quarter (45 features × 5 provinces).

    New fields (v2):
      display_name   — human-readable label for the dashboard waterfall.
      feature_group  — driver category: market | climate | employment |
                       macro_ofw | nlp
      feature_value  — actual input value fed to the model (for tooltip).
      unit           — display unit string (e.g. "% change", "PHP/kg").
      baseline       — SHAP expected value (mean training prob ≈ 0.54).
                       Same for every row in the same model version;
                       stored per-row for convenience.
      final_rfii     — baseline + sum(shap_values) ≈ predicted risk_probability.
                       Same for all rows of the same province-quarter.
    """
    __tablename__ = "shap_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    quarter = Column(String, nullable=False)
    province_code = Column(String, nullable=False)
    feature_name = Column(String, nullable=False)
    shap_value = Column(Float, nullable=False)
    mean_abs_shap = Column(Float, nullable=False)

    # v2 display fields
    display_name = Column(String, nullable=True)
    feature_group = Column(String, nullable=True)
    feature_value = Column(Float, nullable=True)
    unit = Column(String, nullable=True)
    baseline = Column(Float, nullable=True)
    final_rfii = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_shap_quarter_province", "quarter", "province_code"),
    )


class DriverRecord(Base):
    """
    Stores grouped SHAP driver contributions.
    One row per driver group per province per quarter (5 groups × 5 provinces).

    Drives the left-panel "RISK DRIVERS" ranked bar chart and the
    trigger composition stacked bar.
    """
    __tablename__ = "driver_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    quarter = Column(String, nullable=False)
    province_code = Column(String, nullable=False)

    # Raw group key: market | climate | employment | macro_ofw | nlp_sentiment
    driver_group = Column(String, nullable=False)

    # Human-readable label (e.g. "Food Prices")
    driver_label = Column(String, nullable=False)

    # Net SHAP contribution (probability scale, e.g. +0.354)
    group_shap = Column(Float, nullable=False)

    # "increases_risk" or "protective"
    direction = Column(String, nullable=False)

    # Bar width percentage (abs(group_shap) / total_abs_shap * 100)
    display_pct = Column(Float, nullable=False)

    # NLP news signal proportion from trigger_proportions.parquet (0.0–1.0)
    trigger_proportion = Column(Float, nullable=True)

    # Number of NLP articles analysed (stored on the market group row,
    # NULL on others — use max() when querying)
    article_count = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_driver_quarter_province", "quarter", "province_code"),
    )


class AlertRecord(Base):
    """
    Stores early warning alerts.
    States: STAGED (confirmed=False, dismissed=False)
            CONFIRMED (confirmed=True)  — published to dashboard
            DISMISSED (dismissed=True)  — suppressed, kept for audit only
    """
    __tablename__ = "alert_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    quarter = Column(String, nullable=False)
    province_code = Column(String, nullable=False)
    threshold_exceeded = Column(Boolean, nullable=False, default=False)
    confirmed = Column(Boolean, nullable=False, default=False)
    dismissed = Column(Boolean, nullable=False, default=False)
    # Sudden-rise detection fields
    prev_risk_probability = Column(Float, nullable=True)   # previous quarter's probability
    risk_delta = Column(Float, nullable=True)              # current − previous probability
    alert_reason = Column(String, nullable=True)           # e.g. "SUDDEN_RISE"
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_alert_quarter_province", "quarter", "province_code"),
    )

# ===========================================================================
# Admin console: accounts, the review pipeline, audit and feedback
# ===========================================================================


class AdminUser(Base):
    """
    A DA CALABARZON account.

    Replaces both client-side login checks. There is no seeded default
    account: passwords are Argon2 hashes created through
    scripts/create_admin.py, so no credential ever ships in the repo or the
    JS bundle.
    """
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, unique=True, index=True)
    full_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="admin")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, server_default=func.now())
    last_login_at = Column(DateTime, nullable=True)


class RevokedToken(Base):
    """
    Tokens invalidated by an explicit logout.

    A JWT is valid until it expires, so without this table logout would only
    clear the browser's copy while the token itself kept working. Rows are
    dropped once past `expires_at` -- after that the token fails on its own.
    """
    __tablename__ = "revoked_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    jti = Column(String, nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, server_default=func.now())


class ReviewRecord(Base):
    """
    One province-quarter forecast moving through review.

    This is the ONLY source of publication status. A rejection is this row at
    status="Rejected" carrying its reason and actor, not a second table -- two
    tables would be free to disagree about whether a forecast is published,
    which is the exact failure the frontend already has with its two province
    datasets.

    id is deterministic ("rv_2025Q4_quezon") so seeding a quarter twice is a
    no-op rather than a duplicate.
    """
    __tablename__ = "review_records"

    id = Column(String, primary_key=True)
    province_code = Column(String, nullable=False)   # PSGC, e.g. PH040300000
    province_id = Column(String, nullable=False)     # slug, e.g. quezon
    province_name = Column(String, nullable=False)
    quarter = Column(String, nullable=False)

    # Snapshot of the model output at the time of review, so the queue shows
    # what the reviewer acted on even if the model is later retrained.
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)

    status = Column(String, nullable=False, default="Staged")  # Staged|Approved|Rejected

    # Populated only while status == "Rejected".
    rejection_reason = Column(String, nullable=True)
    rejection_notes = Column(Text, nullable=True)

    generated_on = Column(String, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    updated_by = Column(Integer, nullable=True)
    updated_by_name = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("province_id", "quarter", name="uq_review_province_quarter"),
        Index("ix_review_quarter_status", "quarter", "status"),
    )


class AuditRecord(Base):
    """
    Append-only log of every state change an admin makes.

    Written in the same transaction as the change it describes, so the queue
    and the log cannot drift.
    """
    __tablename__ = "audit_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    actor_id = Column(Integer, nullable=True)
    actor_email = Column(String, nullable=False)
    action = Column(String, nullable=False)        # approve|reject|undo|restore|login
    subject_type = Column(String, nullable=False)  # review|rejection|session
    subject_id = Column(String, nullable=True)
    province_id = Column(String, nullable=True)
    quarter = Column(String, nullable=True)
    reason = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)


class FeedbackRecord(Base):
    """
    One System Usability Scale submission.

    `score` is recomputed server-side from `answers`; the client's own total is
    never trusted. `sus_version` pins which question wording produced the
    answers so historical responses stay interpretable if the survey changes.
    """
    __tablename__ = "feedback_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    submitted_at = Column(DateTime, server_default=func.now(), index=True)
    score = Column(Float, nullable=False)
    answers = Column(JSON, nullable=False)      # {"0": 4, "1": 2, ... "9": 2}
    sus_version = Column(String, nullable=False, default="sus-v1")

    full_name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    agency = Column(String, nullable=True)
    designation = Column(String, nullable=True)
    age_band = Column(String, nullable=True)
    sex = Column(String, nullable=True)
    client_type = Column(String, nullable=True)
    province = Column(String, nullable=True)
    municipality = Column(String, nullable=True)

    liked = Column(Text, nullable=True)
    improvements = Column(Text, nullable=True)
