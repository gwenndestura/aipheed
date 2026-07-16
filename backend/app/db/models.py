from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Index
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

    # Human-readable label (e.g. "Market / Prices")
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