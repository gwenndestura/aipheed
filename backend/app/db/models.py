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
    One row per feature per province per quarter.
    """
    __tablename__ = "shap_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    quarter = Column(String, nullable=False)
    province_code = Column(String, nullable=False)
    feature_name = Column(String, nullable=False)
    shap_value = Column(Float, nullable=False)
    mean_abs_shap = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_shap_quarter_province", "quarter", "province_code"),
    )


class AlertRecord(Base):
    """
    Stores early warning alerts.
    Requires human admin confirmation before publication.
    confirmed=False until DSWD admin explicitly confirms.
    """
    __tablename__ = "alert_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    quarter = Column(String, nullable=False)
    province_code = Column(String, nullable=False)
    threshold_exceeded = Column(Boolean, nullable=False, default=False)
    confirmed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_alert_quarter_province", "quarter", "province_code"),
    )