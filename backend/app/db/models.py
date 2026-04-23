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
    """
    __tablename__ = "forecast_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    quarter = Column(String, nullable=False)        # format: "2025-Q4"
    province_code = Column(String, nullable=False)  # PSGC code e.g. "PH-BTG"
    province_name = Column(String, nullable=False)  # e.g. "Batangas"
    risk_probability = Column(Float, nullable=False) # 0.0 to 1.0
    risk_label = Column(String, nullable=False)      # "HIGH" or "LOW"
    data_sufficiency_flag = Column(String, nullable=True)  # "LIMITED_SIGNAL" or None
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_forecast_quarter_province", "quarter", "province_code"),
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