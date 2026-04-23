from pydantic import BaseModel
from datetime import datetime


class ForecastRecord(BaseModel):
    """Represents one province forecast for one quarter."""
    quarter: str                        # format: "2025-Q4"
    province_code: str                  # PSGC code e.g. "PH-BTG"
    province_name: str                  # e.g. "Batangas"
    risk_probability: float             # 0.0 to 1.0
    risk_label: str                     # "HIGH" or "LOW"
    data_sufficiency_flag: str | None   # "LIMITED_SIGNAL" or None

    model_config = {"from_attributes": True}


class ForecastLatestResponse(BaseModel):
    """Response for GET /v1/forecasts/latest"""
    data: list[ForecastRecord]
    generated_at: datetime | None = None

    model_config = {"from_attributes": True}


class ForecastHistoryResponse(BaseModel):
    """Response for GET /v1/forecasts/history"""
    province_code: str
    data: list[ForecastRecord]

    model_config = {"from_attributes": True}


class MunicipalForecastRecord(BaseModel):
    """Represents one municipal disaggregated forecast."""
    quarter: str
    province_code: str
    municipality_name: str
    risk_index: float
    disaggregation_label: str = "Spatially disaggregated estimate derived from province-level forecast. Municipal uncertainty is higher than province uncertainty."

    model_config = {"from_attributes": True}


class MunicipalForecastResponse(BaseModel):
    """Response for GET /v1/forecasts/municipal"""
    province_code: str
    quarter: str
    data: list[MunicipalForecastRecord]

    model_config = {"from_attributes": True}