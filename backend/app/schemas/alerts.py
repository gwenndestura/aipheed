from pydantic import BaseModel
from datetime import datetime


class AlertRecord(BaseModel):
    """Represents one early warning alert."""
    id: int
    quarter: str
    province_code: str
    threshold_exceeded: bool
    confirmed: bool
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class AlertResponse(BaseModel):
    """Response for GET /v1/alerts/active"""
    data: list[AlertRecord]
    total: int

    model_config = {"from_attributes": True}