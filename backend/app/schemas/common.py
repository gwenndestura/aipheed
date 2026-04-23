from pydantic import BaseModel


class HealthOK(BaseModel):
    """Response for GET /v1/health"""
    status: str = "ok"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
    code: int


class Pagination(BaseModel):
    """Reusable pagination info"""
    total: int
    page: int = 1
    per_page: int = 20