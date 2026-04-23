from pydantic import BaseModel


class SHAPFeature(BaseModel):
    """Represents one SHAP feature value for one province in one quarter."""
    quarter: str
    province_code: str
    feature_name: str
    shap_value: float
    mean_abs_shap: float

    model_config = {"from_attributes": True}


class SHAPResponse(BaseModel):
    """Response for GET /v1/shap/{province_code}"""
    province_code: str
    quarter: str
    features: list[SHAPFeature]

    model_config = {"from_attributes": True}