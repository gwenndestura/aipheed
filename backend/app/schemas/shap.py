"""
app/schemas/shap.py
--------------------
Pydantic schemas for the SHAP explainability API.

Dashboard panels served by these schemas:
  - Feature Waterfall   → SHAPResponse.features  (sorted by |shap_value|)
  - Risk Drivers panel  → DriversResponse.drivers (grouped SHAP by category)

SHAPResponse (GET /v1/shap/{province_code}?quarter=2025-Q4)
-------------------------------------------------------------
  baseline     — SHAP expected value (mean training probability = 0.54)
                 This is the "By Baseline" number shown in the waterfall.
  final_rfii   — baseline + sum(all shap_values) ≈ model's predicted
                 risk_probability. Used for the "FINAL RFII" waterfall total.
  features     — list of SHAPFeature, sorted by |shap_value| descending.
                 Top 6–8 are shown in the waterfall; all 45 are stored.

DriversResponse (GET /v1/shap/{province_code}/drivers?quarter=2025-Q4)
----------------------------------------------------------------------
  drivers      — list of DriverRecord, one per driver group (5 total).
                 Drives the left-panel "RISK DRIVERS" ranked bar chart.
  article_count — total NLP news articles analysed for this province-quarter.
"""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# SHAP feature-level schema
# ---------------------------------------------------------------------------

class SHAPFeature(BaseModel):
    """
    One SHAP feature contribution for a province-quarter prediction.

    Stored in shap_records table.  All 45 features are stored; the dashboard
    displays the top N by |shap_value|.
    """

    quarter: str
    province_code: str

    # Raw feature identifier (e.g. "food_cpi_yoy")
    feature_name: str

    # Human-readable label shown on the dashboard waterfall
    # e.g. "Food CPI YoY Growth"
    display_name: str

    # Driver group this feature belongs to
    # one of: market | climate | employment | macro_ofw | nlp
    feature_group: str

    # Probability-space SHAP contribution for THIS province-quarter.
    # Positive → pushes risk up.  Negative → suppresses risk.
    # Sum of all shap_values + baseline ≈ predicted risk_probability.
    shap_value: float

    # Mean |shap_value| across the training set for this feature —
    # used for global feature importance ranking.
    mean_abs_shap: float

    # Actual input value fed to the model for this province-quarter
    # (e.g. food_cpi_yoy = 1.2 means +1.2 % YoY food inflation).
    # Shown in tooltip / detail view alongside the SHAP bar.
    feature_value: float | None = None

    # Display unit for feature_value  (e.g. "% change", "PHP/kg")
    unit: str | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# SHAP response envelope
# ---------------------------------------------------------------------------

class SHAPResponse(BaseModel):
    """
    Response for GET /v1/shap/{province_code}?quarter=...

    Contains everything the Feature Waterfall panel needs.
    """

    province_code: str
    quarter: str

    # Mean training probability — the waterfall "By Baseline" anchor.
    baseline: float

    # baseline + sum(all shap_values).  Equals the model's predicted
    # risk_probability.  Shown as "FINAL RFII" at the bottom of the waterfall.
    final_rfii: float

    # All 45 features sorted by |shap_value| descending.
    # Slice [:6] or [:8] for the compact waterfall; use all for the
    # "Advanced / all features" expanded view.
    features: list[SHAPFeature]

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Risk Drivers schema
# ---------------------------------------------------------------------------

class DriverRecord(BaseModel):
    """
    Aggregated SHAP contribution for one driver group (e.g. "market").

    Displayed as one bar in the left-panel RISK DRIVERS ranked list.

    group_shap is the SUM of shap_values for every feature belonging to
    the group (see DRIVER_GROUP_FEATURES in feature_display.py).

    direction is derived from sign(group_shap):
      "increases_risk"  — positive group_shap  (shown in red)
      "protective"      — negative group_shap   (shown in yellow, labelled PROTECTIVE)

    trigger_proportion is the fraction of NLP news articles for this
    province-quarter that triggered the matching keyword category.
    It represents the "news signal strength" for that driver — useful
    context alongside the SHAP contribution.

    display_pct is abs(group_shap) expressed as a percentage of the
    total absolute SHAP mass, used for the bar width.
    """

    # Raw group key: market | climate | employment | macro_ofw | nlp_sentiment
    driver_group: str

    # Human-readable label shown on the dashboard (e.g. "Food Prices")
    driver_label: str

    # Net SHAP contribution of the group (probability scale, e.g. +0.35)
    group_shap: float

    # "increases_risk" or "protective"
    direction: str

    # Bar width percentage (abs(group_shap) / total_abs_shap * 100)
    display_pct: float

    # NLP news signal proportion (0.0–1.0); None if corpus is empty
    trigger_proportion: float | None = None

    model_config = {"from_attributes": True}


class DriversResponse(BaseModel):
    """
    Response for GET /v1/shap/{province_code}/drivers?quarter=...

    Drives the Risk Drivers panel (ranked bars) and the Trigger
    Composition stacked bar.
    """

    province_code: str
    quarter: str

    # 5 driver groups, sorted by abs(group_shap) descending
    drivers: list[DriverRecord]

    # Number of NLP articles analysed for this province-quarter
    article_count: int

    model_config = {"from_attributes": True}
