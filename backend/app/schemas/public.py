"""
app/schemas/public.py
----------------------
The public dashboard contract.

These models are camelCase and address subjects by slug ("quezon",
"quezon-infanta") because that is how the frontend addresses them. The
internal schemas in forecasts.py / shap.py stay snake_case and PSGC-keyed for
the DB and ML layers; app/services/reference.py is the only translation point.

Every forecast response carries `indicator`. The number is a food AVAILABILITY
shock share, not a household food-insecurity probability, and the label
travels with the value so no screen can rename it in isolation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class Indicator(BaseModel):
    """What the score on screen actually measures."""
    id: str
    name: str
    definition: str
    unit: str
    horizon: str
    source: str
    caveat: str


class Centroid(BaseModel):
    lat: float
    lng: float


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------------------
# Config & timeline
# ---------------------------------------------------------------------------

class Thresholds(BaseModel):
    riskDisplayCutoff: float
    alertThreshold: float
    triggerRedCutoff: int
    limitedSignalMinArticles: int


class ConfigResponse(BaseModel):
    """
    Server-owned display rules, so the frontend stops hardcoding them.

    `riskLevelsActive` reports the bands the model actually produces. The
    frontend type carries four; only two are ever returned.
    """
    region: str
    thresholds: Thresholds
    riskLevelsActive: list[str]
    riskLevelsDefined: list[str]
    indicator: Indicator
    quarterRange: dict
    # Both served models, keyed by horizon: what each answers, the quarters it
    # reaches, and its metrics with baselines.
    horizons: dict
    modelPerformance: dict


class Quarter(BaseModel):
    id: str
    year: int
    q: int
    label: str
    monthsLabel: str
    state: str  # "actual" | "current"


class QuartersResponse(BaseModel):
    """
    The quarters the model can score. `current` is the newest of those, not the
    calendar quarter — PSA's publication lag, not the clock, sets the edge.

    `horizon` says which model's timeline this is: the forecast horizon reaches
    one quarter further than the nowcast, because its inputs all predate the
    quarter being scored.
    """
    current: str
    horizon: str = "nowcast"
    serverTime: str
    calendarQuarter: str
    lagQuarters: int
    quarters: list[Quarter]


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

class ProvinceSummary(BaseModel):
    id: str
    name: str
    provinceCode: str
    centroid: Centroid
    population: int | None = None
    povertyRate: float | None = None
    currentQuarter: str
    # Null when withheld. The province still appears so the map can draw it,
    # but the number an admin held back is not served alongside a flag.
    riskScore: float | None = None
    riskLevel: str | None = None
    qoqChange: float | None = None
    qoqChangePct: float | None = None
    seriesMonitored: int | None = None
    seriesAtRisk: int | None = None
    topAtRiskCommodities: list[str] = Field(default_factory=list)
    articleCount: int = 0
    limitedSignal: bool = False
    horizon: str = "nowcast"
    withheld: bool = False


class ProvincesResponse(BaseModel):
    quarter: str
    horizon: str = "nowcast"
    indicator: Indicator
    data: list[ProvinceSummary]


class MunicipalitySummary(BaseModel):
    id: str
    name: str
    provinceId: str
    lguCode: str
    classification: str
    population: int
    densityPerKm2: float | None = None
    povertyRate: float | None = None
    quarter: str
    riskIndex: float
    riskLevel: str
    # Never null. These are reweighted province numbers, not independent
    # municipal forecasts, and the label says so on every row.
    disaggregationLabel: str
    limitedSignal: bool = False


class MunicipalitiesResponse(BaseModel):
    provinceId: str
    quarter: str
    indicator: Indicator
    data: list[MunicipalitySummary]


class SearchHit(BaseModel):
    type: str  # "province" | "municipality"
    id: str
    name: str
    provinceId: str | None = None
    provinceName: str | None = None


class SearchResponse(BaseModel):
    data: list[SearchHit]


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class ForecastResponse(BaseModel):
    scope: str  # "region" | "province" | "municipality"
    id: str
    name: str
    quarter: str
    quarterLabel: str
    indicator: Indicator
    # Null only for a region whose every province is withheld; a withheld
    # province or municipality 404s instead.
    riskScore: float | None = None
    riskLevel: str | None = None
    qoqChange: float | None = None
    isForecast: bool = False
    isCurrent: bool = False
    limitedSignal: bool = False
    alert: bool = False
    provinceId: str | None = None
    provinceCounts: dict | None = None
    provincesIncluded: int | None = None
    horizon: str = "nowcast"
    seriesMonitored: int | None = None
    seriesAtRisk: int | None = None
    topAtRiskCommodities: list[str] | None = None
    disaggregationLabel: str | None = None
    derivation: str | None = None


class TimeseriesPoint(BaseModel):
    quarter: str
    riskScore: float
    riskLevel: str
    isForecast: bool = False


class TimeseriesResponse(BaseModel):
    scope: str
    id: str
    name: str
    indicator: Indicator
    series: list[TimeseriesPoint]


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

class Trigger(BaseModel):
    """
    One SHAP driver group.

    `pct` values across the list are integers summing to exactly 100.
    `direction` is "increases_risk" or "protective" — a group can hold a large
    share of the explanation while pushing risk DOWN, so bar size alone does
    not tell the reader the sign.
    """
    key: str
    label: str
    share: float
    pct: int
    signedContribution: float
    direction: str
    color: str
    newsSignalProportion: float | None = None


class ExplainabilityResponse(BaseModel):
    scope: str
    id: str
    name: str
    quarter: str
    riskScore: float
    triggers: list[Trigger]
    articleCount: int
    triggerMatchedArticles: int
    narrative: str
    note: str


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

class Topic(BaseModel):
    key: str
    label: str
    count: int
    pct: int


class Article(BaseModel):
    id: str
    title: str
    source: str
    date: str | None = None
    url: str
    excerpt: str
    topicKey: str
    topicLabel: str
    relevanceScore: float | None = None


class NewsResponse(BaseModel):
    scope: str
    id: str
    quarter: str
    articleCount: int
    # How many of those are geocoded to a CALABARZON province. Most of the
    # corpus is not: at region scope this is far below articleCount, which is
    # why the five province counts do not add up to the regional one.
    provinceAttributed: int = 0
    topics: list[Topic]
    data: list[Article]
    page: int
    pageSize: int
    total: int
