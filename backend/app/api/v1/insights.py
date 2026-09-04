"""
app/api/v1/insights.py
-----------------------
Why the score is what it is, and the news corpus behind it.

GET /api/v1/explainability?scope=&id=&quarter=
GET /api/v1/news?scope=&id=&quarter=&page=&pageSize=
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.public import ExplainabilityResponse, NewsResponse
from app.services import dashboard as svc
from app.services import reference as ref

router = APIRouter()

# Seven groups, not five. `seasonal` and `series_history` were added with the
# food-availability target and are usually the largest contributors; a panel
# built for five categories has to grow rather than drop them.
EXPLAINABILITY_NOTE = (
    "Contributions are grouped SHAP values from the food-availability shock "
    "model, rescaled so the seven groups sum to 100%. A group can hold a large "
    "share while pushing risk down -- read `direction`, not bar size alone. "
    "`seasonal` and `series_history` describe where the series sits in its own "
    "annual cycle and how it behaved recently; they are typically the strongest "
    "drivers, which is why the model beats a naive baseline at all. "
    "`color` takes three values, not two: yellow below the 20% line, red above "
    "it while raising the score, GREEN above it while lowering the score. The "
    "original two-colour rule predates signed contributions and would paint a "
    "large protective driver red."
)


def _resolve_province(scope: str, subject_id: str) -> tuple[str | None, str]:
    """Return (province PSGC code or None for region, display name)."""
    if scope == "region":
        return None, ref.REGION_NAME
    if scope == "province":
        code = ref.province_code(subject_id)
        if code is None:
            raise svc.SubjectNotFound(f"Unknown province id '{subject_id}'.")
        return code, ref.province_name(subject_id)
    row = ref.municipality_row(subject_id)
    if row is None:
        raise svc.SubjectNotFound(f"Unknown municipality id '{subject_id}'.")
    return row["province_code"], row["lgu_name"]


@router.get("/explainability", response_model=ExplainabilityResponse)
async def get_explainability(
    scope: str = Query("province", pattern="^(region|province|municipality)$"),
    id: str = Query(..., description="calabarzon | quezon | quezon-infanta"),
    quarter: str | None = Query(None),
) -> ExplainabilityResponse:
    """
    Grouped SHAP for one subject-quarter.

    Region scope averages the five province breakdowns. Municipality scope
    returns its parent province's breakdown: the disaggregation reweights the
    province score without re-explaining it, so a municipality has no
    explanation of its own.
    """
    q = svc.resolve_quarter(quarter)
    code, name = _resolve_province(scope, id)

    if code is None:
        result = svc.explainability_region(q)
        triggers = result["triggers"]
        article_count = result["articleCount"]
        matched = result["triggerMatchedArticles"]
        risk_score = svc.region_forecast(q)["riskScore"]
    else:
        result = svc.explainability(code, q)
        triggers = result["triggers"]
        article_count = result["articleCount"]
        matched = result["triggerMatchedArticles"]
        slug = ref.province_slug(code)
        match = [p for p in svc.province_summary(q) if p["id"] == slug]
        risk_score = match[0]["riskScore"] if match else 0.0

    return ExplainabilityResponse(
        scope=scope,
        id=id.lower(),
        name=name,
        quarter=q,
        riskScore=risk_score,
        triggers=triggers,
        articleCount=article_count,
        triggerMatchedArticles=matched,
        narrative=svc.compose_narrative(name, q, triggers),
        note=EXPLAINABILITY_NOTE,
    )


@router.get("/news", response_model=NewsResponse)
async def get_news(
    scope: str = Query("province", pattern="^(region|province|municipality)$"),
    id: str = Query(..., description="calabarzon | quezon | quezon-infanta"),
    quarter: str | None = Query(None),
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
) -> NewsResponse:
    """
    The analysed article corpus for a subject-quarter, with its topic mix.

    Articles are geocoded to province, so municipality scope returns the parent
    province's corpus.
    """
    q = svc.resolve_quarter(quarter)
    code, _ = _resolve_province(scope, id)
    body = svc.news(code, q, page, pageSize)
    return NewsResponse(
        scope=scope,
        id=id.lower(),
        quarter=q,
        page=page,
        pageSize=pageSize,
        **body,
    )
