"""
app/api/v1/dashboard.py
------------------------
Catalog endpoints: display config, the timeline, and the geography the map
draws attribute data onto.

GET /api/v1/config
GET /api/v1/quarters
GET /api/v1/provinces
GET /api/v1/provinces/{province_id}/municipalities
GET /api/v1/search
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas.public import (
    ConfigResponse,
    MunicipalitiesResponse,
    ProvincesResponse,
    QuartersResponse,
    SearchResponse,
)
from app.services import dashboard as svc
from app.services import reference as ref
from app.services import review as rv

router = APIRouter()

PST = timezone(timedelta(hours=8))

# The frontend type declares four risk bands; the model produces two. Both are
# reported so the unused pair reads as a deliberate gap, not an oversight.
RISK_LEVELS_DEFINED = ["low", "moderate", "high", "severe"]


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Thresholds, active risk bands, what the score means, and both models.

    `horizons` describes the nowcast and the one-quarter-ahead forecast side by
    side -- what each answers, the quarters it reaches, and its metrics with
    their baselines -- so the UI can offer the choice without hardcoding either.
    """
    quarters = svc.available_quarters()
    horizons = {}
    for h in svc.HORIZONS:
        hq = svc.available_quarters(h)
        horizons[h] = {
            **svc.HORIZON_INFO[h],
            "earliest": hq[0] if hq else None,
            "latest": hq[-1] if hq else None,
            "count": len(hq),
            "performance": svc.model_performance(h),
        }
    return ConfigResponse(
        horizons=horizons,
        region=ref.REGION_ID,
        thresholds={
            "riskDisplayCutoff": ref.RISK_DISPLAY_CUTOFF,
            "alertThreshold": ref.ALERT_THRESHOLD,
            "triggerRedCutoff": ref.TRIGGER_RED_CUTOFF,
            "limitedSignalMinArticles": ref.LIMITED_SIGNAL_MIN_ARTICLES,
        },
        riskLevelsActive=ref.ACTIVE_RISK_LEVELS,
        riskLevelsDefined=RISK_LEVELS_DEFINED,
        indicator=svc.INDICATOR,
        quarterRange={
            "earliest": quarters[0] if quarters else None,
            "latest": quarters[-1] if quarters else None,
            "count": len(quarters),
            "forecastQuarters": [],
            "note": (
                "Every scorable quarter is an observed one. The model scores "
                "quarters PSA has already published production volumes for, so "
                "there are no forward forecast quarters to display."
            ),
        },
        modelPerformance=svc.model_performance(),
    )


@router.get("/quarters", response_model=QuartersResponse)
async def get_quarters(
    horizon: str = Query(svc.DEFAULT_HORIZON, pattern="^(nowcast|forecast)$"),
) -> QuartersResponse:
    """
    The timeline the slider should render.

    `current` is the newest scorable quarter, which trails the calendar by the
    PSA publication lag. `lagQuarters` reports that gap so the UI can say why
    the slider stops where it does instead of looking broken.
    """
    h = svc.check_horizon(horizon)
    now = datetime.now(PST)
    calendar_quarter = f"{now.year}-Q{(now.month - 1) // 3 + 1}"
    current = svc.latest_quarter(h)
    lag = ref.quarter_index(calendar_quarter) - ref.quarter_index(current)
    return QuartersResponse(
        current=current,
        horizon=h,
        serverTime=now.isoformat(),
        calendarQuarter=calendar_quarter,
        lagQuarters=max(lag, 0),
        quarters=svc.timeline(h),
    )


@router.get("/provinces", response_model=ProvincesResponse)
async def get_provinces(
    quarter: str | None = Query(None, description="Defaults to the latest scorable quarter"),
    horizon: str = Query(svc.DEFAULT_HORIZON, pattern="^(nowcast|forecast)$"),
    db: AsyncSession = Depends(get_db),
) -> ProvincesResponse:
    """
    The five provinces with headline stats, ranked by risk descending.

    A province a reviewer has withheld still appears -- the map needs its
    geography -- but with `withheld: true` and no score.
    """
    h = svc.check_horizon(horizon)
    q = svc.resolve_quarter(quarter, h)
    withheld = frozenset(pid for pid, rq in await rv.rejected_pairs(db, q))
    return ProvincesResponse(
        quarter=q,
        horizon=h,
        indicator=svc.INDICATOR,
        data=svc.province_summary(q, withheld, h),
    )


@router.get(
    "/provinces/{province_id}/municipalities",
    response_model=MunicipalitiesResponse,
)
async def get_municipalities(
    province_id: str,
    quarter: str | None = Query(None),
) -> MunicipalitiesResponse:
    """
    The province's cities and municipalities, ranked by risk index.

    Each row is the province score reweighted by poverty (60%) and population
    density (40%), not a model run of its own.
    """
    slug = province_id.lower()
    if slug not in ref.PROVINCES:
        raise svc.SubjectNotFound(f"Unknown province id '{province_id}'.")
    q = svc.resolve_quarter(quarter)
    return MunicipalitiesResponse(
        provinceId=slug,
        quarter=q,
        indicator=svc.INDICATOR,
        data=svc.municipality_summary(q, province_id=slug),
    )


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Substring of a province or LGU name"),
    limit: int = Query(10, ge=1, le=50),
) -> SearchResponse:
    """Substring match over the five provinces and 142 LGUs."""
    needle = q.strip().lower()
    hits = [
        {"type": "province", "id": slug, "name": meta["name"]}
        for slug, meta in ref.PROVINCES.items()
        if needle in meta["name"].lower()
    ]
    roster = ref.lgu_roster()
    for _, r in roster[roster["lgu_name"].str.lower().str.contains(needle, regex=False)].iterrows():
        hits.append({
            "type": "municipality",
            "id": r["id"],
            "name": r["lgu_name"],
            "provinceId": r["provinceId"],
            "provinceName": r["province_name"],
        })
    return SearchResponse(data=hits[:limit])
