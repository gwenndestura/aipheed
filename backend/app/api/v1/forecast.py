"""
app/api/v1/forecast.py
-----------------------
The score behind the gauge, the HIGH/LOW pill, the map colouring and the
trend chart.

GET /api/v1/forecast?scope=&id=&quarter=
GET /api/v1/forecast/timeseries?scope=&id=&from=&to=

Asking for a quarter the model cannot score returns 404 forecast_not_found
rather than a zero. A grey province reads as "no forecast published"; a 0.00
reads as "no risk", and those are opposite claims.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas.public import ForecastResponse, TimeseriesResponse
from app.services import dashboard as svc
from app.services import reference as ref
from app.services import review as rv

router = APIRouter()

SCOPES = ("region", "province", "municipality")


async def _withheld_for(db: AsyncSession, quarter: str) -> frozenset[str]:
    """Province slugs an admin has withheld for this quarter."""
    return frozenset(pid for pid, q in await rv.rejected_pairs(db, quarter))


def _quarter_label(quarter: str) -> str:
    year, qn = ref.quarter_parts(quarter)
    return f"Q{qn} {year} ({ref.QUARTER_MONTHS[qn]})"


def _subject_name(scope: str, subject_id: str) -> str:
    if scope == "region":
        return ref.REGION_NAME
    if scope == "province":
        name = ref.province_name(subject_id)
        if name is None:
            raise svc.SubjectNotFound(f"Unknown province id '{subject_id}'.")
        return name
    row = ref.municipality_row(subject_id)
    if row is None:
        raise svc.SubjectNotFound(f"Unknown municipality id '{subject_id}'.")
    return row["lgu_name"]


def _score_for(
    scope: str,
    subject_id: str,
    quarter: str,
    withheld: frozenset[str] = frozenset(),
    horizon: str = svc.DEFAULT_HORIZON,
) -> dict:
    """
    One subject's score for one quarter, in the shape the gauge needs.

    A withheld province -- or a municipality inside one -- raises rather than
    returning a number, so the score cannot reach the client through a caller
    that forgot to cross-check the rejection list.
    """
    if scope == "region":
        region = svc.region_forecast(quarter, withheld, horizon)
        return {
            "riskScore": region["riskScore"],
            "riskLevel": region["riskLevel"],
            "qoqChange": region["qoqChange"],
            "limitedSignal": region["limitedSignal"],
            "provinceCounts": region["provinceCounts"],
            "provincesIncluded": region["provincesIncluded"],
            "horizon": horizon,
            "derivation": region["derivation"],
        }

    if scope == "province":
        slug = subject_id.lower()
        if slug in withheld:
            raise svc.ForecastWithheld(
                f"The {quarter} forecast for this province has been withheld from "
                "publication by a reviewer."
            )
        match = [p for p in svc.province_summary(quarter, horizon=horizon) if p["id"] == slug]
        if not match:
            raise svc.SubjectNotFound(f"Unknown province id '{subject_id}'.")
        p = match[0]
        return {
            "riskScore": p["riskScore"],
            "riskLevel": p["riskLevel"],
            "qoqChange": p["qoqChange"],
            "limitedSignal": p["limitedSignal"],
            "seriesMonitored": p["seriesMonitored"],
            "seriesAtRisk": p["seriesAtRisk"],
            "topAtRiskCommodities": p["topAtRiskCommodities"],
            "horizon": horizon,
        }

    m = svc.municipality_forecast(subject_id, quarter)
    if m["provinceId"] in withheld:
        raise svc.ForecastWithheld(
            f"The {quarter} forecast for {m['name']} is derived from its province, "
            "which a reviewer has withheld from publication."
        )
    return {
        "riskScore": m["riskIndex"],
        "riskLevel": m["riskLevel"],
        "qoqChange": None,
        "limitedSignal": m["limitedSignal"],
        "provinceId": m["provinceId"],
        "disaggregationLabel": m["disaggregationLabel"],
    }


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    scope: str = Query("region", pattern="^(region|province|municipality)$"),
    id: str = Query("calabarzon", description="calabarzon | quezon | quezon-infanta"),
    quarter: str | None = Query(None, description="Defaults to the latest scorable quarter"),
    horizon: str = Query(svc.DEFAULT_HORIZON, pattern="^(nowcast|forecast)$",
                         description="nowcast = what happened; forecast = one quarter ahead"),
    db: AsyncSession = Depends(get_db),
) -> ForecastResponse:
    """
    The single most-used endpoint: one subject's score for one quarter.

    `horizon=forecast` reaches one quarter further than the nowcast, because
    its inputs are all taken from before the quarter being scored.
    """
    h = svc.check_horizon(horizon)
    q = svc.resolve_quarter(quarter, h)
    body = _score_for(scope, id, q, await _withheld_for(db, q), h)
    return ForecastResponse(
        scope=scope,
        id=id.lower(),
        name=_subject_name(scope, id),
        quarter=q,
        quarterLabel=_quarter_label(q),
        indicator=svc.INDICATOR,
        # Nothing is ever a forecast quarter: the model scores quarters whose
        # production volumes PSA has already published.
        isForecast=False,
        isCurrent=(q == svc.latest_quarter(h)),
        alert=(body["riskScore"] or 0) >= ref.ALERT_THRESHOLD,
        **body,
    )


@router.get("/forecast/timeseries", response_model=TimeseriesResponse)
async def get_timeseries(
    scope: str = Query("province", pattern="^(region|province|municipality)$"),
    id: str = Query(..., description="calabarzon | quezon | quezon-infanta"),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    horizon: str = Query(svc.DEFAULT_HORIZON, pattern="^(nowcast|forecast)$"),
    db: AsyncSession = Depends(get_db),
) -> TimeseriesResponse:
    """
    A score per quarter over a range, for the trend chart.

    A municipality series is the parent province's series under a fixed
    poverty/density weight, so its shape matches the province exactly. It is
    served because the chart offers the choice, not because the model resolves
    municipalities independently.
    """
    h = svc.check_horizon(horizon)
    quarters = svc.available_quarters(h)
    start = from_ or (quarters[0] if quarters else None)
    end = to or (quarters[-1] if quarters else None)
    lo, hi = ref.quarter_index(start), ref.quarter_index(end)
    window = [q for q in quarters if lo <= ref.quarter_index(q) <= hi]

    # Withheld quarters drop out of the line rather than plotting a gap at
    # zero, which would read as a real fall to no risk.
    rejected = await rv.rejected_pairs(db)

    series = []
    for q in window:
        withheld = frozenset(pid for pid, rq in rejected if rq == q)
        try:
            body = _score_for(scope, id, q, withheld, h)
        except svc.SubjectNotFound:
            raise  # bad id: every quarter would fail, so say so once
        except svc.ForecastWithheld:
            continue
        except Exception:  # a quarter the subject has no rows for
            continue
        if body["riskScore"] is None:
            continue
        series.append({
            "quarter": q,
            "riskScore": body["riskScore"],
            "riskLevel": body["riskLevel"],
            "isForecast": False,
        })

    return TimeseriesResponse(
        scope=scope,
        id=id.lower(),
        name=_subject_name(scope, id),
        indicator=svc.INDICATOR,
        series=series,
    )
