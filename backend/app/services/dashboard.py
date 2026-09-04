"""
app/services/dashboard.py
--------------------------
Read model output and shape it for the dashboard.

Everything the frontend renders resolves here. The rules this module enforces:

* A quarter is scorable only if the model panel carries it. The calendar does
  not decide; PSA's publication lag does. Quarters outside that range raise
  QuarterNotAvailable rather than returning a zero-filled forecast, so the map
  greys out instead of showing a confident-looking 0.00.
* riskScore is the share of a province's monitored commodity series predicted
  to fall more than 10% below their own seasonal baseline. It is a food
  AVAILABILITY measure, not a household food-insecurity probability, and every
  response says so in `indicator`.
* Driver percentages are integers summing to exactly 100 (largest remainder).

Scoring a quarter costs a model pass and SHAP costs a TreeExplainer pass, so
both are memoised per quarter for the life of the process.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import pandas as pd

from app.ml.inference.disaggregator import Disaggregator
from app.ml.inference.explainer import Explainer
from app.ml.inference.predictor import Predictor
from app.services import reference as ref

logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/processed/corpus_geocoded.parquet")
LGU_POVERTY_PATH = Path("data/processed/lgu_poverty.parquet")
RESULTS_PATH = Path("data/processed/final_results.json")

# What the number on screen actually is. Carried on every forecast response so
# no surface can relabel it on its own.
INDICATOR = {
    "id": "food_availability_shock_share",
    "name": "Food Availability Shock Share",
    "definition": (
        "Share of the province's monitored PSA commodity series predicted to "
        "fall more than 10% below that series' own expanding seasonal baseline."
    ),
    "unit": "share of monitored series (0-1)",
    "horizon": "same-quarter nowcast",
    "source": "PSA OpenStat production volumes, CALABARZON, 2021-2026",
    "caveat": (
        "This is a food availability signal, not a household food-insecurity "
        "probability. It measures production shortfall against seasonal "
        "expectation for crops, vegetables and fisheries."
    ),
}


# The corpus labels topics with two vocabularies: the NLI hypothesis sentence
# the classifier scored against, and a handful of bare slugs from an earlier
# pass. Neither belongs on screen, and the two overlap, so both fold into one
# set of stable keys here.
TOPIC_MAP: dict[str, tuple[str, str]] = {
    "This article is about food prices, food supply problems, or difficulty accessing food":
        ("food_prices", "Food prices & supply"),
    "food_price_change": ("food_prices", "Food prices & supply"),
    "This article is about farmland loss, crop damage, or reduced harvests":
        ("crop_damage", "Crop damage & harvest loss"),
    "This article is about government food assistance, rice subsidies, or relief distribution":
        ("food_assistance", "Food assistance & relief"),
    "This article is about hunger, malnutrition, or nutrition and feeding programs":
        ("hunger_nutrition", "Hunger & nutrition"),
    "food_insecurity_general": ("hunger_nutrition", "Hunger & nutrition"),
    "This article is about overseas Filipino workers or remittances supporting families":
        ("ofw", "OFW & remittances"),
    "This article is about fish kills, fishing bans, or aquaculture losses":
        ("fishkill", "Fish kill & fisheries"),
    "This article is about evacuation or displacement of families due to disaster":
        ("displacement", "Disaster displacement"),
    "disaster_displacement": ("displacement", "Disaster displacement"),
    "This article is about strikes, protests, or unrest disrupting food or livelihoods":
        ("unrest", "Strikes & unrest"),
    "This article is about roads, transport, or storage problems affecting food supply":
        ("logistics", "Transport & storage"),
    "This article is about poverty, unemployment, or economic hardship of families":
        ("poverty", "Poverty & unemployment"),
    "poverty_hardship": ("poverty", "Poverty & unemployment"),
}

UNCLASSIFIED = ("unclassified", "Unclassified")


class QuarterNotAvailable(Exception):
    """Raised when a quarter is outside the range the model can score."""


class SubjectNotFound(Exception):
    """Raised when a province or municipality id does not resolve."""


class ForecastWithheld(Exception):
    """
    Raised when an admin has rejected this province-quarter.

    Deliberately distinct from QuarterNotAvailable: one forecast was never
    generated, the other was generated and held back. The map shows different
    text for each, and conflating them would misreport an editorial decision
    as a data gap.
    """

    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

# Two models are served. They answer different questions and reach different
# quarters, so the horizon is a first-class parameter rather than a setting.
NOWCAST = "nowcast"
FORECAST = "forecast"
HORIZONS = (NOWCAST, FORECAST)
DEFAULT_HORIZON = NOWCAST

HORIZON_INFO = {
    NOWCAST: {
        "id": NOWCAST,
        "name": "Same-quarter nowcast",
        "question": "Which commodity series fell below their seasonal baseline last quarter?",
        "definition": (
            "Scored using the same quarter's CPI, typhoon count, ENSO phase and "
            "news volume, so it cannot run ahead of published data."
        ),
        "resultsFile": "final_results.json",
    },
    FORECAST: {
        "id": FORECAST,
        "name": "One-quarter-ahead forecast",
        "question": "Which commodity series should be watched next quarter?",
        "definition": (
            "Every input is taken from before the quarter being scored, so it "
            "reaches one quarter further than the nowcast. Slightly less "
            "accurate, and it does not beat a seasonal-persistence rule on "
            "binary accuracy -- its value is the forward ranking, which "
            "persistence cannot express."
        ),
        "resultsFile": "forecast_results.json",
    },
}


def _engine(horizon: str):
    if horizon == FORECAST:
        from app.ml.inference.forecaster import Forecaster
        return Forecaster()
    return Predictor()


def check_horizon(horizon: str | None) -> str:
    h = (horizon or DEFAULT_HORIZON).lower()
    if h not in HORIZONS:
        raise SubjectNotFound(
            f"Unknown horizon '{horizon}'. Expected one of {', '.join(HORIZONS)}."
        )
    return h


@functools.lru_cache(maxsize=4)
def available_quarters(horizon: str = DEFAULT_HORIZON) -> list[str]:
    return _engine(horizon).available_quarters()


def latest_quarter(horizon: str = DEFAULT_HORIZON) -> str:
    quarters = available_quarters(horizon)
    if not quarters:
        raise QuarterNotAvailable("The model panel is empty; no quarter is scorable.")
    return quarters[-1]


def resolve_quarter(quarter: str | None, horizon: str = DEFAULT_HORIZON) -> str:
    """Default to the latest scorable quarter; reject anything unscorable."""
    if quarter is None:
        return latest_quarter(horizon)
    if not ref.is_valid_quarter(quarter):
        raise QuarterNotAvailable(f"'{quarter}' is not a quarter id (expected YYYY-Qn).")
    if quarter not in available_quarters(horizon):
        raise QuarterNotAvailable(
            f"No {horizon} exists for {quarter}. The {horizon} model can score "
            f"{available_quarters(horizon)[0]} through {latest_quarter(horizon)}."
        )
    return quarter


def timeline(horizon: str = DEFAULT_HORIZON) -> list[dict]:
    """
    Every scorable quarter for this horizon, oldest first.

    For the nowcast every entry is `actual` -- it scores quarters PSA has
    already published volumes for. The forecast horizon reaches one quarter
    further, and that newest quarter is marked `forecast` because its inputs
    all predate it.
    """
    quarters = available_quarters(horizon)
    latest = quarters[-1] if quarters else None
    out = []
    for q in quarters:
        year, qn = ref.quarter_parts(q)
        if q == latest:
            state = "forecast" if horizon == FORECAST else "current"
        else:
            state = "actual"
        out.append({
            "id": q,
            "year": year,
            "q": qn,
            "label": f"Q{qn}",
            "monthsLabel": ref.QUARTER_MONTHS[qn],
            "state": state,
        })
    return out


# ---------------------------------------------------------------------------
# Province-level forecasts
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _province_forecasts(quarter: str, horizon: str = DEFAULT_HORIZON) -> list[dict]:
    """Raw model output for one quarter, from whichever engine the horizon names."""
    return _engine(horizon).forecast_quarter(quarter)


def province_forecast_map(quarter: str, horizon: str = DEFAULT_HORIZON) -> dict[str, dict]:
    return {r["province_code"]: r for r in _province_forecasts(quarter, horizon)}


def _previous_quarter(quarter: str, horizon: str = DEFAULT_HORIZON) -> str | None:
    quarters = available_quarters(horizon)
    i = quarters.index(quarter) if quarter in quarters else -1
    return quarters[i - 1] if i > 0 else None


def _qoq(current: float, quarter: str, code: str,
         horizon: str = DEFAULT_HORIZON) -> tuple[float | None, float | None]:
    """Absolute and percentage change against the previous scorable quarter."""
    prev_q = _previous_quarter(quarter, horizon)
    if prev_q is None:
        return None, None
    prev = province_forecast_map(prev_q, horizon).get(code)
    if prev is None:
        return None, None
    prev_score = prev["risk_probability"]
    delta = round(current - prev_score, 4)
    pct = round((current - prev_score) / prev_score * 100, 1) if prev_score else None
    return delta, pct


@functools.lru_cache(maxsize=1)
def _province_poverty() -> dict[str, float]:
    """Province-mean PSA 2021 poverty incidence, from the LGU table."""
    if not LGU_POVERTY_PATH.exists():
        return {}
    df = pd.read_parquet(LGU_POVERTY_PATH)
    return df.groupby("province_code")["poverty_incidence_pct"].mean().round(1).to_dict()


@functools.lru_cache(maxsize=1)
def _province_population() -> dict[str, int]:
    roster = ref.lgu_roster()
    return roster.groupby("province_code")["population_2020"].sum().astype(int).to_dict()


def province_summary(
    quarter: str,
    withheld: frozenset[str] = frozenset(),
    horizon: str = DEFAULT_HORIZON,
) -> list[dict]:
    """
    One headline row per province, ranked by risk descending.

    A province in `withheld` keeps its geography and population but its score
    is blanked. Returning the number alongside a "rejected" flag and trusting
    every caller to hide it would defeat the rejection -- and with five
    provinces and a published regional mean, one leaked score is enough to
    reconstruct a withheld one.
    """
    poverty = _province_poverty()
    population = _province_population()
    rows = []
    for record in _province_forecasts(quarter, horizon):
        code = record["province_code"]
        slug = ref.province_slug(code)
        if slug is None:
            continue
        score = record["risk_probability"]
        delta, pct = _qoq(score, quarter, code, horizon)
        meta = ref.PROVINCES[slug]
        rows.append({
            "id": slug,
            "name": meta["name"],
            "provinceCode": code,
            "centroid": {"lat": meta["lat"], "lng": meta["lng"]},
            "population": population.get(code),
            "povertyRate": poverty.get(code),
            "currentQuarter": quarter,
            "riskScore": score,
            "riskLevel": ref.risk_level(score),
            "qoqChange": delta,
            "qoqChangePct": pct,
            "seriesMonitored": record.get("series_monitored"),
            "seriesAtRisk": record.get("series_at_risk"),
            "topAtRiskCommodities": record.get("top_at_risk_commodities", []),
            "articleCount": article_count(code, quarter),
            "limitedSignal": record.get("data_sufficiency_flag") == "LIMITED_SIGNAL",
            "horizon": horizon,
            "withheld": False,
        })

    for row in rows:
        if row["id"] in withheld:
            row.update({
                "riskScore": None,
                "riskLevel": None,
                "qoqChange": None,
                "qoqChangePct": None,
                "seriesAtRisk": None,
                "topAtRiskCommodities": [],
                "withheld": True,
            })

    rows.sort(key=lambda r: (r["riskScore"] is not None, r["riskScore"] or 0), reverse=True)
    return rows


def region_forecast(quarter: str, withheld: frozenset[str] = frozenset(),
                    horizon: str = DEFAULT_HORIZON) -> dict:
    """
    CALABARZON roll-up: the mean of the published province scores.

    The model has no region-level series of its own, so this is an average of
    province output rather than a separate prediction.

    Withheld provinces are excluded from the mean, not counted as zero. With
    five provinces, a mean over all five plus four published scores gives the
    fifth exactly -- so a rejection has to shrink the denominator, and the
    response says how many provinces it covers.
    """
    provinces = province_summary(quarter, withheld, horizon)
    published = [p for p in provinces if not p["withheld"]]
    scores = [p["riskScore"] for p in published]
    score = round(sum(scores) / len(scores), 4) if scores else None

    delta = None
    prev_q = _previous_quarter(quarter, horizon)
    if prev_q and score is not None:
        # Compare like with like: the same provinces, one quarter earlier.
        prev_scores = [
            p["riskScore"] for p in province_summary(prev_q, withheld, horizon)
            if not p["withheld"]
        ]
        if prev_scores:
            delta = round(score - sum(prev_scores) / len(prev_scores), 4)

    return {
        "scope": "region",
        "id": ref.REGION_ID,
        "name": ref.REGION_NAME,
        "riskScore": score,
        "riskLevel": ref.risk_level(score) if score is not None else None,
        "qoqChange": delta,
        "provinceCounts": {
            "high": sum(1 for p in published if p["riskLevel"] == "high"),
            "low": sum(1 for p in published if p["riskLevel"] == "low"),
            "limited": sum(1 for p in published if p["limitedSignal"]),
            "withheld": len(provinces) - len(published),
        },
        "provincesIncluded": len(published),
        "horizon": horizon,
        "limitedSignal": all(p["limitedSignal"] for p in published) if published else True,
        "derivation": (
            f"Mean of {len(published)} published province scores; the model has "
            "no region-level series."
        ),
    }


# ---------------------------------------------------------------------------
# Municipal disaggregation
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _municipal_frame(quarter: str) -> pd.DataFrame:
    forecasts = _province_forecasts(quarter)
    return Disaggregator().disaggregate(forecasts, quarter=quarter)


@functools.lru_cache(maxsize=1)
def _lgu_poverty_map() -> dict[str, float]:
    if not LGU_POVERTY_PATH.exists():
        return {}
    df = pd.read_parquet(LGU_POVERTY_PATH)
    return {
        str(c): float(v)
        for c, v in zip(df["lgu_code"], df["poverty_incidence_pct"])
    }


def municipality_summary(quarter: str, province_id: str | None = None) -> list[dict]:
    """
    The 142 LGU indices for a quarter, optionally filtered to one province.

    These are reweighted province numbers, not independent municipal
    forecasts; disaggregationLabel travels with every row and says so.
    """
    frame = _municipal_frame(quarter)
    roster = ref.lgu_roster().set_index("lgu_psgc")
    poverty = _lgu_poverty_map()

    rows = []
    for _, r in frame.iterrows():
        code = str(r["lgu_code"])
        if code not in roster.index:
            continue
        meta = roster.loc[code]
        if province_id and meta["provinceId"] != province_id:
            continue
        score = round(float(r["risk_index"]), 4)
        rows.append({
            "id": meta["id"],
            "name": meta["lgu_name"],
            "provinceId": meta["provinceId"],
            "lguCode": code,
            "classification": meta["classification"],
            "population": int(meta["population_2020"]),
            "densityPerKm2": float(meta["densityPerKm2"]),
            "povertyRate": poverty.get(code),
            "quarter": quarter,
            "riskIndex": score,
            "riskLevel": ref.risk_level(score),
            "disaggregationLabel": r["disaggregation_label"],
            "limitedSignal": r.get("data_sufficiency_flag") == "LIMITED_SIGNAL",
        })
    rows.sort(key=lambda x: x["riskIndex"], reverse=True)
    return rows


def municipality_forecast(municipality_id: str, quarter: str) -> dict:
    match = [
        m for m in municipality_summary(quarter)
        if m["id"] == municipality_id.lower()
    ]
    if not match:
        raise SubjectNotFound(f"Unknown municipality id '{municipality_id}'.")
    return match[0]


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def _integer_shares(values: list[float]) -> list[int]:
    """
    Largest-remainder rounding so the percentages sum to exactly 100.

    Naive rounding drifts to 99 or 101 and the UI renders a bar chart that
    does not close.
    """
    total = sum(values)
    if total <= 0:
        return [0] * len(values)
    exact = [v / total * 100 for v in values]
    floors = [int(x) for x in exact]
    remainder = 100 - sum(floors)
    order = sorted(range(len(exact)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


@functools.lru_cache(maxsize=256)
def _drivers(province_code: str, quarter: str) -> dict:
    explainer = Explainer()
    records = explainer.explain_province_quarter(province_code, quarter)
    return explainer.build_drivers(province_code, quarter, records)


def trigger_color(pct: int, direction: str) -> str:
    """
    Bar colour for a driver group.

    The frontend contract froze this as "pct > 20 => red", which was written
    when every contribution was assumed to push risk up. Grouped SHAP is
    signed, so that rule paints a large PROTECTIVE driver red -- the reader
    sees the biggest bar in the alarm colour while it is holding the score
    down. Size decides whether a group is prominent; direction decides which
    prominent colour it gets.
    """
    if pct <= ref.TRIGGER_RED_CUTOFF:
        return "yellow"
    return "red" if direction == "increases_risk" else "green"


def explainability(province_code: str, quarter: str) -> dict:
    """
    Grouped SHAP for one province-quarter, ranked and normalised to 100%.

    All seven groups are returned, including `seasonal` and `series_history`.
    Those two are usually the largest, and dropping them to fit a
    five-category panel would hide the model's strongest drivers.
    """
    built = _drivers(province_code, quarter)
    drivers = built["drivers"]
    pcts = _integer_shares([abs(d["group_shap"]) for d in drivers])

    triggers = []
    for d, pct in zip(drivers, pcts):
        triggers.append({
            "key": d["driver_group"],
            "label": d["driver_label"],
            "share": round(pct / 100, 4),
            "pct": pct,
            "signedContribution": d["group_shap"],
            "direction": d["direction"],
            "color": trigger_color(pct, d["direction"]),
            "newsSignalProportion": d["trigger_proportion"],
        })
    triggers.sort(key=lambda t: t["pct"], reverse=True)
    return {
        "quarter": quarter,
        "triggers": triggers,
        # The corpus is the single article count every surface quotes. The
        # trigger table keeps its own tally over a narrower keyword-matched
        # subset; it is reported separately rather than silently substituted.
        "articleCount": article_count(province_code, quarter),
        "triggerMatchedArticles": built["article_count"],
    }


def explainability_region(quarter: str) -> dict:
    """
    Regional breakdown: the five province explanations pooled.

    Groups are ranked by summed absolute contribution across provinces, while
    `signedContribution` averages so the direction still reads correctly -- a
    group that raises risk in one province and lowers it in another should not
    disappear from the ranking just because the two cancel.
    """
    per_province = [
        explainability(ref.province_code(slug), quarter) for slug in ref.PROVINCES
    ]

    totals: dict[str, dict] = {}
    for result in per_province:
        for t in result["triggers"]:
            acc = totals.setdefault(t["key"], {
                "key": t["key"], "label": t["label"],
                "signed": 0.0, "raw": 0.0, "news": [],
            })
            acc["raw"] += abs(t["signedContribution"])
            acc["signed"] += t["signedContribution"]
            if t["newsSignalProportion"] is not None:
                acc["news"].append(t["newsSignalProportion"])

    keys = list(totals)
    pcts = _integer_shares([totals[k]["raw"] for k in keys])

    triggers = []
    for key, pct in zip(keys, pcts):
        acc = totals[key]
        direction = "increases_risk" if acc["signed"] >= 0 else "protective"
        triggers.append({
            "key": key,
            "label": acc["label"],
            "share": round(pct / 100, 4),
            "pct": pct,
            "signedContribution": round(acc["signed"] / len(per_province), 4),
            "direction": direction,
            "color": trigger_color(pct, direction),
            "newsSignalProportion": (
                round(sum(acc["news"]) / len(acc["news"]), 4) if acc["news"] else None
            ),
        })
    triggers.sort(key=lambda t: t["pct"], reverse=True)

    return {
        "quarter": quarter,
        "triggers": triggers,
        "articleCount": sum(r["articleCount"] for r in per_province),
        "triggerMatchedArticles": sum(r["triggerMatchedArticles"] for r in per_province),
    }


def compose_narrative(name: str, quarter: str, triggers: list[dict]) -> str:
    """
    Plain-ASCII summary of a breakdown, for the chart caption and the PDF.

    ASCII only: it is drawn verbatim into PDFs by both jsPDF on the client and
    reportlab here, and a typographic dash breaks the export.
    """
    if not triggers:
        return f"No explainability is available for {name} in {quarter}."

    ranked = ", ".join(f"{t['label']} {t['pct']}%" for t in triggers[:5])
    top = triggers[0]
    direction = (
        "pushing the score up" if top["direction"] == "increases_risk"
        else "holding the score down"
    )

    cutoff = ref.TRIGGER_RED_CUTOFF
    raising = [t["label"] for t in triggers
               if t["pct"] > cutoff and t["direction"] == "increases_risk"]
    lowering = [t["label"] for t in triggers
                if t["pct"] > cutoff and t["direction"] == "protective"]

    def phrase(labels: list[str]) -> str:
        joined = (
            " and ".join([", ".join(labels[:-1]), labels[-1]])
            if len(labels) > 1 else labels[0]
        )
        return f"{joined} {'clears' if len(labels) == 1 else 'clear'}"

    parts = []
    if raising:
        parts.append(f"{phrase(raising)} the {cutoff}% line while raising the score.")
    if lowering:
        parts.append(f"{phrase(lowering)} the {cutoff}% line while lowering it.")
    flagged = " ".join(parts) or f"No single group clears the {cutoff}% line this quarter."

    return (
        f"For {name} in {quarter}, the model's prediction breaks down as: {ranked}. "
        f"{top['label']} is the largest single contributor, {direction}. {flagged} "
        f"Shares are SHAP contributions to a production-shortfall prediction, "
        f"recomputed every quarter."
    )


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def model_performance(horizon: str = DEFAULT_HORIZON) -> dict:
    """
    Headline metrics with the baselines they must be quoted against.

    Accuracy alone is misleading here: the majority class already scores 0.73
    and seasonal persistence 0.81, so the skill margins travel with the number
    wherever it is displayed.
    """
    path = Path("data/processed") / HORIZON_INFO[horizon]["resultsFile"]
    if not path.exists():
        return {}
    results = json.loads(path.read_text())
    op = results.get("operating", {})
    return {
        "horizon": horizon,
        "evaluation": "operating set (>=12 quarters history, 90% coverage, mature folds)",
        "n": op.get("n"),
        "accuracy": round(op.get("accuracy", 0), 4),
        "f1": round(op.get("f1", 0), 4),
        "rocAuc": round(op.get("roc_auc", 0), 4),
        "baselines": {
            "majorityClass": round(op.get("majority_class", 0), 4),
            "seasonalPersistence": round(op.get("seasonal_persistence", 0), 4),
            "naivePersistence": round(op.get("naive_persistence", 0), 4),
        },
        "skill": {
            "vsMajority": round(op.get("skill_vs_majority", 0), 4),
            "vsSeasonal": round(op.get("skill_vs_seasonal", 0), 4),
            "vsNaive": round(op.get("skill_vs_naive", 0), 4),
        },
        "abstainPct": results.get("operating_spec", {}).get("abstain_pct"),
    }


# ---------------------------------------------------------------------------
# News corpus
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _corpus() -> pd.DataFrame:
    """
    Relevant, geocoded articles with display fields normalised once.

    `published` arrives as a mix of ISO dates and raw RSS timestamps, and the
    topic column carries classifier hypothesis sentences. Both are cleaned here
    so no route has to reformat them.
    """
    if not CORPUS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(CORPUS_PATH)
    if "is_relevant" in df.columns:
        df = df[df["is_relevant"].astype("boolean").fillna(False)].copy()

    published = pd.to_datetime(df.get("published"), errors="coerce", utc=True, format="mixed")
    df["_date"] = published.dt.strftime("%Y-%m-%d")
    df["_sort_date"] = published

    topics = df.get("top_topic_name")
    mapped = topics.map(lambda t: TOPIC_MAP.get(t, UNCLASSIFIED)) if topics is not None else None
    df["_topic_key"] = [m[0] for m in mapped] if mapped is not None else UNCLASSIFIED[0]
    df["_topic_label"] = [m[1] for m in mapped] if mapped is not None else UNCLASSIFIED[1]
    return df


def article_count(province_code: str | None, quarter: str) -> int:
    df = _corpus()
    if df.empty:
        return 0
    mask = df["quarter"] == quarter
    if province_code:
        mask &= df["province_code"] == province_code
    return int(mask.sum())


def news(province_code: str | None, quarter: str, page: int, page_size: int) -> dict:
    """
    Analysed articles for a subject-quarter, with the topic mix.

    `provinceAttributed` is reported alongside the total because most of the
    corpus is not pinned to a province: for region scope the two differ sharply,
    and without it the regional count looks irreconcilable with the five
    province counts that are supposed to add up to it.
    """
    df = _corpus()
    if df.empty:
        return {"articleCount": 0, "provinceAttributed": 0,
                "topics": [], "data": [], "total": 0}

    subset = df[df["quarter"] == quarter]
    if province_code:
        subset = subset[subset["province_code"] == province_code]

    province_codes = {meta["code"] for meta in ref.PROVINCES.values()}
    attributed = int(subset["province_code"].isin(province_codes).sum())

    topics = []
    if not subset.empty:
        counts = subset["_topic_label"].value_counts()
        pcts = _integer_shares(counts.tolist())
        key_of = {label: key for key, label in TOPIC_MAP.values()}
        key_of[UNCLASSIFIED[1]] = UNCLASSIFIED[0]
        topics = [
            {"key": key_of.get(label, UNCLASSIFIED[0]), "label": label,
             "count": int(v), "pct": p}
            for (label, v), p in zip(counts.items(), pcts)
        ]

    window = subset.sort_values("_sort_date", ascending=False, na_position="last")
    start = (page - 1) * page_size
    window = window.iloc[start:start + page_size]
    articles = [
        {
            "id": str(r.get("article_id") or ""),
            "title": str(r.get("title") or ""),
            "source": str(r.get("source_domain") or ""),
            "date": r["_date"] if pd.notna(r["_date"]) else None,
            "url": str(r.get("link") or ""),
            "excerpt": str(r.get("summary") or "")[:400],
            "topicKey": r["_topic_key"],
            "topicLabel": r["_topic_label"],
            "relevanceScore": (
                round(float(r["food_insecurity_score"]), 4)
                if pd.notna(r.get("food_insecurity_score")) else None
            ),
        }
        for _, r in window.iterrows()
    ]
    return {
        "articleCount": int(len(subset)),
        "provinceAttributed": attributed,
        "topics": topics,
        "data": articles,
        "total": int(len(subset)),
    }
