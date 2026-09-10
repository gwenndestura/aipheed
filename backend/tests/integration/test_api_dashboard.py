"""
tests/integration/test_api_dashboard.py
-----------------------------------------
Contract tests for the public dashboard API.

These hit the real model artifact through TestClient rather than a fixture, so
they assert the contract the frontend depends on AND that the served numbers
are the model's own. They need no DB and no event loop.

The invariants worth protecting are the ones that would silently mislead a
reader if they broke: percentages that do not sum to 100, an unscorable
quarter answered with a zero instead of a 404, and a protective driver painted
in the alarm colour.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def latest_quarter() -> str:
    return client.get("/api/v1/quarters").json()["current"]


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def test_quarters_reports_only_scorable_quarters(latest_quarter):
    body = client.get("/api/v1/quarters").json()
    assert body["quarters"], "timeline must not be empty"
    # Nothing forward-looking: every quarter has published PSA volumes behind it.
    assert {q["state"] for q in body["quarters"]} <= {"actual", "current"}
    assert body["quarters"][-1]["id"] == latest_quarter
    assert body["quarters"][-1]["state"] == "current"


def test_quarters_declares_its_lag_behind_the_calendar():
    body = client.get("/api/v1/quarters").json()
    # The slider stopping short of today is a data fact, not a bug, so the
    # API states the gap rather than leaving the UI to infer it.
    assert body["lagQuarters"] >= 0
    assert body["calendarQuarter"] != body["current"] or body["lagQuarters"] == 0


# ---------------------------------------------------------------------------
# Missing quarters
# ---------------------------------------------------------------------------

def test_unscorable_quarter_is_404_not_a_zero_score():
    r = client.get("/api/v1/forecast?scope=province&id=quezon&quarter=2027-Q1")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "forecast_not_found"


def test_malformed_quarter_is_rejected():
    r = client.get("/api/v1/forecast?scope=region&id=calabarzon&quarter=not-a-quarter")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "forecast_not_found"


def test_unknown_subject_is_404():
    r = client.get("/api/v1/forecast?scope=province&id=atlantis")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "subject_not_found"


# ---------------------------------------------------------------------------
# Provinces
# ---------------------------------------------------------------------------

def test_provinces_returns_five_ranked_by_risk(latest_quarter):
    body = client.get("/api/v1/provinces").json()
    data = body["data"]
    assert len(data) == 5
    assert [p["riskScore"] for p in data] == sorted(
        (p["riskScore"] for p in data), reverse=True
    )
    assert body["quarter"] == latest_quarter


def test_every_forecast_carries_what_the_number_means():
    body = client.get("/api/v1/forecast?scope=region&id=calabarzon").json()
    indicator = body["indicator"]
    # The score is a production-shortfall share. A surface that relabels it as
    # a household food-insecurity probability is the failure this guards.
    assert indicator["id"] == "food_availability_shock_share"
    assert "availability" in indicator["caveat"].lower()


def test_risk_level_matches_the_published_cutoff():
    cutoff = client.get("/api/v1/config").json()["thresholds"]["riskDisplayCutoff"]
    for p in client.get("/api/v1/provinces").json()["data"]:
        expected = "high" if p["riskScore"] >= cutoff else "low"
        assert p["riskLevel"] == expected


def test_config_distinguishes_active_from_defined_risk_bands():
    body = client.get("/api/v1/config").json()
    assert set(body["riskLevelsActive"]) <= set(body["riskLevelsDefined"])
    # moderate/severe exist in the frontend type but the model never emits them.
    assert set(body["riskLevelsActive"]) == {"low", "high"}


def test_accuracy_is_served_with_its_baselines():
    perf = client.get("/api/v1/config").json()["modelPerformance"]
    # Accuracy alone overstates the result: the majority class already scores
    # ~0.73. The baselines must travel with the headline number.
    assert perf["baselines"]["majorityClass"] > 0
    assert perf["baselines"]["seasonalPersistence"] > 0
    assert perf["skill"]["vsMajority"] == pytest.approx(
        perf["accuracy"] - perf["baselines"]["majorityClass"], abs=1e-3
    )


# ---------------------------------------------------------------------------
# Municipalities
# ---------------------------------------------------------------------------

def test_municipalities_cover_the_full_roster():
    total = sum(
        len(client.get(f"/api/v1/provinces/{slug}/municipalities").json()["data"])
        for slug in ("cavite", "laguna", "quezon", "rizal", "batangas")
    )
    assert total == 142


def test_every_municipal_row_declares_it_is_disaggregated():
    data = client.get("/api/v1/provinces/quezon/municipalities").json()["data"]
    assert data
    for row in data:
        # Rule 8: the label is never null. These are reweighted province
        # numbers and a reader must not mistake one for a municipal model run.
        assert row["disaggregationLabel"]
        assert "disaggregated" in row["disaggregationLabel"].lower()


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scope,subject", [
    ("province", "quezon"),
    ("province", "laguna"),
    ("region", "calabarzon"),
])
def test_driver_percentages_sum_to_exactly_100(scope, subject):
    body = client.get(f"/api/v1/explainability?scope={scope}&id={subject}").json()
    assert sum(t["pct"] for t in body["triggers"]) == 100


def test_drivers_are_ranked_descending():
    body = client.get("/api/v1/explainability?scope=province&id=quezon").json()
    pcts = [t["pct"] for t in body["triggers"]]
    assert pcts == sorted(pcts, reverse=True)


def test_protective_drivers_are_never_painted_red():
    for slug in ("quezon", "laguna", "cavite", "rizal", "batangas"):
        body = client.get(f"/api/v1/explainability?scope=province&id={slug}").json()
        for t in body["triggers"]:
            if t["direction"] == "protective":
                assert t["color"] != "red", (
                    f"{slug}/{t['key']} lowers the score but is coloured red"
                )
            if t["color"] == "red":
                assert t["direction"] == "increases_risk"


def test_narrative_is_plain_ascii_for_pdf_export():
    body = client.get("/api/v1/explainability?scope=region&id=calabarzon").json()
    # jsPDF renders the string verbatim; a typographic dash breaks the export.
    body["narrative"].encode("ascii")


def test_seasonal_and_history_groups_are_not_dropped():
    body = client.get("/api/v1/explainability?scope=province&id=laguna").json()
    keys = {t["key"] for t in body["triggers"]}
    # These two are usually the largest contributors. Collapsing the response
    # to the frontend's original five categories would hide them.
    assert {"seasonal", "series_history"} <= keys


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def test_news_topic_percentages_sum_to_100_when_articles_exist():
    body = client.get("/api/v1/news?scope=province&id=quezon").json()
    if body["articleCount"]:
        assert sum(t["pct"] for t in body["topics"]) == 100


def test_article_dates_are_iso_or_absent():
    body = client.get("/api/v1/news?scope=province&id=quezon&pageSize=20").json()
    for article in body["data"]:
        if article["date"] is not None:
            # The corpus mixes ISO and RSS timestamps; the API must not leak
            # "Thu, 23 Oct 2025" into a field the UI formats as a date.
            assert len(article["date"]) == 10
            assert article["date"][4] == article["date"][7] == "-"


def test_news_count_agrees_with_the_province_summary():
    quarter = client.get("/api/v1/quarters").json()["current"]
    provinces = {p["id"]: p for p in client.get("/api/v1/provinces").json()["data"]}
    for slug, province in provinces.items():
        news = client.get(f"/api/v1/news?scope=province&id={slug}&quarter={quarter}").json()
        assert news["articleCount"] == province["articleCount"]


# ---------------------------------------------------------------------------
# Timeseries & search
# ---------------------------------------------------------------------------

def test_timeseries_is_chronological_and_never_forecast():
    body = client.get("/api/v1/forecast/timeseries?scope=province&id=quezon").json()
    quarters = [p["quarter"] for p in body["series"]]
    assert quarters == sorted(quarters)
    assert not any(p["isForecast"] for p in body["series"])


def test_search_matches_provinces_and_municipalities():
    hits = client.get("/api/v1/search?q=infan").json()["data"]
    assert any(h["id"] == "quezon-infanta" for h in hits)
    hits = client.get("/api/v1/search?q=quezon").json()["data"]
    assert any(h["type"] == "province" and h["id"] == "quezon" for h in hits)


# ---------------------------------------------------------------------------
# Two horizons
# ---------------------------------------------------------------------------

def test_config_describes_both_models():
    horizons = client.get("/api/v1/config").json()["horizons"]
    assert set(horizons) == {"nowcast", "forecast"}
    for h in horizons.values():
        # Each must carry its own metrics with baselines, not share the other's.
        assert h["performance"]["baselines"]["majorityClass"] > 0
        assert h["question"]


def ref_index(quarter: str) -> int:
    year, q = quarter.split("-Q")
    return int(year) * 4 + int(q) - 1


def next_quarter(quarter: str) -> str:
    i = ref_index(quarter) + 1
    return f"{i // 4}-Q{i % 4 + 1}"


def test_both_horizons_reach_the_same_edge():
    """
    The forecast horizon used to reach one quarter further than the nowcast,
    and until 2026-09-09 it did: the feature matrix was capped at 2025-Q4 by a
    hardcoded MODEL_QUARTERS range, so joining the government features at t-1
    bought the forecast model 2026-Q1 that the nowcast had no features for.

    That cap is gone -- the window now tracks the calendar -- so the binding
    constraint is the PSA production panel itself. Neither horizon can score a
    quarter with no production row to compare against, and the t-1 join no
    longer buys any reach.

    The forecast model is still worth serving, but for the other reason: it
    uses no same-quarter information, so it is the one that could be run before
    a quarter closes.
    """
    now = client.get("/api/v1/quarters?horizon=nowcast").json()
    fut = client.get("/api/v1/quarters?horizon=forecast").json()

    assert ref_index(fut["current"]) == ref_index(now["current"])
    assert fut["lagQuarters"] == now["lagQuarters"]
    # Both are bounded by the same published panel, so they expose the same
    # selectable range rather than the forecast carrying one extra entry.
    assert [q["id"] for q in fut["quarters"]] == [q["id"] for q in now["quarters"]]


def test_neither_horizon_scores_past_the_panel_edge():
    """A quarter with no production row is absent for both horizons, not zero."""
    edge = client.get("/api/v1/quarters?horizon=nowcast").json()["current"]
    beyond = next_quarter(edge)

    for horizon in ("nowcast", "forecast"):
        ok = client.get(
            f"/api/v1/forecast?scope=province&id=batangas&quarter={edge}&horizon={horizon}"
        )
        assert ok.status_code == 200, horizon
        assert ok.json()["riskScore"] is not None, horizon

        gone = client.get(
            f"/api/v1/forecast?scope=province&id=batangas&quarter={beyond}&horizon={horizon}"
        )
        assert gone.status_code == 404, horizon
        assert gone.json()["error"]["code"] == "forecast_not_found", horizon


def test_responses_declare_which_horizon_produced_them():
    for h in ("nowcast", "forecast"):
        body = client.get(f"/api/v1/forecast?scope=region&id=calabarzon&horizon={h}").json()
        assert body["horizon"] == h
        provinces = client.get(f"/api/v1/provinces?horizon={h}").json()
        assert provinces["horizon"] == h
        assert all(p["horizon"] == h for p in provinces["data"])


def test_forecast_timeline_marks_its_edge_as_forecast():
    quarters = client.get("/api/v1/quarters?horizon=forecast").json()["quarters"]
    assert quarters[-1]["state"] == "forecast"
    assert {q["state"] for q in quarters[:-1]} == {"actual"}


def test_unknown_horizon_is_rejected():
    r = client.get("/api/v1/forecast?scope=region&id=calabarzon&horizon=crystalball")
    assert r.status_code == 422
