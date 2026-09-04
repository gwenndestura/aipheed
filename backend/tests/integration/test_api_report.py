"""
tests/integration/test_api_report.py
--------------------------------------
Contract tests for the PDF assessments.

These render real documents and read their text back, because the risks here
are about what a printed page claims. A PDF that renders cleanly while calling
the number a food-insecurity probability, or while quietly including a figure
a reviewer withheld, is worse than one that fails to render.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from pdfminer.high_level import extract_text
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings
from app.db.database import Base
from app.db.models import AdminUser
from app.dependencies import get_db
from app.main import app
from app.services import security

EMAIL = "analyst@calabarzon.da.gov.ph"
PASSWORD = "correct-horse-battery-staple"


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("report") / "test.db"

    sync_engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}", poolclass=NullPool)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with Session() as session:
            yield session

    original_key = settings.SECRET_KEY
    settings.SECRET_KEY = "test-signing-key-not-for-production"
    app.dependency_overrides[get_db] = override_get_db

    import asyncio

    async def _seed():
        async with Session() as db:
            db.add(AdminUser(
                email=EMAIL, full_name="J. Dela Cruz",
                password_hash=security.hash_password(PASSWORD),
                role="admin", is_active=True,
            ))
            await db.commit()

    asyncio.run(_seed())

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    settings.SECRET_KEY = original_key


@pytest.fixture(scope="module")
def quarter(api) -> str:
    return api.get("/api/v1/quarters").json()["current"]


def text_of(response) -> str:
    return extract_text(io.BytesIO(response.content))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_region_report_renders(api, quarter):
    r = api.get(f"/api/v1/report?scope=region&id=calabarzon&quarter={quarter}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"
    assert f'filename="aipheed-calabarzon-{quarter}.pdf"' in r.headers["content-disposition"]


@pytest.mark.parametrize("province", ["quezon", "laguna", "cavite", "rizal", "batangas"])
def test_every_province_renders(api, quarter, province):
    r = api.get(f"/api/v1/report?scope=province&id={province}&quarter={quarter}")
    assert r.status_code == 200
    assert r.content[:5] == b"%PDF-"
    assert f'filename="aipheed-{province}-{quarter}.pdf"' in r.headers["content-disposition"]


def test_report_is_not_cached(api, quarter):
    r = api.get(f"/api/v1/report?scope=region&id=calabarzon&quarter={quarter}")
    # Review state can change between requests; a cached copy could republish
    # a figure an admin has just withheld.
    assert "no-store" in r.headers.get("cache-control", "")


def test_municipality_scope_is_rejected(api, quarter):
    r = api.get(f"/api/v1/report?scope=municipality&id=quezon-infanta&quarter={quarter}")
    # Municipal values reweight the province figure rather than resolving
    # independently, so a per-LGU document would imply evidence that does not
    # exist. They appear as a table inside the province report instead.
    assert r.status_code == 422


def test_unknown_province_is_404(api, quarter):
    r = api.get(f"/api/v1/report?scope=province&id=atlantis&quarter={quarter}")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "subject_not_found"


def test_unscorable_quarter_is_404(api):
    r = api.get("/api/v1/report?scope=region&id=calabarzon&quarter=2027-Q1")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "forecast_not_found"


# ---------------------------------------------------------------------------
# What the page actually claims
# ---------------------------------------------------------------------------

def test_report_states_what_the_number_is_not(api, quarter):
    body = text_of(api.get(f"/api/v1/report?scope=province&id=quezon&quarter={quarter}"))
    assert "food availability" in body.lower()
    # The single most important line in the document.
    assert "not a household food-insecurity probability" in body.lower()


def test_report_does_not_recommend_interventions(api, quarter):
    body = text_of(api.get(f"/api/v1/report?scope=region&id=calabarzon&quarter={quarter}")).lower()
    # No validated mapping exists between this indicator and a response
    # protocol. Generated operational advice under a government heading is the
    # worst thing this document could carry.
    for phrase in ("we recommend", "it is recommended", "should distribute",
                   "deploy relief", "recommended action"):
        assert phrase not in body
    assert "reports findings only" in body


def test_report_quotes_accuracy_with_its_baselines(api, quarter):
    body = text_of(api.get(f"/api/v1/report?scope=region&id=calabarzon&quarter={quarter}"))
    assert "Majority-class baseline" in body
    assert "Seasonal-persistence baseline" in body
    # Accuracy alone reads far better than it is.
    assert "must be read against its baselines" in body.replace("\n", " ")


def test_province_report_labels_municipal_figures_as_derived(api, quarter):
    body = text_of(api.get(f"/api/v1/report?scope=province&id=quezon&quarter={quarter}"))
    flat = body.replace("\n", " ")
    assert "not" in flat and "independent municipal forecasts" in flat
    assert "60%" in flat and "40%" in flat


def test_report_declares_it_is_a_nowcast(api, quarter):
    body = text_of(api.get(f"/api/v1/report?scope=region&id=calabarzon&quarter={quarter}"))
    flat = body.replace("\n", " ")
    assert "not a forecast" in flat.lower()


def test_region_report_reconciles_its_article_count(api, quarter):
    """The regional count exceeds the five province counts; the page says why."""
    body = text_of(api.get(f"/api/v1/report?scope=region&id=calabarzon&quarter={quarter}"))
    flat = body.replace("\n", " ")
    news = api.get(f"/api/v1/news?scope=region&id=calabarzon&quarter={quarter}").json()
    if news["articleCount"] > news["provinceAttributed"]:
        assert "located to a specific province" in flat


# ---------------------------------------------------------------------------
# A withheld figure must not reach the page
# ---------------------------------------------------------------------------

class TestWithheld:

    @pytest.fixture
    def rejected(self, api, quarter):
        r = api.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
        headers = {"Authorization": f"Bearer {r.json()['token']}"}
        api.get(f"/api/v1/admin/review?quarter={quarter}", headers=headers)
        api.post(f"/api/v1/admin/review/rv_{quarter.replace('-', '')}_laguna/reject",
                 headers=headers,
                 json={"reason": "Sensitive context - withhold publication"})
        yield headers
        api.post(f"/api/v1/admin/review/rv_{quarter.replace('-', '')}_laguna/undo",
                 headers=headers)

    def test_withheld_province_has_no_report(self, api, quarter, rejected):
        r = api.get(f"/api/v1/report?scope=province&id=laguna&quarter={quarter}")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "forecast_withheld"

    def test_region_report_omits_the_withheld_score(self, api, quarter, rejected):
        r = api.get(f"/api/v1/report?scope=region&id=calabarzon&quarter={quarter}")
        assert r.status_code == 200
        body = text_of(r)

        published = api.get(f"/api/v1/provinces?quarter={quarter}").json()["data"]
        laguna = next(p for p in published if p["id"] == "laguna")
        assert laguna["withheld"] is True

        # The province is named as withheld, but its number never appears.
        assert "Withheld from this assessment" in body.replace("\n", " ")
        assert "0.4730" not in body

    def test_region_report_reports_the_smaller_denominator(self, api, quarter, rejected):
        body = text_of(api.get(f"/api/v1/report?scope=region&id=calabarzon&quarter={quarter}"))
        flat = body.replace("\n", " ")
        assert "4 of 5" in flat
        assert "mean of 4 published province scores" in flat.lower()
