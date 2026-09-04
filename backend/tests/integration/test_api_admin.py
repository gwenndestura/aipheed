"""
tests/integration/test_api_admin.py
-------------------------------------
Contract tests for auth, the review pipeline and feedback.

Each test class gets its own temporary SQLite file through a dependency
override, so nothing here touches the real aipheed.db. These need no event-loop
plugin: TestClient drives the async app, and the one piece of setup that must
run outside a request (CREATE TABLE) uses a synchronous engine on the same file.

The invariants worth protecting are the security ones. A test that only proved
"login returns a token" would pass just as happily against the console this
replaces, where any password worked.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
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

# A complete SUS response and the score it must produce.
#   positive items (0,2,4,6,8): 3+4+3+4+3 = 17
#   negative items (1,3,5,7,9): 3+3+3+4+3 = 16
#   (17+16) * 2.5 = 82.5
FULL_ANSWERS = {"0": 4, "1": 2, "2": 5, "3": 2, "4": 4, "5": 2, "6": 5, "7": 1, "8": 4, "9": 2}
EXPECTED_SUS = 82.5


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A client bound to a throwaway database with a usable signing key."""
    db_file = tmp_path / "test.db"

    # DDL through a sync engine: it runs outside any request's event loop.
    sync_engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    # NullPool: TestClient may serve requests on more than one event loop, and
    # a pooled aiosqlite connection cannot cross loops.
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}", poolclass=NullPool)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with Session() as session:
            yield session

    monkeypatch.setattr(settings, "SECRET_KEY", "test-signing-key-not-for-production")
    # Off by default here: the rest of the suite makes far more requests
    # than a real client would, and TestRateLimiting re-enables it.
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        client.session_factory = Session
        yield client

    app.dependency_overrides.clear()


def make_user(api, *, email=EMAIL, password=PASSWORD, role="admin", active=True):
    import asyncio

    async def _create():
        async with api.session_factory() as db:
            db.add(AdminUser(
                email=email,
                full_name="J. Dela Cruz",
                password_hash=security.hash_password(password),
                role=role,
                is_active=active,
            ))
            await db.commit()

    asyncio.run(_create())


def auth_headers(api, email=EMAIL, password=PASSWORD):
    r = api.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuth:

    def test_admin_route_requires_a_token(self, api):
        r = api.get("/api/v1/admin/review")
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "missing_token"
        # Lets a client tell "sign in" apart from "signed in but not allowed".
        assert r.headers.get("WWW-Authenticate") == "Bearer"

    def test_wrong_password_is_rejected(self, api):
        make_user(api)
        r = api.post("/api/v1/auth/login", json={"email": EMAIL, "password": "wrong"})
        assert r.status_code == 401
        # The old Login.tsx accepted any password against a DA address.
        assert r.json()["error"]["code"] == "invalid_credentials"

    def test_unknown_account_is_indistinguishable_from_a_wrong_password(self, api):
        make_user(api)
        wrong = api.post("/api/v1/auth/login", json={"email": EMAIL, "password": "nope"})
        absent = api.post("/api/v1/auth/login",
                          json={"email": "ghost@calabarzon.da.gov.ph", "password": "nope"})
        # Differing messages would let anyone enumerate registered addresses.
        assert wrong.json() == absent.json()
        assert wrong.status_code == absent.status_code == 401

    def test_login_returns_a_working_token(self, api):
        make_user(api)
        r = api.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
        assert r.status_code == 200
        body = r.json()
        assert body["expiresIn"] == settings.ACCESS_TOKEN_TTL_SECONDS
        assert body["user"]["role"] == "admin"
        assert "password" not in r.text.lower() or "passwordHash" not in r.text

        me = api.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
        assert me.status_code == 200
        assert me.json()["email"] == EMAIL

    def test_a_forged_token_is_rejected(self, api):
        make_user(api)
        headers = auth_headers(api)
        tampered = headers["Authorization"][:-4] + "AAAA"
        r = api.get("/api/v1/auth/me", headers={"Authorization": tampered})
        assert r.status_code == 401
        assert r.json()["error"]["code"] in ("invalid_token", "token_expired")

    def test_logout_actually_revokes_the_token(self, api):
        make_user(api)
        headers = auth_headers(api)
        assert api.post("/api/v1/auth/logout", headers=headers).json()["revoked"] is True
        # A JWT is otherwise valid until it expires; clearing the browser copy
        # alone would leave a working admin token in anything that saw it.
        assert api.get("/api/v1/auth/me", headers=headers).status_code == 401
        assert api.get("/api/v1/admin/review", headers=headers).status_code == 401

    def test_deactivated_account_stops_working_immediately(self, api):
        import asyncio
        make_user(api)
        headers = auth_headers(api)
        assert api.get("/api/v1/auth/me", headers=headers).status_code == 200

        async def _deactivate():
            async with api.session_factory() as db:
                user = await db.scalar(select(AdminUser).where(AdminUser.email == EMAIL))
                user.is_active = False
                await db.commit()

        asyncio.run(_deactivate())
        # The account is re-read per request, so this must not wait for expiry.
        r = api.get("/api/v1/auth/me", headers=headers)
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "account_inactive"

    def test_non_admin_role_is_forbidden_not_unauthorised(self, api):
        make_user(api, role="viewer")
        headers = auth_headers(api)
        r = api.get("/api/v1/admin/review", headers=headers)
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "forbidden"

    def test_default_secret_key_refuses_to_sign(self, api, monkeypatch):
        make_user(api)
        monkeypatch.setattr(settings, "SECRET_KEY", "changeme")
        r = api.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
        # With a known key anyone can mint an admin token; failing loudly beats
        # issuing a forgeable one.
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "server_misconfigured"

    def test_outside_domain_cannot_sign_in(self, api):
        make_user(api, email="outsider@gmail.com", password=PASSWORD)
        r = api.post("/api/v1/auth/login",
                     json={"email": "outsider@gmail.com", "password": PASSWORD})
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Review pipeline
# ---------------------------------------------------------------------------

class TestReviewPipeline:

    def test_queue_seeds_from_real_forecasts(self, api):
        make_user(api)
        headers = auth_headers(api)
        r = api.get("/api/v1/admin/review?quarter=2025-Q4", headers=headers)
        assert r.status_code == 200
        data = r.json()["data"]
        assert len(data) == 5
        assert {i["status"] for i in data} == {"Staged"}
        # Scores are the model's, not generated from a seed like the mock console.
        live = {p["id"]: p["riskScore"] for p in api.get("/api/v1/provinces?quarter=2025-Q4").json()["data"]}
        for item in data:
            assert item["riskScore"] == pytest.approx(live[item["provinceId"]])

    def test_seeding_twice_does_not_reset_a_decision(self, api):
        make_user(api)
        headers = auth_headers(api)
        api.get("/api/v1/admin/review?quarter=2025-Q4", headers=headers)
        api.post("/api/v1/admin/review/rv_2025Q4_quezon/approve", headers=headers)

        again = api.get("/api/v1/admin/review?quarter=2025-Q4", headers=headers).json()["data"]
        quezon = next(i for i in again if i["provinceId"] == "quezon")
        assert quezon["status"] == "Approved"
        assert len(again) == 5

    def test_rejection_reason_must_be_recognised(self, api):
        make_user(api)
        headers = auth_headers(api)
        api.get("/api/v1/admin/review?quarter=2025-Q4", headers=headers)
        r = api.post("/api/v1/admin/review/rv_2025Q4_quezon/reject",
                     headers=headers, json={"reason": "just because"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "unknown_reason"

    def test_rejection_records_who_and_why(self, api):
        make_user(api)
        headers = auth_headers(api)
        api.get("/api/v1/admin/review?quarter=2025-Q4", headers=headers)
        r = api.post("/api/v1/admin/review/rv_2025Q4_quezon/reject", headers=headers,
                     json={"reason": "Data quality issue", "notes": "PSA revision pending."})
        assert r.status_code == 200
        item = r.json()
        assert item["status"] == "Rejected"
        assert item["rejectionReason"] == "Data quality issue"
        assert item["updatedByName"] == "J. Dela Cruz"

    def test_undo_on_a_staged_item_is_an_error(self, api):
        make_user(api)
        headers = auth_headers(api)
        api.get("/api/v1/admin/review?quarter=2025-Q4", headers=headers)
        r = api.post("/api/v1/admin/review/rv_2025Q4_quezon/undo", headers=headers)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "already_staged"

    def test_unknown_review_item_is_404(self, api):
        make_user(api)
        headers = auth_headers(api)
        r = api.post("/api/v1/admin/review/rv_nope_nowhere/approve", headers=headers)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "review_not_found"

    def test_audit_trail_captures_the_transition(self, api):
        make_user(api)
        headers = auth_headers(api)
        api.get("/api/v1/admin/review?quarter=2025-Q4", headers=headers)
        api.post("/api/v1/admin/review/rv_2025Q4_quezon/reject", headers=headers,
                 json={"reason": "Model anomaly / outlier prediction"})

        rows = api.get("/api/v1/admin/audit", headers=headers).json()["data"]
        actions = {r["action"] for r in rows}
        assert "login" in actions and "reject" in actions
        reject_row = next(r for r in rows if r["action"] == "reject")
        assert reject_row["actorEmail"] == EMAIL
        assert reject_row["provinceId"] == "quezon"
        assert reject_row["reason"] == "Model anomaly / outlier prediction"


# ---------------------------------------------------------------------------
# What a rejection does to the public API
# ---------------------------------------------------------------------------

class TestRejectionIsEnforced:

    @pytest.fixture
    def rejected(self, api):
        make_user(api)
        headers = auth_headers(api)
        api.get("/api/v1/admin/review?quarter=2025-Q4", headers=headers)
        api.post("/api/v1/admin/review/rv_2025Q4_laguna/reject", headers=headers,
                 json={"reason": "Sensitive context - withhold publication"})
        return headers

    def test_public_rejections_list_needs_no_auth(self, api, rejected):
        r = api.get("/api/v1/rejections?quarter=2025-Q4")
        assert r.status_code == 200
        rows = r.json()["data"]
        assert [row["provinceId"] for row in rows] == ["laguna"]
        assert rows[0]["rejectedByName"] == "J. Dela Cruz"

    def test_withheld_forecast_is_404_with_its_own_code(self, api, rejected):
        r = api.get("/api/v1/forecast?scope=province&id=laguna&quarter=2025-Q4")
        assert r.status_code == 404
        # Distinct from forecast_not_found: this one was generated, then held
        # back. The map says different things about the two.
        assert r.json()["error"]["code"] == "forecast_withheld"

    def test_withheld_province_keeps_its_row_but_loses_its_score(self, api, rejected):
        rows = {p["id"]: p for p in api.get("/api/v1/provinces?quarter=2025-Q4").json()["data"]}
        assert len(rows) == 5, "the map still needs the geography"
        assert rows["laguna"]["withheld"] is True
        assert rows["laguna"]["riskScore"] is None
        assert rows["laguna"]["population"] > 0
        assert rows["quezon"]["riskScore"] is not None

    def test_region_mean_excludes_the_withheld_province(self, api, rejected):
        body = api.get("/api/v1/forecast?scope=region&id=calabarzon&quarter=2025-Q4").json()
        assert body["provincesIncluded"] == 4
        assert body["provinceCounts"]["withheld"] == 1

        published = [
            p["riskScore"] for p in api.get("/api/v1/provinces?quarter=2025-Q4").json()["data"]
            if p["riskScore"] is not None
        ]
        # Averaging over all five would let anyone recover the withheld score
        # from the mean and the four that are published.
        assert body["riskScore"] == pytest.approx(sum(published) / len(published), abs=1e-4)

    def test_municipality_inside_a_withheld_province_is_withheld(self, api, rejected):
        r = api.get("/api/v1/forecast?scope=municipality&id=laguna-san-pablo&quarter=2025-Q4")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "forecast_withheld"

    def test_timeseries_drops_the_quarter_rather_than_plotting_zero(self, api, rejected):
        series = api.get("/api/v1/forecast/timeseries?scope=province&id=laguna").json()["series"]
        quarters = [p["quarter"] for p in series]
        # A zero here would read as a real fall to no risk.
        assert "2025-Q4" not in quarters
        assert "2025-Q3" in quarters

    def test_undo_republishes(self, api, rejected):
        api.post("/api/v1/admin/review/rv_2025Q4_laguna/undo", headers=rejected)
        r = api.get("/api/v1/forecast?scope=province&id=laguna&quarter=2025-Q4")
        assert r.status_code == 200
        assert r.json()["riskScore"] > 0
        assert api.get("/api/v1/rejections?quarter=2025-Q4").json()["data"] == []

    def test_restore_by_province_and_quarter(self, api, rejected):
        r = api.delete("/api/v1/admin/rejections/laguna/2025-Q4", headers=rejected)
        assert r.status_code == 200
        assert r.json()["status"] == "Staged"
        assert api.get("/api/v1/forecast?scope=province&id=laguna&quarter=2025-Q4").status_code == 200

    def test_restoring_something_not_rejected_is_404(self, api, rejected):
        r = api.delete("/api/v1/admin/rejections/quezon/2025-Q4", headers=rejected)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "review_not_found"

    def test_changing_publication_state_requires_admin(self, api, rejected):
        # The rejection exists; an anonymous caller must not be able to lift it.
        assert api.delete("/api/v1/admin/rejections/laguna/2025-Q4").status_code == 401
        assert api.post("/api/v1/admin/review/rv_2025Q4_laguna/undo").status_code == 401
        assert api.get("/api/v1/rejections?quarter=2025-Q4").json()["data"] != []


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class TestFeedback:

    def test_score_is_computed_server_side(self, api):
        r = api.post("/api/v1/feedback", json={"answers": FULL_ANSWERS})
        assert r.status_code == 201
        assert r.json()["score"] == EXPECTED_SUS

    def test_a_client_supplied_score_is_ignored(self, api):
        r = api.post("/api/v1/feedback", json={"answers": FULL_ANSWERS, "score": 100})
        assert r.status_code == 201
        # The number goes into a thesis result; it is never taken on trust.
        assert r.json()["score"] == EXPECTED_SUS

    def test_partial_submissions_are_refused(self, api):
        r = api.post("/api/v1/feedback", json={"answers": {"0": 4, "1": 2}})
        assert r.status_code == 400
        # Defaulting the gaps would quietly invent survey data.
        assert r.json()["error"]["code"] == "incomplete_submission"

    def test_out_of_range_answers_are_refused(self, api):
        answers = dict(FULL_ANSWERS, **{"3": 9})
        r = api.post("/api/v1/feedback", json={"answers": answers})
        assert r.status_code == 400

    def test_submissions_are_visible_across_users(self, api):
        make_user(api)
        headers = auth_headers(api)
        for client_type in ("Government", "Citizen", "Government"):
            api.post("/api/v1/feedback", json={
                "answers": FULL_ANSWERS,
                "demographics": {"clientType": client_type},
            })

        body = api.get("/api/v1/admin/feedback", headers=headers).json()
        # Previously each admin saw only what their own browser had submitted.
        assert body["summary"]["count"] == 3
        assert body["summary"]["byClientType"] == {"Government": 2, "Citizen": 1}
        assert body["summary"]["meanScore"] == EXPECTED_SUS
        assert len(body["questions"]) == 10

    def test_reading_feedback_requires_admin(self, api):
        assert api.get("/api/v1/admin/feedback").status_code == 401

    def test_response_version_is_pinned(self, api):
        r = api.post("/api/v1/feedback", json={"answers": FULL_ANSWERS})
        # Older responses stay interpretable if the wording ever changes.
        assert r.json()["susVersion"] == "sus-v1"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """
    The limiter is disabled for the rest of the suite (the fixture turns it
    off), so these tests enable it explicitly and reset the counters, rather
    than depending on how many requests earlier tests happened to make.
    """

    @pytest.fixture(autouse=True)
    def enabled(self, api, monkeypatch):
        # Depends on `api` so it runs AFTER it. The api fixture turns the
        # limiter off for the rest of the suite; without this ordering that
        # would silently undo the flag these tests need.
        from app.services import ratelimit
        ratelimit.reset()
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
        yield
        ratelimit.reset()

    def test_login_attempts_are_capped(self, api, monkeypatch):
        make_user(api)
        monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_15MIN", 3)

        for _ in range(3):
            r = api.post("/api/v1/auth/login", json={"email": EMAIL, "password": "wrong"})
            assert r.status_code == 401

        blocked = api.post("/api/v1/auth/login", json={"email": EMAIL, "password": "wrong"})
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "rate_limited"
        # A well-behaved client needs to know how long to wait.
        assert int(blocked.headers["Retry-After"]) > 0

    def test_the_cap_counts_successful_logins_too(self, api, monkeypatch):
        make_user(api)
        monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_15MIN", 2)
        assert api.post("/api/v1/auth/login",
                        json={"email": EMAIL, "password": PASSWORD}).status_code == 200
        assert api.post("/api/v1/auth/login",
                        json={"email": EMAIL, "password": PASSWORD}).status_code == 200
        # If only failures counted, an attacker could reset the window with one
        # valid login and keep guessing.
        assert api.post("/api/v1/auth/login",
                        json={"email": EMAIL, "password": PASSWORD}).status_code == 429

    def test_report_generation_is_capped(self, api, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_REPORT_PER_HOUR", 2)
        for _ in range(2):
            assert api.get("/api/v1/report?scope=region&id=calabarzon").status_code == 200
        # Each call costs a SHAP pass per province; this is the DoS surface.
        assert api.get("/api/v1/report?scope=region&id=calabarzon").status_code == 429

    def test_feedback_submission_is_capped(self, api, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_FEEDBACK_PER_HOUR", 2)
        for _ in range(2):
            assert api.post("/api/v1/feedback",
                            json={"answers": FULL_ANSWERS}).status_code == 201
        # Flooding this corrupts a research result, not just a table.
        assert api.post("/api/v1/feedback",
                        json={"answers": FULL_ANSWERS}).status_code == 429

    def test_limits_are_per_client(self, api, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_FEEDBACK_PER_HOUR", 1)
        monkeypatch.setattr(settings, "RATE_LIMIT_TRUST_FORWARDED", True)
        first = {"X-Forwarded-For": "203.0.113.1"}
        second = {"X-Forwarded-For": "203.0.113.2"}
        assert api.post("/api/v1/feedback", json={"answers": FULL_ANSWERS},
                        headers=first).status_code == 201
        assert api.post("/api/v1/feedback", json={"answers": FULL_ANSWERS},
                        headers=first).status_code == 429
        # One caller exhausting its allowance must not lock everyone out.
        assert api.post("/api/v1/feedback", json={"answers": FULL_ANSWERS},
                        headers=second).status_code == 201
