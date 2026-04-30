import pytest


class TestHealthEndpoints:
    """Tests for GET /v1/health and GET /v1/readiness"""

    async def test_health_returns_200(self, client):
        response = await client.get("/v1/health")
        assert response.status_code == 200

    async def test_health_returns_correct_json(self, client):
        response = await client.get("/v1/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    async def test_readiness_returns_200(self, client):
        response = await client.get("/v1/readiness")
        assert response.status_code == 200

    async def test_readiness_returns_correct_json(self, client):
        response = await client.get("/v1/readiness")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ready"