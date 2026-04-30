import pytest


class TestForecastEndpoints:
    """
    Tests for forecast API endpoints.
    Stubs — will be completed in W14.
    """

    async def test_get_latest_forecasts_returns_200(self, client):
        """GET /v1/forecasts/latest should return 200 OK"""
        # TODO W14: complete this test
        pass

    async def test_get_latest_forecasts_returns_list(self, client):
        """GET /v1/forecasts/latest should return a list of 5 provinces"""
        # TODO W14: complete this test
        pass

    async def test_get_forecast_history_returns_200(self, client):
        """GET /v1/forecasts/history should return 200 OK"""
        # TODO W14: complete this test
        pass

    async def test_get_municipal_forecasts_returns_200(self, client):
        """GET /v1/forecasts/municipal should return 200 OK"""
        # TODO W14: complete this test
        pass

    async def test_municipal_response_has_disaggregation_label(self, client):
        """Every municipal row must include disaggregation_label"""
        # TODO W14: complete this test
        pass

    async def test_invalid_province_returns_404(self, client):
        """Invalid province should return 404"""
        # TODO W14: complete this test
        pass