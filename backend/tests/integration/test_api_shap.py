import pytest


class TestSHAPEndpoints:
    """
    Tests for SHAP API endpoints.
    Stubs — will be completed in W14.
    """

    async def test_get_shap_returns_200(self, client):
        """GET /v1/shap/{province_code} should return 200 OK"""
        # TODO W14: complete this test
        pass

    async def test_get_shap_returns_features_list(self, client):
        """GET /v1/shap/{province_code} should return a list of features"""
        # TODO W14: complete this test
        pass

    async def test_get_shap_sorted_by_magnitude(self, client):
        """Features should be sorted by mean_abs_shap descending"""
        # TODO W14: complete this test
        pass

    async def test_get_shap_invalid_province_returns_404(self, client):
        """GET /v1/shap/{province_code} with invalid province returns 404"""
        # TODO W14: complete this test
        pass