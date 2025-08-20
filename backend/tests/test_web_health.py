# test_web_health.py
"""Basic health check test for web API."""

import pytest
from fastapi.testclient import TestClient

from web.app import app


@pytest.fixture
def client():
    """Create a test client."""
    with TestClient(app) as test_client:
        yield test_client


class TestWebHealth:
    """Test cases for web API health endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_database_health_endpoint(self, client):
        """Test database health check endpoint."""
        response = client.get("/health/database")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database_info" in data