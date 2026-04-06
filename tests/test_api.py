"""Tests for the LangGraph service API endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "langgraph"


class TestAnalyzeEndpoint:
    def test_analyze_missing_query(self, client: TestClient):
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_analyze_with_empty_db(self, client: TestClient):
        """Should gracefully handle when no DB connection is provided."""
        response = client.post(
            "/analyze",
            json={"query": "test query", "session_id": "test"},
        )
        # Should return 200 with error in status (LLM won't have schema)
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test"
        assert data["user_query"] == "test query"
