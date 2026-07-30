"""
Phase 1 Tests — Application startup, configuration, and health endpoint.

Design Decision:
    - TestClient from httpx is used (sync wrapper around async FastAPI).
    - Settings cache is cleared before each test to allow env var overrides.
    - Tests are self-contained with no external service dependencies.
    - No LLM credentials are required for any test in this suite.
"""

import os

import pytest
from fastapi.testclient import TestClient

# Ensure a minimal env is set for testing
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")


from app.core.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the settings LRU cache before each test for isolation."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    """Return a TestClient for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestConfiguration:
    """Tests for the Settings configuration module."""

    def test_settings_loads_successfully(self):
        """Settings should load from environment without raising."""
        settings = get_settings()
        assert settings.app_name is not None
        assert settings.app_name == "Resume RAG Service"

    def test_settings_singleton(self):
        """get_settings() should return the same instance each call."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_max_upload_size_bytes(self):
        """upload_path property should compute bytes correctly."""
        settings = get_settings()
        assert settings.max_upload_size_bytes == settings.max_upload_size_mb * 1024 * 1024

    def test_invalid_log_level_raises(self):
        """Invalid log level should raise validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            from app.core.config import Settings

            Settings(log_level="INVALID")


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client):
        """Health endpoint should return HTTP 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_shape(self, client):
        """Health response must contain status, version, environment."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data

    def test_health_status_is_healthy(self, client):
        """Health endpoint should report 'healthy' status."""
        response = client.get("/api/v1/health")
        assert response.json()["status"] == "healthy"

    def test_health_version_matches_config(self, client):
        """Health endpoint version should match configured app version."""
        settings = get_settings()
        response = client.get("/api/v1/health")
        assert response.json()["version"] == settings.app_version


class TestExceptionHandlers:
    """Tests for global exception handler registration."""

    def test_unknown_route_returns_404(self, client):
        """Unknown routes should return HTTP 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
