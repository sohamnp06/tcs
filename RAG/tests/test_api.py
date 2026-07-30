"""
Integration tests for the Resume Upload API endpoint.

Design Decision:
    - Uses FastAPI TestClient (no network required).
    - PDFs are created in-memory using PyMuPDF so tests run without
      depending on fixture files.
    - ChromaDB and the embedding model ARE loaded in these tests —
      they are integration tests, not mocked unit tests.
    - A separate test chroma_db directory is used to avoid polluting
      the development database.
"""

import os
import io
import tempfile

import fitz
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")
# Use a temp chroma directory for tests
os.environ["CHROMA_PERSIST_DIRECTORY"] = "./chroma_db_test"

from app.core.config import get_settings
from app.main import create_app

get_settings.cache_clear()


SAMPLE_RESUME_TEXT = """John Doe
john.doe@email.com | +1-555-0199

Summary
Experienced Python developer with expertise in FastAPI and machine learning.

Experience
Senior Engineer — TechCorp (2020–Present)
Developed scalable REST APIs using FastAPI and Python.
Managed PostgreSQL databases and Redis caching.

Education
B.Tech Computer Science — State University (2016–2020)

Skills
Python, FastAPI, PostgreSQL, Docker, Redis, Machine Learning

Projects
Resume Parser: Built an NLP pipeline for extracting structured data from resumes.

Certifications
AWS Certified Developer (2022)
"""


def _make_pdf_bytes(text: str) -> bytes:
    """Create a PDF in memory from the given text using BytesIO."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """TestClient with a fresh app instance."""
    app = create_app()
    return TestClient(app)


@pytest.fixture(scope="module")
def valid_pdf_bytes():
    """Valid PDF bytes for upload tests."""
    return _make_pdf_bytes(SAMPLE_RESUME_TEXT)


class TestResumeUpload:
    """Integration tests for POST /api/v1/resume/upload."""

    def test_upload_valid_pdf_returns_201(self, client, valid_pdf_bytes):
        """Valid PDF upload should return HTTP 201."""
        response = client.post(
            "/api/v1/resume/upload",
            files={"file": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201

    def test_upload_response_has_resume_id(self, client, valid_pdf_bytes):
        """Response should include a non-empty resume_id."""
        response = client.post(
            "/api/v1/resume/upload",
            files={"file": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert data["success"] is True
        assert "resume_id" in data["data"]
        assert data["data"]["resume_id"]

    def test_upload_response_has_chunks_stored(self, client, valid_pdf_bytes):
        """Response should report at least 1 chunk stored."""
        response = client.post(
            "/api/v1/resume/upload",
            files={"file": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert data["data"]["chunks_stored"] >= 1

    def test_upload_response_has_char_count(self, client, valid_pdf_bytes):
        """Response should report a positive char count."""
        response = client.post(
            "/api/v1/resume/upload",
            files={"file": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert data["data"]["char_count"] > 0

    def test_upload_wrong_type_returns_415(self, client):
        """Non-PDF file should return HTTP 415."""
        response = client.post(
            "/api/v1/resume/upload",
            files={"file": ("resume.docx", b"fake content", "application/octet-stream")},
        )
        assert response.status_code == 415

    def test_upload_corrupt_pdf_returns_422(self, client):
        """Corrupt PDF bytes should return HTTP 422."""
        response = client.post(
            "/api/v1/resume/upload",
            files={"file": ("resume.pdf", b"not a real pdf", "application/pdf")},
        )
        assert response.status_code == 422


class TestRetrievalEndpoint:
    """Integration tests for GET /api/v1/retrieve."""

    @pytest.fixture(autouse=True)
    def ensure_data(self, client, valid_pdf_bytes):
        """Upload a resume before retrieval tests."""
        client.post(
            "/api/v1/resume/upload",
            files={"file": ("resume.pdf", valid_pdf_bytes, "application/pdf")},
        )

    def test_retrieve_returns_200(self, client):
        """Retrieval query should return HTTP 200."""
        response = client.get("/api/v1/retrieve", params={"query": "Python FastAPI developer"})
        assert response.status_code == 200

    def test_retrieve_response_shape(self, client):
        """Response should have success, message, and data fields."""
        response = client.get("/api/v1/retrieve", params={"query": "machine learning skills"})
        data = response.json()
        assert "success" in data
        assert "data" in data
        assert "results" in data["data"]
        assert "total_results" in data["data"]

    def test_retrieve_results_have_scores(self, client):
        """All returned results should have a similarity score."""
        response = client.get("/api/v1/retrieve", params={"query": "Python engineer"})
        results = response.json()["data"]["results"]
        for result in results:
            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0

    def test_retrieve_results_have_text(self, client):
        """All returned results should have non-empty text."""
        response = client.get("/api/v1/retrieve", params={"query": "education degree"})
        results = response.json()["data"]["results"]
        for result in results:
            assert result["text"].strip()

    def test_retrieve_short_query_rejected(self, client):
        """Query shorter than 3 chars should be rejected."""
        response = client.get("/api/v1/retrieve", params={"query": "ab"})
        assert response.status_code == 422

    def test_retrieve_top_k_respected(self, client):
        """top_k parameter should limit results."""
        response = client.get(
            "/api/v1/retrieve",
            params={"query": "software engineer", "top_k": 2},
        )
        results = response.json()["data"]["results"]
        assert len(results) <= 2
