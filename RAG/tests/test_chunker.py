"""
Tests for the semantic chunker.

Design Decision:
    - Chunker tests use plain text strings (no file I/O needed).
    - Tests verify both section detection and fallback behavior.
"""

import os

import pytest

os.environ.setdefault("ENVIRONMENT", "development")

from app.core.exceptions import ChunkingException
from app.embeddings.chunker import chunk_resume_text, _detect_section, _is_header_line
from app.models.resume import ChunkSection


SAMPLE_RESUME = """John Doe
john@example.com | +1-555-0100

Summary
Experienced software engineer with 5 years in backend development.

Experience
Senior Software Engineer — Acme Corp (2021–Present)
Built microservices using Python and FastAPI.
Led a team of 4 engineers.

Software Engineer — Beta Inc (2019–2021)
Developed REST APIs and maintained PostgreSQL databases.

Education
B.Tech Computer Science — State University (2015–2019)
GPA: 3.8/4.0

Skills
Python, FastAPI, PostgreSQL, ChromaDB, Docker, Kubernetes

Projects
Resume RAG System
Built a semantic retrieval system for resume analysis.

Certifications
AWS Certified Developer — Associate (2022)
"""


class TestIsHeaderLine:
    """Tests for the header line detection heuristic."""

    def test_detects_section_headers(self):
        """Known section headers should be detected."""
        assert _is_header_line("Experience") is True
        assert _is_header_line("Education") is True
        assert _is_header_line("Skills") is True
        assert _is_header_line("Certifications") is True

    def test_rejects_long_lines(self):
        """Lines longer than 60 chars should not be headers."""
        long_line = "This is a very long line that should not be detected as a section header"
        assert _is_header_line(long_line) is False

    def test_rejects_bullet_points(self):
        """Lines with mid-string periods are not headers."""
        assert _is_header_line("Built REST APIs. Led the team.") is False

    def test_rejects_empty(self):
        """Empty string should not be a header."""
        assert _is_header_line("") is False


class TestDetectSection:
    """Tests for section type detection from header strings."""

    def test_experience_detection(self):
        assert _detect_section("Experience") == ChunkSection.EXPERIENCE
        assert _detect_section("Work History") == ChunkSection.EXPERIENCE

    def test_education_detection(self):
        assert _detect_section("Education") == ChunkSection.EDUCATION
        assert _detect_section("Academic Background") == ChunkSection.EDUCATION

    def test_skills_detection(self):
        assert _detect_section("Skills") == ChunkSection.SKILLS
        assert _detect_section("Technical Skills") == ChunkSection.SKILLS

    def test_projects_detection(self):
        assert _detect_section("Projects") == ChunkSection.PROJECTS

    def test_certifications_detection(self):
        assert _detect_section("Certifications") == ChunkSection.CERTIFICATIONS

    def test_summary_detection(self):
        assert _detect_section("Summary") == ChunkSection.SUMMARY
        assert _detect_section("Profile") == ChunkSection.SUMMARY

    def test_unknown_returns_general(self):
        assert _detect_section("Hobbies") == ChunkSection.GENERAL


class TestChunkResumeText:
    """Tests for the main chunking function."""

    def test_produces_chunks(self):
        """Should return at least one chunk for a valid resume."""
        chunks = chunk_resume_text(SAMPLE_RESUME, resume_id="test-resume")
        assert len(chunks) > 0

    def test_chunk_ids_are_unique(self):
        """All chunk IDs should be unique."""
        chunks = chunk_resume_text(SAMPLE_RESUME, resume_id="test-resume")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_resume_id_set(self):
        """All chunks should have the correct resume_id."""
        chunks = chunk_resume_text(SAMPLE_RESUME, resume_id="resume-abc123")
        for chunk in chunks:
            assert chunk.resume_id == "resume-abc123"

    def test_chunk_text_non_empty(self):
        """All chunks should have non-empty text."""
        chunks = chunk_resume_text(SAMPLE_RESUME, resume_id="test")
        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.char_count > 0

    def test_detects_known_sections(self):
        """Should detect at least some named sections from a structured resume."""
        chunks = chunk_resume_text(SAMPLE_RESUME, resume_id="test")
        sections = {c.section for c in chunks}
        # At minimum, Experience or Skills or Education should be detected
        assert len(sections) > 1

    def test_char_count_matches_text(self):
        """char_count field should match the actual text length."""
        chunks = chunk_resume_text(SAMPLE_RESUME, resume_id="test")
        for chunk in chunks:
            assert chunk.char_count == len(chunk.text)

    def test_empty_text_raises(self):
        """Empty text should raise ChunkingException."""
        with pytest.raises(ChunkingException):
            chunk_resume_text("", resume_id="test")

    def test_whitespace_text_raises(self):
        """Whitespace-only text should raise ChunkingException."""
        with pytest.raises(ChunkingException):
            chunk_resume_text("   \n\n   ", resume_id="test")

    def test_no_headers_produces_general_chunk(self):
        """Text with no detectable headers should produce a GENERAL chunk."""
        plain_text = "John Doe is a software engineer with ten years of experience."
        chunks = chunk_resume_text(plain_text, resume_id="test")
        assert len(chunks) >= 1
        assert any(c.section == ChunkSection.GENERAL for c in chunks)
