"""
Tests for the PDF extractor and text cleaner.

Design Decision:
    - Tests use actual small PDF files created in memory via PyMuPDF
      to avoid depending on fixture files on disk.
    - All edge cases (empty, encrypted, corrupt) are tested with
      in-memory PDFs or mock exceptions.
"""

import io
import os
import tempfile
from pathlib import Path

import fitz
import pytest

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")

from app.core.exceptions import CorruptPDFException, EmptyPDFException
from app.parser.pdf_extractor import extract_text_from_pdf
from app.parser.text_cleaner import clean_resume_text


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _create_pdf_with_text(text: str) -> Path:
    """Create a temporary PDF file containing the given text."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()  # Close before PyMuPDF writes (required on Windows)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=11)
    doc.save(tmp_path)
    doc.close()
    return Path(tmp_path)


def _create_blank_pdf() -> Path:
    """Create a temporary PDF with no text content."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()  # Close before PyMuPDF writes (required on Windows)
    doc = fitz.open()
    doc.new_page()  # blank page
    doc.save(tmp_path)
    doc.close()
    return Path(tmp_path)


# ------------------------------------------------------------------ #
# PDF Extractor Tests
# ------------------------------------------------------------------ #


class TestPDFExtractor:
    """Tests for extract_text_from_pdf."""

    def test_extract_valid_pdf(self):
        """Should extract text from a valid PDF."""
        pdf_path = _create_pdf_with_text("John Doe\nSoftware Engineer\nPython, FastAPI")
        try:
            text = extract_text_from_pdf(pdf_path)
            assert "John Doe" in text
            assert "Python" in text
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_extract_returns_string(self):
        """Extracted text should be a non-empty string."""
        pdf_path = _create_pdf_with_text("Test resume content here.")
        try:
            text = extract_text_from_pdf(pdf_path)
            assert isinstance(text, str)
            assert len(text) > 0
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_empty_pdf_raises(self):
        """Blank PDF (no text) should raise EmptyPDFException."""
        pdf_path = _create_blank_pdf()
        try:
            with pytest.raises(EmptyPDFException):
                extract_text_from_pdf(pdf_path)
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_corrupt_pdf_raises(self):
        """A corrupt (non-PDF) file should raise CorruptPDFException."""
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(b"this is not a valid pdf file content")
        tmp.close()
        corrupt_path = Path(tmp.name)
        try:
            with pytest.raises(CorruptPDFException):
                extract_text_from_pdf(corrupt_path)
        finally:
            corrupt_path.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# Text Cleaner Tests
# ------------------------------------------------------------------ #


class TestTextCleaner:
    """Tests for clean_resume_text."""

    def test_clean_basic_text(self):
        """Should return cleaned text from normal input."""
        result = clean_resume_text("John  Doe\n\nSoftware Engineer")
        assert "John Doe" in result
        assert "Software Engineer" in result

    def test_removes_excess_newlines(self):
        """Three or more consecutive newlines should become two."""
        text = "Section A\n\n\n\n\nSection B"
        result = clean_resume_text(text)
        assert "\n\n\n" not in result
        assert "Section A" in result
        assert "Section B" in result

    def test_collapses_multiple_spaces(self):
        """Multiple spaces should be collapsed to one."""
        result = clean_resume_text("Hello    World")
        assert "Hello World" in result

    def test_ligature_replacement(self):
        """PDF ligature characters should be replaced with ASCII."""
        text = "\ufb01le management"  # 'fi' ligature
        result = clean_resume_text(text)
        assert "file management" in result

    def test_empty_input_raises(self):
        """Empty string input should raise ValueError."""
        with pytest.raises(ValueError):
            clean_resume_text("")

    def test_whitespace_only_raises(self):
        """Whitespace-only input should raise ValueError."""
        with pytest.raises(ValueError):
            clean_resume_text("   \n\n   ")

    def test_output_stripped(self):
        """Output should have no leading or trailing whitespace."""
        result = clean_resume_text("  \n  Hello World  \n  ")
        assert result == result.strip()
