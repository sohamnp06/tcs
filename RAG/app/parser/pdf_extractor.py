"""
PDF text extraction using PyMuPDF (fitz).

Design Decision:
    - Extraction is a pure function: Path in, str out.
    - All PyMuPDF-specific errors are caught and re-raised as typed
      application exceptions so the API layer stays decoupled from fitz.
    - Page-level extraction lets us detect mixed (text + image) PDFs
      and warn when OCR would be needed.
"""

from pathlib import Path

import fitz  # PyMuPDF

from app.core.exceptions import (
    CorruptPDFException,
    EmptyPDFException,
    EncryptedPDFException,
    PDFExtractionException,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

# Minimum characters to consider a page as having real text content
_MIN_PAGE_CHARS: int = 10


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all readable text from a PDF file.

    Handles:
        - Encrypted / password-protected PDFs → EncryptedPDFException
        - Corrupt / invalid PDFs → CorruptPDFException
        - Empty / scanned PDFs with no text → EmptyPDFException
        - General extraction failures → PDFExtractionException

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        str: Concatenated text content from all pages with text.

    Raises:
        EncryptedPDFException: PDF is password-protected.
        CorruptPDFException: PDF cannot be opened by PyMuPDF.
        EmptyPDFException: PDF has no extractable text content.
        PDFExtractionException: Any other extraction failure.
    """
    logger.info("Extracting text from PDF: %s", pdf_path.name)

    try:
        doc = fitz.open(str(pdf_path))
    except fitz.FileDataError as exc:
        raise CorruptPDFException(
            message="The uploaded PDF file is corrupt or invalid.",
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise PDFExtractionException(
            message="Failed to open the PDF file.",
            detail=str(exc),
        ) from exc

    # Check for encryption
    if doc.is_encrypted:
        doc.close()
        raise EncryptedPDFException(
            message="The uploaded PDF is password-protected and cannot be read.",
            detail=f"File: {pdf_path.name}",
        )

    pages_text: list[str] = []
    scanned_pages: int = 0

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")

            if len(text.strip()) < _MIN_PAGE_CHARS:
                scanned_pages += 1
                logger.debug(
                    "Page %d has minimal text (possibly scanned).", page_num + 1
                )
            else:
                pages_text.append(text)
        except Exception as exc:
            logger.warning("Failed to extract page %d: %s", page_num + 1, exc)

    doc.close()

    if scanned_pages > 0:
        logger.warning(
            "%d of %d page(s) appear to be scanned images (no extractable text).",
            scanned_pages,
            len(doc) if not doc.is_closed else scanned_pages + len(pages_text),
        )

    raw_text = "\n".join(pages_text).strip()

    if not raw_text:
        raise EmptyPDFException(
            message="No readable text found in the PDF. It may be a scanned image.",
            detail=f"File: {pdf_path.name}. All {scanned_pages} page(s) returned no text.",
        )

    logger.info(
        "Extraction complete: %d chars from %d page(s).",
        len(raw_text),
        len(pages_text),
    )
    return raw_text
