"""
Resume text normalization and cleaning.

Design Decision:
    - Pure function — no side effects, no I/O, easily unit-testable.
    - Cleans PDF extraction artifacts (ligatures, excessive whitespace,
      non-printable characters) without destroying meaningful structure.
    - Preserves newlines that separate semantic sections.
"""

import re
import unicodedata

from app.core.logger import get_logger

logger = get_logger(__name__)

# Common PDF ligature substitutions
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}


def clean_resume_text(raw_text: str) -> str:
    """
    Normalize and clean raw text extracted from a PDF resume.

    Operations performed (in order):
        1. Replace PDF ligature characters with ASCII equivalents.
        2. Normalize Unicode to NFC form.
        3. Remove non-printable / control characters (except newlines/tabs).
        4. Replace multiple consecutive spaces with a single space.
        5. Collapse 3+ consecutive newlines into exactly two (preserve sections).
        6. Strip leading/trailing whitespace per line.
        7. Strip leading/trailing whitespace from the full text.

    Args:
        raw_text: Raw text string from the PDF extractor.

    Returns:
        str: Cleaned, normalized text suitable for chunking.

    Raises:
        ValueError: If raw_text is empty or None.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must be a non-empty string.")

    text = raw_text

    # 1. Replace ligatures
    for ligature, replacement in _LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)

    # 2. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 3. Remove control characters (keep \n and \t)
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )

    # 4. Collapse multiple spaces (not newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Collapse 3+ consecutive newlines → 2 (preserve section breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Strip trailing spaces on each line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    # 7. Final strip
    text = text.strip()

    logger.debug(
        "Text cleaned: %d chars → %d chars.",
        len(raw_text),
        len(text),
    )
    return text
