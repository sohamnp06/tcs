"""
Semantic resume chunker.

Design Decision:
    - Chunks are defined by resume SECTIONS, not token length.
    - This produces semantically coherent chunks (all of "Experience" together)
      rather than arbitrary splits in the middle of a job entry.
    - Section detection uses header keywords + regex to be robust against
      varied resume formats.
    - Each chunk carries section metadata for filtered retrieval.
    - If a section text exceeds max_chunk_chars, it is split at paragraph
      boundaries to keep chunks manageable without cutting mid-sentence.
"""

import re
import uuid

from app.core.exceptions import ChunkingException
from app.core.logger import get_logger
from app.models.resume import ChunkSection, ResumeChunk

logger = get_logger(__name__)

# Maximum characters per chunk before paragraph-level splitting kicks in
_MAX_CHUNK_CHARS: int = 1500

# Maps regex patterns to ChunkSection enum values.
# Order matters: first match wins.
_SECTION_PATTERNS: list[tuple[re.Pattern[str], ChunkSection]] = [
    (re.compile(r"(?i)(summary|objective|profile|about\s*me)"), ChunkSection.SUMMARY),
    (re.compile(r"(?i)(experience|employment|work\s*history|career)"), ChunkSection.EXPERIENCE),
    (re.compile(r"(?i)(education|academic|qualification|degree)"), ChunkSection.EDUCATION),
    (re.compile(r"(?i)(skill|technolog|tech\s*stack|competenc)"), ChunkSection.SKILLS),
    (re.compile(r"(?i)(project|portfolio)"), ChunkSection.PROJECTS),
    (re.compile(r"(?i)(certif|licens|credential)"), ChunkSection.CERTIFICATIONS),
    (re.compile(r"(?i)(achievement|award|honor|recognition)"), ChunkSection.ACHIEVEMENTS),
]

# Regex to detect a section header line:
# Short line (≤60 chars), all-caps or Title Case, possibly with special chars
_HEADER_LINE_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Z][A-Za-z\s\-\/&]{2,55}$"
)


def _detect_section(header: str) -> ChunkSection:
    """
    Map a header string to a ChunkSection enum value.

    Args:
        header: Candidate section header text.

    Returns:
        ChunkSection: Best matching section, or GENERAL if no match.
    """
    for pattern, section in _SECTION_PATTERNS:
        if pattern.search(header):
            return section
    return ChunkSection.GENERAL


def _is_header_line(line: str) -> bool:
    """
    Heuristically determine if a line is a section header.

    A header line is short, starts with a capital letter, and contains
    mostly alphabetical characters (not a sentence or bullet point).

    Args:
        line: A single stripped line from the resume text.

    Returns:
        bool: True if the line looks like a section header.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    # Reject lines that look like sentences (contain period mid-string)
    if ". " in stripped:
        return False
    return bool(_HEADER_LINE_PATTERN.match(stripped))


def _split_large_section(text: str, max_chars: int) -> list[str]:
    """
    Split a large section into paragraph-sized sub-chunks.

    Splits on double-newline (paragraph boundary) and groups paragraphs
    until the char limit is reached.

    Args:
        text: Section text to split.
        max_chars: Maximum characters per sub-chunk.

    Returns:
        list[str]: List of sub-chunk strings.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sub_chunks: list[str] = []
    current: list[str] = []
    current_len: int = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            sub_chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        sub_chunks.append("\n\n".join(current))

    return sub_chunks


def chunk_resume_text(
    cleaned_text: str,
    resume_id: str,
    max_chunk_chars: int = _MAX_CHUNK_CHARS,
) -> list[ResumeChunk]:
    """
    Split cleaned resume text into semantic chunks by section.

    Algorithm:
        1. Split text into lines.
        2. When a header line is detected, save the previous section and start a new one.
        3. Assign a ChunkSection to each block based on its header.
        4. If a section exceeds max_chunk_chars, split at paragraph boundaries.
        5. Return a list of ResumeChunk objects with metadata.

    Args:
        cleaned_text: Cleaned resume text from the text_cleaner.
        resume_id: Identifier for the source resume (used in chunk IDs).
        max_chunk_chars: Maximum characters per chunk before splitting.

    Returns:
        list[ResumeChunk]: Ordered list of semantic chunks.

    Raises:
        ChunkingException: If no chunks can be produced from the text.
    """
    if not cleaned_text or not cleaned_text.strip():
        raise ChunkingException(
            message="Cannot chunk empty text.",
            detail="cleaned_text was empty or whitespace.",
        )

    lines = cleaned_text.splitlines()
    chunks: list[ResumeChunk] = []

    current_header: str = "General"
    current_section: ChunkSection = ChunkSection.GENERAL
    current_lines: list[str] = []

    def _flush_section(header: str, section: ChunkSection, lines_buf: list[str]) -> None:
        """Save current buffer as one or more chunks."""
        text = "\n".join(lines_buf).strip()
        if not text:
            return

        sub_texts = (
            _split_large_section(text, max_chunk_chars)
            if len(text) > max_chunk_chars
            else [text]
        )

        for idx, sub_text in enumerate(sub_texts):
            chunk_id = f"{resume_id}_{section.value}_{uuid.uuid4().hex[:8]}"
            if idx > 0:
                # Subsequent sub-chunks inherit section with index
                chunk_id = f"{resume_id}_{section.value}_{idx}_{uuid.uuid4().hex[:8]}"
            chunks.append(
                ResumeChunk(
                    chunk_id=chunk_id,
                    resume_id=resume_id,
                    section=section,
                    text=sub_text,
                    char_count=len(sub_text),
                )
            )

    for line in lines:
        if _is_header_line(line):
            # Flush current section before starting a new one
            _flush_section(current_header, current_section, current_lines)
            current_header = line.strip()
            current_section = _detect_section(current_header)
            current_lines = []
        else:
            current_lines.append(line)

    # Flush the final section
    _flush_section(current_header, current_section, current_lines)

    if not chunks:
        # No headers detected — treat entire resume as one general chunk
        logger.warning(
            "No section headers detected. Treating entire resume as GENERAL chunk."
        )
        chunk_id = f"{resume_id}_general_{uuid.uuid4().hex[:8]}"
        chunks.append(
            ResumeChunk(
                chunk_id=chunk_id,
                resume_id=resume_id,
                section=ChunkSection.GENERAL,
                text=cleaned_text.strip(),
                char_count=len(cleaned_text.strip()),
            )
        )

    logger.info(
        "Chunking complete: %d chunk(s) from resume '%s'.",
        len(chunks),
        resume_id,
    )
    return chunks
