"""
Pydantic schemas for the Resume RAG pipeline.

Design Decision:
    - All data that flows between modules is typed via these models.
    - Using Pydantic ensures validation at every boundary.
    - ResumeChunk carries metadata alongside text so ChromaDB can
      filter by section type during retrieval.
    - RetrievalResult is the contract between the retriever and API layer.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ChunkSection(str, Enum):
    """
    Semantic section types for resume chunks.

    Used as ChromaDB metadata to enable section-filtered retrieval.
    """

    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    ACHIEVEMENTS = "achievements"
    GENERAL = "general"


class ResumeChunk(BaseModel):
    """
    A single semantic chunk extracted from a resume.

    Attributes:
        chunk_id: Unique identifier for this chunk (used as ChromaDB document ID).
        resume_id: Identifier linking this chunk back to its source resume file.
        section: The resume section this chunk belongs to.
        text: The raw text content of the chunk.
        char_count: Number of characters in the text.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    resume_id: str = Field(..., description="Source resume file identifier.")
    section: ChunkSection = Field(..., description="Resume section this chunk belongs to.")
    text: str = Field(..., min_length=1, description="Chunk text content.")
    char_count: int = Field(..., gt=0, description="Character count of the text.")


class IngestResponse(BaseModel):
    """
    Response returned after a successful resume ingestion.

    Attributes:
        resume_id: Identifier of the ingested resume (UUID-based filename).
        filename: Original uploaded filename.
        chunks_stored: Number of chunks stored in ChromaDB.
        char_count: Total characters extracted from the PDF.
    """

    resume_id: str = Field(..., description="Unique resume identifier.")
    filename: str = Field(..., description="Original uploaded filename.")
    chunks_stored: int = Field(..., ge=0, description="Number of chunks stored in ChromaDB.")
    char_count: int = Field(..., ge=0, description="Total characters extracted.")


class RetrievalResult(BaseModel):
    """
    A single retrieved chunk with its similarity score.

    Attributes:
        chunk_id: The unique chunk identifier.
        resume_id: Source resume identifier.
        section: Resume section this chunk came from.
        text: The chunk's text content.
        score: Similarity score (higher = more relevant). Range: [0.0, 1.0].
    """

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    resume_id: str = Field(..., description="Source resume identifier.")
    section: str = Field(..., description="Resume section.")
    text: str = Field(..., description="Chunk text content.")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score.")


class RetrievalResponse(BaseModel):
    """
    Response returned from a semantic retrieval query.

    Attributes:
        query: The original query string.
        results: Ordered list of relevant chunks (most relevant first).
        total_results: Total number of results returned.
    """

    query: str = Field(..., description="The original query string.")
    results: list[RetrievalResult] = Field(default_factory=list)
    total_results: int = Field(..., ge=0, description="Number of results returned.")
