"""
Custom exception hierarchy for the Resume RAG Service.

Design Decision:
    - A typed exception hierarchy allows the API layer to catch specific
      error types and return appropriate HTTP status codes without
      business logic leaking into route handlers.
    - All custom exceptions inherit from AppBaseException so a single
      catch-all handler can process them uniformly if needed.
    - detail field carries human-readable error context for API responses.
"""


class AppBaseException(Exception):
    """
    Base class for all application-specific exceptions.

    Attributes:
        message: Human-readable error description.
        detail: Optional additional context (e.g., filename, field name).
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        """
        Initialize the base exception.

        Args:
            message: Short error description.
            detail: Optional extended context.
        """
        self.message = message
        self.detail = detail
        super().__init__(message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, detail={self.detail!r})"


# ------------------------------------------------------------------ #
# PDF / File Exceptions
# ------------------------------------------------------------------ #


class FileUploadException(AppBaseException):
    """Raised when a file upload fails validation or storage."""


class InvalidFileTypeException(AppBaseException):
    """Raised when an uploaded file has an unsupported extension."""


class FileTooLargeException(AppBaseException):
    """Raised when an uploaded file exceeds the configured size limit."""


class FileNotFoundException(AppBaseException):
    """Raised when a requested file does not exist on disk."""


# ------------------------------------------------------------------ #
# PDF Parsing Exceptions
# ------------------------------------------------------------------ #


class PDFExtractionException(AppBaseException):
    """Raised when PyMuPDF fails to extract text from a PDF."""


class EncryptedPDFException(AppBaseException):
    """Raised when a PDF is password-protected and cannot be read."""


class EmptyPDFException(AppBaseException):
    """Raised when a PDF contains no extractable text (e.g., scanned image)."""


class CorruptPDFException(AppBaseException):
    """Raised when a PDF file is malformed or corrupt."""


# ------------------------------------------------------------------ #
# Resume Parsing Exceptions
# ------------------------------------------------------------------ #


class ResumeParsingException(AppBaseException):
    """Raised when Gemini fails to parse a resume into structured JSON."""


class InvalidResumeStructureException(AppBaseException):
    """Raised when the parsed resume JSON fails schema validation."""


# ------------------------------------------------------------------ #
# Embedding Exceptions
# ------------------------------------------------------------------ #


class EmbeddingException(AppBaseException):
    """Raised when the embedding model fails to generate vectors."""


class ChunkingException(AppBaseException):
    """Raised when semantic chunking fails to produce valid chunks."""


# ------------------------------------------------------------------ #
# Vector Database Exceptions
# ------------------------------------------------------------------ #


class ChromaDBException(AppBaseException):
    """Raised when a ChromaDB operation fails."""


class CollectionNotFoundException(AppBaseException):
    """Raised when a requested ChromaDB collection does not exist."""


# ------------------------------------------------------------------ #
# RAG Exceptions
# ------------------------------------------------------------------ #


class RetrievalException(AppBaseException):
    """Raised when semantic retrieval fails."""
