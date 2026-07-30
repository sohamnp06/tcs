"""
Application configuration module.

All settings are loaded from environment variables (or .env file).
Using Pydantic BaseSettings ensures type validation at startup,
failing fast if required variables are missing or malformed.

Design Decision:
    - lru_cache() on get_settings() creates a singleton pattern without
      a global variable. This allows easy override in tests via
      dependency injection or cache clearing.
    - No LLM credentials are required. This service is a pure
      Resume RAG pipeline: extract → chunk → embed → store → retrieve.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    All fields are loaded from environment variables.
    Defaults are provided only for non-sensitive optional settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    app_name: str = Field(
        default="Resume RAG Service",
        description="Human-readable application name.",
    )
    app_version: str = Field(default="0.1.0", description="Semantic version string.")
    debug: bool = Field(default=False, description="Enable debug mode.")
    environment: str = Field(
        default="development",
        description="Runtime environment: development | staging | production.",
    )

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    host: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    port: int = Field(default=8000, description="Uvicorn bind port.")

    # ------------------------------------------------------------------ #
    # Embedding Model
    # ------------------------------------------------------------------ #
    embedding_model_name: str = Field(
        default="static-retrieval-mrl-en-v1",
        description="Sentence-transformers model identifier.",
    )
    embedding_dimension: int = Field(
        default=1024,
        gt=0,
        description="Output dimension of the embedding model.",
    )

    # ------------------------------------------------------------------ #
    # ChromaDB
    # ------------------------------------------------------------------ #
    chroma_persist_directory: str = Field(
        default="./chroma_db",
        description="Directory where ChromaDB persists its data.",
    )
    chroma_collection_name: str = Field(
        default="resume_chunks",
        description="Name of the ChromaDB collection for resume embeddings.",
    )

    # ------------------------------------------------------------------ #
    # File Upload
    # ------------------------------------------------------------------ #
    upload_directory: str = Field(
        default="./app/uploads",
        description="Directory where uploaded PDF files are stored.",
    )
    max_upload_size_mb: int = Field(
        default=10,
        gt=0,
        description="Maximum allowed PDF upload size in megabytes.",
    )
    allowed_file_extensions: list[str] = Field(
        default=[".pdf"],
        description="List of permitted upload file extensions.",
    )

    # ------------------------------------------------------------------ #
    # RAG Retrieval
    # ------------------------------------------------------------------ #
    retrieval_top_k: int = Field(
        default=5,
        gt=0,
        description="Number of top chunks to retrieve per query.",
    )
    retrieval_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include a chunk in results.",
    )

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )
    log_format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Python logging format string.",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure log level is a valid Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{value}'")
        return upper

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Ensure environment is one of the expected values."""
        allowed = {"development", "staging", "production"}
        lower = value.lower()
        if lower not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{value}'")
        return lower

    @property
    def upload_path(self) -> Path:
        """Return the upload directory as a resolved Path object."""
        return Path(self.upload_directory).resolve()

    @property
    def chroma_path(self) -> Path:
        """Return the ChromaDB persist directory as a resolved Path object."""
        return Path(self.chroma_persist_directory).resolve()

    @property
    def max_upload_size_bytes(self) -> int:
        """Return the maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Using lru_cache ensures the .env file is read only once.
    In tests, call get_settings.cache_clear() to reset.

    Returns:
        Settings: Validated application configuration.
    """
    return Settings()
