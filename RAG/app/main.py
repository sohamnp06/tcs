"""
FastAPI application factory and lifespan management.

Design Decision:
    - Application is built via a factory function (create_app) rather than
      a module-level instance. This enables clean test isolation — each test
      can create a fresh app instance with overridden dependencies.
    - Lifespan context manager handles startup/shutdown logic cleanly,
      avoiding the deprecated @app.on_event pattern.
    - Global exception handlers map typed exceptions to HTTP responses,
      keeping route handlers free of error-handling boilerplate.
    - All routers are registered with a /api/v1 prefix for API versioning
      from day one, avoiding painful migration later.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, resume, retrieval
from app.core.config import get_settings
from app.core.exceptions import (
    AppBaseException,
    ChromaDBException,
    CorruptPDFException,
    EmptyPDFException,
    EncryptedPDFException,
    FileNotFoundException,
    FileTooLargeException,
    InvalidFileTypeException,
)
from app.core.logger import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup initialization and graceful shutdown.
    Replaces the deprecated @app.on_event("startup") pattern.

    Startup:
        - Configure logging.
        - Ensure required directories exist.
        - Log application start.

    Shutdown:
        - Log graceful shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Yields control to the application during its lifetime.
    """
    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #
    configure_logging()
    settings = get_settings()

    logger.info(
        "Starting %s v%s [%s]",
        settings.app_name,
        settings.app_version,
        settings.environment.upper(),
    )

    # Ensure critical directories exist at startup
    _ensure_directories(settings)

    logger.info("Application startup complete. Listening on %s:%s", settings.host, settings.port)

    yield  # Application runs here

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    logger.info("Application shutting down gracefully.")


def _ensure_directories(settings) -> None:
    """
    Create required runtime directories if they do not exist.

    Args:
        settings: Application settings instance.
    """
    directories = [
        settings.upload_path,
        settings.chroma_path,
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug("Verified directory exists: %s", directory)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Returns:
        FastAPI: Fully configured application ready for serving.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Resume RAG Service. "
            "Extracts, chunks, embeds, and stores resume content for semantic retrieval."
        ),
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # CORS Middleware
    # ------------------------------------------------------------------ #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Exception Handlers
    # ------------------------------------------------------------------ #
    _register_exception_handlers(app)

    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(resume.router, prefix="/api/v1")
    app.include_router(retrieval.router, prefix="/api/v1")

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers that map typed exceptions to HTTP responses.

    Design Decision:
        Route handlers raise typed exceptions. This central handler maps them
        to appropriate HTTP status codes. This keeps route code clean and
        ensures consistent error response shape via ErrorResponse model.

    Args:
        app: The FastAPI application instance.
    """

    @app.exception_handler(InvalidFileTypeException)
    async def invalid_file_type_handler(request: Request, exc: InvalidFileTypeException):
        logger.warning("Invalid file type: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"success": False, "error": "INVALID_FILE_TYPE", "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(FileTooLargeException)
    async def file_too_large_handler(request: Request, exc: FileTooLargeException):
        logger.warning("File too large: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"success": False, "error": "FILE_TOO_LARGE", "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(EncryptedPDFException)
    async def encrypted_pdf_handler(request: Request, exc: EncryptedPDFException):
        logger.warning("Encrypted PDF submitted: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"success": False, "error": "ENCRYPTED_PDF", "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(EmptyPDFException)
    async def empty_pdf_handler(request: Request, exc: EmptyPDFException):
        logger.warning("Empty PDF submitted: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"success": False, "error": "EMPTY_PDF", "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(CorruptPDFException)
    async def corrupt_pdf_handler(request: Request, exc: CorruptPDFException):
        logger.warning("Corrupt PDF submitted: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"success": False, "error": "CORRUPT_PDF", "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(FileNotFoundException)
    async def file_not_found_handler(request: Request, exc: FileNotFoundException):
        logger.warning("File not found: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "error": "FILE_NOT_FOUND", "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(ChromaDBException)
    async def chromadb_handler(request: Request, exc: ChromaDBException):
        logger.error("ChromaDB error: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": "VECTOR_DB_ERROR", "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(AppBaseException)
    async def app_base_exception_handler(request: Request, exc: AppBaseException):
        """Catch-all handler for any unhandled AppBaseException subclass."""
        logger.error("Unhandled application exception: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "INTERNAL_ERROR", "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Safety net for any completely unhandled exceptions."""
        logger.critical("Unhandled exception: %s", str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred.",
                "detail": None,
            },
        )


# ------------------------------------------------------------------ #
# Application instance (used by uvicorn)
# ------------------------------------------------------------------ #
app = create_app()
