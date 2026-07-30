"""
File system utility functions.

Design Decision:
    - Pure functions with no FastAPI or business logic dependencies.
    - The API layer delegates all file I/O here so it stays testable in isolation.
    - UUID-based filenames prevent collisions and path traversal attacks.
"""

import uuid
from pathlib import Path

from app.core.exceptions import FileNotFoundException, FileUploadException, InvalidFileTypeException
from app.core.logger import get_logger

logger = get_logger(__name__)


def save_upload(
    file_bytes: bytes,
    original_filename: str,
    upload_dir: Path,
    allowed_extensions: list[str],
) -> Path:
    """
    Persist uploaded file bytes to disk with a UUID-based filename.

    Validates extension before writing. Uses UUID prefix to prevent
    filename collisions and directory traversal attacks.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        original_filename: Original filename from the upload request.
        upload_dir: Directory to save the file into (must already exist).
        allowed_extensions: List of permitted extensions e.g. ['.pdf'].

    Returns:
        Path: Absolute path to the saved file.

    Raises:
        InvalidFileTypeException: If the file extension is not in allowed_extensions.
        FileUploadException: If writing to disk fails.
    """
    suffix = Path(original_filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise InvalidFileTypeException(
            message=f"Only {allowed_extensions} files are accepted.",
            detail=f"Received extension: '{suffix}'",
        )

    safe_name = f"{uuid.uuid4().hex}{suffix}"
    destination = upload_dir / safe_name

    try:
        destination.write_bytes(file_bytes)
        logger.info("Saved upload: %s (%d bytes)", destination.name, len(file_bytes))
        return destination
    except OSError as exc:
        raise FileUploadException(
            message="Failed to save uploaded file to disk.",
            detail=str(exc),
        ) from exc


def get_file_path(filename: str, upload_dir: Path) -> Path:
    """
    Resolve and validate a filename inside the upload directory.

    Args:
        filename: The filename to look up.
        upload_dir: The directory where uploads are stored.

    Returns:
        Path: Resolved absolute path to the file.

    Raises:
        FileNotFoundException: If the file does not exist.
    """
    path = (upload_dir / filename).resolve()

    # Guard against path traversal: resolved path must be inside upload_dir
    if not str(path).startswith(str(upload_dir.resolve())):
        raise FileNotFoundException(
            message="File not found.",
            detail=f"'{filename}' is outside the upload directory.",
        )

    if not path.exists():
        raise FileNotFoundException(
            message=f"File '{filename}' not found.",
            detail=f"Expected at: {path}",
        )

    return path
