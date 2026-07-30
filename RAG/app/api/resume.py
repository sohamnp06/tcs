"""
Resume upload and ingestion API router.

Design Decision:
    - POST /resume/upload accepts a PDF and runs the full pipeline:
      save → extract → clean → chunk → embed → store.
    - The pipeline is synchronous within the async route because the
      embedding model runs on CPU (sentence-transformers); wrapping in
      run_in_executor would add complexity without benefit at this scale.
    - resume_id is derived from the saved filename (UUID-based) so
      the client can later query by resume_id for filtered retrieval.
    - File size is validated before writing to disk.
"""

import asyncio
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import FileTooLargeException, InvalidFileTypeException
from app.core.logger import get_logger
from app.database.vector_store import store_chunks
from app.embeddings.chunker import chunk_resume_text
from app.models.responses import APIResponse
from app.models.resume import IngestResponse
from app.parser.pdf_extractor import extract_text_from_pdf
from app.parser.text_cleaner import clean_resume_text
from app.utils.file_utils import save_upload

router = APIRouter(prefix="/resume", tags=["Resume"])
logger = get_logger(__name__)


@router.post(
    "/upload",
    response_model=APIResponse[IngestResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest Resume",
    description=(
        "Upload a PDF resume. The service will extract text, clean it, "
        "chunk it into semantic sections, generate embeddings using "
        "static-retrieval-mrl-en-v1, and store everything in ChromaDB."
    ),
)
async def upload_resume(
    file: UploadFile = File(..., description="PDF resume file to upload and ingest."),
) -> JSONResponse:
    """
    Upload a PDF resume and run the full RAG ingestion pipeline.

    Pipeline:
        1. Validate file type and size.
        2. Save PDF to uploads/ directory.
        3. Extract text using PyMuPDF.
        4. Clean and normalize text.
        5. Chunk text into semantic sections.
        6. Generate embeddings and store in ChromaDB.

    Args:
        file: Uploaded PDF file from the multipart form.

    Returns:
        APIResponse[IngestResponse]: Result with resume_id and chunk count.

    Raises:
        InvalidFileTypeException: File is not a PDF.
        FileTooLargeException: File exceeds configured size limit.
        Various pipeline exceptions handled by global handlers.
    """
    settings = get_settings()

    # --- Validate file type early (before reading bytes) ---
    suffix = Path(file.filename or "unknown").suffix.lower()
    if suffix not in settings.allowed_file_extensions:
        raise InvalidFileTypeException(
            message="Only PDF files are accepted.",
            detail=f"Received: '{file.filename}'",
        )

    logger.info("Received upload: '%s'", file.filename)

    # --- Read bytes and validate size ---
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise FileTooLargeException(
            message=f"File exceeds the {settings.max_upload_size_mb} MB size limit.",
            detail=f"File size: {len(file_bytes) / (1024*1024):.2f} MB",
        )

    # --- Save to disk ---
    saved_path = save_upload(
        file_bytes=file_bytes,
        original_filename=file.filename or "resume.pdf",
        upload_dir=settings.upload_path,
        allowed_extensions=settings.allowed_file_extensions,
    )

    # resume_id is the UUID-based stem of the saved filename
    resume_id = saved_path.stem

    # --- Extract text ---
    logger.info("Extracting text from '%s'...", saved_path.name)
    raw_text = extract_text_from_pdf(saved_path)

    # --- Clean text ---
    cleaned_text = clean_resume_text(raw_text)
    logger.info("Text cleaned: %d chars.", len(cleaned_text))

    # --- Chunk ---
    chunks = chunk_resume_text(cleaned_text=cleaned_text, resume_id=resume_id)
    logger.info("Produced %d chunk(s).", len(chunks))

    # --- Embed + Store ---
    stored_count = store_chunks(chunks)

    response_data = IngestResponse(
        resume_id=resume_id,
        filename=file.filename or saved_path.name,
        chunks_stored=stored_count,
        char_count=len(cleaned_text),
    )

    logger.info(
        "Ingestion complete: resume_id='%s', %d chunks stored.",
        resume_id,
        stored_count,
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=APIResponse[IngestResponse](
            message="Resume ingested successfully.",
            data=response_data,
        ).model_dump(),
    )
