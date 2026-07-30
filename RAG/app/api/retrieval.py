"""
Semantic retrieval API router.

Design Decision:
    - GET /retrieve accepts a query string and optional resume_id filter.
    - Kept as a GET (not POST) because it is a read-only, idempotent operation.
    - resume_id filter allows callers to restrict results to a single candidate's
      resume rather than searching across all stored resumes.
    - top_k and similarity_threshold are query parameters with sensible defaults
      from settings, allowing per-request tuning without config changes.
"""

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.core.logger import get_logger
from app.database.vector_store import retrieve_chunks
from app.models.responses import APIResponse
from app.models.resume import RetrievalResponse

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])
logger = get_logger(__name__)


@router.get(
    "",
    response_model=APIResponse[RetrievalResponse],
    summary="Semantic Retrieval",
    description=(
        "Query the resume vector store with a natural language query. "
        "Returns the most semantically relevant resume chunks with similarity scores."
    ),
)
async def retrieve(
    query: str = Query(..., min_length=3, description="Natural language query."),
    top_k: int = Query(default=None, ge=1, le=20, description="Max number of results."),
    similarity_threshold: float = Query(
        default=None, ge=0.0, le=1.0, description="Minimum similarity score."
    ),
    resume_id: str | None = Query(
        default=None, description="Optional: restrict results to a specific resume."
    ),
) -> APIResponse[RetrievalResponse]:
    """
    Perform semantic search against stored resume embeddings.

    Args:
        query: Natural language query string (min 3 chars).
        top_k: Override for number of results (default: from settings).
        similarity_threshold: Override for min score (default: from settings).
        resume_id: Optional resume filter.

    Returns:
        APIResponse[RetrievalResponse]: Matched chunks with scores.
    """
    settings = get_settings()
    effective_k = top_k or settings.retrieval_top_k
    effective_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.retrieval_similarity_threshold
    )

    logger.info(
        "Retrieval query='%s' top_k=%d threshold=%.2f resume_id=%s",
        query,
        effective_k,
        effective_threshold,
        resume_id or "ALL",
    )

    results = retrieve_chunks(
        query=query,
        top_k=effective_k,
        similarity_threshold=effective_threshold,
        resume_id=resume_id,
    )

    response = RetrievalResponse(
        query=query,
        results=results,
        total_results=len(results),
    )

    return APIResponse[RetrievalResponse](
        message=f"Retrieved {len(results)} result(s).",
        data=response,
    )
