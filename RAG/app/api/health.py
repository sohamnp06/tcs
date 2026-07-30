"""
Health check API router.

Design Decision:
    - Health endpoint is isolated in its own router to keep main.py clean.
    - Returns application version and environment so ops teams can verify
      the correct build is deployed without digging into logs.
    - No authentication required — monitoring tools must reach this freely.
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logger import get_logger
from app.models.responses import HealthResponse

router = APIRouter(tags=["Health"])
logger = get_logger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns application health status, version, and environment.",
)
async def health_check() -> HealthResponse:
    """
    Perform a lightweight health check.

    This endpoint is called by load balancers and monitoring tools.
    It should return quickly with no external dependencies checked.

    Returns:
        HealthResponse: Current application status and metadata.
    """
    settings = get_settings()
    logger.debug("Health check requested.")

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )
