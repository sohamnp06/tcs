"""
Pydantic models for standardized API responses.

Design Decision:
    - All API endpoints return a typed response wrapper.
    - This ensures consistent JSON structure across all routes.
    - Consumers (frontend, other services) always know the shape of a response.
    - success flag allows clients to handle errors without relying solely on HTTP codes.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = Field(..., description="Application health status.")
    version: str = Field(..., description="Current application version.")
    environment: str = Field(..., description="Runtime environment.")


class APIResponse(BaseModel, Generic[DataT]):
    """
    Generic wrapper for all successful API responses.

    Attributes:
        success: Always True for success responses.
        message: Human-readable description of the operation.
        data: Typed payload. None if no data to return.
    """

    success: bool = Field(default=True)
    message: str = Field(..., description="Human-readable operation result.")
    data: DataT | None = Field(default=None)


class ErrorResponse(BaseModel):
    """
    Standardized error response model.

    Attributes:
        success: Always False for error responses.
        error: Short error category identifier (e.g., 'INVALID_PDF').
        message: Human-readable error description.
        detail: Optional additional context.
    """

    success: bool = Field(default=False)
    error: str = Field(..., description="Error category identifier.")
    message: str = Field(..., description="Human-readable error description.")
    detail: str | None = Field(default=None, description="Extended error context.")
