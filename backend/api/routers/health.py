"""Health check endpoint."""

from fastapi import APIRouter

from backend.models.queries import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check application health status.

    Returns:
        HealthResponse with status and version.
    """
    return HealthResponse(status="healthy", version="0.1.0")
