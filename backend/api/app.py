"""FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routers import chat, documents, health
from backend.core.exceptions import LumaError
from backend.core.logging import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Sets up CORS, exception handlers, and registers all routers.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Luma",
        description="Evidence-based AI document intelligence platform",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(LumaError)
    async def luma_error_handler(request: Request, exc: LumaError) -> JSONResponse:
        """Handle all Luma application errors with consistent JSON responses."""
        logger.error("Application error", error=exc.message, path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors with a generic response."""
        logger.error("Unexpected error", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
            },
        )

    # Register routers
    app.include_router(health.router)
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")

    return app
