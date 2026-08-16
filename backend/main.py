"""Application entry point: creates and runs the Luma backend server."""

import uvicorn

from backend.api.app import create_app
from backend.core.config import get_settings
from backend.core.logging import setup_logging

settings = get_settings()
setup_logging(debug=settings.app_debug)

app = create_app()


def main() -> None:
    """Run the Luma backend with uvicorn."""
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="debug" if settings.app_debug else "info",
    )


if __name__ == "__main__":
    main()
