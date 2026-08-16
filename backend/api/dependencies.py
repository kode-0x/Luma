"""FastAPI dependency injection: provides service instances to route handlers."""

from functools import lru_cache

from backend.core.config import Settings, get_settings
from backend.core.container import Container


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Get or create the singleton application container.

    Returns:
        The application dependency container.
    """
    settings = get_settings()
    return Container(settings)


def get_settings_dep() -> Settings:
    """FastAPI dependency for application settings.

    Returns:
        Application settings instance.
    """
    return get_container().settings


def get_document_service() -> "DocumentService":
    """FastAPI dependency for the document service.

    Returns:
        Document service instance.
    """
    return get_container().document_service


def get_chat_service() -> "ChatService":
    """FastAPI dependency for the chat service.

    Returns:
        Chat service instance.
    """
    return get_container().chat_service


# Type hints for lazy imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.services.chat_service import ChatService
    from backend.services.document_service import DocumentService
