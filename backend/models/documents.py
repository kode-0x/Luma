"""Domain models for documents and their metadata."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    """Processing status of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(StrEnum):
    """Supported document file types."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"
    CSV = "csv"


SUPPORTED_EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".txt": FileType.TXT,
    ".md": FileType.MARKDOWN,
    ".csv": FileType.CSV,
}


class Document(BaseModel):
    """Represents an uploaded document and its processing state.

    Attributes:
        id: Unique identifier for the document.
        filename: Original filename as uploaded by the user.
        file_type: Detected file type.
        file_size_bytes: Size of the uploaded file in bytes.
        status: Current processing status.
        chunk_count: Number of chunks generated from this document.
        created_at: Timestamp when the document was uploaded.
        updated_at: Timestamp of the last status update.
        error_message: Error details if processing failed.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: FileType
    file_size_bytes: int
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error_message: str | None = None
