"""Document metadata storage and file management."""

import shutil
from datetime import UTC
from pathlib import Path

from backend.core.exceptions import DocumentNotFoundError
from backend.core.logging import get_logger
from backend.models.documents import Document, DocumentStatus

logger = get_logger(__name__)


class DocumentRepository:
    """Manages document metadata and uploaded files on disk.

    Stores document metadata in memory (suitable for development).
    In production, this would be backed by a persistent database.

    Attributes:
        upload_dir: Directory where uploaded files are stored.
    """

    def __init__(self, upload_dir: Path) -> None:
        """Initialize the document repository.

        Args:
            upload_dir: Directory for storing uploaded files.
        """
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._documents: dict[str, Document] = {}

    def save(self, document: Document) -> Document:
        """Save or update a document record.

        Args:
            document: The document to persist.

        Returns:
            The saved document.
        """
        self._documents[document.id] = document
        logger.info("Saved document metadata", document_id=document.id, filename=document.filename)
        return document

    def get_by_id(self, document_id: str) -> Document:
        """Retrieve a document by its ID.

        Args:
            document_id: The unique document identifier.

        Returns:
            The matching document.

        Raises:
            DocumentNotFoundError: If no document exists with the given ID.
        """
        document = self._documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        return document

    def list_all(self) -> list[Document]:
        """List all documents ordered by creation time (newest first).

        Returns:
            List of all stored documents.
        """
        return sorted(self._documents.values(), key=lambda d: d.created_at, reverse=True)

    def delete(self, document_id: str) -> None:
        """Delete a document and its associated file.

        Args:
            document_id: The document ID to delete.

        Raises:
            DocumentNotFoundError: If no document exists with the given ID.
        """
        document = self.get_by_id(document_id)

        # Remove file from disk
        file_path = self.get_file_path(document_id, document.filename)
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted file from disk", path=str(file_path))

        # Remove document directory if empty
        doc_dir = self.upload_dir / document_id
        if doc_dir.exists() and not any(doc_dir.iterdir()):
            doc_dir.rmdir()

        del self._documents[document_id]
        logger.info("Deleted document", document_id=document_id)

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> Document:
        """Update the processing status of a document.

        Args:
            document_id: The document ID to update.
            status: The new processing status.
            chunk_count: Number of chunks (set when processing completes).
            error_message: Error message (set when processing fails).

        Returns:
            The updated document.

        Raises:
            DocumentNotFoundError: If no document exists with the given ID.
        """
        from datetime import datetime

        document = self.get_by_id(document_id)
        document.status = status
        document.updated_at = datetime.now(UTC)

        if chunk_count is not None:
            document.chunk_count = chunk_count
        if error_message is not None:
            document.error_message = error_message

        logger.info("Updated document status", document_id=document_id, status=status)
        return document

    def get_file_path(self, document_id: str, filename: str) -> Path:
        """Get the file path for a stored document.

        Files are stored under upload_dir/<document_id>/<filename>.

        Args:
            document_id: The document ID.
            filename: The original filename.

        Returns:
            Path to the stored file.
        """
        return self.upload_dir / document_id / filename

    async def save_upload(self, document_id: str, filename: str, content: bytes) -> Path:
        """Save uploaded file content to disk.

        Args:
            document_id: The document ID.
            filename: The original filename.
            content: Raw file bytes.

        Returns:
            Path where the file was saved.
        """
        doc_dir = self.upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        file_path = doc_dir / filename
        file_path.write_bytes(content)

        logger.info("Saved uploaded file", path=str(file_path), size_bytes=len(content))
        return file_path

    def cleanup_upload(self, document_id: str) -> None:
        """Remove all uploaded files for a document.

        Args:
            document_id: The document ID whose files should be removed.
        """
        doc_dir = self.upload_dir / document_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
            logger.info("Cleaned up upload directory", document_id=document_id)
