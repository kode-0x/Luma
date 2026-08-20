"""Document service: orchestrates upload, parsing, chunking, embedding, and storage."""

import asyncio
from pathlib import Path

from backend.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from backend.core.logging import get_logger
from backend.ingestion.chunker import TextChunker
from backend.ingestion.embedder import EmbeddingService
from backend.ingestion.parser import DocumentParser
from backend.models.documents import SUPPORTED_EXTENSIONS, Document, DocumentStatus
from backend.repository.document_repository import DocumentRepository
from backend.repository.vector_store import QdrantVectorStore
from backend.retrieval.bm25_search import BM25Searcher

logger = get_logger(__name__)


class DocumentService:
    """Service layer for document lifecycle management.

    Orchestrates the full document ingestion pipeline: validation,
    storage, parsing, chunking, embedding, and vector indexing.
    Now also updates the BM25 index during ingestion and deletion.
    """

    def __init__(
        self,
        parser: DocumentParser,
        chunker: TextChunker,
        embedder: EmbeddingService,
        vector_store: QdrantVectorStore,
        repository: DocumentRepository,
        bm25_searcher: BM25Searcher | None = None,
    ) -> None:
        """Initialize the document service with its dependencies.

        Args:
            parser: Document text extraction component.
            chunker: Text chunking component.
            embedder: Embedding generation component.
            vector_store: Vector database for chunk storage/retrieval.
            repository: Document metadata and file storage.
            bm25_searcher: Optional BM25 index for lexical search updates.
        """
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._repository = repository
        self._bm25_searcher = bm25_searcher

    async def upload_document(
        self,
        filename: str,
        content: bytes,
        max_size_bytes: int,
    ) -> Document:
        """Upload a document and kick off background processing.

        Validates the file, saves it to disk, and returns immediately
        with PROCESSING status. The heavy work (parsing, chunking,
        embedding, vector storage) runs in the background.

        Args:
            filename: Original filename.
            content: Raw file bytes.
            max_size_bytes: Maximum allowed file size.

        Returns:
            The created Document with PROCESSING status.

        Raises:
            UnsupportedFileTypeError: If the file extension is not supported.
            FileTooLargeError: If the file exceeds the size limit.
        """
        # Validate file type
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(filename, suffix)

        # Validate file size
        size_mb = len(content) / (1024 * 1024)
        max_mb = max_size_bytes // (1024 * 1024)
        if len(content) > max_size_bytes:
            raise FileTooLargeError(filename, size_mb, max_mb)

        # Create document record
        file_type = SUPPORTED_EXTENSIONS[suffix]
        document = Document(
            filename=filename,
            file_type=file_type,
            file_size_bytes=len(content),
            status=DocumentStatus.PENDING,
        )
        self._repository.save(document)

        # Save file to disk immediately
        self._repository.update_status(document.id, DocumentStatus.PROCESSING)
        file_path = await self._repository.save_upload(document.id, filename, content)

        # Process in background (non-blocking)
        asyncio.create_task(self._process_document(document.id, file_path))

        return self._repository.get_by_id(document.id)

    async def _process_document(self, document_id: str, file_path: Path) -> None:
        """Process a document in the background: parse, chunk, embed, store.

        Args:
            document_id: The document to process.
            file_path: Path to the saved file on disk.
        """
        try:
            # Run the CPU-heavy work in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._process_sync, document_id, file_path)
        except Exception as exc:
            logger.error("Background processing failed", document_id=document_id, error=str(exc))
            self._repository.update_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(exc),
            )

    def _process_sync(self, document_id: str, file_path: Path) -> None:
        """Synchronous processing pipeline (runs in thread pool).

        Args:
            document_id: The document to process.
            file_path: Path to the saved file on disk.
        """
        try:
            # Parse
            parsed = self._parser.parse(file_path)

            # Chunk
            chunks = self._chunker.chunk_document(parsed, document_id)

            if not chunks:
                self._repository.update_status(
                    document_id,
                    DocumentStatus.COMPLETED,
                    chunk_count=0,
                )
                return

            # Embed
            texts = [chunk.content for chunk in chunks]
            embeddings = self._embedder.embed_batch(texts)
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding

            # Store in vector database
            self._vector_store.upsert_chunks(chunks)

            # Update BM25 index for hybrid retrieval
            if self._bm25_searcher is not None:
                self._bm25_searcher.add_chunks(chunks)

            # Update status
            self._repository.update_status(
                document_id,
                DocumentStatus.COMPLETED,
                chunk_count=len(chunks),
            )

            logger.info(
                "Document processed successfully",
                document_id=document_id,
                chunk_count=len(chunks),
            )

        except Exception as exc:
            logger.error("Document processing failed", document_id=document_id, error=str(exc))
            self._repository.update_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(exc),
            )

    def get_document(self, document_id: str) -> Document:
        """Retrieve a document by ID.

        Args:
            document_id: The document identifier.

        Returns:
            The matching document.

        Raises:
            DocumentNotFoundError: If the document does not exist.
        """
        return self._repository.get_by_id(document_id)

    def list_documents(self) -> list[Document]:
        """List all documents.

        Returns:
            All documents ordered by creation time (newest first).
        """
        return self._repository.list_all()

    def delete_document(self, document_id: str) -> None:
        """Delete a document and all its associated data.

        Removes the document metadata, uploaded file, vector store chunks,
        and BM25 index entries.

        Args:
            document_id: The document to delete.

        Raises:
            DocumentNotFoundError: If the document does not exist.
        """
        # Verify document exists (raises if not found)
        self._repository.get_by_id(document_id)

        # Remove from vector store
        self._vector_store.delete_by_document_id(document_id)

        # Remove from BM25 index
        if self._bm25_searcher is not None:
            self._bm25_searcher.remove_document(document_id)

        # Remove metadata and file
        self._repository.delete(document_id)

        logger.info("Document deleted", document_id=document_id)
