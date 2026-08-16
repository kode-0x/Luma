"""Document management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from backend.api.dependencies import get_document_service, get_settings_dep
from backend.core.config import Settings
from backend.core.exceptions import (
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from backend.models.queries import (
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
    ErrorResponse,
)
from backend.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or size"},
    },
)
async def upload_document(
    file: UploadFile,
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings_dep),
) -> DocumentUploadResponse:
    """Upload a document for processing.

    Accepts PDF, DOCX, TXT, Markdown, and CSV files. The document will be
    parsed, chunked, embedded, and indexed for retrieval.

    Args:
        file: The uploaded file.
        service: Document service (injected).
        settings: Application settings (injected).

    Returns:
        Document upload confirmation with ID and status.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()

    try:
        document = await service.upload_document(
            filename=file.filename,
            content=content,
            max_size_bytes=settings.max_upload_size_bytes,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc

    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status.value,
        message=f"Document '{document.filename}' uploaded and processing {document.status.value}",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List all uploaded documents.

    Returns:
        List of document summaries with metadata.
    """
    documents = service.list_documents()
    summaries = [
        DocumentSummary(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type.value,
            status=doc.status.value,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
        )
        for doc in documents
    ]
    return DocumentListResponse(documents=summaries, total=len(summaries))


@router.get(
    "/{document_id}",
    response_model=DocumentSummary,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentSummary:
    """Get details for a specific document.

    Args:
        document_id: The document identifier.
        service: Document service (injected).

    Returns:
        Document summary with metadata.
    """
    try:
        doc = service.get_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc

    return DocumentSummary(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type.value,
        status=doc.status.value,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> None:
    """Delete a document and all its indexed data.

    Args:
        document_id: The document to delete.
        service: Document service (injected).
    """
    try:
        service.delete_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc
