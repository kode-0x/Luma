"""Request and response schemas for the chat/query API."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat/query request from the user.

    Attributes:
        query: The natural language question.
        document_ids: Optional list of document IDs to restrict the search scope.
            If empty, searches across all documents.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] = Field(default_factory=list)
    model: str | None = None


class Citation(BaseModel):
    """A citation pointing to a specific location in a source document.

    Attributes:
        document_id: ID of the source document.
        filename: Original filename of the source document.
        page_number: Page number of the cited passage (None for non-paged formats).
        content: The relevant text passage that supports the answer.
        score: Relevance score of this citation.
    """

    document_id: str
    filename: str
    page_number: int | None = None
    content: str
    score: float


class ChatResponse(BaseModel):
    """Response to a user query including the answer and supporting citations.

    Attributes:
        answer: The generated answer text.
        citations: List of source citations supporting the answer.
        query: The original user query (echoed back for reference).
        timestamp: When the response was generated.
    """

    answer: str
    citations: list[Citation]
    query: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StreamChunk(BaseModel):
    """A single chunk in a streaming response.

    Attributes:
        type: The type of stream event (token, citation, done, error).
        content: The payload content for this chunk.
    """

    type: str
    content: str


class DocumentUploadResponse(BaseModel):
    """Response after successfully uploading a document.

    Attributes:
        document_id: The assigned document identifier.
        filename: The original filename.
        status: Current processing status.
        message: Human-readable status message.
    """

    document_id: str
    filename: str
    status: str
    message: str


class DocumentListResponse(BaseModel):
    """Response containing a list of documents.

    Attributes:
        documents: List of document summaries.
        total: Total number of documents.
    """

    documents: list["DocumentSummary"]
    total: int


class DocumentSummary(BaseModel):
    """Summary view of a document for listing purposes.

    Attributes:
        id: Document identifier.
        filename: Original filename.
        file_type: Detected file type.
        status: Current processing status.
        chunk_count: Number of chunks generated.
        created_at: Upload timestamp.
    """

    id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int
    created_at: datetime


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Application health status.
        version: Application version string.
    """

    status: str = "healthy"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """Standardized error response.

    Attributes:
        error: Error type identifier.
        message: Human-readable error description.
        detail: Optional additional detail.
    """

    error: str
    message: str
    detail: str | None = None
