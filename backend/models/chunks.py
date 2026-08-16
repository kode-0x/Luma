"""Domain models for document chunks and their metadata."""

import uuid

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata associated with a document chunk.

    Attributes:
        document_id: ID of the parent document.
        filename: Original filename of the source document.
        page_number: Page number where the chunk originates (1-indexed, None for non-paged formats).
        section: Section heading or label if detected.
        chunk_index: Sequential index of the chunk within the document.
    """

    document_id: str
    filename: str
    page_number: int | None = None
    section: str | None = None
    chunk_index: int = 0


class DocumentChunk(BaseModel):
    """A single chunk of text extracted from a document.

    Attributes:
        id: Unique identifier for the chunk.
        content: The text content of the chunk.
        metadata: Structured metadata about the chunk origin.
        embedding: Vector embedding of the content (None if not yet embedded).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class ScoredChunk(BaseModel):
    """A document chunk paired with a relevance score.

    Used as output from retrieval and reranking stages.

    Attributes:
        chunk: The retrieved document chunk.
        score: Relevance score (higher is more relevant).
    """

    chunk: DocumentChunk
    score: float
