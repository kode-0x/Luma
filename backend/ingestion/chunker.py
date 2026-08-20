"""Text chunking using LangChain's RecursiveCharacterTextSplitter."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.logging import get_logger
from backend.ingestion.parser import ParsedDocument
from backend.models.chunks import ChunkMetadata, DocumentChunk

logger = get_logger(__name__)


class TextChunker:
    """Splits parsed documents into overlapping text chunks using LangChain.

    Uses RecursiveCharacterTextSplitter which intelligently splits on
    paragraph breaks, newlines, sentences, and words — in that priority order —
    to produce semantically coherent chunks.

    Attributes:
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        """Initialize the chunker with size parameters.

        Args:
            chunk_size: Target characters per chunk.
            chunk_overlap: Characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
            length_function=len,
        )

    def chunk_document(self, parsed: ParsedDocument, document_id: str) -> list[DocumentChunk]:
        """Split a parsed document into chunks with metadata.

        If the document has page-level text, chunks are created per page
        to preserve page attribution. Otherwise, the full text is chunked.

        Args:
            parsed: The parsed document output.
            document_id: Unique ID assigned to this document.

        Returns:
            List of DocumentChunk instances with metadata.
        """
        chunks: list[DocumentChunk] = []

        if parsed.pages:
            for page_index, page_text in enumerate(parsed.pages):
                if not page_text.strip():
                    continue
                page_chunks = self._splitter.split_text(page_text)
                for chunk_text in page_chunks:
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata=ChunkMetadata(
                            document_id=document_id,
                            filename=parsed.filename,
                            page_number=page_index + 1,
                            chunk_index=len(chunks),
                        ),
                    )
                    chunks.append(chunk)
        else:
            text_chunks = self._splitter.split_text(parsed.text)
            for idx, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        filename=parsed.filename,
                        chunk_index=idx,
                    ),
                )
                chunks.append(chunk)

        logger.info("Chunked document", document_id=document_id, chunk_count=len(chunks))
        return chunks
