"""Text chunking: split documents into overlapping segments for embedding."""

from backend.core.logging import get_logger
from backend.ingestion.parser import ParsedDocument
from backend.models.chunks import ChunkMetadata, DocumentChunk

logger = get_logger(__name__)


class TextChunker:
    """Splits parsed documents into overlapping text chunks.

    Uses a character-based sliding window approach with configurable
    chunk size and overlap. Respects sentence boundaries where possible.

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
                page_chunks = self._split_text(page_text)
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
            text_chunks = self._split_text(parsed.text)
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

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks using sentence-aware boundaries.

        Attempts to break at sentence boundaries (periods followed by spaces)
        when possible, falling back to hard character splits.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        if not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to find a sentence boundary near the end
            boundary = self._find_sentence_boundary(text, start, end)
            chunk = text[start:boundary].strip()

            if chunk:
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            start = boundary - self.chunk_overlap
            if start <= (boundary - self.chunk_size):
                start = boundary

        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary near the target end position.

        Looks for sentence-ending punctuation (. ! ?) followed by whitespace
        within the last 20% of the chunk. Falls back to the hard end position.

        Args:
            text: The full text being chunked.
            start: Start index of the current chunk.
            end: Target end index.

        Returns:
            The chosen boundary index.
        """
        search_start = end - (self.chunk_size // 5)
        search_start = max(search_start, start)

        best_boundary = end
        for i in range(end, search_start, -1):
            if i < len(text) and text[i - 1] in ".!?" and (i >= len(text) or text[i] == " " or text[i] == "\n"):
                best_boundary = i
                break

        return best_boundary
