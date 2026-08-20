"""BM25 lexical search using LangChain's BM25Retriever."""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument

from backend.core.logging import get_logger
from backend.models.chunks import ChunkMetadata, DocumentChunk, ScoredChunk

logger = get_logger(__name__)


class BM25Searcher:
    """BM25-based lexical search using LangChain's BM25Retriever.

    Maintains an in-memory BM25 index that can be rebuilt when
    documents are added or removed. Complements semantic vector
    search for hybrid retrieval.
    """

    def __init__(self) -> None:
        """Initialize the BM25 searcher with empty state."""
        self._retriever: BM25Retriever | None = None
        self._chunks: list[DocumentChunk] = []

    def _build_retriever(self, top_k: int = 10) -> None:
        """Build or rebuild the BM25 retriever from stored chunks.

        Args:
            top_k: Default number of results to return.
        """
        if not self._chunks:
            self._retriever = None
            return

        # Convert chunks to LangChain documents with metadata
        lc_docs = [
            LCDocument(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.id,
                    "document_id": chunk.metadata.document_id,
                    "filename": chunk.metadata.filename,
                    "page_number": chunk.metadata.page_number,
                    "section": chunk.metadata.section,
                    "chunk_index": chunk.metadata.chunk_index,
                },
            )
            for chunk in self._chunks
        ]

        self._retriever = BM25Retriever.from_documents(lc_docs, k=top_k)
        logger.info("Built BM25 retriever", chunk_count=len(self._chunks))

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Build or rebuild the BM25 index from a list of chunks.

        Replaces the current index entirely with the provided chunks.

        Args:
            chunks: Document chunks to index.
        """
        self._chunks = chunks
        self._build_retriever()

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to the existing index by rebuilding it.

        Args:
            chunks: Additional chunks to include.
        """
        self._chunks.extend(chunks)
        self._build_retriever()

    def remove_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document and rebuild the index.

        Args:
            document_id: The document whose chunks should be removed.
        """
        original_count = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.metadata.document_id != document_id]

        if len(self._chunks) < original_count:
            self._build_retriever()
            logger.info(
                "Rebuilt BM25 index after removal",
                document_id=document_id,
                removed=original_count - len(self._chunks),
            )

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        """Search the BM25 index for chunks matching the query.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            document_ids: Optional filter to restrict results to specific documents.

        Returns:
            List of ScoredChunk results ordered by descending BM25 score.
        """
        if self._retriever is None or not self._chunks:
            return []

        if not query.strip():
            return []

        # Update k for this search
        self._retriever.k = top_k * 2 if document_ids else top_k

        # Invoke the retriever
        results = self._retriever.invoke(query)

        # Convert back to ScoredChunks with document_id filtering
        scored_chunks: list[ScoredChunk] = []
        for rank, doc in enumerate(results):
            metadata = doc.metadata
            doc_id = metadata.get("document_id", "")

            # Apply document filter
            if document_ids and doc_id not in document_ids:
                continue

            chunk = DocumentChunk(
                id=metadata.get("chunk_id", ""),
                content=doc.page_content,
                metadata=ChunkMetadata(
                    document_id=doc_id,
                    filename=metadata.get("filename", ""),
                    page_number=metadata.get("page_number"),
                    section=metadata.get("section"),
                    chunk_index=metadata.get("chunk_index", 0),
                ),
            )

            # BM25Retriever doesn't return scores, assign rank-based score
            score = 1.0 / (rank + 1)
            scored_chunks.append(ScoredChunk(chunk=chunk, score=score))

            if len(scored_chunks) >= top_k:
                break

        return scored_chunks
