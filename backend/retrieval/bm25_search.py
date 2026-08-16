"""BM25 lexical search over document chunks."""

import re

from rank_bm25 import BM25Okapi

from backend.core.logging import get_logger
from backend.models.chunks import DocumentChunk, ScoredChunk

logger = get_logger(__name__)


class BM25Searcher:
    """BM25-based lexical search for keyword and exact-match retrieval.

    Maintains an in-memory BM25 index that can be rebuilt when
    documents are added or removed. Complements semantic vector
    search for hybrid retrieval.
    """

    def __init__(self) -> None:
        """Initialize the BM25 searcher with empty state."""
        self._index: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Build or rebuild the BM25 index from a list of chunks.

        Replaces the current index entirely with the provided chunks.

        Args:
            chunks: Document chunks to index.
        """
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]

        if self._tokenized_corpus:
            self._index = BM25Okapi(self._tokenized_corpus)
            logger.info("Built BM25 index", chunk_count=len(chunks))
        else:
            self._index = None
            logger.info("BM25 index is empty")

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to the existing index by rebuilding it.

        Args:
            chunks: Additional chunks to include.
        """
        self._chunks.extend(chunks)
        self._tokenized_corpus.extend([self._tokenize(chunk.content) for chunk in chunks])

        if self._tokenized_corpus:
            self._index = BM25Okapi(self._tokenized_corpus)
            logger.info("Rebuilt BM25 index after addition", total_chunks=len(self._chunks))

    def remove_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document and rebuild the index.

        Args:
            document_id: The document whose chunks should be removed.
        """
        original_count = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.metadata.document_id != document_id]

        if len(self._chunks) < original_count:
            self._tokenized_corpus = [self._tokenize(chunk.content) for chunk in self._chunks]
            self._index = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
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
        if self._index is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._index.get_scores(tokenized_query)

        # Pair chunks with scores and apply document filter
        scored_pairs: list[tuple[DocumentChunk, float]] = []
        for chunk, score in zip(self._chunks, scores, strict=True):
            if score <= 0:
                continue
            if document_ids and chunk.metadata.document_id not in document_ids:
                continue
            scored_pairs.append((chunk, float(score)))

        # Sort by score descending and limit
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_pairs[:top_k]

        return [ScoredChunk(chunk=chunk, score=score) for chunk, score in top_results]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words for BM25 indexing.

        Strips punctuation and splits on whitespace.

        Args:
            text: Raw text to tokenize.

        Returns:
            List of lowercase word tokens.
        """
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
