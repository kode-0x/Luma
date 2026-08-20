"""Cross-encoder reranking using LangChain's HuggingFaceCrossEncoder."""

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document as LCDocument

from backend.core.logging import get_logger
from backend.models.chunks import ScoredChunk

logger = get_logger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model via LangChain.

    Cross-encoders jointly encode the query and each candidate passage,
    producing more accurate relevance scores than bi-encoder similarity.
    The trade-off is higher latency, so reranking is applied only to
    the top candidates from the initial retrieval stage.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, top_n: int = 5) -> None:
        """Initialize the reranker.

        Args:
            model_name: Hugging Face model identifier for the cross-encoder.
            top_n: Default number of top results to return.
        """
        self.model_name = model_name
        self._top_n = top_n
        self._cross_encoder: HuggingFaceCrossEncoder | None = None

    @property
    def cross_encoder(self) -> HuggingFaceCrossEncoder:
        """Lazily load and cache the cross-encoder model.

        Returns:
            The loaded HuggingFaceCrossEncoder instance.
        """
        if self._cross_encoder is None:
            logger.info("Loading cross-encoder reranker model", model=self.model_name)
            self._cross_encoder = HuggingFaceCrossEncoder(model_name=self.model_name)
            logger.info("Cross-encoder reranker model loaded")
        return self._cross_encoder

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        """Rerank chunks by computing cross-encoder relevance scores.

        Args:
            query: The user's original query.
            chunks: Candidate chunks from the retrieval stage.
            top_k: Number of top results to return after reranking.

        Returns:
            List of ScoredChunk reordered and truncated by cross-encoder score.
        """
        if not chunks:
            return []

        # Score pairs using the cross-encoder directly for score access
        pairs = [[query, chunk.chunk.content] for chunk in chunks]
        scores = self.cross_encoder.score(pairs)

        # Pair scores with chunks
        scored_results: list[tuple[ScoredChunk, float]] = []
        for chunk, score in zip(chunks, scores, strict=True):
            scored_results.append((chunk, float(score)))

        # Sort by cross-encoder score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k with updated scores
        results = [
            ScoredChunk(chunk=original.chunk, score=rerank_score) for original, rerank_score in scored_results[:top_k]
        ]

        logger.debug(
            "Reranking complete",
            input_count=len(chunks),
            output_count=len(results),
            top_score=results[0].score if results else 0.0,
        )

        return results

    def rerank_documents(self, query: str, documents: list[LCDocument], top_k: int = 5) -> list[LCDocument]:
        """Rerank LangChain documents using the cross-encoder directly.

        This method is useful for integration with LangChain chains and pipelines.

        Args:
            query: The user's original query.
            documents: LangChain documents to rerank.
            top_k: Number of top results to return.

        Returns:
            Reranked list of LangChain documents.
        """
        if not documents:
            return []

        # Score all query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.cross_encoder.score(pairs)

        # Pair scores with documents and sort
        scored_docs = sorted(
            zip(documents, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        return [doc for doc, _ in scored_docs[:top_k]]
