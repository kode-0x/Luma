"""Cross-encoder reranking for refining retrieval results."""

from backend.core.logging import get_logger
from backend.models.chunks import ScoredChunk

logger = get_logger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model.

    Cross-encoders jointly encode the query and each candidate passage,
    producing more accurate relevance scores than bi-encoder similarity.
    The trade-off is higher latency, so reranking is applied only to
    the top candidates from the initial retrieval stage.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        """Initialize the reranker.

        Args:
            model_name: Hugging Face model identifier for the cross-encoder.
        """
        self.model_name = model_name
        self._model: object | None = None

    @property
    def model(self) -> "CrossEncoder":
        """Lazily load and cache the cross-encoder model.

        Returns:
            The loaded CrossEncoder model instance.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker model", model=self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder reranker model loaded")
        return self._model  # type: ignore[return-value]

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

        # Prepare query-passage pairs for the cross-encoder
        pairs: list[list[str]] = [[query, chunk.chunk.content] for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)  # type: ignore[union-attr]

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


# Type hint for lazy import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder
