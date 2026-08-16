"""Hybrid retrieval combining semantic and lexical search with rank fusion."""

from backend.core.logging import get_logger
from backend.models.chunks import ScoredChunk

logger = get_logger(__name__)


class HybridRetriever:
    """Combines semantic (vector) and lexical (BM25) search results using Reciprocal Rank Fusion.

    RRF is a rank-based fusion method that does not require score normalization,
    making it robust across different scoring distributions from vector and BM25 search.
    """

    def __init__(self, rrf_k: int = 60, semantic_weight: float = 0.6) -> None:
        """Initialize the hybrid retriever.

        Args:
            rrf_k: The RRF constant (default 60, as per the original paper).
            semantic_weight: Weight given to semantic results vs lexical (0.0-1.0).
        """
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight

    def fuse(
        self,
        semantic_results: list[ScoredChunk],
        lexical_results: list[ScoredChunk],
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """Fuse semantic and lexical search results using weighted Reciprocal Rank Fusion.

        Each chunk receives an RRF score based on its rank in each result list,
        weighted by the configured semantic/lexical balance.

        Args:
            semantic_results: Results from vector similarity search (ordered by score desc).
            lexical_results: Results from BM25 search (ordered by score desc).
            top_k: Maximum number of fused results to return.

        Returns:
            List of ScoredChunk with RRF fusion scores, ordered by descending score.
        """
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, ScoredChunk] = {}

        # Score semantic results using RRF
        for rank, scored_chunk in enumerate(semantic_results):
            chunk_id = scored_chunk.chunk.id
            rrf_score = self.semantic_weight * (1.0 / (self.rrf_k + rank + 1))
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + rrf_score
            chunk_map[chunk_id] = scored_chunk

        # Score lexical results using RRF
        for rank, scored_chunk in enumerate(lexical_results):
            chunk_id = scored_chunk.chunk.id
            rrf_score = self.lexical_weight * (1.0 / (self.rrf_k + rank + 1))
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + rrf_score
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = scored_chunk

        # Sort by fused score and return top_k
        sorted_ids = sorted(chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True)
        top_ids = sorted_ids[:top_k]

        results = [
            ScoredChunk(chunk=chunk_map[chunk_id].chunk, score=chunk_scores[chunk_id]) for chunk_id in top_ids
        ]

        logger.debug(
            "Hybrid fusion complete",
            semantic_count=len(semantic_results),
            lexical_count=len(lexical_results),
            fused_count=len(results),
        )

        return results
