"""Chat service: orchestrates retrieval, reranking, and RAG generation for user queries."""

from collections.abc import AsyncGenerator

from backend.core.logging import get_logger
from backend.generation.llm_client import LLMClient
from backend.generation.rag_pipeline import RAGPipeline
from backend.ingestion.embedder import EmbeddingService
from backend.models.chunks import ScoredChunk
from backend.models.queries import ChatResponse, Citation
from backend.repository.vector_store import QdrantVectorStore
from backend.retrieval.bm25_search import BM25Searcher
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)


class ChatService:
    """Service layer for handling user chat queries.

    Orchestrates the full RAG pipeline: embedding the query, performing
    hybrid retrieval, reranking results, and generating a cited answer.
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: QdrantVectorStore,
        bm25_searcher: BM25Searcher,
        reranker: CrossEncoderReranker,
        llm_client: LLMClient,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> None:
        """Initialize the chat service.

        Args:
            embedder: Service for generating query embeddings.
            vector_store: Vector store for semantic search.
            bm25_searcher: BM25 index for lexical search.
            reranker: Cross-encoder for reranking candidates.
            llm_client: LLM client for answer generation.
            retrieval_top_k: Number of candidates from each retrieval method.
            rerank_top_k: Number of final results after reranking.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_searcher = bm25_searcher
        self._reranker = reranker
        self._llm_client = llm_client
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._hybrid_retriever = HybridRetriever()
        self._rag_pipeline = RAGPipeline(llm_client)

    async def query(self, query: str, document_ids: list[str] | None = None, model: str | None = None) -> ChatResponse:
        """Process a user query and return a complete answer with citations.

        Pipeline:
        1. Embed the query.
        2. Perform semantic search (vector store).
        3. Perform lexical search (BM25).
        4. Fuse results with Reciprocal Rank Fusion.
        5. Rerank with cross-encoder.
        6. Generate answer with citations using RAG.

        Args:
            query: The user's natural language question.
            document_ids: Optional document scope filter.

        Returns:
            ChatResponse with the answer and supporting citations.
        """
        logger.info("Processing query", query=query[:100], document_ids=document_ids)

        # Retrieve evidence
        evidence = self._retrieve_and_rerank(query, document_ids)

        # Generate answer
        answer, citations = await self._rag_pipeline.generate_answer(query, evidence, model=model)

        return ChatResponse(
            answer=answer,
            citations=citations,
            query=query,
        )

    async def query_stream(
        self, query: str, document_ids: list[str] | None = None, model: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response token-by-token for a user query.

        Performs the same retrieval pipeline as `query()` but streams
        the generation step for lower time-to-first-token.

        Args:
            query: The user's natural language question.
            document_ids: Optional document scope filter.

        Yields:
            Text tokens as they are generated.
        """
        logger.info("Processing streaming query", query=query[:100])

        # Retrieve evidence
        evidence = self._retrieve_and_rerank(query, document_ids)

        # Stream answer generation
        async for token in self._rag_pipeline.generate_answer_stream(query, evidence, model=model):
            yield token

    def get_citations_for_query(self, query: str, document_ids: list[str] | None = None) -> list[Citation]:
        """Retrieve citations/evidence without generating an answer.

        Useful for "show sources" functionality.

        Args:
            query: The user's question.
            document_ids: Optional document scope filter.

        Returns:
            List of citations from the retrieval pipeline.
        """
        evidence = self._retrieve_and_rerank(query, document_ids)
        return self._rag_pipeline._extract_citations(evidence)

    def _retrieve_and_rerank(
        self, query: str, document_ids: list[str] | None = None
    ) -> list[ScoredChunk]:
        """Run the full retrieval and reranking pipeline.

        Args:
            query: The search query.
            document_ids: Optional document scope filter.

        Returns:
            Reranked list of evidence chunks.
        """
        # Embed query for vector search
        query_embedding = self._embedder.embed_text(query)

        # Semantic search
        semantic_results = self._vector_store.search(
            query_vector=query_embedding,
            top_k=self._retrieval_top_k,
            document_ids=document_ids or None,
        )

        # Lexical search
        lexical_results = self._bm25_searcher.search(
            query=query,
            top_k=self._retrieval_top_k,
            document_ids=document_ids or None,
        )

        # Hybrid fusion
        fused_results = self._hybrid_retriever.fuse(
            semantic_results=semantic_results,
            lexical_results=lexical_results,
            top_k=self._retrieval_top_k,
        )

        # Rerank
        if fused_results:
            reranked = self._reranker.rerank(
                query=query,
                chunks=fused_results,
                top_k=self._rerank_top_k,
            )
        else:
            reranked = []

        logger.debug(
            "Retrieval complete",
            semantic_count=len(semantic_results),
            lexical_count=len(lexical_results),
            fused_count=len(fused_results),
            reranked_count=len(reranked),
        )

        return reranked
