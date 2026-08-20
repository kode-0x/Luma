"""Chat service: orchestrates RAG queries using the LangGraph RAG workflow."""

from collections.abc import AsyncGenerator

from backend.core.logging import get_logger
from backend.generation.rag_graph import RAGGraph
from backend.models.queries import ChatResponse, Citation

logger = get_logger(__name__)


class ChatService:
    """Service layer for handling user chat queries.

    Delegates the full RAG pipeline (retrieval, fusion, reranking, generation)
    to the LangGraph-based RAGGraph workflow.
    """

    def __init__(self, rag_graph: RAGGraph) -> None:
        """Initialize the chat service.

        Args:
            rag_graph: The LangGraph RAG workflow instance.
        """
        self._rag_graph = rag_graph

    async def query(self, query: str, document_ids: list[str] | None = None, model: str | None = None) -> ChatResponse:
        """Process a user query and return a complete answer with citations.

        Delegates to the LangGraph RAG workflow which handles:
        1. Query embedding
        2. Semantic search (vector store)
        3. Lexical search (BM25)
        4. Hybrid fusion (RRF)
        5. Cross-encoder reranking
        6. Context building and LLM generation

        Args:
            query: The user's natural language question.
            document_ids: Optional document scope filter.
            model: Optional model override.

        Returns:
            ChatResponse with the answer and supporting citations.
        """
        logger.info("Processing query", query=query[:100], document_ids=document_ids)

        answer, citations = await self._rag_graph.ainvoke(
            query=query,
            document_ids=document_ids,
            model=model,
        )

        return ChatResponse(
            answer=answer,
            citations=citations,
            query=query,
        )

    async def query_stream(
        self, query: str, document_ids: list[str] | None = None, model: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response token-by-token for a user query.

        Uses the LangGraph RAG workflow for retrieval, then streams
        the generation step for lower time-to-first-token.

        Args:
            query: The user's natural language question.
            document_ids: Optional document scope filter.
            model: Optional model override.

        Yields:
            Text tokens as they are generated.
        """
        logger.info("Processing streaming query", query=query[:100])

        async for token in self._rag_graph.astream(
            query=query,
            document_ids=document_ids,
            model=model,
        ):
            yield token

    def get_citations_for_query(self, query: str, document_ids: list[str] | None = None) -> list[Citation]:
        """Retrieve citations/evidence without generating an answer.

        Useful for "show sources" functionality. Runs the retrieval
        portion of the pipeline synchronously.

        Args:
            query: The user's question.
            document_ids: Optional document scope filter.

        Returns:
            List of citations from the retrieval pipeline.
        """
        # Perform retrieval steps manually for citation-only queries
        from backend.generation.rag_graph import MINIMUM_EVIDENCE_SCORE

        query_embedding = self._rag_graph._embedder.embed_text(query)

        semantic_results = self._rag_graph._vector_store.search(
            query_vector=query_embedding,
            top_k=self._rag_graph._retrieval_top_k,
            document_ids=document_ids,
        )

        lexical_results = self._rag_graph._bm25_searcher.search(
            query=query,
            top_k=self._rag_graph._retrieval_top_k,
            document_ids=document_ids,
        )

        fused = self._rag_graph._hybrid_retriever.fuse(
            semantic_results=semantic_results,
            lexical_results=lexical_results,
            top_k=self._rag_graph._retrieval_top_k,
        )

        if fused:
            reranked = self._rag_graph._reranker.rerank(
                query=query,
                chunks=fused,
                top_k=self._rag_graph._rerank_top_k,
            )
        else:
            reranked = []

        relevant = [e for e in reranked if e.score >= MINIMUM_EVIDENCE_SCORE]
        return self._rag_graph._extract_citations(relevant)
