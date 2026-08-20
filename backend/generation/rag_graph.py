"""LangGraph-based RAG workflow for chat orchestration.

Defines a stateful graph with nodes for:
1. Retrieval (semantic + lexical)
2. Hybrid fusion (RRF)
3. Reranking (cross-encoder)
4. Context building
5. Answer generation (LLM)

This replaces the manual orchestration previously done in ChatService.
"""

from collections.abc import AsyncGenerator
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.core.logging import get_logger
from backend.generation.context_builder import ContextBuilder
from backend.generation.llm_client import LLMClient
from backend.generation.prompts import INSUFFICIENT_EVIDENCE_RESPONSE, RAG_PROMPT
from backend.ingestion.embedder import EmbeddingService
from backend.models.chunks import ScoredChunk
from backend.models.queries import Citation
from backend.repository.vector_store import QdrantVectorStore
from backend.retrieval.bm25_search import BM25Searcher
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)

MINIMUM_EVIDENCE_SCORE = 0.1


class RAGState(TypedDict):
    """State flowing through the RAG graph.

    Attributes:
        query: The user's natural language question.
        document_ids: Optional document scope filter.
        model: Optional model override for generation.
        query_embedding: The embedded query vector.
        semantic_results: Results from vector similarity search.
        lexical_results: Results from BM25 search.
        fused_results: Results after hybrid RRF fusion.
        reranked_results: Final evidence after cross-encoder reranking.
        context: Assembled context string for the prompt.
        answer: The generated answer text.
        citations: Extracted citations from evidence.
        is_sufficient: Whether enough evidence was found.
    """

    query: str
    document_ids: list[str] | None
    model: str | None
    query_embedding: list[float]
    semantic_results: list[ScoredChunk]
    lexical_results: list[ScoredChunk]
    fused_results: list[ScoredChunk]
    reranked_results: list[ScoredChunk]
    context: str
    answer: str
    citations: list[Citation]
    is_sufficient: bool


class RAGGraph:
    """LangGraph-powered RAG workflow.

    Encapsulates the full query pipeline as a compiled state graph,
    providing clear node boundaries for observability, testing, and
    potential future branching logic (e.g., query routing, multi-hop).
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: QdrantVectorStore,
        bm25_searcher: BM25Searcher,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm_client: LLMClient,
        context_builder: ContextBuilder | None = None,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> None:
        """Initialize the RAG graph with all required components.

        Args:
            embedder: Service for generating query embeddings.
            vector_store: Vector store for semantic search.
            bm25_searcher: BM25 index for lexical search.
            hybrid_retriever: RRF fusion component.
            reranker: Cross-encoder for reranking candidates.
            llm_client: LLM client for answer generation.
            context_builder: Optional custom context builder.
            retrieval_top_k: Number of candidates from each retrieval method.
            rerank_top_k: Number of results after reranking.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_searcher = bm25_searcher
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._llm_client = llm_client
        self._context_builder = context_builder or ContextBuilder()
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph state graph.

        The graph follows this flow:
        embed_query -> semantic_search -> lexical_search -> fuse -> rerank -> check_evidence
            -> (sufficient) -> build_context -> generate -> END
            -> (insufficient) -> insufficient_evidence -> END

        Returns:
            Compiled LangGraph graph.
        """
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("embed_query", self._embed_query)
        workflow.add_node("semantic_search", self._semantic_search)
        workflow.add_node("lexical_search", self._lexical_search)
        workflow.add_node("fuse_results", self._fuse_results)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("check_evidence", self._check_evidence)
        workflow.add_node("build_context", self._build_context)
        workflow.add_node("generate", self._generate)
        workflow.add_node("insufficient_evidence", self._insufficient_evidence)

        # Define edges
        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "semantic_search")
        workflow.add_edge("semantic_search", "lexical_search")
        workflow.add_edge("lexical_search", "fuse_results")
        workflow.add_edge("fuse_results", "rerank")
        workflow.add_edge("rerank", "check_evidence")

        # Conditional: check if evidence is sufficient
        workflow.add_conditional_edges(
            "check_evidence",
            self._route_evidence,
            {
                "sufficient": "build_context",
                "insufficient": "insufficient_evidence",
            },
        )

        workflow.add_edge("build_context", "generate")
        workflow.add_edge("generate", END)
        workflow.add_edge("insufficient_evidence", END)

        return workflow.compile()

    # --- Node implementations ---

    def _embed_query(self, state: RAGState) -> dict[str, Any]:
        """Embed the user query for vector search."""
        query_embedding = self._embedder.embed_text(state["query"])
        return {"query_embedding": query_embedding}

    def _semantic_search(self, state: RAGState) -> dict[str, Any]:
        """Perform semantic vector search."""
        results = self._vector_store.search(
            query_vector=state["query_embedding"],
            top_k=self._retrieval_top_k,
            document_ids=state.get("document_ids"),
        )
        return {"semantic_results": results}

    def _lexical_search(self, state: RAGState) -> dict[str, Any]:
        """Perform BM25 lexical search."""
        results = self._bm25_searcher.search(
            query=state["query"],
            top_k=self._retrieval_top_k,
            document_ids=state.get("document_ids"),
        )
        return {"lexical_results": results}

    def _fuse_results(self, state: RAGState) -> dict[str, Any]:
        """Fuse semantic and lexical results using RRF."""
        fused = self._hybrid_retriever.fuse(
            semantic_results=state["semantic_results"],
            lexical_results=state["lexical_results"],
            top_k=self._retrieval_top_k,
        )
        return {"fused_results": fused}

    def _rerank(self, state: RAGState) -> dict[str, Any]:
        """Rerank fused results with cross-encoder."""
        if state["fused_results"]:
            reranked = self._reranker.rerank(
                query=state["query"],
                chunks=state["fused_results"],
                top_k=self._rerank_top_k,
            )
        else:
            reranked = []
        return {"reranked_results": reranked}

    def _check_evidence(self, state: RAGState) -> dict[str, Any]:
        """Check if sufficient evidence was found."""
        relevant = [e for e in state["reranked_results"] if e.score >= MINIMUM_EVIDENCE_SCORE]
        is_sufficient = len(relevant) > 0
        # Update reranked_results to only include relevant evidence
        return {"reranked_results": relevant, "is_sufficient": is_sufficient}

    def _route_evidence(self, state: RAGState) -> str:
        """Route based on evidence sufficiency."""
        if state.get("is_sufficient", False):
            return "sufficient"
        return "insufficient"

    def _build_context(self, state: RAGState) -> dict[str, Any]:
        """Build the context string from reranked evidence."""
        context = self._context_builder.build(state["reranked_results"])
        return {"context": context}

    def _generate(self, state: RAGState) -> dict[str, Any]:
        """Generate answer — this is a sync placeholder; actual generation is async."""
        # This node is used for graph structure; actual async generation
        # is handled in the invoke/stream methods below
        citations = self._extract_citations(state["reranked_results"])
        return {"citations": citations}

    def _insufficient_evidence(self, state: RAGState) -> dict[str, Any]:
        """Handle insufficient evidence case."""
        return {
            "answer": INSUFFICIENT_EVIDENCE_RESPONSE,
            "citations": [],
            "is_sufficient": False,
        }

    # --- Public interface ---

    async def ainvoke(
        self,
        query: str,
        document_ids: list[str] | None = None,
        model: str | None = None,
    ) -> tuple[str, list[Citation]]:
        """Execute the full RAG pipeline and return answer with citations.

        Args:
            query: The user's natural language question.
            document_ids: Optional document scope filter.
            model: Optional model override.

        Returns:
            Tuple of (answer_text, list_of_citations).
        """
        logger.info("RAG graph invoked", query=query[:100], document_ids=document_ids)

        initial_state: dict[str, Any] = {
            "query": query,
            "document_ids": document_ids,
            "model": model,
            "query_embedding": [],
            "semantic_results": [],
            "lexical_results": [],
            "fused_results": [],
            "reranked_results": [],
            "context": "",
            "answer": "",
            "citations": [],
            "is_sufficient": False,
        }

        # Run the graph up to the generate node to get context and evidence
        # We use the graph for retrieval orchestration and do async generation separately
        final_state = await self._graph.ainvoke(initial_state)

        # If insufficient evidence, return early
        if not final_state.get("is_sufficient", False):
            return final_state["answer"], final_state["citations"]

        # Perform async LLM generation (outside the sync graph nodes)
        from langchain_core.output_parsers import StrOutputParser

        llm = self._llm_client._get_llm(model)
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = await chain.ainvoke({"context": final_state["context"], "query": query})

        citations = final_state["citations"]

        logger.info(
            "RAG graph complete",
            answer_length=len(answer),
            citation_count=len(citations),
        )

        return answer.strip(), citations

    async def astream(
        self,
        query: str,
        document_ids: list[str] | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute retrieval pipeline and stream the answer generation.

        Args:
            query: The user's natural language question.
            document_ids: Optional document scope filter.
            model: Optional model override.

        Yields:
            Text tokens as they are generated.
        """
        logger.info("RAG graph streaming invoked", query=query[:100])

        initial_state: dict[str, Any] = {
            "query": query,
            "document_ids": document_ids,
            "model": model,
            "query_embedding": [],
            "semantic_results": [],
            "lexical_results": [],
            "fused_results": [],
            "reranked_results": [],
            "context": "",
            "answer": "",
            "citations": [],
            "is_sufficient": False,
        }

        # Run the graph for retrieval/reranking
        final_state = await self._graph.ainvoke(initial_state)

        # If insufficient evidence, yield the message
        if not final_state.get("is_sufficient", False):
            yield INSUFFICIENT_EVIDENCE_RESPONSE
            return

        # Stream LLM generation
        from langchain_core.output_parsers import StrOutputParser

        llm = self._llm_client._get_llm(model)
        chain = RAG_PROMPT | llm | StrOutputParser()

        async for token in chain.astream({"context": final_state["context"], "query": query}):
            if token:
                yield token

    def _extract_citations(self, evidence: list[ScoredChunk]) -> list[Citation]:
        """Convert scored chunks into deduplicated citation objects.

        Args:
            evidence: The evidence chunks used for generation.

        Returns:
            List of unique Citation objects with source information.
        """
        citations: list[Citation] = []
        seen: set[tuple[str, int | None, str]] = set()

        for scored_chunk in evidence:
            metadata = scored_chunk.chunk.metadata
            content = scored_chunk.chunk.content[:300]
            key = (metadata.document_id, metadata.page_number, content)

            if key in seen:
                continue
            seen.add(key)

            citation = Citation(
                document_id=metadata.document_id,
                filename=metadata.filename,
                page_number=metadata.page_number,
                content=content,
                score=scored_chunk.score,
            )
            citations.append(citation)

        return citations
