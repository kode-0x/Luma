"""RAG pipeline: orchestrates context building, generation, and citation extraction."""

from collections.abc import AsyncGenerator

from backend.core.logging import get_logger
from backend.generation.context_builder import ContextBuilder
from backend.generation.llm_client import LLMClient
from backend.generation.prompts import INSUFFICIENT_EVIDENCE_RESPONSE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from backend.models.chunks import ScoredChunk
from backend.models.queries import Citation

logger = get_logger(__name__)

MINIMUM_EVIDENCE_SCORE = 0.1


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Coordinates context assembly, prompt construction, LLM generation,
    and citation extraction into a single coherent workflow.
    """

    def __init__(self, llm_client: LLMClient, context_builder: ContextBuilder | None = None) -> None:
        """Initialize the RAG pipeline.

        Args:
            llm_client: Client for LLM text generation.
            context_builder: Optional custom context builder. Uses default if not provided.
        """
        self._llm = llm_client
        self._context_builder = context_builder or ContextBuilder()

    async def generate_answer(self, query: str, evidence: list[ScoredChunk], model: str | None = None) -> tuple[str, list[Citation]]:
        """Generate a complete answer with citations from retrieved evidence.

        If evidence is insufficient (empty or all below threshold), returns
        a standard insufficient-evidence message.

        Args:
            query: The user's question.
            evidence: Reranked evidence chunks.

        Returns:
            Tuple of (answer_text, list_of_citations).
        """
        # Filter out very low-score evidence
        relevant_evidence = [e for e in evidence if e.score >= MINIMUM_EVIDENCE_SCORE]

        if not relevant_evidence:
            logger.info("Insufficient evidence for query", query=query[:100])
            return INSUFFICIENT_EVIDENCE_RESPONSE, []

        # Build context and prompt
        context = self._context_builder.build(relevant_evidence)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)

        # Generate answer
        answer = await self._llm.generate(SYSTEM_PROMPT, user_prompt, model_override=model)

        # Extract citations from the evidence used
        citations = self._extract_citations(relevant_evidence)

        logger.info("Generated answer", query_length=len(query), answer_length=len(answer), citation_count=len(citations))
        return answer, citations

    async def generate_answer_stream(
        self, query: str, evidence: list[ScoredChunk], model: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream an answer token-by-token from the LLM.

        Args:
            query: The user's question.
            evidence: Reranked evidence chunks.

        Yields:
            Text tokens as they are generated.
        """
        relevant_evidence = [e for e in evidence if e.score >= MINIMUM_EVIDENCE_SCORE]

        if not relevant_evidence:
            yield INSUFFICIENT_EVIDENCE_RESPONSE
            return

        context = self._context_builder.build(relevant_evidence)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)

        async for token in self._llm.generate_stream(SYSTEM_PROMPT, user_prompt, model_override=model):
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
