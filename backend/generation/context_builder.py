"""Context builder: assembles retrieved chunks into a prompt-ready context block."""

from langchain_core.documents import Document as LCDocument

from backend.core.logging import get_logger
from backend.models.chunks import ScoredChunk

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 6000


class ContextBuilder:
    """Builds a structured context string from retrieved and reranked chunks.

    Formats each evidence chunk with source attribution (filename, page) so
    that the LLM can generate answers with inline citations.

    Attributes:
        max_context_chars: Maximum total characters to include in the context block.
    """

    def __init__(self, max_context_chars: int = MAX_CONTEXT_CHARS) -> None:
        """Initialize the context builder.

        Args:
            max_context_chars: Character budget for the assembled context.
        """
        self.max_context_chars = max_context_chars

    def build(self, scored_chunks: list[ScoredChunk]) -> str:
        """Assemble scored chunks into a numbered context string.

        Each evidence block is labeled with its source document and page.
        Chunks are included in order until the character budget is exhausted.

        Args:
            scored_chunks: Reranked chunks ordered by relevance (best first).

        Returns:
            Formatted context string ready for inclusion in a prompt.
        """
        if not scored_chunks:
            return ""

        blocks: list[str] = []
        total_chars = 0

        for idx, scored_chunk in enumerate(scored_chunks, start=1):
            metadata = scored_chunk.chunk.metadata
            source_label = metadata.filename
            if metadata.page_number:
                source_label += f" (Page {metadata.page_number})"

            block = f"[{idx}] Source: {source_label}\n{scored_chunk.chunk.content}"

            if total_chars + len(block) > self.max_context_chars:
                # Include partial block if we have room for at least half
                remaining = self.max_context_chars - total_chars
                if remaining > len(block) // 2:
                    blocks.append(block[:remaining])
                break

            blocks.append(block)
            total_chars += len(block)

        context = "\n\n".join(blocks)
        logger.debug("Built context", chunk_count=len(blocks), total_chars=len(context))
        return context

    def build_from_documents(self, documents: list[LCDocument]) -> str:
        """Build context from LangChain documents (for chain-native workflows).

        Args:
            documents: LangChain documents with metadata containing source info.

        Returns:
            Formatted context string ready for inclusion in a prompt.
        """
        if not documents:
            return ""

        blocks: list[str] = []
        total_chars = 0

        for idx, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            source_label = metadata.get("filename", "Unknown")
            page_number = metadata.get("page_number")
            if page_number:
                source_label += f" (Page {page_number})"

            block = f"[{idx}] Source: {source_label}\n{doc.page_content}"

            if total_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - total_chars
                if remaining > len(block) // 2:
                    blocks.append(block[:remaining])
                break

            blocks.append(block)
            total_chars += len(block)

        context = "\n\n".join(blocks)
        logger.debug("Built context from documents", chunk_count=len(blocks), total_chars=len(context))
        return context
