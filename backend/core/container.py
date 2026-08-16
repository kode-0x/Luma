"""Dependency injection container for wiring application components."""

from backend.core.config import Settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class Container:
    """Application dependency container.

    Lazily constructs and holds references to all major services,
    repositories, and infrastructure components. Used as a single
    source of truth for dependency resolution at application startup.

    Attributes:
        settings: Application configuration.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the container with application settings.

        Args:
            settings: Application configuration instance.
        """
        self.settings = settings

        # Lazy-initialized components
        self._embedding_service: object | None = None
        self._vector_store: object | None = None
        self._document_repository: object | None = None
        self._document_parser: object | None = None
        self._chunker: object | None = None
        self._bm25_searcher: object | None = None
        self._reranker: object | None = None
        self._llm_client: object | None = None
        self._document_service: object | None = None
        self._chat_service: object | None = None

    @property
    def embedding_service(self) -> "EmbeddingService":
        """Get or create the embedding service."""
        if self._embedding_service is None:
            from backend.ingestion.embedder import EmbeddingService

            self._embedding_service = EmbeddingService(
                model_name=self.settings.embedding_model,
            )
            logger.info("Initialized embedding service", model=self.settings.embedding_model)
        return self._embedding_service  # type: ignore[return-value]

    @property
    def vector_store(self) -> "QdrantVectorStore":
        """Get or create the vector store client."""
        if self._vector_store is None:
            from backend.repository.vector_store import QdrantVectorStore

            self._vector_store = QdrantVectorStore(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
                collection_name=self.settings.qdrant_collection_name,
                embedding_dimension=self.settings.embedding_dimension,
            )
            logger.info("Initialized vector store", url=self.settings.qdrant_url)
        return self._vector_store  # type: ignore[return-value]

    @property
    def document_repository(self) -> "DocumentRepository":
        """Get or create the document repository."""
        if self._document_repository is None:
            from backend.repository.document_repository import DocumentRepository

            self._document_repository = DocumentRepository(
                upload_dir=self.settings.upload_dir,
            )
            logger.info("Initialized document repository")
        return self._document_repository  # type: ignore[return-value]

    @property
    def document_parser(self) -> "DocumentParser":
        """Get or create the document parser."""
        if self._document_parser is None:
            from backend.ingestion.parser import DocumentParser

            self._document_parser = DocumentParser()
            logger.info("Initialized document parser")
        return self._document_parser  # type: ignore[return-value]

    @property
    def chunker(self) -> "TextChunker":
        """Get or create the text chunker."""
        if self._chunker is None:
            from backend.ingestion.chunker import TextChunker

            self._chunker = TextChunker(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            logger.info("Initialized text chunker")
        return self._chunker  # type: ignore[return-value]

    @property
    def bm25_searcher(self) -> "BM25Searcher":
        """Get or create the BM25 search component."""
        if self._bm25_searcher is None:
            from backend.retrieval.bm25_search import BM25Searcher

            self._bm25_searcher = BM25Searcher()
            logger.info("Initialized BM25 searcher")
        return self._bm25_searcher  # type: ignore[return-value]

    @property
    def reranker(self) -> "CrossEncoderReranker":
        """Get or create the cross-encoder reranker."""
        if self._reranker is None:
            from backend.retrieval.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
            logger.info("Initialized cross-encoder reranker")
        return self._reranker  # type: ignore[return-value]

    @property
    def llm_client(self) -> "LLMClient":
        """Get or create the LLM client."""
        if self._llm_client is None:
            from backend.generation.llm_client import LLMClient

            self._llm_client = LLMClient(
                model_name=self.settings.llm_model,
                api_token=self.settings.openrouter_api_key,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
            )
            logger.info("Initialized LLM client", model=self.settings.llm_model)
        return self._llm_client  # type: ignore[return-value]

    @property
    def document_service(self) -> "DocumentService":
        """Get or create the document service."""
        if self._document_service is None:
            from backend.services.document_service import DocumentService

            self._document_service = DocumentService(
                parser=self.document_parser,
                chunker=self.chunker,
                embedder=self.embedding_service,
                vector_store=self.vector_store,
                repository=self.document_repository,
            )
            logger.info("Initialized document service")
        return self._document_service  # type: ignore[return-value]

    @property
    def chat_service(self) -> "ChatService":
        """Get or create the chat service."""
        if self._chat_service is None:
            from backend.services.chat_service import ChatService

            self._chat_service = ChatService(
                embedder=self.embedding_service,
                vector_store=self.vector_store,
                bm25_searcher=self.bm25_searcher,
                reranker=self.reranker,
                llm_client=self.llm_client,
                retrieval_top_k=self.settings.retrieval_top_k,
                rerank_top_k=self.settings.rerank_top_k,
            )
            logger.info("Initialized chat service")
        return self._chat_service  # type: ignore[return-value]


# Type hints for lazy properties (import only for type checking)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.generation.llm_client import LLMClient
    from backend.ingestion.chunker import TextChunker
    from backend.ingestion.embedder import EmbeddingService
    from backend.ingestion.parser import DocumentParser
    from backend.repository.document_repository import DocumentRepository
    from backend.repository.vector_store import QdrantVectorStore
    from backend.retrieval.bm25_search import BM25Searcher
    from backend.retrieval.reranker import CrossEncoderReranker
    from backend.services.chat_service import ChatService
    from backend.services.document_service import DocumentService
