"""Text embedding service using LangChain HuggingFace embeddings."""

from langchain_huggingface import HuggingFaceEmbeddings

from backend.core.exceptions import EmbeddingError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Generates vector embeddings for text using LangChain's HuggingFace integration.

    Wraps langchain_huggingface.HuggingFaceEmbeddings to provide a consistent
    interface for both single-text and batch embedding operations.

    Attributes:
        model_name: Name of the sentence-transformers model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the embedding service.

        Args:
            model_name: Hugging Face model identifier for sentence-transformers.
        """
        self.model_name = model_name
        self._embeddings: HuggingFaceEmbeddings | None = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazily initialize and return the LangChain embeddings model.

        Returns:
            The HuggingFaceEmbeddings instance.

        Raises:
            EmbeddingError: If the model fails to load.
        """
        if self._embeddings is None:
            try:
                logger.info("Loading embedding model", model=self.model_name)
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("Embedding model loaded successfully")
            except Exception as exc:
                raise EmbeddingError(f"Failed to load embedding model '{self.model_name}': {exc}") from exc
        return self._embeddings

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each batch (used for chunked processing).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        try:
            # Process in batches to manage memory for large document sets
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed batch of {len(texts)} texts: {exc}") from exc

    @property
    def dimension(self) -> int:
        """Get the embedding dimension of the loaded model.

        Returns:
            The dimensionality of the embedding vectors.
        """
        # Embed a short text to determine dimension
        sample = self.embed_text("test")
        return len(sample)
