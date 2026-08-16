"""Text embedding service using sentence-transformers."""

from backend.core.exceptions import EmbeddingError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Generates vector embeddings for text using sentence-transformers.

    Lazily loads the model on first use to avoid blocking application startup.

    Attributes:
        model_name: Name of the sentence-transformers model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the embedding service.

        Args:
            model_name: Hugging Face model identifier for sentence-transformers.
        """
        self.model_name = model_name
        self._model: object | None = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazily load and cache the sentence-transformer model.

        Returns:
            The loaded SentenceTransformer model instance.

        Raises:
            EmbeddingError: If the model fails to load.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model", model=self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as exc:
                raise EmbeddingError(f"Failed to load embedding model '{self.model_name}': {exc}") from exc
        return self._model  # type: ignore[return-value]

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
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()  # type: ignore[union-attr]
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each batch.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [embedding.tolist() for embedding in embeddings]  # type: ignore[union-attr]
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed batch of {len(texts)} texts: {exc}") from exc

    @property
    def dimension(self) -> int:
        """Get the embedding dimension of the loaded model.

        Returns:
            The dimensionality of the embedding vectors.
        """
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value, union-attr]


# Type hint for lazy import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
