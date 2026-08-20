"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings populated from environment variables.

    Attributes:
        app_env: The deployment environment (development, staging, production).
        app_debug: Whether debug mode is enabled.
        app_host: Host address for the server.
        app_port: Port number for the server.
        openrouter_api_key: OpenRouter API key for LLM access.
        embedding_model: Name of the sentence-transformers embedding model.
        embedding_dimension: Dimensionality of the embedding vectors.
        llm_model: Name of the LLM model on OpenRouter.
        llm_max_tokens: Maximum tokens for LLM generation.
        llm_temperature: Sampling temperature for the LLM.
        qdrant_url: URL for the Qdrant vector database.
        qdrant_api_key: API key for Qdrant authentication.
        qdrant_collection_name: Name of the Qdrant collection for documents.
        retrieval_top_k: Number of candidates retrieved before reranking.
        rerank_top_k: Number of results after reranking.
        chunk_size: Maximum number of tokens per document chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        upload_dir: Directory for storing uploaded files.
        max_upload_size_mb: Maximum file upload size in megabytes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Hugging Face (embeddings only)
    hf_api_token: str = ""

    # OpenRouter (LLM)
    openrouter_api_key: str = ""

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # LLM
    llm_model: str = "meta-llama/llama-3.1-8b-instruct"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    # Qdrant (use ":memory:" for in-memory mode without a Qdrant server)
    qdrant_url: str = ":memory:"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "luma_documents"

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Upload
    upload_dir: Path = Path("./uploads")
    max_upload_size_mb: int = 50

    @property
    def is_production(self) -> bool:
        """Check if the application is running in production."""
        return self.app_env == "production"

    @property
    def max_upload_size_bytes(self) -> int:
        """Get the maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


def get_settings() -> Settings:
    """Create and return application settings.

    Returns:
        Populated Settings instance from environment variables.
    """
    return Settings()
