"""Qdrant vector store integration using LangChain's Qdrant wrapper."""

from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore as LCQdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from backend.core.exceptions import VectorStoreError
from backend.core.logging import get_logger
from backend.models.chunks import ChunkMetadata, DocumentChunk, ScoredChunk

logger = get_logger(__name__)


class QdrantVectorStore:
    """Qdrant-backed vector store using LangChain's QdrantVectorStore.

    Provides both LangChain-native retriever access and direct operations
    for upsert, delete, and filtered search. Creates the collection on first use.

    Attributes:
        url: Qdrant server URL.
        collection_name: Name of the Qdrant collection.
        embedding_dimension: Dimensionality of stored vectors.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        embedding_dimension: int,
        api_key: str | None = None,
        embeddings: HuggingFaceEmbeddings | None = None,
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            url: Qdrant server URL. Use ":memory:" for in-memory mode (no server needed).
            collection_name: Name of the collection to use.
            embedding_dimension: Expected vector dimension.
            api_key: Optional API key for Qdrant authentication.
            embeddings: LangChain embeddings instance for retriever operations.
        """
        self.url = url
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._api_key = api_key
        self._embeddings = embeddings
        self._client: QdrantClient | None = None
        self._langchain_store: LCQdrantVectorStore | None = None

    @property
    def client(self) -> QdrantClient:
        """Lazily initialize and return the Qdrant client.

        Returns:
            An initialized QdrantClient instance.

        Raises:
            VectorStoreError: If the client cannot be initialized.
        """
        if self._client is None:
            try:
                if self.url == ":memory:":
                    self._client = QdrantClient(location=":memory:")
                else:
                    self._client = QdrantClient(
                        url=self.url,
                        api_key=self._api_key,
                    )
                self._ensure_collection()
                logger.info("Connected to Qdrant", url=self.url, collection=self.collection_name)
            except Exception as exc:
                raise VectorStoreError(f"Failed to connect to Qdrant at {self.url}: {exc}") from exc
        return self._client

    @property
    def langchain_store(self) -> LCQdrantVectorStore:
        """Get or create the LangChain QdrantVectorStore wrapper.

        This is used for retriever-based access patterns (similarity search).

        Returns:
            LangChain QdrantVectorStore instance.

        Raises:
            VectorStoreError: If the store cannot be initialized.
        """
        if self._langchain_store is None:
            if self._embeddings is None:
                raise VectorStoreError("Embeddings must be provided for LangChain store access")
            try:
                # Ensure client and collection are ready
                _ = self.client
                self._langchain_store = LCQdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self._embeddings,
                )
                logger.info("Initialized LangChain Qdrant store")
            except Exception as exc:
                raise VectorStoreError(f"Failed to initialize LangChain Qdrant store: {exc}") from exc
        return self._langchain_store

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist.

        Raises:
            VectorStoreError: If collection creation fails.
        """
        try:
            collections = self._client.get_collections().collections  # type: ignore[union-attr]
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self._client.create_collection(  # type: ignore[union-attr]
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection", collection=self.collection_name)

            # Ensure payload index exists for document_id filtering
            self._client.create_payload_index(  # type: ignore[union-attr]
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to ensure collection '{self.collection_name}': {exc}") from exc

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Store document chunks with their embeddings in Qdrant.

        Args:
            chunks: List of document chunks with embeddings populated.

        Raises:
            VectorStoreError: If the upsert operation fails.
        """
        if not chunks:
            return

        try:
            points = []
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning("Skipping chunk without embedding", chunk_id=chunk.id)
                    continue

                payload = {
                    "content": chunk.content,
                    "document_id": chunk.metadata.document_id,
                    "filename": chunk.metadata.filename,
                    "page_number": chunk.metadata.page_number,
                    "section": chunk.metadata.section,
                    "chunk_index": chunk.metadata.chunk_index,
                }

                points.append(
                    PointStruct(
                        id=chunk.id,
                        vector=chunk.embedding,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                logger.info("Upserted chunks to Qdrant", count=len(points))

        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert {len(chunks)} chunks: {exc}") from exc

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        """Search for chunks similar to the query vector.

        Args:
            query_vector: The embedding vector of the search query.
            top_k: Maximum number of results to return.
            document_ids: Optional filter to restrict search to specific documents.

        Returns:
            List of ScoredChunk results ordered by descending similarity.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        try:
            query_filter = None
            if document_ids:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchAny(any=document_ids),
                        )
                    ]
                )

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )

            scored_chunks: list[ScoredChunk] = []
            for result in results.points:
                payload = result.payload or {}
                chunk = DocumentChunk(
                    id=str(result.id),
                    content=payload.get("content", ""),
                    metadata=ChunkMetadata(
                        document_id=payload.get("document_id", ""),
                        filename=payload.get("filename", ""),
                        page_number=payload.get("page_number"),
                        section=payload.get("section"),
                        chunk_index=payload.get("chunk_index", 0),
                    ),
                )
                scored_chunks.append(ScoredChunk(chunk=chunk, score=result.score))

            return scored_chunks

        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

    def similarity_search_with_filter(
        self,
        query: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        """Perform similarity search using the LangChain store (embeds query internally).

        This method uses LangChain's similarity_search_with_score which handles
        query embedding internally.

        Args:
            query: The text query to search for.
            top_k: Maximum number of results to return.
            document_ids: Optional filter to restrict search to specific documents.

        Returns:
            List of ScoredChunk results ordered by descending similarity.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        try:
            filter_dict = None
            if document_ids:
                filter_dict = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchAny(any=document_ids),
                        )
                    ]
                )

            results = self.langchain_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_dict,
            )

            scored_chunks: list[ScoredChunk] = []
            for doc, score in results:
                metadata = doc.metadata
                chunk = DocumentChunk(
                    id=metadata.get("_id", ""),
                    content=doc.page_content,
                    metadata=ChunkMetadata(
                        document_id=metadata.get("document_id", ""),
                        filename=metadata.get("filename", ""),
                        page_number=metadata.get("page_number"),
                        section=metadata.get("section"),
                        chunk_index=metadata.get("chunk_index", 0),
                    ),
                )
                scored_chunks.append(ScoredChunk(chunk=chunk, score=score))

            return scored_chunks

        except Exception as exc:
            raise VectorStoreError(f"LangChain similarity search failed: {exc}") from exc

    def as_retriever(self, top_k: int = 10, document_ids: list[str] | None = None) -> Any:
        """Get a LangChain retriever interface for this vector store.

        Args:
            top_k: Number of documents to retrieve.
            document_ids: Optional filter to restrict search to specific documents.

        Returns:
            A LangChain retriever instance.
        """
        search_kwargs: dict[str, Any] = {"k": top_k}

        if document_ids:
            search_kwargs["filter"] = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=document_ids),
                    )
                ]
            )

        return self.langchain_store.as_retriever(search_kwargs=search_kwargs)

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific document.

        Args:
            document_id: The document ID whose chunks should be removed.

        Raises:
            VectorStoreError: If the delete operation fails.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            logger.info("Deleted chunks from Qdrant", document_id=document_id)

        except Exception as exc:
            raise VectorStoreError(f"Failed to delete chunks for document '{document_id}': {exc}") from exc

    def get_all_chunks_by_document(self, document_id: str, limit: int = 1000) -> list[DocumentChunk]:
        """Retrieve all chunks belonging to a document (for BM25 indexing).

        Args:
            document_id: The document ID to retrieve chunks for.
            limit: Maximum number of chunks to retrieve.

        Returns:
            List of DocumentChunk instances (without embeddings).

        Raises:
            VectorStoreError: If the scroll operation fails.
        """
        try:
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_vectors=False,
            )

            chunks: list[DocumentChunk] = []
            for point in results:
                payload = point.payload or {}
                chunk = DocumentChunk(
                    id=str(point.id),
                    content=payload.get("content", ""),
                    metadata=ChunkMetadata(
                        document_id=payload.get("document_id", ""),
                        filename=payload.get("filename", ""),
                        page_number=payload.get("page_number"),
                        section=payload.get("section"),
                        chunk_index=payload.get("chunk_index", 0),
                    ),
                )
                chunks.append(chunk)

            return chunks

        except Exception as exc:
            raise VectorStoreError(f"Failed to retrieve chunks for document '{document_id}': {exc}") from exc
