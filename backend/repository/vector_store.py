"""Qdrant vector store integration for storing and searching document embeddings."""

from typing import Any

from backend.core.exceptions import VectorStoreError
from backend.core.logging import get_logger
from backend.models.chunks import ChunkMetadata, DocumentChunk, ScoredChunk

logger = get_logger(__name__)


class QdrantVectorStore:
    """Qdrant-backed vector store for document chunk storage and retrieval.

    Manages a single Qdrant collection, handling upserts and similarity
    searches. Creates the collection on first use if it does not exist.

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
    ) -> None:
        """Initialize the Qdrant vector store client.

        Args:
            url: Qdrant server URL.
            collection_name: Name of the collection to use.
            embedding_dimension: Expected vector dimension.
            api_key: Optional API key for Qdrant authentication.
        """
        self.url = url
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._api_key = api_key
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        """Lazily initialize and return the Qdrant client.

        Returns:
            An initialized QdrantClient instance.

        Raises:
            VectorStoreError: If the client cannot be initialized.
        """
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(
                    url=self.url,
                    api_key=self._api_key,
                )
                self._ensure_collection()
                logger.info("Connected to Qdrant", url=self.url, collection=self.collection_name)
            except Exception as exc:
                raise VectorStoreError(f"Failed to connect to Qdrant at {self.url}: {exc}") from exc
        return self._client

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist.

        Raises:
            VectorStoreError: If collection creation fails.
        """
        try:
            from qdrant_client.models import Distance, VectorParams

            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection", collection=self.collection_name)

            # Ensure payload index exists for document_id filtering
            from qdrant_client.models import PayloadSchemaType

            self._client.create_payload_index(
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
            from qdrant_client.models import PointStruct

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
            from qdrant_client.models import FieldCondition, Filter, MatchAny

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

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )

            scored_chunks: list[ScoredChunk] = []
            for result in results:
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

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific document.

        Args:
            document_id: The document ID whose chunks should be removed.

        Raises:
            VectorStoreError: If the delete operation fails.
        """
        try:
            from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

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
            from qdrant_client.models import FieldCondition, Filter, MatchValue

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
            raise VectorStoreError(f"Failed to scroll chunks for document '{document_id}': {exc}") from exc
