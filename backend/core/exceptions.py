"""Application-wide exception hierarchy."""


class LumaError(Exception):
    """Base exception for all Luma application errors.

    Args:
        message: Human-readable error description.
    """

    def __init__(self, message: str = "An unexpected error occurred") -> None:
        self.message = message
        super().__init__(self.message)


# --- Document Errors ---


class DocumentNotFoundError(LumaError):
    """Raised when a requested document does not exist.

    Args:
        document_id: The identifier of the missing document.
    """

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")


class DocumentParsingError(LumaError):
    """Raised when a document cannot be parsed.

    Args:
        filename: The name of the file that failed parsing.
        reason: Description of why parsing failed.
    """

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to parse '{filename}': {reason}")


class UnsupportedFileTypeError(LumaError):
    """Raised when an uploaded file has an unsupported format.

    Args:
        filename: The name of the unsupported file.
        file_type: The detected file extension or MIME type.
    """

    def __init__(self, filename: str, file_type: str) -> None:
        self.filename = filename
        self.file_type = file_type
        super().__init__(f"Unsupported file type '{file_type}' for file '{filename}'")


class FileTooLargeError(LumaError):
    """Raised when an uploaded file exceeds the size limit.

    Args:
        filename: The name of the oversized file.
        size_mb: The file size in megabytes.
        max_size_mb: The maximum allowed size in megabytes.
    """

    def __init__(self, filename: str, size_mb: float, max_size_mb: int) -> None:
        self.filename = filename
        self.size_mb = size_mb
        self.max_size_mb = max_size_mb
        super().__init__(f"File '{filename}' is {size_mb:.1f}MB, exceeds limit of {max_size_mb}MB")


# --- Retrieval Errors ---


class RetrievalError(LumaError):
    """Raised when the retrieval pipeline encounters an error."""

    def __init__(self, message: str = "Retrieval failed") -> None:
        super().__init__(message)


class EmbeddingError(LumaError):
    """Raised when text embedding fails.

    Args:
        message: Description of the embedding failure.
    """

    def __init__(self, message: str = "Embedding generation failed") -> None:
        super().__init__(message)


# --- Generation Errors ---


class GenerationError(LumaError):
    """Raised when the LLM generation step fails.

    Args:
        message: Description of the generation failure.
    """

    def __init__(self, message: str = "LLM generation failed") -> None:
        super().__init__(message)


class InsufficientEvidenceError(LumaError):
    """Raised when not enough evidence is found to answer a query.

    Args:
        query: The original user query.
    """

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__("Insufficient evidence found in uploaded documents to answer this query reliably")


# --- Vector Store Errors ---


class VectorStoreError(LumaError):
    """Raised when a vector store operation fails.

    Args:
        message: Description of the vector store failure.
    """

    def __init__(self, message: str = "Vector store operation failed") -> None:
        super().__init__(message)
