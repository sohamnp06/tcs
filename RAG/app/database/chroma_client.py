"""
ChromaDB client initialization and collection management.

Design Decision:
    - ChromaDB client is a singleton (lru_cache) so one persistent
      connection is reused across all requests.
    - PersistentClient is used (not in-memory) so embeddings survive
      server restarts — essential for a real retrieval service.
    - get_or_create_collection() is idempotent: safe to call on every
      startup without creating duplicates.
    - cosine distance is used because our embeddings are L2-normalized
      (normalize_embeddings=True in the embedder), making cosine distance
      the most meaningful similarity metric.
"""

from functools import lru_cache

import chromadb
from chromadb import Collection, PersistentClient

from app.core.config import get_settings
from app.core.exceptions import ChromaDBException
from app.core.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client() -> PersistentClient:
    """
    Return the singleton ChromaDB persistent client.

    The client is created once and cached for the process lifetime.
    In tests, call get_chroma_client.cache_clear() to reset.

    Returns:
        PersistentClient: Connected ChromaDB client.

    Raises:
        ChromaDBException: If the client cannot be initialized.
    """
    settings = get_settings()
    persist_dir = str(settings.chroma_path)

    logger.info("Initializing ChromaDB persistent client at: %s", persist_dir)
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB client initialized successfully.")
        return client
    except Exception as exc:
        raise ChromaDBException(
            message="Failed to initialize ChromaDB client.",
            detail=str(exc),
        ) from exc


def get_or_create_collection(collection_name: str | None = None) -> Collection:
    """
    Get or create the ChromaDB collection for resume embeddings.

    Uses cosine distance because embeddings are L2-normalized.
    Safe to call on every request — idempotent.

    Args:
        collection_name: Optional collection name override. Defaults to
                         the value from settings.

    Returns:
        Collection: The ChromaDB collection instance.

    Raises:
        ChromaDBException: If collection creation/retrieval fails.
    """
    settings = get_settings()
    name = collection_name or settings.chroma_collection_name
    client = get_chroma_client()

    try:
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("Collection '%s' ready (%d items).", name, collection.count())
        return collection
    except Exception as exc:
        raise ChromaDBException(
            message=f"Failed to get or create ChromaDB collection '{name}'.",
            detail=str(exc),
        ) from exc
