"""
Embedding model wrapper using sentence-transformers.

Design Decision:
    - The embedder is a class (not a function) so the model is loaded ONCE
      and reused across all requests — model loading takes ~2-5 seconds.
    - Singleton via module-level cached instance (_embedder_instance).
    - get_embedder() is the dependency injection entry point used by FastAPI
      and the vector store.
    - static-retrieval-mrl-en-v1 is a Matryoshka model: high accuracy at
      truncated dimensions. We use the full 1024-dim output for maximum recall.
    - Batch encoding is used for efficiency when storing multiple chunks.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.exceptions import EmbeddingException
from app.core.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Wrapper around a SentenceTransformer model.

    Attributes:
        model_name: The HuggingFace model identifier.
        _model: The loaded SentenceTransformer instance.
    """

    def __init__(self, model_name: str) -> None:
        """
        Load the embedding model from HuggingFace / local cache.

        Args:
            model_name: SentenceTransformer model name or path.

        Raises:
            EmbeddingException: If the model fails to load.
        """
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        try:
            self._model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            raise EmbeddingException(
                message=f"Failed to load embedding model '{model_name}'.",
                detail=str(exc),
            ) from exc

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            list[float]: Dense embedding vector.

        Raises:
            EmbeddingException: If embedding generation fails.
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        try:
            vector = self._model.encode(text, normalize_embeddings=True)
            return vector.tolist()
        except Exception as exc:
            raise EmbeddingException(
                message="Failed to generate embedding for text.",
                detail=str(exc),
            ) from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a batch of text strings.

        Batch encoding is significantly faster than encoding one-by-one
        for large numbers of chunks.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of dense embedding vectors.

        Raises:
            EmbeddingException: If batch embedding fails.
            ValueError: If texts list is empty or contains empty strings.
        """
        if not texts:
            raise ValueError("Cannot embed an empty batch.")

        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if empty_indices:
            raise ValueError(f"Empty strings found at indices: {empty_indices}")

        try:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            return [v.tolist() for v in vectors]
        except Exception as exc:
            raise EmbeddingException(
                message="Failed to generate batch embeddings.",
                detail=str(exc),
            ) from exc

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """
    Return the singleton Embedder instance.

    Uses lru_cache to ensure the model is loaded only once per process.
    In tests, call get_embedder.cache_clear() to reset.

    Returns:
        Embedder: Ready-to-use embedding model wrapper.
    """
    settings = get_settings()
    return Embedder(model_name=settings.embedding_model_name)
