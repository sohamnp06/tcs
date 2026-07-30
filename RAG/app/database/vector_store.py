"""
Vector store operations — store and retrieve resume embeddings in ChromaDB.

Design Decision:
    - store_chunks() and retrieve_chunks() are the only two public functions.
      Single Responsibility: this module only handles ChromaDB I/O.
    - Metadata is stored alongside embeddings so retrieval results are
      self-contained — no need to join with another data store.
    - Batch upsert is used (not add) to make ingestion idempotent:
      re-ingesting the same resume updates rather than duplicates.
    - Distance scores from ChromaDB are in [0, 2] for cosine space.
      We convert to similarity via: similarity = 1 - (distance / 2).
"""

from app.core.config import get_settings
from app.core.exceptions import ChromaDBException, RetrievalException
from app.core.logger import get_logger
from app.database.chroma_client import get_or_create_collection
from app.embeddings.embedder import get_embedder
from app.models.resume import ResumeChunk, RetrievalResult

logger = get_logger(__name__)


def store_chunks(chunks: list[ResumeChunk]) -> int:
    """
    Embed and store a list of resume chunks in ChromaDB.

    Uses batch embedding for efficiency and upsert semantics for idempotency.
    Re-ingesting the same resume will update existing chunks rather than
    creating duplicates.

    Args:
        chunks: List of ResumeChunk objects to embed and store.

    Returns:
        int: Number of chunks successfully stored.

    Raises:
        ChromaDBException: If the storage operation fails.
        EmbeddingException: If embedding generation fails.
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("chunks list must not be empty.")

    collection = get_or_create_collection()
    embedder = get_embedder()

    texts = [chunk.text for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]
    metadatas = [
        {
            "resume_id": chunk.resume_id,
            "section": chunk.section.value,
            "char_count": str(chunk.char_count),
        }
        for chunk in chunks
    ]

    logger.info("Generating embeddings for %d chunk(s)...", len(chunks))
    embeddings = embedder.embed_batch(texts)

    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(
            "Stored %d chunk(s) in ChromaDB collection '%s'.",
            len(chunks),
            collection.name,
        )
        return len(chunks)
    except Exception as exc:
        raise ChromaDBException(
            message="Failed to store chunks in ChromaDB.",
            detail=str(exc),
        ) from exc


def retrieve_chunks(
    query: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    resume_id: str | None = None,
) -> list[RetrievalResult]:
    """
    Perform semantic search against stored resume embeddings.

    Converts the query to an embedding, queries ChromaDB, and returns
    the top-k results above the similarity threshold.

    Args:
        query: Natural language query (e.g. an interview question).
        top_k: Number of results to return. Defaults to settings value.
        similarity_threshold: Minimum similarity score [0.0, 1.0].
                              Defaults to settings value.
        resume_id: Optional filter to restrict results to a specific resume.

    Returns:
        list[RetrievalResult]: Ordered list of relevant chunks (best first).

    Raises:
        RetrievalException: If the query operation fails.
        EmbeddingException: If query embedding fails.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    threshold = similarity_threshold if similarity_threshold is not None else settings.retrieval_similarity_threshold

    collection = get_or_create_collection()
    embedder = get_embedder()

    logger.info("Retrieving top-%d chunks for query (threshold=%.2f).", k, threshold)

    query_embedding = embedder.embed(query)

    where_filter: dict | None = {"resume_id": resume_id} if resume_id else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, max(collection.count(), 1)),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RetrievalException(
            message="ChromaDB query failed.",
            detail=str(exc),
        ) from exc

    retrieved: list[RetrievalResult] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine distance ∈ [0, 2]. Convert to similarity ∈ [0, 1].
        similarity = max(0.0, 1.0 - (dist / 2.0))

        if similarity < threshold:
            logger.debug(
                "Chunk '%s' below threshold (score=%.3f). Skipping.", chunk_id, similarity
            )
            continue

        retrieved.append(
            RetrievalResult(
                chunk_id=chunk_id,
                resume_id=meta.get("resume_id", ""),
                section=meta.get("section", "general"),
                text=doc,
                score=round(similarity, 4),
            )
        )

    logger.info("Retrieval returned %d result(s) above threshold.", len(retrieved))
    return retrieved
