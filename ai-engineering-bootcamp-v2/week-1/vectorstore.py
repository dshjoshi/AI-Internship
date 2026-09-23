"""Pinecone connection, shared embedding model, and a health check for /ask's RAG extension."""

import os

from openai import OpenAI
from pinecone import Pinecone

# Single source of truth so ingest and query can never drift onto different models or sizes.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1024  # truncated from the model's native 1536 to match the "glorious-palm" Pinecone index

_pinecone_client: Pinecone | None = None
_openai_client: OpenAI | None = None


def _get_pinecone_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return _pinecone_client


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _openai_client


def get_index():
    """Return a handle to the configured Pinecone index."""
    pc = _get_pinecone_client()
    return pc.Index(os.environ["PINECONE_INDEX_NAME"])


def embed_text(text: str) -> list[float]:
    """Embed one string with the shared model — call this at both ingest and query time."""
    response = _get_openai_client().embeddings.create(
        model=EMBEDDING_MODEL, input=text, dimensions=EMBEDDING_DIMENSION
    )
    return response.data[0].embedding


def pinecone_health() -> dict:
    """Confirm Pinecone is reachable and the configured index is usable."""
    index_name = os.environ.get("PINECONE_INDEX_NAME", "<unset>")
    try:
        stats = get_index().describe_index_stats()
        return {
            "ok": True,
            "index": index_name,
            "dimension": stats.dimension,
            "total_vectors": stats.total_vector_count,
        }
    except Exception as exc:
        return {"ok": False, "index": index_name, "error": str(exc)}
