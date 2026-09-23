"""Pinecone connection, shared embedding model, and a health check for /ask's RAG extension."""

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

# Single source of truth so ingest and query can never drift onto different models or sizes.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1024  # truncated from the model's native 1536 to match the "glorious-palm" Pinecone index

# Chunking — env-configurable like the rest of this app's settings, with the requested defaults.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

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


def chunk_text(text: str) -> list[str]:
    """Split one document's text into overlapping chunks using CHUNK_SIZE / CHUNK_OVERLAP."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)


def upsert_chunks(document_id: str, chunks: list[str], source: str | None) -> int:
    """Embed and upsert a pre-made list of chunks under one document_id. Returns the count."""
    vectors = [
        {
            "id": f"{document_id}-{i}",
            "values": embed_text(chunk),
            "metadata": {
                "document_id": document_id,
                "chunk_index": i,
                "source": source or "",
                "text": chunk,  # kept so a later /ask query can show what was actually retrieved
            },
        }
        for i, chunk in enumerate(chunks)
    ]

    if vectors:
        get_index().upsert(vectors=vectors)
    return len(vectors)


def upsert_document(document_id: str, text: str, source: str | None) -> int:
    """Chunk (via CHUNK_SIZE/CHUNK_OVERLAP), embed, and upsert one document. Returns chunk count."""
    return upsert_chunks(document_id, chunk_text(text), source)


def query_similar(query_text: str, top_k: int = 5) -> list[dict]:
    """Embed a query and return its top-k nearest chunks — retrieval only, no LLM call."""
    matches = get_index().query(
        vector=embed_text(query_text), top_k=top_k, include_metadata=True
    )["matches"]

    return [
        {
            "score": match["score"],
            "document_id": match["metadata"].get("document_id"),
            "chunk_index": match["metadata"].get("chunk_index"),
            "source": match["metadata"].get("source"),
            "text": match["metadata"].get("text"),
        }
        for match in matches
    ]


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
