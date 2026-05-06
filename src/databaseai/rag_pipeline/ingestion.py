"""
RAG Pipeline — Ingestion
=========================

Chunks movie knowledge documents and stores them in the vector DB.

DB Architect notes:
  - Chunking strategy: 200-500 token chunks with 50-token overlap
  - Chunk size is a hyperparameter: larger = more context per result,
    smaller = more precise retrieval
  - Overlap prevents splitting a sentence across chunks, losing context
  - Each chunk stores its source document ID as metadata (provenance)
  - Deduplication: hash-based IDs prevent re-indexing unchanged content
"""

import hashlib
from typing import Optional
import chromadb
from ..vector_db.movie_vectors import DeterministicEmbeddingFn


class RAGIngestion:
    """
    Chunks and indexes movie knowledge for RAG retrieval.

    Real-world parallel:
      A customer support chatbot chunks the entire product documentation,
      FAQs, and changelog into a vector DB. When a user asks a question,
      the relevant chunks are retrieved and injected into the LLM prompt.
    """

    COLLECTION = "movie_knowledge"
    CHUNK_SIZE = 200          # characters (use tokens in production)
    CHUNK_OVERLAP = 40

    def __init__(self, persist_dir: Optional[str] = None):
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

        self._embed_fn = DeterministicEmbeddingFn()
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_document(self, doc_id: str, title: str, content: str, metadata: dict = None) -> int:
        """
        Chunk a document and store all chunks.
        Returns number of chunks created.
        """
        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = self._chunk_id(doc_id, i, chunk)
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({
                "doc_id": doc_id,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {}),
            })

        self._col.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(chunks)

    def ingest_batch(self, documents: list[dict]) -> dict:
        """
        Ingest multiple documents.
        Each dict: {id, title, content, metadata (optional)}
        """
        total_chunks = 0
        results = {}
        for doc in documents:
            n = self.ingest_document(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                metadata=doc.get("metadata"),
            )
            results[doc["id"]] = n
            total_chunks += n
        return {"documents_ingested": len(documents), "total_chunks": total_chunks, "per_doc": results}

    def chunk_count(self) -> int:
        return self._col.count()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """Simple character-level chunking with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    @staticmethod
    def _chunk_id(doc_id: str, index: int, content: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk{index}_{content_hash}"

    def get_collection(self):
        return self._col
