"""
RAG Pipeline — Retrieval
=========================

Retrieves relevant context chunks and builds a grounded prompt for an LLM.

DB Architect notes:
  - Retrieval = vector search (semantic) + optional keyword filter (lexical)
  - Hybrid search: combine vector similarity + BM25 score (best of both)
  - Re-ranking: a cross-encoder re-scores top-k candidates (more expensive)
  - Context window budget: summarize chunks if they exceed token limit
  - Without an LLM key, we return the retrieved context as the "answer"
    so the whole pipeline is demonstrable without any API key.
"""

import chromadb
from typing import Optional
from ..vector_db.movie_vectors import DeterministicEmbeddingFn


class RAGRetrieval:
    """
    Retrieves relevant document chunks and builds a grounded LLM prompt.

    In a production system, the built prompt is sent to an LLM API.
    Here we return the prompt + context so it works with any LLM.
    """

    COLLECTION = "movie_knowledge"
    MAX_CONTEXT_CHARS = 2000

    def __init__(self, ingestion_client=None):
        if ingestion_client:
            self._client = ingestion_client
        else:
            self._client = chromadb.EphemeralClient()

        self._embed_fn = DeterministicEmbeddingFn()
        try:
            self._col = self._client.get_collection(
                name=self.COLLECTION,
                embedding_function=self._embed_fn,
            )
        except Exception:
            self._col = None

    @classmethod
    def from_ingestion(cls, ingestion) -> "RAGRetrieval":
        """Share the same ChromaDB client as an RAGIngestion instance."""
        obj = cls.__new__(cls)
        obj._embed_fn = DeterministicEmbeddingFn()
        obj._col = ingestion.get_collection()
        return obj

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, n_chunks: int = 5) -> list[dict]:
        """
        Retrieve the n most relevant chunks for a question.

        Returns list of {text, title, doc_id, chunk_index, similarity}.
        """
        if not self._col or self._col.count() == 0:
            return []

        n_chunks = min(n_chunks, self._col.count())
        results = self._col.query(query_texts=[question], n_results=n_chunks)

        chunks = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            chunks.append({
                "text": results["documents"][0][i],
                "title": meta.get("title", ""),
                "doc_id": meta.get("doc_id", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "similarity": round(1 - dist, 4),
            })
        return chunks

    def build_prompt(self, question: str, n_chunks: int = 5) -> dict:
        """
        Build a grounded prompt for an LLM.

        Returns:
          {
            "question": str,
            "context_chunks": list[dict],
            "prompt": str,          ← ready to send to any LLM API
            "context_char_count": int,
          }

        DB Architect note:
          The prompt engineering here is the "retrieval-augmented" part.
          Without good context, even the best LLM hallucinates.
          With good context, a small LLM beats a large one on domain tasks.
        """
        chunks = self.retrieve(question, n_chunks)

        context_parts = []
        total_chars = 0
        used_chunks = []

        for chunk in chunks:
            text = f"[Source: {chunk['title']}]\n{chunk['text']}"
            if total_chars + len(text) > self.MAX_CONTEXT_CHARS:
                break
            context_parts.append(text)
            total_chars += len(text)
            used_chunks.append(chunk)

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

        prompt = (
            "You are a movie expert assistant. Answer the question using ONLY "
            "the context provided below. If the context does not contain enough "
            "information, say so honestly.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )

        return {
            "question": question,
            "context_chunks": used_chunks,
            "prompt": prompt,
            "context_char_count": total_chars,
        }

    def answer_without_llm(self, question: str, n_chunks: int = 5) -> dict:
        """
        Return the retrieved context as the answer — no LLM needed.

        Useful for: testing retrieval quality, showing what context
        would be injected, and running this project with zero API keys.
        """
        chunks = self.retrieve(question, n_chunks)

        if not chunks:
            summary = "No relevant information found in the knowledge base."
        else:
            parts = []
            for c in chunks:
                parts.append(f"From '{c['title']}' (relevance: {c['similarity']}):\n  {c['text'][:200]}...")
            summary = "\n\n".join(parts)

        return {
            "question": question,
            "retrieved_chunks": len(chunks),
            "grounded_answer": summary,
            "note": "This is the retrieved context. Pass the built prompt to an LLM for a natural answer.",
        }
