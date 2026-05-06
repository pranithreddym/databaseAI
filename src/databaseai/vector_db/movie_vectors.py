"""
Vector Database — ChromaDB
==========================

Stores movie plot/description embeddings and enables semantic similarity search.

DB Architect notes:
  - Embeddings are dense float vectors (e.g. 384 dims from a sentence-transformer)
  - ChromaDB uses HNSW index: O(log n) ANN search, not exact but very close
  - Collections = tables; documents = rows; embeddings = the indexed column
  - Metadata filters run BEFORE vector search (pre-filter) to narrow the space
  - Distance metrics: cosine (direction), L2 (magnitude), inner product
"""

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from typing import Optional
import hashlib
import numpy as np


class DeterministicEmbeddingFn(EmbeddingFunction):
    """
    Hash-based pseudo-embeddings — no model download needed.

    Produces 384-dimensional vectors deterministically from text.
    Good enough for demonstrating vector search mechanics.
    In production use: all-MiniLM-L6-v2, text-embedding-3-small, etc.
    """

    DIMS = 384

    def __call__(self, input: list[str]) -> Embeddings:
        results = []
        for text in input:
            seed = int(hashlib.sha256(text.lower().encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.DIMS).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9  # unit-normalise for cosine
            results.append(vec.tolist())
        return results


class MovieVectorStore:
    """
    Vector store for movie plots and descriptions.

    Real-world parallel:
      Netflix stores content embeddings to power "Because you watched X"
      recommendations via approximate nearest-neighbour (ANN) search.
    """

    COLLECTION = "movies"

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

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert_movies(self, movies: list[dict]) -> None:
        """
        Insert or update movie records.

        Each dict must have: id (str), title, description.
        Optional: genre, year, director, rating.
        """
        ids, docs, metas = [], [], []
        for m in movies:
            ids.append(str(m["id"]))
            # The text we embed — richer text = better retrieval
            docs.append(f"{m['title']}. {m['description']}")
            metas.append({
                "title": m["title"],
                "genre": m.get("genre", "unknown"),
                "year": int(m.get("year", 0)),
                "director": m.get("director", "unknown"),
                "rating": float(m.get("rating", 0.0)),
            })

        self._col.upsert(ids=ids, documents=docs, metadatas=metas)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query: str,
        n: int = 5,
        genre_filter: Optional[str] = None,
        min_year: Optional[int] = None,
    ) -> list[dict]:
        """
        Semantic similarity search.

        The query is embedded and compared against all stored vectors.
        Returns the n most similar movies ordered by cosine similarity.
        """
        where: dict = {}
        if genre_filter:
            where["genre"] = {"$eq": genre_filter}
        if min_year:
            where["year"] = {"$gte": min_year}

        total = self.count()
        if total == 0:
            return []

        kwargs: dict = {"query_texts": [query], "n_results": min(n, total)}
        if where:
            kwargs["where"] = where

        results = self._col.query(**kwargs)

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            output.append({
                "id": doc_id,
                "title": meta["title"],
                "genre": meta["genre"],
                "year": meta["year"],
                "director": meta["director"],
                "rating": meta["rating"],
                "similarity_score": round(1 - dist, 4),  # cosine: dist=0 → sim=1
                "matched_text": results["documents"][0][i],
            })
        return output

    def get_by_id(self, movie_id: str) -> Optional[dict]:
        result = self._col.get(ids=[movie_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        return {"id": movie_id, "document": result["documents"][0], **meta}

    def count(self) -> int:
        return self._col.count()

    def delete_movie(self, movie_id: str) -> None:
        self._col.delete(ids=[movie_id])

    def reset(self) -> None:
        self._client.delete_collection(self.COLLECTION)
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
