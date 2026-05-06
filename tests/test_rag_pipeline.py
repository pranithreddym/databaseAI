"""Tests for the RAG Pipeline (ingestion + retrieval)."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.rag_pipeline import RAGIngestion, RAGRetrieval
from databaseai.seed_data import MOVIES


class TestRAGIngestion:

    def test_ingest_single_document(self):
        ing = RAGIngestion()
        n = ing.ingest_document("doc1", "Test Movie", "A" * 500)
        assert n > 1  # long text should produce multiple chunks

    def test_ingest_short_document(self):
        ing = RAGIngestion()
        n = ing.ingest_document("doc1", "Short", "Short description")
        assert n == 1

    def test_ingest_batch(self, rag):
        ingestion, _ = rag
        assert ingestion.chunk_count() > len(MOVIES)  # multiple chunks per movie

    def test_chunk_ids_are_deterministic(self):
        ing1 = RAGIngestion()
        ing2 = RAGIngestion()
        ing1.ingest_document("d1", "Title", "Some content about movies")
        ing2.ingest_document("d1", "Title", "Some content about movies")
        assert ing1.chunk_count() == ing2.chunk_count()


class TestRAGRetrieval:

    def test_retrieve_returns_chunks(self, rag):
        _, retrieval = rag
        chunks = retrieval.retrieve("space travel wormhole astronauts", n_chunks=3)
        assert len(chunks) == 3

    def test_retrieve_includes_similarity(self, rag):
        _, retrieval = rag
        chunks = retrieval.retrieve("dream heist layers", n_chunks=5)
        for c in chunks:
            assert "similarity" in c
            assert 0.0 <= c["similarity"] <= 1.0

    def test_retrieve_includes_title(self, rag):
        _, retrieval = rag
        chunks = retrieval.retrieve("superhero Gotham crime", n_chunks=3)
        assert all("title" in c for c in chunks)

    def test_build_prompt_contains_question(self, rag):
        _, retrieval = rag
        result = retrieval.build_prompt("Who directed Inception?")
        assert "Who directed Inception?" in result["prompt"]
        assert "CONTEXT:" in result["prompt"]
        assert "QUESTION:" in result["prompt"]

    def test_build_prompt_has_context(self, rag):
        _, retrieval = rag
        result = retrieval.build_prompt("What is the plot of Interstellar?")
        assert result["context_char_count"] > 0
        assert len(result["context_chunks"]) > 0

    def test_answer_without_llm(self, rag):
        _, retrieval = rag
        result = retrieval.answer_without_llm("animated film about the spirit world")
        assert "question" in result
        assert "grounded_answer" in result
        assert result["retrieved_chunks"] > 0

    def test_empty_retrieval_graceful(self, tmp_path):
        # Use a unique persistent dir to get a truly isolated ChromaDB instance
        empty_ing = RAGIngestion(persist_dir=str(tmp_path / "empty_rag"))
        retrieval = RAGRetrieval.from_ingestion(empty_ing)
        chunks = retrieval.retrieve("anything")
        assert chunks == []

    def test_retrieve_respects_n_chunks(self, rag):
        _, retrieval = rag
        for n in [1, 3, 5]:
            chunks = retrieval.retrieve("movie", n_chunks=n)
            assert len(chunks) == n
